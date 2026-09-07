#!/usr/bin/env python3
"""Runtime helpers for replacement-only Fusion B over live CTC batches."""

from __future__ import annotations

from typing import Any, Callable
import unicodedata

import torch

from fusion_b_student_model import FusionBStudentConfig
from fusion_b_visual_top16_model import FusionBVisualTop16


_NON_TEXT_LETTERLIKE = set("ー〜～〰")


def is_text_character(character: str) -> bool:
    return (
        len(character) == 1
        and unicodedata.category(character)[0] in {"L", "N"}
        and character not in _NON_TEXT_LETTERLIKE
    )


def batch_ctc_candidates(
    probabilities: torch.Tensor, top_k: int = 16
) -> list[dict[str, torch.Tensor]]:
    """Return greedy CTC cells and their interval-max top-k character IDs."""
    values = probabilities.float()
    sums = values.sum(dim=-1)
    if bool((values < 0).any()) or not torch.allclose(
        sums, torch.ones_like(sums), rtol=1e-3, atol=1e-3
    ):
        values = values.softmax(dim=-1)
    greedy_probability, greedy_id = values.max(dim=-1)
    frame_values, frame_indices = values[..., 1:].topk(top_k, dim=-1)
    frame_indices = frame_indices + 1
    greedy_probability = greedy_probability.cpu()
    greedy_id = greedy_id.cpu()
    frame_values = frame_values.cpu()
    frame_indices = frame_indices.cpu()
    results = []
    for row_ids, row_scores, row_values, row_indices in zip(
        greedy_id,
        greedy_probability,
        frame_values,
        frame_indices,
        strict=True,
    ):
        runs: list[tuple[int, int]] = []
        ids = row_ids.tolist()
        scores = row_scores.tolist()
        start = 0
        while start < len(ids):
            token_id = ids[start]
            end = start + 1
            while end < len(ids) and ids[end] == token_id:
                end += 1
            if token_id != 0:
                centre = start + max(
                    range(end - start), key=lambda offset: scores[start + offset]
                )
                runs.append((token_id, centre))
            start = end
        greedy, candidate_ids = [], []
        for index, (token_id, centre) in enumerate(runs):
            left = 0 if index == 0 else (runs[index - 1][1] + centre) // 2 + 1
            right = (
                len(ids)
                if index + 1 == len(runs)
                else (centre + runs[index + 1][1]) // 2 + 1
            )
            maxima: dict[int, float] = {}
            for candidate, score in zip(
                row_indices[left:right].reshape(-1).tolist(),
                row_values[left:right].reshape(-1).tolist(),
                strict=True,
            ):
                maxima[candidate] = max(maxima.get(candidate, 0.0), score)
            ranked = sorted(maxima.items(), key=lambda item: (-item[1], item[0]))[
                :top_k
            ]
            if len(ranked) < top_k:
                raise RuntimeError("CTC interval yielded fewer than top-k candidates")
            greedy.append(token_id - 1)
            candidate_ids.append([item[0] - 1 for item in ranked])
        results.append(
            {
                "greedy_ids": torch.tensor(greedy, dtype=torch.long),
                "candidate_ids": torch.tensor(candidate_ids, dtype=torch.long).reshape(
                    -1, top_k
                ),
            }
        )
    return results


def _advance_levenshtein_row(
    row: list[int], emitted: str, target: str
) -> list[int]:
    current = row
    for character in emitted:
        following = [current[0] + 1]
        for index, target_character in enumerate(target, start=1):
            following.append(
                min(
                    current[index] + 1,
                    following[index - 1] + 1,
                    current[index - 1] + (character != target_character),
                )
            )
        current = following
    return current


def text_only_top16_oracle_distance(
    parts: list[dict[str, torch.Tensor]],
    characters: list[str],
    reference: str,
    normalizer: Callable[[str], str],
) -> tuple[int, int]:
    """Exact minimum ED through a per-cell text-only top16 candidate lattice."""
    cells: list[tuple[str, set[str]]] = []
    for part in parts:
        for greedy_id, candidate_ids in zip(
            part["greedy_ids"].tolist(),
            part["candidate_ids"].tolist(),
            strict=True,
        ):
            source = characters[greedy_id + 1]
            options = {source}
            if is_text_character(source):
                options.update(
                    candidate
                    for candidate in (
                        characters[candidate_id + 1]
                        for candidate_id in candidate_ids
                    )
                    if is_text_character(candidate)
                )
            cells.append((source, options))

    # Merge immutable typography spans so whole-run normalization (ellipsis,
    # !/?, long marks) stays exactly equivalent to normalizing the transcript.
    units: list[set[str]] = []
    fixed_span: list[str] = []
    for source, options in cells:
        if len(options) == 1:
            fixed_span.append(source)
            continue
        if fixed_span:
            units.append({normalizer("".join(fixed_span))})
            fixed_span.clear()
        units.append({normalizer(option) for option in options})
    if fixed_span:
        units.append({normalizer("".join(fixed_span))})

    target = normalizer(reference)
    row = list(range(len(target) + 1))
    for options in units:
        candidate_rows = [
            _advance_levenshtein_row(row, option, target) for option in options
        ]
        row = [min(values) for values in zip(*candidate_rows, strict=True)]
    return row[-1], len(target)


def load_fusion_b(checkpoint_path, device: torch.device) -> FusionBVisualTop16:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = FusionBStudentConfig(**checkpoint["config"])
    model = FusionBVisualTop16(config)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


def batch_ctc_evidence(
    probabilities: torch.Tensor,
    neck: torch.Tensor,
    top_k: int = 16,
) -> list[dict[str, torch.Tensor]]:
    """Collapse a CTC batch and retain exactly the evidence used in training.

    Per-frame top-k is sufficient to reconstruct the top-k of the max over any
    cell interval: every member of the latter must be top-k in at least one
    constituent frame.  This avoids a T x 6808 CPU transfer.
    """
    values = probabilities.float()
    sums = values.sum(dim=-1)
    if bool((values < 0).any()) or not torch.allclose(
        sums, torch.ones_like(sums), rtol=1e-3, atol=1e-3
    ):
        values = values.softmax(dim=-1)
    greedy_probability, greedy_id = values.max(dim=-1)
    frame_values, frame_indices = values[..., 1:].topk(top_k, dim=-1)
    frame_indices = frame_indices + 1
    greedy_probability = greedy_probability.cpu()
    greedy_id = greedy_id.cpu()
    frame_values = frame_values.cpu()
    frame_indices = frame_indices.cpu()
    neck = neck.to(dtype=torch.float16).cpu()
    results = []
    for row_ids, row_scores, row_values, row_indices, row_neck in zip(
        greedy_id, greedy_probability, frame_values, frame_indices, neck, strict=True
    ):
        runs: list[tuple[int, int, int, int, float]] = []
        ids = row_ids.tolist()
        scores = row_scores.tolist()
        start = 0
        while start < len(ids):
            token_id = ids[start]
            end = start + 1
            while end < len(ids) and ids[end] == token_id:
                end += 1
            if token_id != 0:
                peak_offset = max(range(end - start), key=lambda i: scores[start + i])
                centre = start + peak_offset
                runs.append((token_id, start, end, centre, scores[centre]))
            start = end
        greedy, visual, candidate_ids, candidate_logp, scalars = [], [], [], [], []
        for index, (token_id, _run_start, _run_end, centre, peak) in enumerate(runs):
            left = 0 if index == 0 else (runs[index - 1][3] + centre) // 2 + 1
            right = len(ids) if index + 1 == len(runs) else (centre + runs[index + 1][3]) // 2 + 1
            maxima: dict[int, float] = {}
            for candidate, score in zip(
                row_indices[left:right].reshape(-1).tolist(),
                row_values[left:right].reshape(-1).tolist(),
                strict=True,
            ):
                maxima[candidate] = max(maxima.get(candidate, 0.0), score)
            ranked = sorted(maxima.items(), key=lambda item: (-item[1], item[0]))[:top_k]
            if len(ranked) < top_k:
                raise RuntimeError("CTC interval yielded fewer than top-k candidates")
            candidates = torch.tensor([item[0] - 1 for item in ranked], dtype=torch.long)
            probs = torch.tensor([item[1] for item in ranked], dtype=torch.float32)
            # Blank statistics are already reduced before the large posterior
            # can be released. `values` is still available on accelerator; use
            # the copied greedy probability only when greedy is blank would be
            # wrong, so these two values are supplied by the caller below.
            greedy.append(token_id - 1)
            visual.append(row_neck[centre])
            candidate_ids.append(candidates)
            candidate_logp.append(probs.clamp_min(1e-30).log().to(torch.float16))
            scalars.append((peak, left, right, float(probs[0] - probs[1])))
        results.append({
            "greedy_ids": torch.tensor(greedy, dtype=torch.long),
            "visual": torch.stack(visual) if visual else torch.empty((0, 120), dtype=torch.float16),
            "candidate_ids": torch.stack(candidate_ids) if candidate_ids else torch.empty((0, top_k), dtype=torch.long),
            "candidate_logp": torch.stack(candidate_logp) if candidate_logp else torch.empty((0, top_k), dtype=torch.float16),
            "scalar_partial": scalars,
            "greedy_path": row_ids,
        })
    # Blank max/mean require the blank posterior. Copy only that B x T plane.
    blank = values[..., 0].cpu()
    for result, row_blank in zip(results, blank, strict=True):
        completed = []
        for peak, left, right, probability_margin in result.pop("scalar_partial"):
            interval = row_blank[left:right]
            completed.append([peak, float(interval.max()), float(interval.mean()), probability_margin])
        result["scalars"] = torch.tensor(completed, dtype=torch.float16).reshape(-1, 4)
        result.pop("greedy_path")
    return results


@torch.inference_mode()
def apply_fusion_b(
    model: FusionBVisualTop16,
    states: list[dict[str, Any]],
    characters: list[str],
    threshold: float,
    device: torch.device,
    proposal_floor: float = 2.0,
) -> None:
    """Attach `b_prediction` and edit telemetry to each GPU state."""
    examples = []
    for state_index, state in enumerate(states):
        gpu = state["gpu"]
        parts = [
            part for part, accepted in zip(gpu["b_parts"], gpu["accepted"], strict=True)
            if accepted and part is not None
        ]
        if not parts:
            gpu["b_prediction"] = ""
            gpu["b_edits"] = 0
            gpu["b_cells"] = 0
            gpu["b_proposals"] = []
            continue
        greedy = torch.cat([part["greedy_ids"] for part in parts])
        visual = torch.cat([part["visual"] for part in parts]).float()
        candidate_ids = torch.cat([part["candidate_ids"] for part in parts])
        candidate_logp = torch.cat([part["candidate_logp"] for part in parts]).float()
        base_scalars = torch.cat([part["scalars"] for part in parts]).float()
        scalars = torch.cat(
            (base_scalars, candidate_logp[:, :1], candidate_logp[:, :1] - candidate_logp[:, 1:2]), dim=1
        )
        starts = torch.cat([
            (
                torch.tensor(
                    [1] + [0] * (len(part["greedy_ids"]) - 1), dtype=torch.long
                )
                if len(part["greedy_ids"])
                else torch.empty(0, dtype=torch.long)
            )
            for part in parts
        ])
        keep_match = candidate_ids == greedy[:, None]
        if not bool(keep_match.any(dim=1).all()):
            raise RuntimeError("greedy character absent from cell top16")
        examples.append({
            "state_index": state_index, "greedy": greedy, "visual": visual,
            "candidate_ids": candidate_ids, "candidate_logp": candidate_logp,
            "scalars": scalars, "track_starts": starts,
            "keep_index": keep_match.long().argmax(dim=1),
        })
    if not examples:
        return
    config = model.config
    width = max(len(item["greedy"]) for item in examples) + 2
    batch = len(examples)
    input_ids = torch.full((batch, width), config.pad_token_id, dtype=torch.long)
    attention = torch.zeros((batch, width), dtype=torch.bool)
    track_starts = torch.zeros((batch, width), dtype=torch.long)
    visual = torch.zeros((batch, width, 120), dtype=torch.float32)
    scalars = torch.zeros((batch, width, 6), dtype=torch.float32)
    candidates = torch.zeros((batch, width, 16), dtype=torch.long)
    logp = torch.zeros((batch, width, 16), dtype=torch.float32)
    keeps = torch.zeros((batch, width), dtype=torch.long)
    cell_mask = torch.zeros((batch, width), dtype=torch.bool)
    input_ids[:, 0] = config.bos_token_id
    for row, item in enumerate(examples):
        length = len(item["greedy"])
        destination = slice(1, length + 1)
        input_ids[row, destination] = item["greedy"]
        input_ids[row, length + 1] = config.eos_token_id
        attention[row, : length + 2] = True
        track_starts[row, destination] = item["track_starts"]
        visual[row, destination] = item["visual"]
        scalars[row, destination] = item["scalars"]
        candidates[row, destination] = item["candidate_ids"]
        logp[row, destination] = item["candidate_logp"]
        keeps[row, destination] = item["keep_index"]
        cell_mask[row, destination] = True
    tensors = [input_ids, attention, track_starts, visual, scalars, candidates, logp, keeps, cell_mask]
    tensors = [tensor.to(device) for tensor in tensors]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        scores = model(*tensors).float()
    offset = 0
    for item in examples:
        length = len(item["greedy"])
        local = scores[offset : offset + length]
        offset += length
        keep_index = item["keep_index"].to(device)
        keep_score = local.gather(1, keep_index[:, None]).squeeze(1)
        alternate = local.clone()
        alternate.scatter_(1, keep_index[:, None], -torch.inf)
        alt_score, alt_index = alternate.max(dim=1)
        take = alt_score - keep_score > threshold
        prediction = item["greedy"].to(device).clone()
        candidate_ids = item["candidate_ids"].to(device)
        prediction[take] = candidate_ids[take, alt_index[take]]
        ids = prediction.cpu().tolist()
        gpu = states[item["state_index"]]["gpu"]
        gpu["b_prediction"] = "".join(characters[index + 1] for index in ids)
        gpu["b_edits"] = int(take.sum())
        gpu["b_cells"] = length
        # Keep a compact offline calibration trace.  Production runs normally
        # retain only plausible edits; threshold-search runs may lower the
        # floor so the expensive DET/REC pass never needs to be repeated.
        margins = (alt_score - keep_score).cpu()
        alternate_ids = candidate_ids[
            torch.arange(length, device=device), alt_index
        ].cpu()
        gpu["b_proposals"] = [
            [position, characters[int(alternate_ids[position]) + 1], round(float(margins[position]), 5)]
            for position in range(length)
            if float(margins[position]) > proposal_floor
        ]

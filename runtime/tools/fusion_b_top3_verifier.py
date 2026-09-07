#!/usr/bin/env python3
"""Device-only top-3 multi-MASK refinement and tiny edit verifier for Fusion B."""

from __future__ import annotations

from dataclasses import dataclass
import math
import unicodedata

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_b_visual_top16_model import FusionBVisualTop16


FAMILY_NAMES = ("hiragana", "katakana", "han", "digit", "latin", "other_text")
NON_TEXT_LETTERLIKE = set("ー〜～〰")


def character_family(character: str) -> int:
    codepoint = ord(character)
    if 0x3040 <= codepoint <= 0x309F:
        return 0
    if 0x30A0 <= codepoint <= 0x30FF or 0xFF66 <= codepoint <= 0xFF9F:
        return 1
    if (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    ):
        return 2
    if unicodedata.category(character) == "Nd":
        return 3
    if "LATIN" in unicodedata.name(character, ""):
        return 4
    return 5


def character_lookups(characters: list[str], vocabulary_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build editable and family tables for zero-based REC/B character IDs."""
    editable = torch.zeros(vocabulary_size, dtype=torch.bool)
    families = torch.full((vocabulary_size,), len(FAMILY_NAMES) - 1, dtype=torch.long)
    for index, character in enumerate(characters):
        if index >= vocabulary_size:
            break
        editable[index] = (
            len(character) == 1
            and unicodedata.category(character)[0] in {"L", "N"}
            and character not in NON_TEXT_LETTERLIKE
        )
        families[index] = character_family(character)
    return editable, families


class EditVerifierRuntime(nn.Module):
    def __init__(self, character_count: int, numeric_dimensions: int, embedding_size: int = 8):
        super().__init__()
        self.character_embedding = nn.Embedding(character_count, embedding_size)
        input_size = numeric_dimensions + embedding_size * 2
        self.network = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, 96),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 3),
        )

    def forward(
        self, numeric: torch.Tensor, source_ids: torch.Tensor, replacement_ids: torch.Tensor
    ) -> torch.Tensor:
        combined = torch.cat(
            (
                numeric,
                self.character_embedding(source_ids),
                self.character_embedding(replacement_ids),
            ),
            dim=-1,
        )
        return self.network(combined)


@dataclass(frozen=True)
class Top3VerifierOutput:
    replacement_ids: torch.Tensor
    take: torch.Tensor
    queried: torch.Tensor
    verifier_score: torch.Tensor


def _gather_cells(tensor: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    tail = tensor.shape[2:]
    index = positions.reshape(*positions.shape, *([1] * len(tail))).expand(
        *positions.shape, *tail
    )
    return tensor.gather(1, index)


def _candidate_scores(
    model: FusionBVisualTop16,
    hidden: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logp: torch.Tensor,
    keep_index: torch.Tensor,
) -> torch.Tensor:
    query = model.candidate_projection(hidden)
    embedding = model.backbone.character_embedding(candidate_ids)
    scores = (embedding * query[:, :, None]).sum(dim=-1) / math.sqrt(model.config.hidden_size)
    scores = scores + torch.exp(model.ctc_log_scale).clamp(max=10.0) * candidate_logp
    scores = scores + model.rank_bias
    scores.scatter_add_(
        2, keep_index[..., None], model.keep_bias.expand_as(keep_index[..., None])
    )
    return scores


class FusionBTop3Verifier(nn.Module):
    """Run normal B, one top-3 multi-MASK pass, and a tiny proposal verifier."""

    def __init__(
        self,
        b: FusionBVisualTop16,
        verifier: EditVerifierRuntime,
        *,
        editable_lookup: torch.Tensor,
        family_lookup: torch.Tensor,
        family_thresholds: torch.Tensor,
        numeric_mean: torch.Tensor,
        numeric_std: torch.Tensor,
        alpha: float = 0.6,
        query_topk: int = 3,
        veto_threshold: float = -1.7643163800239563,
        add_threshold: float = 1.5959173440933228,
    ) -> None:
        super().__init__()
        if query_topk != 3:
            raise ValueError("the v1 verifier contract is calibrated for top-3 queries")
        self.b = b
        self.verifier = verifier
        self.register_buffer("editable_lookup", editable_lookup.bool())
        self.register_buffer("family_lookup", family_lookup.long())
        self.register_buffer("family_thresholds", family_thresholds.float())
        self.register_buffer("numeric_mean", numeric_mean.float())
        self.register_buffer("numeric_std", numeric_std.float())
        self.alpha = alpha
        self.query_topk = query_topk
        self.veto_threshold = veto_threshold
        self.add_threshold = add_threshold

    def _allowed(
        self,
        input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
        keep_index: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> torch.Tensor:
        source = self.editable_lookup[input_ids.clamp_max(self.editable_lookup.numel() - 1)]
        target = self.editable_lookup[
            candidate_ids.clamp_max(self.editable_lookup.numel() - 1)
        ]
        allowed = cell_mask[..., None] & source[..., None] & target
        allowed.scatter_(2, keep_index[..., None], cell_mask[..., None])
        return allowed

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        track_starts: torch.Tensor,
        visual: torch.Tensor,
        scalars: torch.Tensor,
        candidate_ids: torch.Tensor,
        candidate_logp: torch.Tensor,
        keep_index: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> Top3VerifierOutput:
        addition = self.b.scalar_projection(self.b.scalar_norm(scalars))
        addition = addition + self.b.visual_projection(self.b.visual_norm(visual))
        normal_hidden = self.b.backbone.encode(
            input_ids, attention_mask, track_starts, addition
        )
        normal = _candidate_scores(
            self.b, normal_hidden, candidate_ids, candidate_logp, keep_index
        )
        allowed = self._allowed(input_ids, candidate_ids, keep_index, cell_mask)
        normal = normal.masked_fill(~allowed, -torch.inf)

        keep_score = normal.gather(2, keep_index[..., None]).squeeze(-1)
        nonkeep = normal.clone()
        nonkeep.scatter_(2, keep_index[..., None], -torch.inf)
        suspicion = nonkeep.max(dim=-1).values - keep_score
        queryable = cell_mask & torch.isfinite(suspicion)
        query_values, query_positions = suspicion.masked_fill(
            ~queryable, -torch.inf
        ).topk(self.query_topk, dim=1)
        query_valid = torch.isfinite(query_values)
        selected = torch.zeros_like(cell_mask)
        selected.scatter_(1, query_positions, query_valid)

        masked_ids = input_ids.clone()
        masked_ids[selected] = self.b.config.mask_token_id
        masked_hidden = self.b.backbone.encode(masked_ids, attention_mask, track_starts)
        transformed = self.b.backbone.mlm_norm(
            F.gelu(self.b.backbone.mlm_dense(masked_hidden))
        )
        embedding = self.b.backbone.character_embedding(candidate_ids)
        masked = (embedding * transformed[:, :, None]).sum(dim=-1)
        masked = masked + self.b.backbone.mlm_bias[candidate_ids]
        masked = masked.masked_fill(~allowed, -torch.inf)
        blended = torch.where(selected[..., None], normal + self.alpha * masked, normal)

        keep_score = blended.gather(2, keep_index[..., None]).squeeze(-1)
        alternate = blended.clone()
        alternate.scatter_(2, keep_index[..., None], -torch.inf)
        alternate_score, alternate_index = alternate.max(dim=-1)
        # MPS may return -1 as the argmax index when every candidate is masked
        # to -inf (padding cells and cells with no editable alternative).  Such
        # cells are never eligible for editing, but the verifier still gathers
        # their fixed-shape features below.  Clamp to a legal placeholder so
        # the masked tail cannot fault the device kernel.
        alternate_index = alternate_index.clamp_min(0)
        replacement_ids = candidate_ids.gather(
            2, alternate_index[..., None]
        ).squeeze(-1)
        source_ids = candidate_ids.gather(2, keep_index[..., None]).squeeze(-1)
        source_family = self.family_lookup[source_ids]
        base_take = (
            cell_mask
            & self.editable_lookup[source_ids]
            & self.editable_lookup[replacement_ids]
            & ((alternate_score - keep_score) > self.family_thresholds[source_family])
        )

        q_normal = _gather_cells(normal, query_positions)
        q_masked = _gather_cells(masked, query_positions)
        q_blended = _gather_cells(blended, query_positions)
        q_ctc = _gather_cells(candidate_logp, query_positions)
        q_allowed = _gather_cells(allowed, query_positions)
        q_keep = _gather_cells(keep_index[..., None], query_positions).squeeze(-1)
        q_proposal = _gather_cells(alternate_index[..., None], query_positions).squeeze(-1)
        q_source_ids = _gather_cells(source_ids[..., None], query_positions).squeeze(-1)
        q_replacement_ids = _gather_cells(
            replacement_ids[..., None], query_positions
        ).squeeze(-1)

        scalar_parts = []
        for matrix in (q_normal, q_masked, q_blended, q_ctc):
            proposal = matrix.gather(2, q_proposal[..., None]).squeeze(-1)
            keep = matrix.gather(2, q_keep[..., None]).squeeze(-1)
            competitor = matrix.masked_fill(~q_allowed, -torch.inf).clone()
            competitor.scatter_(2, q_keep[..., None], -torch.inf)
            competitor.scatter_(2, q_proposal[..., None], -torch.inf)
            other = competitor.max(dim=-1).values
            other_margin = proposal - other
            other_margin = torch.where(
                torch.isfinite(other_margin), other_margin, torch.full_like(other_margin, 20.0)
            )
            scalar_parts.extend((proposal, keep, proposal - keep, other_margin))

        def best_nonkeep(matrix: torch.Tensor) -> torch.Tensor:
            local = matrix.masked_fill(~q_allowed, -torch.inf).clone()
            local.scatter_(2, q_keep[..., None], -torch.inf)
            return local.argmax(dim=-1)

        lengths = cell_mask.sum(dim=1).clamp_min(1)
        local_position = (query_positions - 1).clamp_min(0)
        q_track_start = _gather_cells(
            track_starts[..., None], query_positions
        ).squeeze(-1)
        scalar_parts.extend(
            (
                q_proposal.float() / 15.0,
                q_keep.float() / 15.0,
                (best_nonkeep(q_normal) == q_proposal).float(),
                (best_nonkeep(q_masked) == q_proposal).float(),
                (best_nonkeep(q_ctc) == q_proposal).float(),
                local_position.float() / lengths[:, None].float(),
                (lengths[:, None] - 1 - local_position).float() / lengths[:, None].float(),
                torch.log1p(lengths.float())[:, None].expand_as(local_position) / math.log(129.0),
                (local_position == 0).float(),
                (local_position == lengths[:, None] - 1).float(),
                q_track_start.float(),
            )
        )
        q_source_family = self.family_lookup[q_source_ids]
        q_replacement_family = self.family_lookup[q_replacement_ids]
        numeric = torch.cat(
            (
                _gather_cells(visual, query_positions),
                _gather_cells(scalars, query_positions),
                torch.stack(scalar_parts, dim=-1),
                F.one_hot(q_source_family, len(FAMILY_NAMES)).float(),
                F.one_hot(q_replacement_family, len(FAMILY_NAMES)).float(),
                (q_source_family == q_replacement_family).float()[..., None],
            ),
            dim=-1,
        )
        numeric = (numeric - self.numeric_mean) / self.numeric_std
        logits = self.verifier(numeric, q_source_ids, q_replacement_ids)
        verifier_score = logits[..., 2] - torch.logsumexp(logits[..., :2], dim=-1)
        q_base = _gather_cells(base_take[..., None], query_positions).squeeze(-1)
        q_take = torch.where(
            q_base,
            verifier_score >= self.veto_threshold,
            verifier_score > self.add_threshold,
        )
        q_take &= query_valid
        final_take = base_take.clone()
        prior = _gather_cells(final_take[..., None], query_positions).squeeze(-1)
        final_take.scatter_(1, query_positions, torch.where(query_valid, q_take, prior))
        return Top3VerifierOutput(
            replacement_ids=replacement_ids,
            take=final_take,
            queried=selected,
            verifier_score=verifier_score,
        )

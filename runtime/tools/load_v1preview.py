#!/usr/bin/env python3
"""Load the self-contained v1Preview bundle on MPS, CUDA, or ROCm."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unicodedata

import torch
import yaml

from fusion_b_runtime import load_fusion_b
from fusion_b_top3_verifier import (
    EditVerifierRuntime,
    FAMILY_NAMES,
    FusionBTop3Verifier,
    character_lookups,
)
from jmanga_ocr.gpu_db import GpuDBConfig, SmartGpuDBConfig, SmartGpuDBPostProcess
from v1preview_all_gpu import V1PreviewConfig, V1PreviewPipeline


SUPPLEMENT_CHARACTERS = set(
    ".,!?！？。、．…・･゛ﾞﾟ"
    "っッゃゅょャュョぁぃぅぇぉァィゥェォゎヮゕヵヶゖ"
)
NON_TEXT_LETTERLIKE = set("ー〜～〰")


def _read_dictionary(path: Path, use_space_char: bool) -> list[str]:
    characters = path.read_text(encoding="utf-8").splitlines()
    if use_space_char and " " not in characters:
        characters.append(" ")
    return ["blank", *characters]


def load_v1preview(
    bundle: str | Path,
    device: str | torch.device,
    *,
    enable_b: bool = True,
) -> tuple[V1PreviewPipeline, list[str], dict]:
    """Return the assembled model, CTC dictionary, and frozen release spec."""
    root = Path(bundle).resolve()
    spec = json.loads((root / "v1preview.json").read_text(encoding="utf-8"))
    device = torch.device(device)
    if device.type not in {"mps", "cuda"}:
        raise ValueError("v1Preview supports Apple MPS and CUDA/ROCm")
    ptocr = root / "third_party" / "PaddleOCR2Pytorch"
    sys.path.insert(0, str(ptocr))
    from pytorchocr.modeling.architectures.base_model import BaseModel

    det_config = yaml.safe_load(
        (root / spec["files"]["det_config"]).read_text(encoding="utf-8")
    )
    det = BaseModel(copy.deepcopy(det_config["Architecture"]))
    det.load_state_dict(
        torch.load(
            root / spec["files"]["det_checkpoint"],
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )

    rec_config = yaml.safe_load(
        (root / spec["files"]["rec_config"]).read_text(encoding="utf-8")
    )
    rec_state = torch.load(
        root / spec["files"]["rec_checkpoint"],
        map_location="cpu",
        weights_only=True,
    )
    ctc_bias = next(
        rec_state[name]
        for name in ("head.ctc_head.fc.bias", "head.ctc_head.fc2.bias")
        if name in rec_state
    )
    class_count = int(ctc_bias.shape[0])
    architecture = copy.deepcopy(rec_config["Architecture"])
    architecture["Head"]["out_channels_list"] = {
        "CTCLabelDecode": class_count,
        "SARLabelDecode": class_count + 2,
        "NRTRLabelDecode": class_count + 4,
    }
    rec = BaseModel(architecture)
    incompatible = rec.load_state_dict(rec_state, strict=False)
    unexpected_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(("head.before_gtc", "head.gtc_head"))
    ]
    if incompatible.unexpected_keys or unexpected_missing:
        raise RuntimeError(f"REC checkpoint mismatch: {incompatible}")

    characters = _read_dictionary(
        root / spec["files"]["dictionary"],
        bool(rec_config.get("Global", {}).get("use_space_char")),
    )
    if len(characters) != class_count:
        raise ValueError("REC dictionary and checkpoint class counts differ")
    b = None
    if enable_b and "b_checkpoint" in spec["files"]:
        b = load_fusion_b(root / spec["files"]["b_checkpoint"], device)
    b_refiner = None
    if enable_b and b is not None and "b_verifier_checkpoint" in spec["files"]:
        verifier_saved = torch.load(
            root / spec["files"]["b_verifier_checkpoint"],
            map_location="cpu",
            weights_only=True,
        )
        verifier = EditVerifierRuntime(
            verifier_saved["character_count"],
            verifier_saved["numeric_dimensions"],
            verifier_saved["embedding_size"],
        )
        verifier.load_state_dict(verifier_saved["model"], strict=True)
        verifier_spec = spec["b"]["top3_verifier"]
        verifier_editable, family_lookup = character_lookups(
            characters[1:], b.config.vocab_size
        )
        family_thresholds = torch.tensor(
            [verifier_spec["family_thresholds"][name] for name in FAMILY_NAMES],
            dtype=torch.float32,
        )
        b_refiner = FusionBTop3Verifier(
            b,
            verifier,
            editable_lookup=verifier_editable,
            family_lookup=family_lookup,
            family_thresholds=family_thresholds,
            numeric_mean=verifier_saved["numeric_mean"],
            numeric_std=verifier_saved["numeric_std"],
            alpha=float(verifier_spec["alpha"]),
            query_topk=int(verifier_spec["query_topk"]),
            veto_threshold=float(verifier_spec["veto_threshold"]),
            add_threshold=float(verifier_spec["add_threshold"]),
        )

    db_spec = spec["a_gpu_db"]
    angles = tuple(
        float(value)
        for value in torch.arange(
            -db_spec["angle_radius_degrees"],
            db_spec["angle_radius_degrees"] + db_spec["angle_step_degrees"] * 0.5,
            db_spec["angle_step_degrees"],
        ).tolist()
    )
    refinements = tuple(
        float(value)
        for value in torch.arange(
            -db_spec["angle_refine_radius_degrees"],
            db_spec["angle_refine_radius_degrees"]
            + db_spec["angle_refine_step_degrees"] * 0.5,
            db_spec["angle_refine_step_degrees"],
        ).tolist()
    )
    common = dict(
        threshold=db_spec["threshold"],
        unclip_ratio=db_spec["unclip_ratio"],
        max_components=db_spec["max_components"],
        angle_offsets_degrees=angles,
        angle_refine_offsets_degrees=refinements,
    )
    primary = GpuDBConfig(
        box_threshold=db_spec["box_threshold"],
        min_size=db_spec["min_size"],
        **common,
    )
    supplement = GpuDBConfig(
        box_threshold=db_spec["supplement_box_threshold"],
        min_size=db_spec["supplement_min_size"],
        **common,
    )
    a = SmartGpuDBPostProcess(
        SmartGpuDBConfig(
            db=primary,
            supplement_db=supplement,
            reuse_supplement_decode=bool(
                db_spec.get("reuse_supplement_decode", False)
            ),
            order_before_expansion=True,
            robust_reading_order=True,
            expanded_as_primary=False,
        )
    )
    editable_size = b.config.vocab_size if b is not None else class_count - 1
    editable = torch.zeros(editable_size, dtype=torch.bool)
    for index, character in enumerate(characters[1:]):
        editable[index] = (
            unicodedata.category(character)[0] in {"L", "N"}
            and character not in NON_TEXT_LETTERLIKE
        )
    supplement_ids = torch.tensor(
        [
            index - 1
            for index, character in enumerate(characters)
            if index > 0 and character in SUPPLEMENT_CHARACTERS
        ],
        dtype=torch.int64,
    )
    runtime_config = V1PreviewConfig(
        **spec["runtime"],
        replacement_threshold=spec.get("b", {}).get(
            "replacement_threshold", float("inf")
        ),
    )
    pipeline = V1PreviewPipeline(
        det.to(device).eval(),
        a,
        rec.to(device).eval(),
        b,
        b_refiner=b_refiner,
        editable_character_lookup=editable,
        supplement_token_ids=supplement_ids,
        config=runtime_config,
    ).to(device).eval()
    return pipeline, characters, spec


def decode_v1preview(output, characters: list[str]) -> list[str]:
    """Presentation helper; intentionally outside the device OCR forward."""
    ids = output.token_ids.detach().cpu()
    counts = output.token_count.detach().cpu()
    return [
        "".join(characters[int(token) + 1] for token in row[: int(count)])
        for row, count in zip(ids, counts, strict=True)
    ]

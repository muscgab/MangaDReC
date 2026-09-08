"""Rebuild MangaDReC/DReCo from declarative config and safe tensor weights."""

from __future__ import annotations

import copy
import json
import sys
import unicodedata
from pathlib import Path

import torch
import yaml
from fusion_b_student_model import FusionBStudentConfig
from fusion_b_visual_top16_model import FusionBVisualTop16
from safetensors.torch import load_model
from v1preview_all_gpu import V1PreviewConfig, V1PreviewPipeline

from jmanga_ocr.gpu_db import GpuDBConfig, SmartGpuDBConfig, SmartGpuDBPostProcess

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


def _postprocessor(spec: dict) -> SmartGpuDBPostProcess:
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
    common = {
        "threshold": db_spec["threshold"],
        "unclip_ratio": db_spec["unclip_ratio"],
        "max_components": db_spec["max_components"],
        "angle_offsets_degrees": angles,
        "angle_refine_offsets_degrees": refinements,
    }
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
    return SmartGpuDBPostProcess(
        SmartGpuDBConfig(
            db=primary,
            supplement_db=supplement,
            reuse_supplement_decode=bool(db_spec.get("reuse_supplement_decode", False)),
            order_before_expansion=True,
            robust_reading_order=True,
            expanded_as_primary=False,
        )
    )


def load_safetensors_pipeline(
    package: str | Path,
    checkpoint: str | Path,
    device: str | torch.device,
) -> tuple[V1PreviewPipeline, list[str], dict]:
    """Build the module graph without pickle and load a safetensors checkpoint."""
    root = Path(package).resolve()
    manifest = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
    spec = manifest["spec"]
    target = torch.device(device)
    if target.type not in {"mps", "cuda"}:
        raise ValueError("MangaDReC supports Apple MPS and CUDA/ROCm")

    ptocr = root / "runtime" / "third_party" / "PaddleOCR2Pytorch"
    if str(ptocr) not in sys.path:
        sys.path.insert(0, str(ptocr))
    from pytorchocr.modeling.architectures.base_model import BaseModel

    det_config = yaml.safe_load(
        (root / manifest["architecture"]["det_config"]).read_text(encoding="utf-8")
    )
    det = BaseModel(copy.deepcopy(det_config["Architecture"]))

    rec_config = yaml.safe_load(
        (root / manifest["architecture"]["rec_config"]).read_text(encoding="utf-8")
    )
    characters = _read_dictionary(
        root / manifest["architecture"]["dictionary"],
        bool(rec_config.get("Global", {}).get("use_space_char")),
    )
    class_count = len(characters)
    rec_architecture = copy.deepcopy(rec_config["Architecture"])
    rec_architecture["Head"]["out_channels_list"] = {
        "CTCLabelDecode": class_count,
        "SARLabelDecode": class_count + 2,
        "NRTRLabelDecode": class_count + 4,
    }
    rec = BaseModel(rec_architecture)

    b = None
    if spec.get("b", {}).get("enabled", False):
        b = FusionBVisualTop16(
            FusionBStudentConfig(**manifest["architecture"]["b_config"])
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
        det.eval(),
        _postprocessor(spec),
        rec.eval(),
        b,
        editable_character_lookup=editable,
        supplement_token_ids=supplement_ids,
        config=runtime_config,
    ).eval()
    missing, unexpected = load_model(
        pipeline, str(Path(checkpoint).resolve()), strict=True, device="cpu"
    )
    if missing or unexpected:
        raise RuntimeError(
            f"safetensors checkpoint mismatch: missing={missing}, unexpected={unexpected}"
        )
    return pipeline.to(target).eval(), characters, spec

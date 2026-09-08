#!/usr/bin/env python3
"""Convert the legacy full-object checkpoint into safe tensor weights."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import save_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--legacy-checkpoint", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    package = args.package.resolve()
    runtime = package / "runtime"
    sys.path[:0] = [
        str(runtime / "tools"),
        str(runtime / "src"),
        str(runtime / "third_party" / "PaddleOCR2Pytorch"),
    ]
    for module in (
        "v1preview_all_gpu",
        "fusion_b_student_model",
        "fusion_b_top3_verifier",
        "fusion_b_visual_top16_model",
    ):
        __import__(module)

    saved = torch.load(
        args.legacy_checkpoint.resolve(), map_location="cpu", weights_only=False
    )
    model = saved["model"].eval()
    output = package / args.output_name
    save_model(
        model,
        output,
        metadata={
            "format": "pt",
            "schema": "mangadrec_full_pipeline_safetensors_v1",
            "release": saved["spec"]["release"],
        },
    )

    config_dir = package / "config"
    config_dir.mkdir(exist_ok=True)
    source = args.source_bundle.resolve()
    shutil.copy2(source / "models/det/config.yml", config_dir / "det.yml")
    shutil.copy2(source / "models/rec/config.yml", config_dir / "rec.yml")
    shutil.copy2(
        source / "models/rec/character_dict_tcy12.txt",
        config_dir / "character_dict_tcy12.txt",
    )

    architecture = {
        "det_config": "config/det.yml",
        "rec_config": "config/rec.yml",
        "dictionary": "config/character_dict_tcy12.txt",
    }
    if saved["spec"].get("b", {}).get("enabled", False):
        b_checkpoint = torch.load(
            source / saved["spec"]["files"]["b_checkpoint"],
            map_location="cpu",
            weights_only=False,
        )
        architecture["b_config"] = b_checkpoint["config"]
    manifest = {
        "schema": "mangadrec_declarative_architecture_v1",
        "weights": args.output_name,
        "architecture": architecture,
        "spec": saved["spec"],
    }
    (package / "model_config.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint": str(output),
                "bytes": output.stat().st_size,
                "tensor_entries": len(model.state_dict()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

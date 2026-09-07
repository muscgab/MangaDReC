#!/usr/bin/env python3
"""Evaluate one packaged MangaDReC family model on a crop manifest."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch


def load_ocr(package: Path, device: str):
    if (package / "manga_dreco.py").exists():
        module_name, filename, class_name = "manga_dreco", "manga_dreco.py", "MangaDReCo"
    else:
        module_name, filename, class_name = "manga_drec", "manga_drec.py", "MangaDReC"
    spec = importlib.util.spec_from_file_location(module_name, package / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return getattr(module, class_name).from_pretrained(device=device)


def make_input(paths: list[Path], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    images = [np.asarray(Image.open(path).convert("RGB")).copy() for path in paths]
    height = max(image.shape[0] for image in images)
    width = max(image.shape[1] for image in images)
    batch = np.zeros((len(images), height, width, 3), dtype=np.uint8)
    sizes = []
    for index, image in enumerate(images):
        h, w = image.shape[:2]
        batch[index, :h, :w] = image[..., ::-1]
        sizes.append((h, w))
    return (
        torch.from_numpy(batch).permute(0, 3, 1, 2).to(device),
        torch.tensor(sizes, device=device),
    )


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for row, a in enumerate(left, 1):
        current = [row]
        for column, b in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (a != b),
                )
            )
        previous = current
    return previous[-1]


class Score:
    def __init__(self) -> None:
        self.samples = self.exact = self.errors = self.characters = 0

    def add(self, reference: str, prediction: str) -> None:
        self.samples += 1
        self.exact += reference == prediction
        self.errors += edit_distance(reference, prediction)
        self.characters += len(reference)

    def report(self) -> dict:
        return {
            "samples": self.samples,
            "exact_matches": self.exact,
            "edit_distance": self.errors,
            "reference_characters": self.characters,
            "em": self.exact / self.samples,
            "cer": self.errors / self.characters,
        }


def main() -> None:
    evaluation_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda"), default="cuda")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--heldout-ids", type=Path)
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=evaluation_dir / "semantic_visual_v2_translation_eval.py",
    )
    parser.add_argument(
        "--normalization-preset",
        type=Path,
        default=evaluation_dir / "normalization_presets_semantic_visual_v2.yaml",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()

    os.environ["JMANGA_NORMALIZATION_PRESET"] = str(args.normalization_preset.resolve())
    sys.path.insert(0, str(args.normalizer.resolve().parent))
    normalizer_spec = importlib.util.spec_from_file_location("metric_normalizer", args.normalizer)
    normalizer = importlib.util.module_from_spec(normalizer_spec)
    assert normalizer_spec.loader is not None
    normalizer_spec.loader.exec_module(normalizer)
    normalize_pair = normalizer.normalize_pair

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    heldout = set(args.heldout_ids.read_text().splitlines()) if args.heldout_ids else set()
    ocr = load_ocr(args.package.resolve(), args.device)
    batch_size = args.batch_size
    warm_rows = rows[: batch_size]
    warm_paths = [args.crop_root / row["image_path"] for row in warm_rows]
    for _ in range(args.warmup_batches):
        tensor, sizes = make_input(warm_paths, args.device)
        ocr.decode(ocr.forward_gpu(tensor, sizes))
    if args.device == "cuda":
        torch.cuda.synchronize()
    else:
        torch.mps.synchronize()

    all_score, heldout_score = Score(), Score()
    edited_blocks = runtime_errors = 0
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with args.predictions.open("w", encoding="utf-8") as sink:
        for offset in range(0, len(rows), batch_size):
            piece = rows[offset : offset + batch_size]
            paths = [args.crop_root / row["image_path"] for row in piece]
            try:
                tensor, sizes = make_input(paths, args.device)
                output = ocr.forward_gpu(tensor, sizes)
                predictions = ocr.decode(output)
                edits = output.edited.any(dim=1).detach().cpu().tolist()
            except Exception as error:
                runtime_errors += len(piece)
                predictions = [""] * len(piece)
                edits = [False] * len(piece)
                print(f"batch {offset} failed: {error!r}", file=sys.stderr, flush=True)
            for row, prediction, edited in zip(piece, predictions, edits, strict=True):
                reference_normalized, prediction_normalized = normalize_pair(
                    row["text"], prediction
                )
                all_score.add(reference_normalized, prediction_normalized)
                sample_id = row.get("sample_id", row.get("id"))
                if sample_id in heldout:
                    heldout_score.add(reference_normalized, prediction_normalized)
                edited_blocks += bool(edited)
                sink.write(json.dumps({
                    "sample_id": sample_id,
                    "reference": row["text"],
                    "prediction": prediction,
                    "normalized_reference": reference_normalized,
                    "normalized_prediction": prediction_normalized,
                    "edited": bool(edited),
                }, ensure_ascii=False) + "\n")
            if offset and offset % (batch_size * 200) == 0:
                print(json.dumps({"processed": offset, "em": all_score.report()["em"], "cer": all_score.report()["cer"]}), flush=True)
    elapsed = time.perf_counter() - started
    payload = {
        "schema": "mangadrec_manga109s_evaluation_v1",
        "release": ocr.spec["release"],
        "training_scope": ocr.spec.get("b", {}).get("training_scope", "no_b"),
        "device": args.device,
        "torch": torch.__version__,
        "accelerator": torch.cuda.get_device_name(0) if args.device == "cuda" else "Apple MPS",
        "batch_size": batch_size,
        "normalization": getattr(normalizer, "EVALUATION_POLICY", "unknown"),
        "all": all_score.report(),
        "heldout": heldout_score.report() if heldout else None,
        "edited_blocks": edited_blocks,
        "runtime_errors": runtime_errors,
        "elapsed_seconds": elapsed,
        "images_per_second_including_io_and_metric": len(rows) / elapsed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

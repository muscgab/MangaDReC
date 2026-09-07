#!/usr/bin/env python3
"""Warm benchmark for one packaged MangaDReC family model."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import random
import statistics
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


def synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    else:
        torch.cuda.synchronize()


def make_input(images: list[np.ndarray], device: str) -> tuple[torch.Tensor, torch.Tensor]:
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


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(q * len(ordered)) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cuda"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line]
    rows = random.Random(args.seed).sample(rows, min(args.samples, len(rows)))
    images = [
        np.asarray(Image.open(args.crop_root / row["image_path"]).convert("RGB")).copy()
        for row in rows
    ]
    load_started = time.perf_counter()
    ocr = load_ocr(args.package.resolve(), args.device)
    synchronize(args.device)
    load_seconds = time.perf_counter() - load_started

    batch_size = args.batch_size
    for offset in range(0, args.warmup_batches * batch_size, batch_size):
        piece = [images[(offset + index) % len(images)] for index in range(batch_size)]
        tensor, sizes = make_input(piece, args.device)
        ocr.decode(ocr.forward_gpu(tensor, sizes))
        synchronize(args.device)

    latencies = []
    predictions = []
    started_all = time.perf_counter()
    for offset in range(0, len(images), batch_size):
        piece = images[offset : offset + batch_size]
        started = time.perf_counter()
        tensor, sizes = make_input(piece, args.device)
        output = ocr.forward_gpu(tensor, sizes)
        predictions.extend(ocr.decode(output))
        synchronize(args.device)
        latencies.append((time.perf_counter() - started) * 1000.0)
    elapsed = time.perf_counter() - started_all
    payload = {
        "schema": "mangadrec_package_speed_v1",
        "release": ocr.spec["release"],
        "device": args.device,
        "torch": torch.__version__,
        "accelerator": (
            torch.cuda.get_device_name(0) if args.device == "cuda" else "Apple MPS"
        ),
        "samples": len(images),
        "seed": args.seed,
        "batch_size": batch_size,
        "warmup_batches": args.warmup_batches,
        "load_seconds": load_seconds,
        "timing_boundary": "preloaded RGB arrays through padding, H2D, model, D2H and UTF-8 decode",
        "batch_mean_ms": statistics.mean(latencies),
        "batch_p50_ms": statistics.median(latencies),
        "batch_p90_ms": percentile(latencies, 0.90),
        "per_image_wall_ms": elapsed * 1000.0 / len(images),
        "images_per_second": len(images) / elapsed,
        "nonempty_predictions": sum(bool(item) for item in predictions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Device-resident DET -> A -> REC -> replacement-only B for v1Preview.

The public forward contract starts with an already decoded, padded BGR image
tensor and ends with fixed-capacity character IDs.  Between those boundaries
all pixels, boxes, routing masks, CTC evidence, B decisions and output IDs stay
on the input MPS/CUDA device.  REC can either execute the three native-width
buckets separately or choose one common 320/480/640 width for the whole image
batch and execute once.

Image decoding and conversion of the returned IDs to UTF-8 are I/O concerns
outside this module.  PyTorch eager mode still uses the host to dispatch device
kernels, as every non-captured PyTorch program does, but the host never reads a
value to make an OCR decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from fusion_b_visual_top16_model import FusionBVisualTop16
from fusion_b_top3_verifier import FusionBTop3Verifier
from jmanga_ocr.gpu_db import SmartGpuDBPostProcess


REC_BUCKETS = (320, 480, 640)


@dataclass(frozen=True)
class V1PreviewConfig:
    det_size: int = 224
    rec_height: int = 48
    rec_buckets: tuple[int, int, int] = REC_BUCKETS
    max_components: int = 64
    # B sees at most 64 characters (99.8815% of the current 123,212-block
    # corpus). Longer blocks are emitted unchanged. The separate output buffer
    # preserves the complete REC string instead of truncating it for B.
    b_max_characters: int = 64
    max_output_characters: int = 320
    top_k: int = 16
    # Choose one common nearest REC bucket from the largest required ROI width,
    # then execute all ROI rows in one REC forward.  Selecting a tensor shape
    # requires reading one bucket index from the accelerator in eager mode.
    rec_single_batch: bool = True
    # Direct projective sampling is faster on MPS and improved the paired
    # 1,000-image translation-normalized result used for this preview.
    rec_two_stage_approximation: bool = False
    replacement_threshold: float = 2.0
    supplement_minimum_confidence: float = 0.70
    supplement_minimum_area_ratio: float = 0.075
    supplement_maximum_distance: float = 50.0


@dataclass(frozen=True)
class V1PreviewOutput:
    token_ids: torch.Tensor
    token_count: torch.Tensor
    source_token_ids: torch.Tensor
    edited: torch.Tensor
    accepted_rois: torch.Tensor
    box_count: torch.Tensor
    boxes: torch.Tensor
    overflow: torch.Tensor


def _normalised_axis(length: int, device: torch.device) -> torch.Tensor:
    return torch.arange(length, device=device, dtype=torch.float32) + 0.5


def gpu_letterbox_224(
    images_bgr: torch.Tensor,
    source_sizes_hw: torch.Tensor,
    size: int = 224,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Letterbox padded BGR images and return DET input plus source geometry.

    ``images_bgr`` is ``[B,3,Hmax,Wmax]`` uint8 or float in the 0..255 range.
    ``source_sizes_hw`` contains the unpadded height and width of each image.
    """
    if images_bgr.ndim != 4 or images_bgr.shape[1] != 3:
        raise ValueError("images_bgr must have shape [B,3,H,W]")
    if source_sizes_hw.shape != (images_bgr.shape[0], 2):
        raise ValueError("source_sizes_hw must have shape [B,2]")
    if images_bgr.device.type not in {"mps", "cuda"}:
        raise ValueError("v1Preview requires MPS, CUDA, or ROCm through cuda")
    device = images_bgr.device
    source = images_bgr.float()
    sizes = source_sizes_hw.to(device=device, dtype=torch.float32)
    height, width = sizes.unbind(dim=1)
    scale = float(size) / torch.maximum(height, width).clamp_min(1.0)
    resized_h = torch.round(height * scale).clamp(1.0, float(size))
    resized_w = torch.round(width * scale).clamp(1.0, float(size))
    top = torch.floor((float(size) - resized_h) * 0.5)
    left = torch.floor((float(size) - resized_w) * 0.5)

    oy = _normalised_axis(size, device)[None, :, None]
    ox = _normalised_axis(size, device)[None, None, :]
    inside = (
        (oy >= top[:, None, None])
        & (oy < (top + resized_h)[:, None, None])
        & (ox >= left[:, None, None])
        & (ox < (left + resized_w)[:, None, None])
    )
    # Reproduce OpenCV INTER_LINEAR's separable uint8 resize. Its resize path
    # quantizes each linear coefficient to 11 fixed-point bits; a generic
    # grid_sample differs by around half a grey level and can move a DB pixel
    # across threshold on otherwise borderline components.
    resize_scale_y = resized_h / height.clamp_min(1.0)
    resize_scale_x = resized_w / width.clamp_min(1.0)
    sy = (oy - top[:, None, None]) / resize_scale_y[:, None, None] - 0.5
    sx = (ox - left[:, None, None]) / resize_scale_x[:, None, None] - 0.5
    y0 = torch.floor(sy)
    x0 = torch.floor(sx)
    fy = sy - y0
    fx = sx - x0
    low_y = y0 < 0
    low_x = x0 < 0
    high_y = y0 >= (height - 1.0)[:, None, None]
    high_x = x0 >= (width - 1.0)[:, None, None]
    y0 = torch.where(low_y, torch.zeros_like(y0), y0)
    x0 = torch.where(low_x, torch.zeros_like(x0), x0)
    y0 = torch.minimum(y0, (height - 1.0)[:, None, None])
    x0 = torch.minimum(x0, (width - 1.0)[:, None, None])
    fy = torch.where(low_y | high_y, torch.zeros_like(fy), fy)
    fx = torch.where(low_x | high_x, torch.zeros_like(fx), fx)
    y1 = torch.minimum(y0 + 1.0, (height - 1.0)[:, None, None])
    x1 = torch.minimum(x0 + 1.0, (width - 1.0)[:, None, None])
    coefficient_scale = 2048.0
    wy1 = torch.round(fy * coefficient_scale)
    wx1 = torch.round(fx * coefficient_scale)
    wy0 = torch.round((1.0 - fy) * coefficient_scale)
    wx0 = torch.round((1.0 - fx) * coefficient_scale)
    padded_h, padded_w = images_bgr.shape[-2:]
    x0 = x0.to(torch.int64).expand(-1, size, size)
    x1 = x1.to(torch.int64).expand(-1, size, size)
    y0 = y0.to(torch.int64).expand(-1, size, size)
    y1 = y1.to(torch.int64).expand(-1, size, size)
    flat = source.reshape(source.shape[0], 3, padded_h * padded_w)

    def gather(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        index = y * padded_w + x
        return flat.gather(2, index[:, None].expand(-1, 3, -1, -1).reshape(source.shape[0], 3, -1)).reshape(source.shape[0], 3, size, size)

    p00 = gather(y0, x0)
    p01 = gather(y0, x1)
    p10 = gather(y1, x0)
    p11 = gather(y1, x1)
    top_value = p00 * wx0[:, None] + p01 * wx1[:, None]
    bottom_value = p10 * wx0[:, None] + p11 * wx1[:, None]
    canvas = torch.round(
        (top_value * wy0[:, None] + bottom_value * wy1[:, None])
        / (coefficient_scale * coefficient_scale)
    )
    canvas = torch.where(inside[:, None], canvas, torch.full_like(canvas, 255.0))
    mean = torch.tensor((0.485, 0.456, 0.406), device=device)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), device=device)[:, None, None]
    det_input = (canvas / 255.0 - mean) / std
    geometry = torch.stack((scale, left, top, width, height), dim=1)
    return det_input, geometry


def _order_quad_x(boxes: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Match the production cropper's x-then-y corner convention on device."""
    x_order = boxes[..., 0].argsort(dim=2)
    x_sorted = boxes.gather(2, x_order[..., None].expand(-1, -1, -1, 2))
    left = x_sorted[:, :, :2]
    right = x_sorted[:, :, 2:]
    left = left.gather(
        2, left[..., 1].argsort(dim=2)[..., None].expand(-1, -1, -1, 2)
    )
    right = right.gather(
        2, right[..., 1].argsort(dim=2)[..., None].expand(-1, -1, -1, 2)
    )
    return left[:, :, 0], right[:, :, 0], right[:, :, 1], left[:, :, 1]


def _gpu_narrow_peer_repair(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    *,
    aspect_ratio: float = 4.0,
    peer_outlier_ratio: float = 1.5,
) -> torch.Tensor:
    """Match the deployed narrow ellipsis/symbol peer-width crop repair."""
    tl, tr, br, bl = _order_quad_x(boxes.float())
    width = 0.5 * (
        torch.linalg.vector_norm(tr - tl, dim=-1)
        + torch.linalg.vector_norm(br - bl, dim=-1)
    )
    height = 0.5 * (
        torch.linalg.vector_norm(bl - tl, dim=-1)
        + torch.linalg.vector_norm(br - tr, dim=-1)
    )
    vertical = height > width * 1.05
    cross = torch.where(vertical, width, height)
    short = torch.minimum(width, height)
    long = torch.maximum(width, height)
    capacity = boxes.shape[1]
    identity = torch.eye(capacity, device=boxes.device, dtype=torch.bool)[None]
    peers = (
        valid[:, :, None]
        & valid[:, None, :]
        & (vertical[:, :, None] == vertical[:, None, :])
        & ~identity
    )
    candidates = cross[:, None, :].expand(-1, capacity, -1)
    ordered = torch.where(
        peers, candidates, torch.full_like(candidates, float("inf"))
    ).sort(dim=2).values
    peer_count = peers.sum(dim=2)
    lower_index = ((peer_count - 1).clamp_min(0) // 2).to(torch.int64)
    upper_index = (peer_count.clamp_min(1) // 2).to(torch.int64)
    median = 0.5 * (
        ordered.gather(2, lower_index[..., None]).squeeze(2)
        + ordered.gather(2, upper_index[..., None]).squeeze(2)
    )
    repair = (
        valid
        & (peer_count > 0)
        & (long / short.clamp_min(1.0e-6) >= aspect_ratio)
        & (median >= cross * peer_outlier_ratio)
    )
    target = torch.where(repair, median, short).clamp_min(1.0e-6)
    short_axis = torch.where(
        (width <= height)[..., None],
        (tr - tl) + (br - bl),
        (bl - tl) + (br - tr),
    )
    short_axis = short_axis / torch.linalg.vector_norm(
        short_axis, dim=-1, keepdim=True
    ).clamp_min(1.0e-6)
    centre = boxes.mean(dim=2, keepdim=True)
    offsets = boxes - centre
    projection = (offsets * short_axis[:, :, None]).sum(dim=-1)
    scale = target / short
    repaired = boxes + (
        (scale - 1.0)[:, :, None, None]
        * projection[..., None]
        * short_axis[:, :, None]
    )
    return torch.where(repair[:, :, None, None], repaired, boxes)


def gpu_rec_bucket_crops(
    images_bgr: torch.Tensor,
    boxes: torch.Tensor,
    valid: torch.Tensor,
    buckets: Sequence[int] = REC_BUCKETS,
    rec_height: int = 48,
    two_stage_approximation: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Compact and perspective-sample A slots into three REC buckets.

    ``nonzero`` and ``index_select`` create each compact queue on device.  A
    masked dummy is appended so all three REC shapes can execute without a
    host-side empty-queue branch.  No count is read back to launch a bucket.
    """
    batch, capacity = boxes.shape[:2]
    boxes = _gpu_narrow_peer_repair(boxes, valid)
    tl, tr, br, bl = _order_quad_x(boxes.float())
    width = torch.maximum(
        torch.linalg.vector_norm(tr - tl, dim=-1),
        torch.linalg.vector_norm(br - bl, dim=-1),
    ).round().clamp_min(2.0)
    height = torch.maximum(
        torch.linalg.vector_norm(bl - tl, dim=-1),
        torch.linalg.vector_norm(br - tr, dim=-1),
    ).round().clamp_min(2.0)
    rotate = height / width >= 1.05
    long_side = torch.where(rotate, height, width)
    short_side = torch.where(rotate, width, height)
    resized_width = torch.ceil(long_side * float(rec_height) / short_side).clamp(
        1.0, float(max(buckets))
    )
    bucket_values = torch.as_tensor(
        buckets, device=boxes.device, dtype=resized_width.dtype
    )
    bucket_index = (
        resized_width[..., None] > bucket_values[None, None]
    ).sum(dim=-1).clamp_max(len(buckets) - 1)

    # Inverse projective map from the rectified rectangle to the source quad.
    delta = tl - tr + br - bl
    v1 = tr - br
    v2 = bl - br
    denominator = v1[..., 0] * v2[..., 1] - v2[..., 0] * v1[..., 1]
    safe_denominator = torch.where(
        denominator.abs() > 1.0e-8, denominator, torch.ones_like(denominator)
    )
    g = (delta[..., 0] * v2[..., 1] - v2[..., 0] * delta[..., 1]) / safe_denominator
    k = (v1[..., 0] * delta[..., 1] - delta[..., 0] * v1[..., 1]) / safe_denominator
    a0 = tr[..., 0] - tl[..., 0] + g * tr[..., 0]
    a1 = bl[..., 0] - tl[..., 0] + k * bl[..., 0]
    a2 = tl[..., 0]
    b0 = tr[..., 1] - tl[..., 1] + g * tr[..., 1]
    b1 = bl[..., 1] - tl[..., 1] + k * bl[..., 1]
    b2 = tl[..., 1]

    source_h, source_w = images_bgr.shape[-2:]
    source = images_bgr.float()
    outputs: list[torch.Tensor] = []
    selected_slots: list[torch.Tensor] = []
    # Construct sampling grids only for compacted ROI rows.  The previous
    # implementation first broadcast every grid over the full fixed DB
    # capacity (normally 64 slots) and discarded almost all of that work after
    # ``nonzero``.  Selection depends only on the scalar box geometry, so it is
    # equivalent and substantially cheaper to compact those scalars first.
    flat_width_all = width.reshape(-1)
    flat_height_all = height.reshape(-1)
    flat_long_all = long_side.reshape(-1)
    flat_short_all = short_side.reshape(-1)
    flat_rotate_all = rotate.reshape(-1)
    flat_resized_width_all = resized_width.reshape(-1)
    flat_coefficients = [item.reshape(-1) for item in (a0, a1, a2, b0, b1, b2, g, k)]
    yy = _normalised_axis(rec_height, boxes.device)[None, :, None]
    for bucket_slot, bucket_width in enumerate(buckets):
        slot_valid = valid & (bucket_index == bucket_slot) & (denominator.abs() > 1.0e-8)
        selected = torch.nonzero(slot_valid.reshape(-1), as_tuple=False).flatten()
        selected_slots.append(selected)
        # One dummy guarantees a legal non-empty REC batch for an unused bucket.
        selected_with_dummy = torch.cat(
            (selected, torch.zeros(1, device=selected.device, dtype=selected.dtype))
        )
        source_rows = torch.div(selected_with_dummy, capacity, rounding_mode="floor")
        flat_width = flat_width_all.index_select(0, selected_with_dummy)
        flat_height = flat_height_all.index_select(0, selected_with_dummy)
        flat_long = flat_long_all.index_select(0, selected_with_dummy)
        flat_short = flat_short_all.index_select(0, selected_with_dummy)
        flat_rotate = flat_rotate_all.index_select(0, selected_with_dummy)
        active_width_selected = flat_resized_width_all.index_select(
            0, selected_with_dummy
        ).clamp_max(float(bucket_width))
        coefficients = [
            item.index_select(0, selected_with_dummy) for item in flat_coefficients
        ]
        xx = _normalised_axis(bucket_width, boxes.device)[None, None, :]
        rx = xx * flat_long[:, None, None] / active_width_selected[:, None, None] - 0.5
        ry = yy * flat_short[:, None, None] / float(rec_height) - 0.5
        selected_crop_x = torch.where(
            flat_rotate[:, None, None], flat_width[:, None, None] - 1.0 - ry, rx
        )
        selected_crop_y = torch.where(flat_rotate[:, None, None], rx, ry)
        selected_crop_x = selected_crop_x.clamp_min(0.0)
        selected_crop_y = selected_crop_y.clamp_min(0.0)
        selected_crop_x = torch.minimum(
            selected_crop_x, flat_width[:, None, None] - 1.0
        )
        selected_crop_y = torch.minimum(
            selected_crop_y, flat_height[:, None, None] - 1.0
        )

        def project_grid(cx: torch.Tensor, cy: torch.Tensor) -> torch.Tensor:
            u = cx / (flat_width[:, None, None] - 1.0).clamp_min(1.0)
            v = cy / (flat_height[:, None, None] - 1.0).clamp_min(1.0)
            pa0, pa1, pa2, pb0, pb1, pb2, pg, pk = coefficients
            projective_denominator = (
                pg[:, None, None] * u + pk[:, None, None] * v + 1.0
            )
            epsilon = torch.full_like(projective_denominator, 1.0e-8)
            projective_denominator = torch.where(
                projective_denominator.abs() >= epsilon,
                projective_denominator,
                torch.where(projective_denominator < 0, -epsilon, epsilon),
            )
            sx = (
                pa0[:, None, None] * u
                + pa1[:, None, None] * v
                + pa2[:, None, None]
            ) / projective_denominator
            sy = (
                pb0[:, None, None] * u
                + pb1[:, None, None] * v
                + pb2[:, None, None]
            ) / projective_denominator
            # OpenCV's first interpolation uses a 1/32 fixed-point table.
            if two_stage_approximation:
                sx = torch.round(sx * 32.0) / 32.0
                sy = torch.round(sy * 32.0) / 32.0
            sx = sx.clamp(0.0, float(source_w - 1))
            sy = sy.clamp(0.0, float(source_h - 1))
            return torch.stack(
                (
                    (sx + 0.5) * (2.0 / source_w) - 1.0,
                    (sy + 0.5) * (2.0 / source_h) - 1.0,
                ),
                dim=-1,
            )

        selected_source = source.index_select(0, source_rows)
        if two_stage_approximation:
            x0 = torch.floor(selected_crop_x)
            y0 = torch.floor(selected_crop_y)
            x1 = torch.minimum(x0 + 1.0, flat_width[:, None, None] - 1.0)
            y1 = torch.minimum(y0 + 1.0, flat_height[:, None, None] - 1.0)
            grids = torch.cat(
                (
                    project_grid(x0, y0),
                    project_grid(x1, y0),
                    project_grid(x0, y1),
                    project_grid(x1, y1),
                ),
                dim=1,
            )
            stage = F.grid_sample(
                selected_source,
                grids,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).round()
            top_left, top_right, bottom_left, bottom_right = stage.split(
                rec_height, dim=2
            )
            fx = selected_crop_x - x0
            fy = selected_crop_y - y0
            sampled = (
                (top_left * (1.0 - fx[:, None]) + top_right * fx[:, None])
                * (1.0 - fy[:, None])
                + (bottom_left * (1.0 - fx[:, None]) + bottom_right * fx[:, None])
                * fy[:, None]
            ).round()
        else:
            sampled = F.grid_sample(
                selected_source,
                project_grid(selected_crop_x, selected_crop_y),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        pixel_valid = (
            xx
            < active_width_selected[:, None, None]
        )
        real_row = torch.arange(
            selected_with_dummy.numel(), device=boxes.device
        ) < selected.numel()
        sampled = torch.where(
            pixel_valid[:, None] & real_row[:, None, None, None],
            sampled / 127.5 - 1.0,
            torch.zeros_like(sampled),
        )
        outputs.append(sampled)
    return outputs, selected_slots, bucket_index, resized_width.to(torch.int64)


def gpu_rec_required_width(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    rec_height: int = 48,
) -> torch.Tensor:
    """Return the largest post-rectification REC width required by a batch."""
    boxes = _gpu_narrow_peer_repair(boxes, valid)
    tl, tr, br, bl = _order_quad_x(boxes.float())
    width = torch.maximum(
        torch.linalg.vector_norm(tr - tl, dim=-1),
        torch.linalg.vector_norm(br - bl, dim=-1),
    ).round().clamp_min(2.0)
    height = torch.maximum(
        torch.linalg.vector_norm(bl - tl, dim=-1),
        torch.linalg.vector_norm(br - tr, dim=-1),
    ).round().clamp_min(2.0)
    long_side = torch.maximum(width, height)
    short_side = torch.minimum(width, height)
    required = torch.ceil(long_side * float(rec_height) / short_side)
    return torch.where(valid, required, torch.zeros_like(required)).amax()


def _ctc_cells(
    probabilities: torch.Tensor,
    neck: torch.Tensor,
    top_k: int,
    max_cells: int,
    supplement_label_lookup: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Build the same peak/cell top-k evidence as B training, on device."""
    values = probabilities.float()
    labels = values.argmax(dim=-1)
    greedy_score = values.gather(2, labels[..., None]).squeeze(-1)
    previous = torch.cat((torch.full_like(labels[:, :1], -1), labels[:, :-1]), 1)
    run_start = (labels != 0) & (labels != previous)
    run_index = run_start.cumsum(dim=1) - 1
    # Every frame belongs to its current nonblank run only for peak selection.
    active_run = labels != 0
    current_run = torch.where(active_run, run_index, torch.full_like(run_index, -1))
    rows, steps = labels.shape
    flat_rows = torch.arange(rows, device=values.device)[:, None].expand(rows, steps)
    keys = (flat_rows * max_cells + current_run.clamp_min(0)).reshape(-1)
    peak_scores = torch.full(
        (rows * max_cells,), -torch.inf, device=values.device, dtype=values.dtype
    )
    peak_scores.scatter_reduce_(
        0,
        keys,
        torch.where(active_run, greedy_score, torch.full_like(greedy_score, -torch.inf)).reshape(-1),
        reduce="amax",
        include_self=True,
    )
    peak_scores = peak_scores.reshape(rows, max_cells)
    frame_position = torch.arange(steps, device=values.device)[None].expand(rows, -1)
    is_peak = active_run & (
        greedy_score == peak_scores.gather(1, current_run.clamp_min(0))
    )
    # MPS implements floating-point scatter reductions but not int64 amin.
    peak_positions = torch.full(
        (rows * max_cells,), float(steps), device=values.device, dtype=torch.float32
    )
    peak_positions.scatter_reduce_(
        0,
        keys,
        torch.where(is_peak, frame_position, torch.full_like(frame_position, steps)).float().reshape(-1),
        reduce="amin",
        include_self=True,
    )
    peak_positions = peak_positions.reshape(rows, max_cells).to(torch.int64)
    raw_cell_count = run_start.sum(dim=1)
    cell_count = raw_cell_count.clamp_max(max_cells)
    cell_slot = torch.arange(max_cells, device=values.device)[None]
    cell_valid = cell_slot < cell_count[:, None]

    # Assign each frame to the closest peak. argmin resolves midpoint ties in
    # favour of the earlier cell, matching the production integer boundaries.
    distance = (
        frame_position[:, :, None] - peak_positions[:, None, :]
    ).abs().float()
    distance = torch.where(
        cell_valid[:, None], distance, torch.full_like(distance, float(steps + 1))
    )
    frame_cell = distance.argmin(dim=2)
    frame_top_value, frame_top_id = values[..., 1:].topk(top_k, dim=-1)
    # Reduce duplicate frame candidates by (row, cell, character), then top-k.
    character_count = values.shape[-1] - 1
    dense = torch.full(
        (rows * max_cells * character_count,),
        -torch.inf,
        device=values.device,
        dtype=values.dtype,
    )
    dense_key = (
        (flat_rows[:, :, None] * max_cells + frame_cell[:, :, None])
        * character_count
        + frame_top_id
    ).reshape(-1)
    dense.scatter_reduce_(
        0,
        dense_key,
        frame_top_value.reshape(-1),
        reduce="amax",
        include_self=True,
    )
    dense = dense.reshape(rows, max_cells, character_count)
    candidate_probability, candidate_ids = dense.topk(top_k, dim=-1)

    peak_safe = peak_positions.clamp_max(steps - 1)
    greedy_ids = labels.gather(1, peak_safe) - 1
    visual = neck.gather(
        1, peak_safe[..., None].expand(-1, -1, neck.shape[-1])
    )
    cell_frame_mask = frame_cell[:, None, :] == cell_slot[:, :, None]
    blank = values[..., 0][:, None, :]
    blank_max = torch.where(
        cell_frame_mask, blank, torch.full_like(blank, -torch.inf)
    ).amax(dim=2)
    blank_mean = (blank * cell_frame_mask).sum(dim=2) / cell_frame_mask.sum(dim=2).clamp_min(1)
    peak_probability = greedy_score.gather(1, peak_safe)
    probability_margin = candidate_probability[..., 0] - candidate_probability[..., 1]
    candidate_logp = candidate_probability.clamp_min(1.0e-30).log()
    scalars = torch.stack(
        (
            peak_probability,
            blank_max,
            blank_mean,
            probability_margin,
            candidate_logp[..., 0],
            candidate_logp[..., 0] - candidate_logp[..., 1],
        ),
        dim=-1,
    )
    keep_match = candidate_ids == greedy_ids[..., None]
    keep_index = keep_match.to(torch.int64).argmax(dim=-1)
    cell_valid &= keep_match.any(dim=-1)
    result = {
        "greedy_ids": greedy_ids.clamp_min(0),
        "visual": visual,
        "candidate_ids": candidate_ids,
        "candidate_logp": candidate_logp,
        "scalars": scalars,
        "keep_index": keep_index,
        "cell_valid": cell_valid,
        "cell_count": cell_count,
    }
    if supplement_label_lookup is not None:
        label_allowed = supplement_label_lookup[labels.clamp_max(
            supplement_label_lookup.numel() - 1
        )]
        all_labels_allowed = torch.where(
            run_start, label_allowed, torch.ones_like(label_allowed)
        ).all(dim=1)
        first_frame_confidence = (greedy_score * run_start).sum(dim=1) / (
            raw_cell_count.clamp_min(1)
        )
        first_frame_confidence = torch.where(
            raw_cell_count > 0,
            first_frame_confidence,
            torch.zeros_like(first_frame_confidence),
        )
        result.update(
            gate_token_count=raw_cell_count,
            gate_confidence=first_frame_confidence,
            gate_all_labels_allowed=all_labels_allowed,
        )
    return result


def _ctc_greedy_cells(
    probabilities: torch.Tensor,
    max_cells: int,
    supplement_label_lookup: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Collapse CTC runs without constructing B's H120/top-k evidence."""
    values = probabilities.float()
    labels = values.argmax(dim=-1)
    greedy_score = values.gather(2, labels[..., None]).squeeze(-1)
    previous = torch.cat((torch.full_like(labels[:, :1], -1), labels[:, :-1]), 1)
    run_start = (labels != 0) & (labels != previous)
    run_index = run_start.cumsum(dim=1) - 1
    raw_cell_count = run_start.sum(dim=1)
    cell_count = raw_cell_count.clamp_max(max_cells)
    rows = labels.shape[0]
    greedy_ids = torch.zeros(
        (rows, max_cells), device=labels.device, dtype=torch.int64
    )
    # A CTC run has one nonblank start. Scatter-add only that start into its
    # compact slot; all other frames contribute zero.
    greedy_ids.scatter_add_(
        1,
        run_index.clamp(0, max_cells - 1),
        torch.where(run_start, labels - 1, torch.zeros_like(labels)),
    )
    slot = torch.arange(max_cells, device=labels.device)[None]
    result = {
        "greedy_ids": greedy_ids.clamp_min(0),
        "cell_valid": slot < cell_count[:, None],
        "cell_count": cell_count,
    }
    if supplement_label_lookup is not None:
        label_allowed = supplement_label_lookup[
            labels.clamp_max(supplement_label_lookup.numel() - 1)
        ]
        all_labels_allowed = torch.where(
            run_start, label_allowed, torch.ones_like(label_allowed)
        ).all(dim=1)
        first_frame_confidence = (greedy_score * run_start).sum(dim=1) / (
            raw_cell_count.clamp_min(1)
        )
        result.update(
            gate_token_count=raw_cell_count,
            gate_confidence=torch.where(
                raw_cell_count > 0,
                first_frame_confidence,
                torch.zeros_like(first_frame_confidence),
            ),
            gate_all_labels_allowed=all_labels_allowed,
        )
    return result


def _pad_cell_capacity(
    cells: dict[str, torch.Tensor], target: int
) -> dict[str, torch.Tensor]:
    """Pad a bucket's cell axis after doing expensive work at its true limit."""
    current = cells["cell_valid"].shape[1]
    if current == target:
        return cells
    padded: dict[str, torch.Tensor] = {}
    for name, value in cells.items():
        if name in {
            "cell_count",
            "gate_token_count",
            "gate_confidence",
            "gate_all_labels_allowed",
        }:
            padded[name] = value
            continue
        tail = value.shape[2:]
        zeros = torch.zeros(
            (value.shape[0], target - current, *tail),
            device=value.device,
            dtype=value.dtype,
        )
        padded[name] = torch.cat((value, zeros), dim=1)
    return padded


def _merge_bucket_cells(
    bucket_cells: list[dict[str, torch.Tensor]],
    selected_slots: list[torch.Tensor],
    batch: int,
    capacity: int,
    roi_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    merged: dict[str, torch.Tensor] = {}
    for name in ("greedy_ids", "visual", "candidate_ids", "candidate_logp", "scalars", "keep_index"):
        tail = bucket_cells[0][name].shape[1:]
        result = torch.zeros(
            (batch * capacity, *tail),
            device=bucket_cells[0][name].device,
            dtype=bucket_cells[0][name].dtype,
        )
        for selected, piece in zip(selected_slots, bucket_cells, strict=True):
            result.index_copy_(0, selected, piece[name])
        merged[name] = result.reshape(batch, capacity, *tail)
    cell_valid = torch.zeros(
        (batch * capacity, bucket_cells[0]["cell_valid"].shape[1]),
        device=roi_valid.device,
        dtype=torch.bool,
    )
    for selected, piece in zip(selected_slots, bucket_cells, strict=True):
        cell_valid.index_copy_(0, selected, piece["cell_valid"])
    cell_valid = cell_valid.reshape(batch, capacity, -1)
    merged["cell_valid"] = cell_valid & roi_valid[..., None]
    return merged


def _merge_greedy_bucket_cells(
    bucket_cells: list[dict[str, torch.Tensor]],
    selected_slots: list[torch.Tensor],
    batch: int,
    capacity: int,
    roi_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Merge the compact CTC payload used by the public no-B variant."""
    cell_capacity = bucket_cells[0]["greedy_ids"].shape[1]
    greedy = torch.zeros(
        (batch * capacity, cell_capacity),
        device=roi_valid.device,
        dtype=torch.int64,
    )
    valid = torch.zeros_like(greedy, dtype=torch.bool)
    for selected, piece in zip(selected_slots, bucket_cells, strict=True):
        greedy.index_copy_(0, selected, piece["greedy_ids"])
        valid.index_copy_(0, selected, piece["cell_valid"])
    return {
        "greedy_ids": greedy.reshape(batch, capacity, cell_capacity),
        "cell_valid": valid.reshape(batch, capacity, cell_capacity)
        & roi_valid[..., None],
    }


def _packed_cell_indices(
    cell_valid: torch.Tensor,
    maximum: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map packed output positions to flattened ROI/cell positions.

    Valid CTC cells are a contiguous prefix inside every ROI.  Use per-ROI
    counts and a small fixed ``[output, ROI]`` comparison to locate them.  This
    avoids flattening thousands of capacity slots followed by repeated boolean
    advanced-index assignments, which are particularly expensive on MPS.
    """
    batch, capacity, per_roi = cell_valid.shape
    roi_count = cell_valid.sum(dim=2, dtype=torch.int64)
    roi_end = roi_count.cumsum(dim=1)
    total = roi_end[:, -1].clamp_max(maximum)
    output_position = torch.arange(
        maximum, device=cell_valid.device, dtype=torch.int64
    )[None, :]
    # ``right=True`` skips zero-length ROIs and maps every compact output
    # position to the first cumulative end strictly greater than it.  On MPS
    # this is also cheaper than materialising [output, ROI] comparisons.
    roi_index = torch.searchsorted(
        roi_end.contiguous(),
        output_position.expand(batch, -1).contiguous(),
        right=True,
    )
    roi_index = roi_index.clamp_max(capacity - 1)
    previous_roi = (roi_index - 1).clamp_min(0)
    previous_end = roi_end.gather(1, previous_roi)
    previous_end = torch.where(
        roi_index == 0, torch.zeros_like(previous_end), previous_end
    )
    cell_index = output_position - previous_end
    packed_valid = output_position < total[:, None]
    # Invalid tail positions are masked after gather; clamp them to a legal
    # address so no data-dependent branch or host readback is required.
    cell_index = cell_index.clamp(0, per_roi - 1)
    flat_index = roi_index * per_roi + cell_index
    return flat_index, packed_valid, total, cell_index


def _gather_packed(
    source: torch.Tensor,
    cell_valid: torch.Tensor,
    flat_index: torch.Tensor,
) -> torch.Tensor:
    """Gather arbitrary cell payloads with one fixed-shape device operation."""
    batch = cell_valid.shape[0]
    flattened = cell_valid.shape[1] * cell_valid.shape[2]
    tail = source.shape[cell_valid.ndim :]
    flat = source.reshape(batch, flattened, *tail)
    index = flat_index.reshape(batch, flat_index.shape[1], *([1] * len(tail)))
    index = index.expand(batch, flat_index.shape[1], *tail)
    return flat.gather(1, index)


def _pack_b_input(
    cells: dict[str, torch.Tensor],
    track_ids: torch.Tensor,
    config,
    max_characters: int,
    packing: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    cell_valid = cells["cell_valid"]
    batch = cell_valid.shape[0]
    if packing is None:
        flat_index, packed_valid, count, cell_index = _packed_cell_indices(
            cell_valid, max_characters
        )
    else:
        flat_index, packed_valid, count, cell_index = packing
    width = max_characters + 2

    device = cell_valid.device
    input_ids = torch.full((batch, width), config.pad_token_id, device=device, dtype=torch.int64)
    attention = torch.zeros((batch, width), device=device, dtype=torch.bool)
    starts = torch.zeros((batch, width), device=device, dtype=torch.int64)
    visual = torch.zeros((batch, width, 120), device=device)
    scalars = torch.zeros((batch, width, 6), device=device)
    candidates = torch.zeros((batch, width, 16), device=device, dtype=torch.int64)
    candidate_logp = torch.zeros((batch, width, 16), device=device)
    keep_index = torch.zeros((batch, width), device=device, dtype=torch.int64)
    cell_mask = torch.zeros((batch, width), device=device, dtype=torch.bool)
    input_ids[:, 0] = config.bos_token_id
    attention[:, 0] = True
    gathered_ids = _gather_packed(cells["greedy_ids"], cell_valid, flat_index)
    gathered_visual = _gather_packed(cells["visual"], cell_valid, flat_index)
    gathered_scalars = _gather_packed(cells["scalars"], cell_valid, flat_index)
    gathered_candidates = _gather_packed(cells["candidate_ids"], cell_valid, flat_index)
    gathered_candidate_logp = _gather_packed(cells["candidate_logp"], cell_valid, flat_index)
    gathered_keep_index = _gather_packed(cells["keep_index"], cell_valid, flat_index)
    input_ids[:, 1 : max_characters + 1] = torch.where(
        packed_valid, gathered_ids, torch.full_like(gathered_ids, config.pad_token_id)
    )
    visual[:, 1 : max_characters + 1] = torch.where(
        packed_valid[..., None], gathered_visual, torch.zeros_like(gathered_visual)
    )
    scalars[:, 1 : max_characters + 1] = torch.where(
        packed_valid[..., None], gathered_scalars, torch.zeros_like(gathered_scalars)
    )
    candidates[:, 1 : max_characters + 1] = torch.where(
        packed_valid[..., None], gathered_candidates, torch.zeros_like(gathered_candidates)
    )
    candidate_logp[:, 1 : max_characters + 1] = torch.where(
        packed_valid[..., None], gathered_candidate_logp, torch.zeros_like(gathered_candidate_logp)
    )
    keep_index[:, 1 : max_characters + 1] = torch.where(
        packed_valid, gathered_keep_index, torch.zeros_like(gathered_keep_index)
    )
    cell_mask[:, 1 : max_characters + 1] = packed_valid
    attention[:, 1 : max_characters + 1] = packed_valid
    # Every compacted ROI begins a new training track, including fragments that
    # A assigns to one geometric track.  Contiguous-prefix cells make this
    # exactly equivalent to the previous per-ROI cumsum implementation.
    starts[:, 1 : max_characters + 1] = (packed_valid & (cell_index == 0)).to(
        torch.int64
    )

    eos_position = count + 1
    input_ids.scatter_(1, eos_position[:, None], config.eos_token_id)
    attention.scatter_(1, eos_position[:, None], True)
    return [input_ids, attention, starts, visual, scalars, candidates, candidate_logp, keep_index, cell_mask], count, packed_valid


def _pack_source_ids(
    cells: dict[str, torch.Tensor], max_output_characters: int
) -> tuple[torch.Tensor, torch.Tensor]:
    cell_valid = cells["cell_valid"]
    flat_index, packed_valid, count, _ = _packed_cell_indices(
        cell_valid, max_output_characters
    )
    gathered = _gather_packed(cells["greedy_ids"], cell_valid, flat_index)
    return torch.where(packed_valid, gathered, torch.zeros_like(gathered)), count


class V1PreviewPipeline(nn.Module):
    """Assembled v1Preview OCR data path with device-only decisions."""

    def __init__(
        self,
        det: nn.Module,
        a_postprocess: SmartGpuDBPostProcess,
        rec: nn.Module,
        b: FusionBVisualTop16 | None,
        *,
        b_refiner: FusionBTop3Verifier | None = None,
        editable_character_lookup: torch.Tensor,
        supplement_token_ids: torch.Tensor,
        config: V1PreviewConfig,
    ) -> None:
        super().__init__()
        self.det = det.eval()
        self.a = a_postprocess
        self.rec = rec.eval()
        # The refiner owns the same B module when enabled.  Keep only one
        # registered path so state/device traversal does not duplicate it.
        self.b_refiner = b_refiner.eval() if b_refiner is not None else None
        self.b = b.eval() if b_refiner is None and b is not None else None
        self.register_buffer("editable_character_lookup", editable_character_lookup.bool())
        self.register_buffer("supplement_token_ids", supplement_token_ids.long())
        supplement_label_lookup = torch.zeros(
            editable_character_lookup.numel() + 1, dtype=torch.bool
        )
        supplement_label_lookup[(supplement_token_ids.long() + 1).clamp_max(
            supplement_label_lookup.numel() - 1
        )] = True
        self.register_buffer("supplement_label_lookup", supplement_label_lookup)
        self.config = config
        self._neck: torch.Tensor | None = None
        self._hook = self.rec.head.ctc_encoder.register_forward_hook(self._capture_neck)

    def _capture_neck(self, _module, _inputs, output) -> None:
        self._neck = output

    def _run_rec(self, crops: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._neck = None
        probabilities = self.rec(crops)
        if self._neck is None:
            raise RuntimeError("REC ctc_encoder hook did not capture H120")
        return probabilities, self._neck

    @torch.inference_mode()
    def forward(
        self, images_bgr: torch.Tensor, source_sizes_hw: torch.Tensor
    ) -> V1PreviewOutput:
        # DET letterboxing and REC warping both need the same float image.  Keep
        # one device copy alive instead of converting the uint8 source twice.
        source_bgr = images_bgr.float()
        det_input, geometry = gpu_letterbox_224(
            source_bgr, source_sizes_hw, self.config.det_size
        )
        probability = self.det(det_input)["maps"]
        a_result = self.a(probability, source_geometry=geometry)
        rec_buckets: Sequence[int] = self.config.rec_buckets
        if self.config.rec_single_batch:
            required_width = gpu_rec_required_width(
                a_result.boxes, a_result.valid, self.config.rec_height
            )
            choices = torch.as_tensor(
                self.config.rec_buckets,
                device=required_width.device,
                dtype=required_width.dtype,
            )
            # A dense tensor batch has one shared width.  Eager MPS/CUDA needs
            # this one scalar synchronization to select which tensor shape to
            # allocate; pixels and OCR decisions remain on the accelerator.
            common_index = int((choices - required_width).abs().argmin().item())
            rec_buckets = (self.config.rec_buckets[common_index],)
        crops, selected_slots, bucket_index, _ = gpu_rec_bucket_crops(
            source_bgr,
            a_result.boxes,
            a_result.valid,
            rec_buckets,
            self.config.rec_height,
            self.config.rec_two_stage_approximation,
        )
        batch, capacity = a_result.valid.shape
        max_cells = self.config.b_max_characters
        bucket_probabilities: list[torch.Tensor] = []
        bucket_necks: list[torch.Tensor] = []
        for crop in crops:
            probabilities, neck = self._run_rec(crop)
            bucket_probabilities.append(probabilities)
            bucket_necks.append(neck)
        shared_cell_capacity = min(
            max_cells, max(item.shape[1] for item in bucket_probabilities)
        )
        bucket_cells: list[dict[str, torch.Tensor]] = []
        for probabilities, neck in zip(bucket_probabilities, bucket_necks, strict=True):
            # The final row is the masked dummy used only to make an empty REC
            # bucket executable.  Exclude it before the vocabulary-wide CTC
            # aggregation: processing fake rows was a sizeable fraction of
            # this glue stage for the common one-to-four ROI case.  MPS topk
            # cannot execute a zero-row tensor, so a genuinely empty bucket
            # retains its dummy for the aggregation and drops it immediately
            # afterwards.  The three-call fallback launches every configured
            # bucket; the default common-bucket path contains only one item.
            real_rows = probabilities.shape[0] - 1
            processed_rows = max(real_rows, 1)
            if self.b is None and self.b_refiner is None:
                processed = _ctc_greedy_cells(
                    probabilities[:processed_rows],
                    min(shared_cell_capacity, probabilities.shape[1]),
                    self.supplement_label_lookup,
                )
            else:
                processed = _ctc_cells(
                    probabilities[:processed_rows],
                    neck[:processed_rows],
                    self.config.top_k,
                    min(shared_cell_capacity, probabilities.shape[1]),
                    self.supplement_label_lookup,
                )
            bucket_cells.append(
                _pad_cell_capacity(
                    {name: value[:real_rows] for name, value in processed.items()},
                    shared_cell_capacity,
                )
            )

        # Choose the matching bucket's REC sequence and apply the existing
        # supplement gate.  DET-empty recall-first accepts any nonempty CTC
        # output; no alternate-view result is read by the host.
        accepted_flat = torch.zeros(batch * capacity, device=images_bgr.device, dtype=torch.bool)
        for selected, piece in zip(selected_slots, bucket_cells, strict=True):
            flat_supplemental = a_result.supplemental.reshape(-1).index_select(0, selected)
            supplement_accepted = (
                (piece["gate_token_count"] >= 1)
                & piece["gate_all_labels_allowed"]
                & (
                    piece["gate_confidence"]
                    >= self.config.supplement_minimum_confidence
                )
                & (
                    a_result.supplement_area_ratio.reshape(-1).index_select(
                        0, selected
                    )
                    >= self.config.supplement_minimum_area_ratio
                )
                & (
                    a_result.supplement_distance.reshape(-1).index_select(
                        0, selected
                    )
                    <= self.config.supplement_maximum_distance
                )
            )
            gate_accepted = ~flat_supplemental.bool() | supplement_accepted
            fallback = a_result.fallback_fullcrop[:, None].expand(-1, capacity).reshape(-1).index_select(0, selected)
            accepted = torch.where(
                fallback, piece["gate_token_count"] > 0, gate_accepted
            )
            accepted_flat.index_copy_(0, selected, accepted)
        roi_accepted = accepted_flat.reshape(batch, capacity) & a_result.valid

        if self.b is None and self.b_refiner is None:
            cells = _merge_greedy_bucket_cells(
                bucket_cells, selected_slots, batch, capacity, roi_accepted
            )
        else:
            cells = _merge_bucket_cells(
                bucket_cells, selected_slots, batch, capacity, roi_accepted
            )
        # Compute the compact cell-to-source mapping once.  B consumes its
        # prefix, while the full mapping preserves unusually long REC outputs.
        # The old path independently repeated cumsum/search/gather setup for the
        # source output and B input.
        full_flat_index, full_packed_valid, full_token_count, full_cell_index = (
            _packed_cell_indices(cells["cell_valid"], self.config.max_output_characters)
        )
        full_greedy = _gather_packed(
            cells["greedy_ids"], cells["cell_valid"], full_flat_index
        )
        full_source_ids = torch.where(
            full_packed_valid, full_greedy, torch.zeros_like(full_greedy)
        )
        if self.b is None and self.b_refiner is None:
            active = (
                torch.arange(
                    self.config.max_output_characters,
                    device=full_source_ids.device,
                )[None]
                < full_token_count[:, None]
            )
            source_ids = torch.where(
                active, full_source_ids, torch.zeros_like(full_source_ids)
            )
            return V1PreviewOutput(
                token_ids=source_ids,
                token_count=full_token_count,
                source_token_ids=source_ids,
                edited=torch.zeros_like(source_ids, dtype=torch.bool),
                accepted_rois=roi_accepted,
                box_count=a_result.count,
                boxes=a_result.boxes,
                overflow=a_result.overflow,
            )
        b_core = self.b_refiner.b if self.b_refiner is not None else self.b
        if b_core is None:
            raise RuntimeError("Fusion B is not configured")
        tensors, token_count, _ = _pack_b_input(
            cells,
            a_result.track_ids,
            b_core.config,
            self.config.b_max_characters,
            packing=(
                full_flat_index[:, : self.config.b_max_characters],
                full_packed_valid[:, : self.config.b_max_characters],
                full_token_count.clamp_max(self.config.b_max_characters),
                full_cell_index[:, : self.config.b_max_characters],
            ),
        )
        (
            input_ids,
            attention,
            track_starts,
            visual,
            scalars,
            candidate_ids,
            candidate_logp,
            keep_index,
            cell_mask,
        ) = tensors
        if self.b_refiner is not None:
            refined = self.b_refiner(
                input_ids,
                attention,
                track_starts,
                visual,
                scalars,
                candidate_ids,
                candidate_logp,
                keep_index,
                cell_mask,
            )
            alternate_ids = refined.replacement_ids
            take = refined.take
        else:
            addition = b_core.scalar_projection(b_core.scalar_norm(scalars))
            addition = addition + b_core.visual_projection(b_core.visual_norm(visual))
            hidden = b_core.backbone.encode(input_ids, attention, track_starts, addition)
            query = b_core.candidate_projection(hidden)
            candidate_embedding = b_core.backbone.character_embedding(candidate_ids)
            scores = (candidate_embedding * query[:, :, None]).sum(dim=-1) / math.sqrt(
                b_core.config.hidden_size
            )
            scores = scores + torch.exp(b_core.ctc_log_scale).clamp(max=10.0) * candidate_logp
            scores = scores + b_core.rank_bias
            scores.scatter_add_(2, keep_index[..., None], b_core.keep_bias.expand_as(keep_index[..., None]))
            keep_score = scores.gather(2, keep_index[..., None]).squeeze(-1)
            alternate = scores.scatter(2, keep_index[..., None], -torch.inf)
            alt_score, alt_index = alternate.max(dim=2)
            alternate_ids = candidate_ids.gather(2, alt_index[..., None]).squeeze(-1)
            source_editable = self.editable_character_lookup[input_ids.clamp_max(self.editable_character_lookup.numel() - 1)]
            target_editable = self.editable_character_lookup[alternate_ids.clamp_max(self.editable_character_lookup.numel() - 1)]
            take = (
                cell_mask
                & source_editable
                & target_editable
                & ((alt_score - keep_score) > self.config.replacement_threshold)
            )
        take = take[:, 1 : self.config.b_max_characters + 1]
        # Do not apply a context-truncated B decision to one of the extremely
        # rare blocks longer than its semantic window.
        take &= (full_token_count <= self.config.b_max_characters)[:, None]
        chosen_prefix = torch.where(
            take,
            alternate_ids[:, 1 : self.config.b_max_characters + 1],
            input_ids[:, 1 : self.config.b_max_characters + 1],
        )
        output_ids = full_source_ids.clone()
        output_ids[:, : self.config.b_max_characters] = chosen_prefix
        active = (
            torch.arange(self.config.max_output_characters, device=output_ids.device)[None]
            < full_token_count[:, None]
        )
        output_ids = torch.where(active, output_ids, torch.zeros_like(output_ids))
        source_ids = torch.where(active, full_source_ids, torch.zeros_like(full_source_ids))
        edited = torch.zeros_like(output_ids, dtype=torch.bool)
        edited[:, : self.config.b_max_characters] = take
        return V1PreviewOutput(
            token_ids=output_ids,
            token_count=full_token_count,
            source_token_ids=source_ids,
            edited=edited & active,
            accepted_rois=roi_accepted,
            box_count=a_result.count,
            boxes=a_result.boxes,
            overflow=a_result.overflow,
        )


def warmup_v1preview(
    model: V1PreviewPipeline,
    images_bgr: torch.Tensor,
    source_sizes_hw: torch.Tensor,
    repetitions: int = 3,
) -> None:
    """Materialize the REC shape selected by the supplied warm-up input."""
    for _ in range(repetitions):
        model(images_bgr, source_sizes_hw)
    if images_bgr.device.type == "mps":
        torch.mps.synchronize()
    elif images_bgr.device.type == "cuda":
        torch.cuda.synchronize(images_bgr.device)

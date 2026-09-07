"""Unified MPS/CUDA smart DB post-processing.

This module adds conservative probability-aware expansion, a whole-image
candidate for zero-box inputs, and device-resident reading order to the fixed
capacity DB decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .postprocess import GpuDBConfig, GpuDBPostProcess, GpuDBResult


@dataclass(frozen=True)
class SmartGpuDBConfig:
    db: GpuDBConfig = field(
        default_factory=lambda: GpuDBConfig(max_components=64)
    )
    support_threshold: float = 0.20
    maximum_support_area_ratio: float = 0.25
    minimum_support_pixels: int = 2
    maximum_support_gap: float = 2.5
    minimum_support_extent: float = 0.5
    support_margin: float = 0.75
    cross_cap_ratio: float = 0.35
    end_cap_ratio: float = 0.85
    minimum_cap: float = 2.0
    maximum_cap: float = 18.0
    fullcrop_on_empty: bool = True
    supplement_db: GpuDBConfig | None = None
    # The primary and supplement decoders normally repeat connected components
    # and rotated-box fitting on the same thresholded probability map.  When
    # their geometric settings match, decode once with the permissive
    # supplement thresholds and derive the primary validity mask from its
    # scores and fitted box size.
    reuse_supplement_decode: bool = False
    supplement_duplicate_tolerance: float = 1.0e-3
    order_before_expansion: bool = True
    robust_reading_order: bool = True
    expanded_as_primary: bool = False
    optional_maximum_area_ratio: float = 0.13
    optional_border_maximum_area_ratio: float = 0.50
    optional_extreme_aspect_ratio: float = 12.0
    optional_maximum_score: float = 0.55
    optional_border_margin: float = 1.5


@dataclass(frozen=True)
class SmartGpuDBResult:
    boxes: torch.Tensor
    base_boxes: torch.Tensor
    expanded_boxes: torch.Tensor
    scores: torch.Tensor
    valid: torch.Tensor
    count: torch.Tensor
    raw_component_count: torch.Tensor
    overflow: torch.Tensor
    expanded_sides: torch.Tensor
    supplemental: torch.Tensor
    supplement_area_ratio: torch.Tensor
    supplement_distance: torch.Tensor
    optional: torch.Tensor
    base_area_ratio: torch.Tensor
    border_contact: torch.Tensor
    fallback_fullcrop: torch.Tensor
    order_from_decoder: torch.Tensor
    track_ids: torch.Tensor
    vertical: torch.Tensor
    skew_radians: torch.Tensor
    labels: torch.Tensor | None = None


def _adaptive_expand(
    probability: torch.Tensor,
    boxes: torch.Tensor,
    valid: torch.Tensor,
    config: SmartGpuDBConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    arguments = {
        "threshold": config.support_threshold,
        "maximum_support_area_ratio": config.maximum_support_area_ratio,
        "minimum_pixels": config.minimum_support_pixels,
        "maximum_gap": config.maximum_support_gap,
        "minimum_extent": config.minimum_support_extent,
        "support_margin": config.support_margin,
        "cross_cap_ratio": config.cross_cap_ratio,
        "end_cap_ratio": config.end_cap_ratio,
        "minimum_cap": config.minimum_cap,
        "maximum_cap": config.maximum_cap,
    }
    if probability.device.type == "mps":
        from .smart_mps import adaptive_expand_mps

        return adaptive_expand_mps(probability, boxes, valid, **arguments)
    if probability.device.type == "cuda":
        from .smart_torch import adaptive_expand_torch

        return adaptive_expand_torch(probability, boxes, valid, **arguments)
    raise ValueError("smart DB post-processing requires CUDA or MPS")


def _reading_order(
    boxes: torch.Tensor,
    valid: torch.Tensor,
    *,
    robust: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if boxes.device.type == "mps":
        from .smart_mps import reading_order_mps

        return reading_order_mps(boxes, valid, robust=robust)
    if boxes.device.type == "cuda":
        from .smart_torch import reading_order_torch

        return reading_order_torch(boxes, valid, robust=robust)
    raise ValueError("smart DB post-processing requires CUDA or MPS")


def _quad_area(boxes: torch.Tensor) -> torch.Tensor:
    following = boxes.roll(shifts=-1, dims=2)
    signed_area = (
        boxes[..., 0] * following[..., 1]
        - boxes[..., 1] * following[..., 0]
    ).sum(dim=2)
    return 0.5 * signed_area.abs()


def _map_to_source(
    boxes: torch.Tensor,
    geometry: torch.Tensor,
) -> torch.Tensor:
    scale = geometry[:, 0].clamp_min(1e-6)
    offset = geometry[:, 1:3]
    mapped = (boxes - offset[:, None, None]) / scale[:, None, None, None]
    x = mapped[..., 0].clamp_min(0.0)
    y = mapped[..., 1].clamp_min(0.0)
    x = torch.minimum(x, (geometry[:, 3] - 1.0)[:, None, None])
    y = torch.minimum(y, (geometry[:, 4] - 1.0)[:, None, None])
    return torch.stack([x, y], dim=-1)


def _median_valid_area(areas: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    infinity = torch.full_like(areas, float("inf"))
    ordered = torch.where(valid, areas, infinity).sort(dim=1).values
    count = valid.sum(dim=1)
    lower_index = ((count - 1).clamp_min(0) // 2).to(torch.int64)
    upper_index = (count.clamp_min(1) // 2).to(torch.int64)
    lower = ordered.gather(1, lower_index[:, None])[:, 0]
    upper = ordered.gather(1, upper_index[:, None])[:, 0]
    median = 0.5 * (lower + upper)
    return torch.where(count > 0, median, torch.ones_like(median)).clamp_min(1e-6)


def _optional_primary_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    valid: torch.Tensor,
    supplemental: torch.Tensor,
    fallback: torch.Tensor,
    coordinate_width: torch.Tensor,
    coordinate_height: torch.Tensor,
    config: SmartGpuDBConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    primary = valid & ~supplemental
    areas = _quad_area(boxes)
    median_area = _median_valid_area(areas, primary)
    area_ratio = areas / median_area[:, None]
    width = 0.5 * (
        torch.linalg.vector_norm(boxes[:, :, 1] - boxes[:, :, 0], dim=-1)
        + torch.linalg.vector_norm(boxes[:, :, 2] - boxes[:, :, 3], dim=-1)
    )
    height = 0.5 * (
        torch.linalg.vector_norm(boxes[:, :, 3] - boxes[:, :, 0], dim=-1)
        + torch.linalg.vector_norm(boxes[:, :, 2] - boxes[:, :, 1], dim=-1)
    )
    aspect = torch.maximum(width, height) / torch.minimum(width, height).clamp_min(
        1.0e-6
    )
    margin = config.optional_border_margin
    border_contact = (
        (boxes[..., 0] <= margin)
        | (boxes[..., 1] <= margin)
        | (boxes[..., 0] >= coordinate_width[:, None, None] - 1.0 - margin)
        | (boxes[..., 1] >= coordinate_height[:, None, None] - 1.0 - margin)
    ).any(dim=2)
    optional = primary & (
        (area_ratio <= config.optional_maximum_area_ratio)
        | (
            border_contact
            & (area_ratio <= config.optional_border_maximum_area_ratio)
        )
        | (aspect >= config.optional_extreme_aspect_ratio)
        | (scores <= config.optional_maximum_score)
    )
    optional = optional & ~(
        fallback[:, None]
        & (torch.arange(boxes.shape[1], device=boxes.device)[None] == 0)
    )
    return optional, area_ratio, border_contact


def _combine_supplements(
    base_boxes: torch.Tensor,
    base_match_boxes: torch.Tensor,
    base_scores: torch.Tensor,
    base_valid: torch.Tensor,
    base_expanded: torch.Tensor,
    supplement_boxes: torch.Tensor,
    supplement_scores: torch.Tensor,
    supplement_valid: torch.Tensor,
    duplicate_tolerance: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    capacity = base_boxes.shape[1]
    difference = (
        supplement_boxes[:, :, None] - base_match_boxes[:, None]
    ).abs().amax(dim=(-1, -2))
    duplicate = (
        (difference <= duplicate_tolerance) & base_valid[:, None]
    ).any(dim=2)
    supplement_valid = supplement_valid & ~duplicate & base_valid.any(dim=1)[:, None]

    base_areas = _quad_area(base_boxes)
    supplement_areas = _quad_area(supplement_boxes)
    median_area = _median_valid_area(base_areas, base_valid)
    supplement_area_ratio = supplement_areas / median_area[:, None]
    base_centres = base_boxes.mean(dim=2)
    supplement_centres = supplement_boxes.mean(dim=2)
    distances = torch.linalg.vector_norm(
        supplement_centres[:, :, None] - base_centres[:, None], dim=-1
    )
    distances = torch.where(
        base_valid[:, None], distances, torch.full_like(distances, float("inf"))
    )
    supplement_distance = distances.amin(dim=2)

    all_boxes = torch.cat([base_boxes, supplement_boxes], dim=1)
    all_scores = torch.cat([base_scores, supplement_scores], dim=1)
    all_valid = torch.cat([base_valid, supplement_valid], dim=1)
    all_supplemental = torch.cat(
        [torch.zeros_like(base_valid), supplement_valid], dim=1
    )
    all_area_ratio = torch.cat(
        [torch.zeros_like(base_scores), supplement_area_ratio], dim=1
    )
    all_distance = torch.cat(
        [torch.zeros_like(base_scores), supplement_distance], dim=1
    )
    all_expanded = torch.cat(
        [base_expanded, torch.zeros_like(base_expanded)], dim=1
    )
    priority = torch.cat(
        [
            torch.where(base_valid, base_scores + 2.0, torch.full_like(base_scores, -1.0)),
            torch.where(
                supplement_valid,
                supplement_scores,
                torch.full_like(supplement_scores, -1.0),
            ),
        ],
        dim=1,
    )
    selected = priority.topk(capacity, dim=1, largest=True, sorted=True).indices
    boxes = all_boxes.gather(
        1, selected[:, :, None, None].expand(-1, -1, 4, 2)
    )
    match_boxes = torch.cat([base_match_boxes, supplement_boxes], dim=1).gather(
        1, selected[:, :, None, None].expand(-1, -1, 4, 2)
    )
    scores = all_scores.gather(1, selected)
    valid = all_valid.gather(1, selected)
    supplemental = all_supplemental.gather(1, selected)
    area_ratio = all_area_ratio.gather(1, selected)
    distance = all_distance.gather(1, selected)
    expanded = all_expanded.gather(
        1, selected[:, :, None].expand(-1, -1, 4)
    )
    return (
        boxes,
        match_boxes,
        scores,
        valid,
        expanded,
        supplemental,
        area_ratio,
        distance,
    )


class SmartGpuDBPostProcess:
    """Decode, refine, order and optionally create an empty-DET fallback slot."""

    def __init__(self, config: SmartGpuDBConfig | None = None) -> None:
        self.config = config or SmartGpuDBConfig()
        self.decoder = GpuDBPostProcess(self.config.db)
        self.supplement_decoder = (
            GpuDBPostProcess(self.config.supplement_db)
            if self.config.supplement_db is not None
            else None
        )

    def _can_reuse_supplement_decode(self) -> bool:
        base = self.config.db
        supplement = self.config.supplement_db
        if not self.config.reuse_supplement_decode or supplement is None:
            return False
        return (
            base.threshold == supplement.threshold
            and base.unclip_ratio == supplement.unclip_ratio
            and base.max_components == supplement.max_components
            and base.union_find_rounds == supplement.union_find_rounds
            and base.angle_offsets_degrees == supplement.angle_offsets_degrees
            and base.angle_refine_offsets_degrees
            == supplement.angle_refine_offsets_degrees
        )

    def _shared_decode(
        self, source: torch.Tensor, *, return_labels: bool
    ) -> tuple[GpuDBResult, GpuDBResult]:
        if self.supplement_decoder is None:
            raise RuntimeError("shared decode requires a supplement decoder")
        supplement = self.supplement_decoder(source, return_labels=return_labels)
        boxes = supplement.boxes
        width = 0.5 * (
            torch.linalg.vector_norm(boxes[:, :, 1] - boxes[:, :, 0], dim=-1)
            + torch.linalg.vector_norm(boxes[:, :, 2] - boxes[:, :, 3], dim=-1)
        )
        height = 0.5 * (
            torch.linalg.vector_norm(boxes[:, :, 3] - boxes[:, :, 0], dim=-1)
            + torch.linalg.vector_norm(boxes[:, :, 2] - boxes[:, :, 1], dim=-1)
        )
        post_unclip_minimum = self.config.db.min_size + 2.0
        # The native decoder checks size before clamping corners to the map.
        # Preserve valid components touching an edge even if the clipped quad
        # is slightly shorter than that internal extent.
        map_height, map_width = source.shape[-2:]
        border = (
            (boxes[..., 0] <= 1.0e-4)
            | (boxes[..., 1] <= 1.0e-4)
            | (boxes[..., 0] >= float(map_width) - 1.0e-4)
            | (boxes[..., 1] >= float(map_height) - 1.0e-4)
        ).any(dim=2)
        primary_valid = (
            supplement.valid
            & (supplement.scores >= self.config.db.box_threshold)
            & (
                (torch.minimum(width, height) >= post_unclip_minimum)
                | border
            )
        )
        primary = GpuDBResult(
            boxes=supplement.boxes,
            scores=supplement.scores,
            valid=primary_valid,
            count=primary_valid.sum(dim=1, dtype=torch.int32),
            raw_component_count=supplement.raw_component_count,
            overflow=supplement.overflow,
            labels=supplement.labels,
        )
        return primary, supplement

    @torch.inference_mode()
    def __call__(
        self,
        probability: torch.Tensor,
        *,
        destination_sizes: torch.Tensor | None = None,
        source_geometry: torch.Tensor | None = None,
        return_labels: bool = False,
    ) -> SmartGpuDBResult:
        if destination_sizes is not None and source_geometry is not None:
            raise ValueError("destination_sizes and source_geometry are mutually exclusive")
        source = probability.contiguous().float()
        shared_decode = self._can_reuse_supplement_decode()
        if shared_decode:
            base, supplement = self._shared_decode(
                source, return_labels=return_labels
            )
        else:
            base = self.decoder(source, return_labels=return_labels)
            supplement = None
        # CUDA uses base boxes in this release.  Avoid the eager adaptive
        # expansion whose diagnostic result is not consumed by REC.
        if source.device.type == "cuda" and not self.config.expanded_as_primary:
            boxes = base.boxes
            expanded_sides = torch.zeros(
                (*base.valid.shape, 4), device=base.boxes.device, dtype=torch.bool
            )
        else:
            boxes, expanded_sides = _adaptive_expand(
                source, base.boxes, base.valid, self.config
            )
        base_match_boxes = base.boxes
        valid = base.valid
        scores = base.scores
        if supplement is None:
            supplement = (
                self.supplement_decoder(source)
                if self.supplement_decoder is not None
                else None
            )
        _, _, map_height, map_width = source.shape
        coordinate_width = torch.full(
            (source.shape[0],), float(map_width), device=source.device
        )
        coordinate_height = torch.full(
            (source.shape[0],), float(map_height), device=source.device
        )
        if source_geometry is not None:
            if source_geometry.shape != (source.shape[0], 5):
                raise ValueError(
                    "source_geometry must have shape [B,5] as scale,left,top,width,height"
                )
            geometry = source_geometry.to(device=source.device, dtype=source.dtype)
            coordinate_width = geometry[:, 3]
            coordinate_height = geometry[:, 4]
            boxes = _map_to_source(boxes, geometry)
            base_match_boxes = _map_to_source(base_match_boxes, geometry)
            valid = valid & (_quad_area(boxes) >= 16.0)
            if supplement is not None:
                supplement_boxes = _map_to_source(supplement.boxes, geometry)
                supplement_valid = supplement.valid & (
                    _quad_area(supplement_boxes) >= 16.0
                )
        elif supplement is not None:
            supplement_boxes = supplement.boxes
            supplement_valid = supplement.valid

        if supplement is not None:
            (
                boxes,
                base_match_boxes,
                scores,
                valid,
                expanded_sides,
                supplemental,
                supplement_area_ratio,
                supplement_distance,
            ) = _combine_supplements(
                boxes,
                base_match_boxes,
                scores,
                valid,
                expanded_sides,
                supplement_boxes,
                supplement.scores,
                supplement_valid,
                self.config.supplement_duplicate_tolerance,
            )
            raw_component_count = torch.maximum(
                base.raw_component_count, supplement.raw_component_count
            )
            overflow = base.overflow | supplement.overflow
        else:
            supplemental = torch.zeros_like(valid)
            supplement_area_ratio = torch.zeros_like(scores)
            supplement_distance = torch.zeros_like(scores)
            raw_component_count = base.raw_component_count
            overflow = base.overflow
        fallback = valid.sum(dim=1) == 0
        if self.config.fullcrop_on_empty:
            zeros = torch.zeros_like(coordinate_width)
            right = coordinate_width - (1.0 if source_geometry is not None else 0.0)
            bottom = coordinate_height - (1.0 if source_geometry is not None else 0.0)
            fullcrop = torch.stack(
                [
                    torch.stack([zeros, zeros], dim=1),
                    torch.stack([right, zeros], dim=1),
                    torch.stack([right, bottom], dim=1),
                    torch.stack([zeros, bottom], dim=1),
                ],
                dim=1,
            )
            first_box = torch.where(fallback[:, None, None], fullcrop, boxes[:, 0])
            boxes = torch.cat([first_box[:, None], boxes[:, 1:]], dim=1)
            first_base_box = torch.where(
                fallback[:, None, None], fullcrop, base_match_boxes[:, 0]
            )
            base_match_boxes = torch.cat(
                [first_base_box[:, None], base_match_boxes[:, 1:]], dim=1
            )
            first_valid = valid[:, 0] | fallback
            valid = torch.cat([first_valid[:, None], valid[:, 1:]], dim=1)
            first_score = torch.where(fallback, torch.zeros_like(scores[:, 0]), scores[:, 0])
            scores = torch.cat([first_score[:, None], scores[:, 1:]], dim=1)
            first_supplemental = torch.where(
                fallback, torch.zeros_like(supplemental[:, 0]), supplemental[:, 0]
            )
            supplemental = torch.cat(
                [first_supplemental[:, None], supplemental[:, 1:]], dim=1
            )
            supplement_area_ratio = torch.cat(
                [
                    torch.where(
                        fallback,
                        torch.zeros_like(supplement_area_ratio[:, 0]),
                        supplement_area_ratio[:, 0],
                    )[:, None],
                    supplement_area_ratio[:, 1:],
                ],
                dim=1,
            )
            supplement_distance = torch.cat(
                [
                    torch.where(
                        fallback,
                        torch.zeros_like(supplement_distance[:, 0]),
                        supplement_distance[:, 0],
                    )[:, None],
                    supplement_distance[:, 1:],
                ],
                dim=1,
            )
        else:
            fallback = torch.zeros_like(fallback)

        optional, base_area_ratio, border_contact = _optional_primary_boxes(
            base_match_boxes,
            scores,
            valid,
            supplemental,
            fallback,
            coordinate_width,
            coordinate_height,
            self.config,
        )

        ordering_boxes = (
            base_match_boxes if self.config.order_before_expansion else boxes
        )
        order, track_ids, vertical, skew, count = _reading_order(
            ordering_boxes,
            valid,
            robust=self.config.robust_reading_order,
        )
        safe_order = order.clamp_min(0).to(torch.int64)
        ordered_expanded_boxes = boxes.gather(
            1, safe_order[:, :, None, None].expand_as(boxes)
        )
        ordered_base_boxes = base_match_boxes.gather(
            1, safe_order[:, :, None, None].expand_as(base_match_boxes)
        )
        ordered_scores = scores.gather(1, safe_order)
        ordered_expanded = expanded_sides.gather(
            1, safe_order[:, :, None].expand_as(expanded_sides)
        )
        ordered_supplemental = supplemental.gather(1, safe_order)
        ordered_supplement_area_ratio = supplement_area_ratio.gather(1, safe_order)
        ordered_supplement_distance = supplement_distance.gather(1, safe_order)
        ordered_optional = optional.gather(1, safe_order)
        ordered_base_area_ratio = base_area_ratio.gather(1, safe_order)
        ordered_border_contact = border_contact.gather(1, safe_order)
        ordered_tracks = track_ids.gather(1, safe_order)
        slots = torch.arange(boxes.shape[1], device=boxes.device)[None]
        ordered_valid = slots < count[:, None]
        ordered_expanded_boxes = torch.where(
            ordered_valid[:, :, None, None],
            ordered_expanded_boxes,
            torch.zeros_like(ordered_expanded_boxes),
        )
        ordered_base_boxes = torch.where(
            ordered_valid[:, :, None, None],
            ordered_base_boxes,
            torch.zeros_like(ordered_base_boxes),
        )
        ordered_scores = torch.where(
            ordered_valid, ordered_scores, torch.zeros_like(ordered_scores)
        )
        ordered_expanded &= ordered_valid[:, :, None]
        ordered_supplemental &= ordered_valid
        ordered_optional &= ordered_valid
        ordered_border_contact &= ordered_valid
        ordered_base_area_ratio = torch.where(
            ordered_valid,
            ordered_base_area_ratio,
            torch.zeros_like(ordered_base_area_ratio),
        )
        ordered_supplement_area_ratio = torch.where(
            ordered_valid,
            ordered_supplement_area_ratio,
            torch.zeros_like(ordered_supplement_area_ratio),
        )
        ordered_supplement_distance = torch.where(
            ordered_valid,
            ordered_supplement_distance,
            torch.zeros_like(ordered_supplement_distance),
        )
        ordered_tracks = torch.where(
            ordered_valid, ordered_tracks, torch.full_like(ordered_tracks, -1)
        )
        if destination_sizes is not None:
            if destination_sizes.shape != (source.shape[0], 2):
                raise ValueError("destination_sizes must have shape [B,2] as width,height")
            scale = destination_sizes.to(device=source.device, dtype=source.dtype)
            scale = scale / torch.tensor(
                [source.shape[3], source.shape[2]],
                device=source.device,
                dtype=source.dtype,
            )
            ordered_expanded_boxes = (
                ordered_expanded_boxes * scale[:, None, None, :]
            )
            ordered_base_boxes = ordered_base_boxes * scale[:, None, None, :]
        ordered_boxes = (
            ordered_expanded_boxes
            if self.config.expanded_as_primary
            else ordered_base_boxes
        )
        return SmartGpuDBResult(
            boxes=ordered_boxes,
            base_boxes=ordered_base_boxes,
            expanded_boxes=ordered_expanded_boxes,
            scores=ordered_scores,
            valid=ordered_valid,
            count=count,
            raw_component_count=raw_component_count,
            overflow=overflow,
            expanded_sides=ordered_expanded,
            supplemental=ordered_supplemental,
            supplement_area_ratio=ordered_supplement_area_ratio,
            supplement_distance=ordered_supplement_distance,
            optional=ordered_optional,
            base_area_ratio=ordered_base_area_ratio,
            border_contact=ordered_border_contact,
            fallback_fullcrop=fallback,
            order_from_decoder=order,
            track_ids=ordered_tracks,
            vertical=vertical,
            skew_radians=skew,
            labels=base.labels,
        )

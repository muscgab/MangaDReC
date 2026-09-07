"""Unified fixed-capacity DB post-processing for Apple MPS and CUDA."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GpuDBConfig:
    threshold: float = 0.20
    box_threshold: float = 0.45
    unclip_ratio: float = 1.4
    min_size: float = 3.0
    max_components: int = 32
    union_find_rounds: int = 4
    # A rectangle orientation is periodic over 90 degrees.  The old decoder
    # only searched PCA +/-8 degrees, which is not sufficient for hollow,
    # punctuated, or partly clipped text components.  Search the full
    # equivalent interval coarsely, then refine around the best angle.
    angle_offsets_degrees: tuple[float, ...] = tuple(
        float(value) for value in range(-44, 45, 4)
    )
    angle_refine_offsets_degrees: tuple[float, ...] = (
        -3.0,
        -2.5,
        -2.0,
        -1.5,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
    )

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0,1]")
        if not 0.0 <= self.box_threshold <= 1.0:
            raise ValueError("box_threshold must be in [0,1]")
        if self.unclip_ratio < 0.0:
            raise ValueError("unclip_ratio must be non-negative")
        if self.min_size < 0.0:
            raise ValueError("min_size must be non-negative")
        if not 1 <= self.max_components <= 64:
            raise ValueError("max_components must be in [1,64]")
        if self.union_find_rounds < 1:
            raise ValueError("union_find_rounds must be positive")
        if not self.angle_offsets_degrees:
            raise ValueError("at least one angle offset is required")
        if not self.angle_refine_offsets_degrees:
            raise ValueError("at least one angle refinement offset is required")


@dataclass(frozen=True)
class GpuDBResult:
    boxes: torch.Tensor
    scores: torch.Tensor
    valid: torch.Tensor
    count: torch.Tensor
    raw_component_count: torch.Tensor
    overflow: torch.Tensor
    labels: torch.Tensor | None = None


def _decode_backend(
    source: torch.Tensor, config: GpuDBConfig
) -> tuple[torch.Tensor, ...]:
    arguments = {
        "threshold": config.threshold,
        "box_threshold": config.box_threshold,
        "unclip_ratio": config.unclip_ratio,
        "min_size": config.min_size,
        "max_components": config.max_components,
        "rounds": config.union_find_rounds,
        "angle_offsets_radians": tuple(
            math.radians(value) for value in config.angle_offsets_degrees
        ),
        "angle_refine_offsets_radians": tuple(
            math.radians(value) for value in config.angle_refine_offsets_degrees
        ),
    }
    if source.device.type == "mps":
        from .mps import decode_db_mps

        return decode_db_mps(source, **arguments)
    if source.device.type == "cuda":
        from .cuda import decode_db_cuda

        return decode_db_cuda(source, **arguments)
    raise ValueError("GpuDBPostProcess requires a CUDA or MPS tensor")


class GpuDBPostProcess:
    """Convert a DB probability map to GPU-resident quadrilateral slots."""

    def __init__(self, config: GpuDBConfig | None = None) -> None:
        self.config = config or GpuDBConfig()

    @torch.inference_mode()
    def __call__(
        self,
        probability: torch.Tensor,
        *,
        destination_sizes: torch.Tensor | None = None,
        return_labels: bool = False,
    ) -> GpuDBResult:
        if probability.ndim != 4 or probability.shape[1] != 1:
            raise ValueError("probability must have shape [B,1,H,W]")
        if not probability.dtype.is_floating_point:
            raise ValueError("probability must be floating point")
        if probability.device.type not in {"mps", "cuda"}:
            raise ValueError("probability must be on MPS or CUDA")

        source = probability.contiguous().float()
        batch, _, height, width = source.shape
        pixels = height * width
        (
            roots_global,
            boxes,
            score,
            valid,
            count,
            raw_component_count,
        ) = _decode_backend(source, self.config)
        if destination_sizes is not None:
            if destination_sizes.shape != (batch, 2):
                raise ValueError(
                    "destination_sizes must have shape [B,2] as width,height"
                )
            destination_sizes = destination_sizes.to(
                device=source.device, dtype=source.dtype
            )
            scale = destination_sizes / torch.tensor(
                [width, height], device=source.device, dtype=source.dtype
            )
            boxes = boxes * scale[:, None, None, :]

        labels = None
        if return_labels:
            offsets = (
                torch.arange(batch, device=source.device, dtype=torch.int32)
                * pixels
            )
            labels = torch.where(
                roots_global >= 0,
                roots_global - offsets[:, None, None],
                torch.full_like(roots_global, -1),
            )
        return GpuDBResult(
            boxes=boxes,
            scores=score,
            valid=valid,
            count=count,
            raw_component_count=raw_component_count,
            overflow=raw_component_count > self.config.max_components,
            labels=labels,
        )

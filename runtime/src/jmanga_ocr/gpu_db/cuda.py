"""Lazy CUDA extension for the fixed-capacity DB decoder."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def _extension():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    from torch.utils.cpp_extension import load

    source = Path(__file__).with_name("csrc") / "ccl_cuda.cu"
    # hipcc forwards CUDA-style flags through clang.  ROCm 7.2 no longer
    # accepts ``--use_fast_math`` there, while ``-ffast-math`` has the same
    # intent and works for the HIP translation of this source.
    fast_math_flag = "-ffast-math" if torch.version.hip is not None else "--use_fast_math"
    return load(
        name="jmanga_gpu_db_ccl_cuda",
        sources=[str(source)],
        extra_cuda_cflags=["-O3", fast_math_flag],
        verbose=False,
    )


def connected_components_cuda(
    probability: torch.Tensor,
    threshold: float,
    rounds: int,
) -> torch.Tensor:
    """Return global root labels for a CUDA ``[B,1,H,W]`` probability map."""
    if probability.device.type != "cuda":
        raise ValueError("connected_components_cuda requires a CUDA tensor")
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError("probability must have shape [B,1,H,W]")
    if rounds < 1:
        raise ValueError("rounds must be positive")
    return _extension().connected_components(
        probability.contiguous().float(), float(threshold), int(rounds)
    )


def decode_db_cuda(
    probability: torch.Tensor,
    *,
    threshold: float,
    box_threshold: float,
    unclip_ratio: float,
    min_size: float,
    max_components: int,
    rounds: int,
    angle_offsets_radians: tuple[float, ...],
    angle_refine_offsets_radians: tuple[float, ...],
) -> tuple[torch.Tensor, ...]:
    """Run complete fixed-slot DB decoding without leaving CUDA memory."""
    if probability.device.type != "cuda":
        raise ValueError("decode_db_cuda requires a CUDA tensor")
    if max_components > 64:
        raise ValueError("CUDA DB decoder supports at most 64 components")
    offsets = torch.tensor(
        angle_offsets_radians,
        device=probability.device,
        dtype=torch.float32,
    )
    refine_offsets = torch.tensor(
        angle_refine_offsets_radians,
        device=probability.device,
        dtype=torch.float32,
    )
    return tuple(
        _extension().decode_db(
            probability.contiguous().float(),
            float(threshold),
            float(box_threshold),
            float(unclip_ratio),
            float(min_size),
            int(max_components),
            int(rounds),
            offsets,
            refine_offsets,
        )
    )

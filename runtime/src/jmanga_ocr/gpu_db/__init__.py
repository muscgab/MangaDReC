"""Device-resident DB detector post-processing for CUDA and Apple MPS."""

from .ctc_gate import (
    MANGA109S_DET_EMPTY_HALLUCINATION_V0,
    CtcConsistencyGateResult,
    CtcHallucinationRateResult,
    CtcSupplementGateResult,
    HallucinationRateCalibration,
    ctc_consistency_gate,
    ctc_hallucination_rate,
    ctc_supplement_gate,
)
from .postprocess import GpuDBConfig, GpuDBPostProcess, GpuDBResult
from .smart import SmartGpuDBConfig, SmartGpuDBPostProcess, SmartGpuDBResult

__all__ = [
    "GpuDBConfig",
    "GpuDBPostProcess",
    "GpuDBResult",
    "SmartGpuDBConfig",
    "SmartGpuDBPostProcess",
    "SmartGpuDBResult",
    "CtcConsistencyGateResult",
    "CtcHallucinationRateResult",
    "CtcSupplementGateResult",
    "HallucinationRateCalibration",
    "MANGA109S_DET_EMPTY_HALLUCINATION_V0",
    "ctc_consistency_gate",
    "ctc_hallucination_rate",
    "ctc_supplement_gate",
]

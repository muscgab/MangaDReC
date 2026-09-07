"""GPU-resident consistency gate for zero-DET whole-image REC candidates."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CtcConsistencyGateResult:
    accepted: torch.Tensor
    token_ids: torch.Tensor
    token_count: torch.Tensor
    confidence: torch.Tensor
    alternate_confidence: torch.Tensor


@dataclass(frozen=True)
class CtcSupplementGateResult:
    accepted: torch.Tensor
    token_ids: torch.Tensor
    token_count: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class HallucinationRateCalibration:
    """Empirical counts for a non-neural DET-empty hallucination estimator.

    Axes are ``[same_sequence, confidence_bucket, length_bucket]``.  The
    bundled v0 table was measured on 91 real DET-empty text crops and 1,333
    DET-empty real-manga non-text crops for which whole-image REC emitted a
    non-empty sequence.
    """

    confidence_edges: tuple[float, ...]
    length_edges: tuple[int, ...]
    text_counts: tuple[tuple[tuple[int, ...], ...], ...]
    nontext_counts: tuple[tuple[tuple[int, ...], ...], ...]
    smoothing: float = 1.0


@dataclass(frozen=True)
class CtcHallucinationRateResult:
    hallucination_rate: torch.Tensor
    likelihood_ratio: torch.Tensor
    same_sequence: torch.Tensor
    token_count: torch.Tensor
    minimum_view_confidence: torch.Tensor
    confidence_bucket: torch.Tensor
    length_bucket: torch.Tensor


MANGA109S_DET_EMPTY_HALLUCINATION_V0 = HallucinationRateCalibration(
    confidence_edges=(0.80, 0.95, 0.995),
    length_edges=(2, 3, 5),
    text_counts=(
        ((4, 1, 1, 1), (1, 2, 0, 9), (0, 0, 0, 5), (0, 0, 0, 0)),
        ((4, 0, 0, 1), (2, 1, 2, 5), (2, 2, 5, 2), (19, 16, 4, 2)),
    ),
    nontext_counts=(
        ((533, 334, 191, 16), (17, 14, 8, 0), (2, 0, 0, 0), (0, 0, 0, 0)),
        ((121, 20, 5, 0), (28, 13, 4, 1), (18, 1, 1, 0), (4, 2, 0, 0)),
    ),
)


def _collapse_ctc(
    probabilities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores, labels = probabilities.max(dim=-1)
    previous = torch.cat(
        [torch.full_like(labels[:, :1], -1), labels[:, :-1]], dim=1
    )
    keep = (labels != 0) & (labels != previous)
    count = keep.sum(dim=1)
    slots = keep.cumsum(dim=1) - 1
    tokens = torch.zeros_like(labels)
    tokens.scatter_add_(
        1,
        slots.clamp_min(0),
        torch.where(keep, labels, torch.zeros_like(labels)),
    )
    confidence = (scores * keep).sum(dim=1) / count.clamp_min(1)
    confidence = torch.where(count > 0, confidence, torch.zeros_like(confidence))
    return tokens, count, confidence


@torch.inference_mode()
def ctc_consistency_gate(
    probabilities: torch.Tensor,
    alternate_probabilities: torch.Tensor,
    *,
    maximum_length: int = 6,
    minimum_confidence: float = 0.995,
    minimum_confidence_by_length: torch.Tensor
    | list[float]
    | tuple[float, ...]
    | None = None,
) -> CtcConsistencyGateResult:
    """Accept only the same short, confident CTC sequence in both views.

    Both inputs stay on their original device.  The alternate view used by the
    measured policy is the same crop with 10% replicated-border padding.
    """
    if probabilities.ndim != 3 or alternate_probabilities.ndim != 3:
        raise ValueError("CTC probabilities must have shape [N,T,C]")
    if probabilities.shape[0] != alternate_probabilities.shape[0]:
        raise ValueError("CTC views must have the same batch size")
    if probabilities.device != alternate_probabilities.device:
        raise ValueError("CTC views must be on the same device")
    tokens, count, confidence = _collapse_ctc(probabilities)
    alternate_tokens, alternate_count, alternate_confidence = _collapse_ctc(
        alternate_probabilities
    )
    shared_steps = min(tokens.shape[1], alternate_tokens.shape[1])
    equal = (count == alternate_count) & (
        tokens[:, :shared_steps] == alternate_tokens[:, :shared_steps]
    ).all(dim=1)
    confidence_threshold = torch.full_like(confidence, minimum_confidence)
    if minimum_confidence_by_length is not None:
        thresholds = torch.as_tensor(
            minimum_confidence_by_length,
            device=probabilities.device,
            dtype=confidence.dtype,
        ).flatten()
        if thresholds.numel() != maximum_length:
            raise ValueError(
                "minimum_confidence_by_length must contain one threshold for "
                "each accepted length"
            )
        length_index = count.clamp(min=1, max=maximum_length).to(torch.int64) - 1
        confidence_threshold = thresholds[length_index]
    accepted = (
        equal
        & (count >= 1)
        & (count <= maximum_length)
        & (confidence >= confidence_threshold)
        & (alternate_confidence >= confidence_threshold)
    )
    return CtcConsistencyGateResult(
        accepted=accepted,
        token_ids=tokens,
        token_count=count,
        confidence=confidence,
        alternate_confidence=alternate_confidence,
    )


@torch.inference_mode()
def ctc_hallucination_rate(
    probabilities: torch.Tensor,
    alternate_probabilities: torch.Tensor,
    *,
    prior_hallucination: float | torch.Tensor,
    calibration: HallucinationRateCalibration = (
        MANGA109S_DET_EMPTY_HALLUCINATION_V0
    ),
) -> CtcHallucinationRateResult:
    """Estimate risk that non-empty whole-image REC came from non-text.

    This is an empirical Bayes lookup, not a neural model and not an output
    gate.  ``prior_hallucination`` is the deployment prior among DET-empty
    inputs that emit a non-empty base-view sequence.  The returned rate is
    zero when the base view emits nothing because there is no output to call a
    hallucination.
    """
    if probabilities.ndim != 3 or alternate_probabilities.ndim != 3:
        raise ValueError("CTC probabilities must have shape [N,T,C]")
    if probabilities.shape[0] != alternate_probabilities.shape[0]:
        raise ValueError("CTC views must have the same batch size")
    if probabilities.device != alternate_probabilities.device:
        raise ValueError("CTC views must be on the same device")
    if isinstance(prior_hallucination, (float, int)) and not (
        0.0 < float(prior_hallucination) < 1.0
    ):
        raise ValueError("prior_hallucination must lie strictly between 0 and 1")
    if calibration.smoothing <= 0.0:
        raise ValueError("calibration smoothing must be positive")

    tokens, count, confidence = _collapse_ctc(probabilities)
    alternate_tokens, alternate_count, alternate_confidence = _collapse_ctc(
        alternate_probabilities
    )
    shared_steps = min(tokens.shape[1], alternate_tokens.shape[1])
    same_sequence = (
        (count == alternate_count)
        & (count <= shared_steps)
        & (
            tokens[:, :shared_steps]
            == alternate_tokens[:, :shared_steps]
        ).all(dim=1)
    )
    minimum_view_confidence = torch.minimum(confidence, alternate_confidence)

    confidence_edges = torch.as_tensor(
        calibration.confidence_edges,
        device=probabilities.device,
        dtype=minimum_view_confidence.dtype,
    )
    length_edges = torch.as_tensor(
        calibration.length_edges,
        device=probabilities.device,
        dtype=count.dtype,
    )
    confidence_bucket = (
        minimum_view_confidence[:, None] >= confidence_edges[None]
    ).sum(dim=1).to(torch.int64)
    length_bucket = (count[:, None] >= length_edges[None]).sum(dim=1).to(
        torch.int64
    )

    text_counts = torch.as_tensor(
        calibration.text_counts,
        device=probabilities.device,
        dtype=minimum_view_confidence.dtype,
    )
    nontext_counts = torch.as_tensor(
        calibration.nontext_counts,
        device=probabilities.device,
        dtype=minimum_view_confidence.dtype,
    )
    expected_shape = (
        2,
        len(calibration.confidence_edges) + 1,
        len(calibration.length_edges) + 1,
    )
    if tuple(text_counts.shape) != expected_shape or tuple(
        nontext_counts.shape
    ) != expected_shape:
        raise ValueError("hallucination calibration count table has invalid shape")

    cell_count = text_counts.numel()
    text_probability = (text_counts + calibration.smoothing) / (
        text_counts.sum() + calibration.smoothing * cell_count
    )
    nontext_probability = (nontext_counts + calibration.smoothing) / (
        nontext_counts.sum() + calibration.smoothing * cell_count
    )
    index = (
        same_sequence.to(torch.int64),
        confidence_bucket,
        length_bucket,
    )
    likelihood_ratio = nontext_probability[index] / text_probability[index]

    prior = torch.as_tensor(
        prior_hallucination,
        device=probabilities.device,
        dtype=minimum_view_confidence.dtype,
    )
    prior = torch.broadcast_to(prior, minimum_view_confidence.shape)
    epsilon = torch.finfo(minimum_view_confidence.dtype).eps
    prior = prior.clamp(min=epsilon, max=1.0 - epsilon)
    numerator = prior * likelihood_ratio
    hallucination_rate = numerator / (numerator + (1.0 - prior))
    has_output = count > 0
    hallucination_rate = torch.where(
        has_output, hallucination_rate, torch.zeros_like(hallucination_rate)
    )
    likelihood_ratio = torch.where(
        has_output, likelihood_ratio, torch.zeros_like(likelihood_ratio)
    )
    return CtcHallucinationRateResult(
        hallucination_rate=hallucination_rate,
        likelihood_ratio=likelihood_ratio,
        same_sequence=same_sequence,
        token_count=count,
        minimum_view_confidence=minimum_view_confidence,
        confidence_bucket=confidence_bucket,
        length_bucket=length_bucket,
    )


@torch.inference_mode()
def ctc_supplement_gate(
    probabilities: torch.Tensor,
    supplemental: torch.Tensor,
    area_ratio: torch.Tensor,
    distance: torch.Tensor,
    allowed_token_ids: torch.Tensor | list[int] | tuple[int, ...],
    *,
    minimum_confidence: float = 0.70,
    minimum_area_ratio: float = 0.075,
    maximum_distance: float = 50.0,
) -> CtcSupplementGateResult:
    """Keep weak DB components only when REC and geometry both support them.

    Ordinary DB boxes always pass. Supplemental boxes must decode to a
    non-empty sequence containing only punctuation or small-kana token IDs,
    meet the measured REC-confidence threshold, have enough area relative to
    the median ordinary box, and lie near an ordinary box. All decisions stay
    on the input device.
    """
    if probabilities.ndim != 3:
        raise ValueError("CTC probabilities must have shape [N,T,C]")
    batch = probabilities.shape[0]
    for name, value in (
        ("supplemental", supplemental),
        ("area_ratio", area_ratio),
        ("distance", distance),
    ):
        if value.shape != (batch,):
            raise ValueError(f"{name} must have shape [N]")
        if value.device != probabilities.device:
            raise ValueError(f"{name} must be on the CTC device")

    allowed = torch.as_tensor(
        allowed_token_ids, device=probabilities.device, dtype=torch.int64
    ).flatten()
    if allowed.numel() == 0:
        raise ValueError("allowed_token_ids cannot be empty")

    tokens, count, confidence = _collapse_ctc(probabilities)
    token_positions = torch.arange(tokens.shape[1], device=tokens.device)[None]
    active = token_positions < count[:, None]
    token_allowed = (tokens[:, :, None] == allowed[None, None]).any(dim=2)
    all_tokens_allowed = torch.where(
        active, token_allowed, torch.ones_like(token_allowed)
    ).all(dim=1)
    supplement_accepted = (
        (count >= 1)
        & all_tokens_allowed
        & (confidence >= minimum_confidence)
        & (area_ratio >= minimum_area_ratio)
        & (distance <= maximum_distance)
    )
    accepted = ~supplemental.bool() | supplement_accepted
    return CtcSupplementGateResult(
        accepted=accepted,
        token_ids=tokens,
        token_count=count,
        confidence=confidence,
    )

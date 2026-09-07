#!/usr/bin/env python3
"""Shared symmetric ``v2_semantic_eval`` normalization for OCR scoring.

Apply :func:`normalize_v2_semantic_eval` to both ground truth and prediction
before computing normalized EM/CER.  Training-label canonicalization remains
owned by the renderer's ``manga_semantic_visual_v2`` preset; the small set of
extra rules below is evaluation-only.
"""

from __future__ import annotations

import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import regex
import yaml


DEFAULT_PRESET = Path(
    os.environ.get(
        "JMANGA_NORMALIZATION_PRESET",
        str(Path(__file__).with_name("normalization_presets_semantic_visual_v2.yaml")),
    )
)

EVALUATION_POLICY = "v2.1_semantic_visual_typography_eval_20260905"


@lru_cache(maxsize=4)
def load_v2_preset(path: str | Path = DEFAULT_PRESET) -> dict[str, object]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return raw["presets"]["manga_semantic_visual_v2"]


def benchmark_adapter(text: str) -> str:
    """Remove layout whitespace and adapt the benchmark's period-run notation."""
    text = re.sub(r"[ \u3000\t\r\n]", "", text)
    text = re.sub(r"\.{3,}", lambda match: "・" * len(match.group(0)), text)
    return re.sub(r"．{3,}", lambda match: "・" * len(match.group(0)), text)


def v2_core(text: str, preset: dict[str, object] | None = None) -> str:
    preset = preset or load_v2_preset()
    source = text.strip()
    if not source:
        return ""
    if any(unicodedata.category(char) in {"Cc", "Cs"} for char in source):
        raise ValueError("contains forbidden control or surrogate character")

    aliases = preset["character_aliases"]
    compatibility = preset["compatibility_aliases"]
    assert isinstance(aliases, dict) and isinstance(compatibility, dict)
    parts: list[str] = []
    for unit in regex.findall(r"\X", source):
        if unit in aliases:
            value = str(aliases[unit])
        elif unit in compatibility:
            value = str(compatibility[unit])
        elif len(unit) == 1 and 0xFF01 <= ord(unit) <= 0xFF5E:
            value = unicodedata.normalize("NFKC", unit)
        elif unit == "￥":
            value = "¥"
        elif all(0xFF61 <= ord(char) <= 0xFF9F for char in unit) and any(
            0xFF61 <= ord(char) <= 0xFF9D for char in unit
        ):
            value = unicodedata.normalize("NFKC", unit)
        else:
            value = unicodedata.normalize("NFC", unit)
        parts.append(unicodedata.normalize("NFC", value))
    return "".join(parts)


def normalize_v2_semantic_eval(
    text: str, preset: dict[str, object] | None = None
) -> str:
    """Return the symmetric semantic-irrelevant evaluation view of ``text``."""
    normalized = benchmark_adapter(text)
    # Some approved rules form chains, for example halfwidth ﾍ -> ヘ -> へ.
    # Evaluate to a fixed point so that equivalent inputs cannot stop at
    # different intermediate representations.
    for _ in range(4):
        updated = v2_core(normalized, preset)
        if updated == normalized:
            break
        normalized = updated
    else:
        raise ValueError("semantic-visual-v2 normalization did not converge")

    # Japanese corner quotes form one family.  The training/evaluation preset
    # folds straight, fullwidth, curly, and Japanese double-prime quote glyphs
    # into a second family; the two families stay distinct.
    normalized = normalized.translate(
        str.maketrans({"『": "「", "』": "」"})
    )

    # For an elongated-mark run, one mark remains one; every run of at least
    # two is represented by exactly two marks.  Long-vowel and wave families
    # remain distinct from one another.
    normalized = re.sub(r"ー{2,}", "ーー", normalized)
    normalized = re.sub(r"〜{2,}", "〜〜", normalized)
    return normalized


def normalize_pair(
    ground_truth: str,
    prediction: str,
    preset: dict[str, object] | None = None,
) -> tuple[str, str]:
    """Normalize a GT/prediction pair with the exact same rules."""
    active_preset = preset or load_v2_preset()
    return (
        normalize_v2_semantic_eval(ground_truth, active_preset),
        normalize_v2_semantic_eval(prediction, active_preset),
    )


# Backward-compatible import name.  New evaluation code must use the policy
# name above so reports no longer suggest that this is a benchmark-only rule.
normalize_jmangabench_visual_eval = normalize_v2_semantic_eval

#!/usr/bin/env python3
"""Translation-oriented punctuation view layered on v2 semantic evaluation.

This module intentionally does not change ``v2_semantic_eval``.  It applies a
coarser, symmetric comparison policy to both reference and OCR prediction:

* inherited v2 rules keep one ``ー``/``〜`` unchanged and map every run of two
  or more to exactly ``ーー``/``〜〜``; visual aliases are folded first;
* a run containing both ``!`` and ``?`` keeps one of each in first-seen order;
* a repeated run containing only one of them keeps exactly two;
* an ellipsis-dot run of 3--6 ``・`` keeps three, a run longer than six keeps
  six, and a run shorter than three keeps its original length.

The scoring key uses a private canonical dot token.  That makes the policy
idempotent even though the requested visible mapping ``7+ -> 6`` and
``3..6 -> 3`` is not idempotent when represented with the same literal dot.
Call :func:`render_translation_normalized` only when visible text is needed.
"""

from __future__ import annotations

import re

from semantic_visual_v2_eval import normalize_v2_semantic_eval


EVALUATION_POLICY = "v2.1_translation_typography_eval_20260905"
APPLICATION_CONTRACT = (
    "The scoring key is idempotent and uses U+E000 internally for each canonical "
    "ellipsis dot. Render U+E000 back to ・ only after scoring or before translation."
)
_EXCLAMATION_QUESTION_RUN = re.compile(r"[!?]{2,}")
_ELLIPSIS_DOT_RUN = re.compile(r"・+")
_ELLIPSIS_DOT_TOKEN = "\ue000"


def normalize_exclamation_question_runs_from_v2(text: str) -> str:
    """Collapse expressive !/? counts while retaining type and first order."""

    def replace(match: re.Match[str]) -> str:
        value = match.group(0)
        has_exclamation = "!" in value
        has_question = "?" in value
        if has_exclamation and has_question:
            return "!?" if value.index("!") < value.index("?") else "?!"
        return value[0] * 2

    return _EXCLAMATION_QUESTION_RUN.sub(replace, text)


def normalize_ellipsis_runs_from_v2(text: str) -> str:
    """Return an idempotent scoring key with 1/2/3/6 canonical dot tokens."""

    return _ELLIPSIS_DOT_RUN.sub(
        lambda match: _ELLIPSIS_DOT_TOKEN
        * (
            len(match.group(0))
            if len(match.group(0)) < 3
            else (3 if len(match.group(0)) <= 6 else 6)
        ),
        text,
    )


def normalize_v2_translation_eval(text: str) -> str:
    """Return the idempotent canonical scoring key, including inherited v2 rules."""
    normalized = normalize_v2_semantic_eval(text)
    normalized = normalize_exclamation_question_runs_from_v2(normalized)
    return normalize_ellipsis_runs_from_v2(normalized)


def render_translation_normalized(text: str) -> str:
    """Render a canonical scoring key as visible middle-dot punctuation."""

    return text.replace(_ELLIPSIS_DOT_TOKEN, "・")


def normalize_v2_translation_display(text: str) -> str:
    """Return visible normalized text for translation or presentation."""

    return render_translation_normalized(normalize_v2_translation_eval(text))


def normalize_pair(reference: str, prediction: str) -> tuple[str, str]:
    return normalize_v2_translation_eval(reference), normalize_v2_translation_eval(prediction)

#!/usr/bin/env python3
"""Public fixed-vector tests for translation punctuation normalization v2.1."""

from __future__ import annotations

from semantic_visual_v2_translation_eval import (
    normalize_v2_translation_display,
    normalize_v2_translation_eval,
)


CASES = {
    "!!?": "!?",
    "!!??": "!?",
    "!??": "!?",
    "??!": "?!",
    "!!!": "!!",
    "????": "??",
    "!": "!",
    "?": "?",
    "・・": "・・",
    "・・・": "・・・",
    "・・・・・・": "・・・",
    "・・・・・・・": "・・・・・・",
    "ー": "ー",
    "ーーー": "ーー",
    "〜": "〜",
    "〜〜〜": "〜〜",
}


def main() -> None:
    for source, expected in CASES.items():
        actual = normalize_v2_translation_display(source)
        assert actual == expected, (source, expected, actual)
        key = normalize_v2_translation_eval(source)
        assert normalize_v2_translation_eval(key) == key, source
    print(f"ok: {len(CASES)} fixed vectors and idempotence")


if __name__ == "__main__":
    main()


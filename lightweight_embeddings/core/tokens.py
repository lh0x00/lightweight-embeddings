"""Token estimation helpers.

Two flavors:

* :func:`estimate_tokens_fast` — character-based heuristic (no tokenizer).
  Used for *pre-flight* quota checks before allocating heavy resources.
* :func:`count_tokens_exact` — uses the model's tokenizer when available.
  Used after embedding completes to commit accurate usage to analytics.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def estimate_tokens_fast(texts: Iterable[str]) -> int:
    """Estimate token count without tokenizing.

    Roughly: ``ceil(len(text)/3)`` per string. Errs on the side of slightly
    over-counting which is the safe direction for quota gating.
    """
    total = 0
    for text in texts:
        # max(1, ...) so empty strings still cost something.
        total += max(1, len(text) // 3)
    return total


def count_tokens_exact(tokenizer: Any, texts: list[str]) -> int:
    """Count tokens precisely using a HuggingFace fast tokenizer.

    Falls back to :func:`estimate_tokens_fast` if the tokenizer call fails.
    """
    if not texts:
        return 0
    try:
        encoded = tokenizer(
            texts,
            add_special_tokens=True,
            truncation=False,
            return_length=True,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        lengths = encoded.get("length")
        if lengths is None:
            ids = encoded["input_ids"]
            return sum(len(x) for x in ids)
        return int(sum(lengths))
    except Exception:
        return estimate_tokens_fast(texts)

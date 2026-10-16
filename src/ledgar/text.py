"""Word-level text processing for the LSTM (S04 p4-6, TP04 Parts A-B)."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers, punctuation dropped (TP04 Ex 1)."""
    return TOKEN_RE.findall(text.lower())

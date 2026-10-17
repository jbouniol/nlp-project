"""Word-level text processing for the LSTM (S04 p4-6, TP04 Parts A-B).

The tokenizer and the vocabulary are built once, on the training set only (S04 p4-5).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers, punctuation dropped (TP04 Ex 1)."""
    return TOKEN_RE.findall(text.lower())


def count_words(texts: Iterable[str]) -> Counter:
    """Word counts over the training texts (TP04 Ex 2): the vocabulary is part of the model."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(tokenize(text))
    return counts


def build_vocab(counts: Counter, min_freq: int = 2) -> tuple[list[str], dict[str, int]]:
    """[<pad>, <unk>] + the words seen >= min_freq times, most frequent first, ties alphabetical (TP04 Ex 3).

    Rare words map to <unk> during training, so the model learns a useful <unk> vector (S04 p5).
    """
    words = sorted((w for w, n in counts.items() if n >= min_freq), key=lambda w: (-counts[w], w))
    itos = [PAD, UNK, *words]
    return itos, {w: i for i, w in enumerate(itos)}


def encode(text: str, stoi: dict[str, int], max_words: int | None = None) -> list[int]:
    """Token ids of a text, unknown words -> UNK_ID, cut to the first `max_words` words (TP04 Ex 4)."""
    ids = [stoi.get(tok, UNK_ID) for tok in tokenize(text)]
    return ids[:max_words] if max_words is not None else ids


def unk_rate(texts: Iterable[str], stoi: dict[str, int]) -> float:
    """Share of the tokens of `texts` that are mapped to <unk>."""
    n_tokens = n_unk = 0
    for text in texts:
        tokens = tokenize(text)
        n_tokens += len(tokens)
        n_unk += sum(tok not in stoi for tok in tokens)
    return n_unk / max(n_tokens, 1)


class ClauseDataset(Dataset):
    """Clauses encoded once into id tensors; an empty clause becomes [UNK_ID] (TP04 Ex 5)."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int], stoi: dict[str, int], max_words: int):
        self.ids = [torch.tensor(encode(t, stoi, max_words) or [UNK_ID], dtype=torch.long) for t in texts]
        self.labels = torch.as_tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ids[i], self.labels[i]


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad to the longest clause **of the batch** (S04 p6); return ids (B, T), lengths (B,), labels (B,)."""
    ids, labels = zip(*batch, strict=True)
    lengths = torch.tensor([len(x) for x in ids], dtype=torch.long)
    return pad_sequence(list(ids), batch_first=True, padding_value=PAD_ID), lengths, torch.stack(labels)


def pad_share(ids: torch.Tensor) -> float:
    """Share of <pad> positions in a padded batch."""
    return float((ids == PAD_ID).float().mean())

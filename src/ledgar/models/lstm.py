"""LSTM clause classifier (S04 p7-17, TP04 Parts C and E)."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ledgar.text import PAD_ID


class LSTMClassifier(nn.Module):
    """Embedding -> dropout -> LSTM over the **packed** sequence -> state at the true last token -> linear.

    Packing (S04 p10-11) means the LSTM never reads a pad: `h_n` is the state after the last real
    token of each clause. Bidirectional: the forward and backward final states are concatenated (2H).
    The output is the logits (B, C); the softmax is inside the cross-entropy (S04 p13).
    """

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        hidden: int,
        n_classes: int,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * (2 if bidirectional else 1), n_classes)
        self.bidirectional = bidirectional

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.emb(ids))  # (B, T, d)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)  # h_n: (directions, B, H)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.bidirectional else h_n[-1]
        return self.fc(self.drop(h))  # (B, C)


def count_params(V: int, d: int, H: int, C: int, bidirectional: bool = False) -> int:
    """N = Vd + k * 4(Hd + H^2 + 2H) + kHC + C, with k = 2 directions if bidirectional (S04 p14)."""
    k = 2 if bidirectional else 1
    return V * d + k * 4 * (H * d + H * H + 2 * H) + k * H * C + C


def last_true_state(out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Output (B, T, H) read at the last real position of each sequence, not at the last column (S04 p12)."""
    return out[torch.arange(out.shape[0]), lengths.to(out.device) - 1]


def load_glove(path: str | Path, itos: list[str], dim: int = 50, seed: int = 0) -> tuple[np.ndarray, float]:
    """Embedding matrix (V, dim) initialised from GloVe (S04 p15, TP04 Ex 14).

    Known words get their GloVe vector, unknown ones N(0, 0.1), the <pad> row stays 0.

    Returns:
        (W, coverage) where coverage is the share of real vocabulary words (not <pad>, <unk>) found in GloVe.
    """
    wanted = {w: i for i, w in enumerate(itos)}
    W = np.random.default_rng(seed).normal(0, 0.1, (len(itos), dim)).astype(np.float32)
    W[PAD_ID] = 0
    found = 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            word, *values = line.rstrip().split(" ")
            if len(values) == dim and word in wanted and wanted[word] > 1:
                W[wanted[word]] = np.asarray(values, dtype=np.float32)
                found += 1
    return W, found / max(len(itos) - 2, 1)

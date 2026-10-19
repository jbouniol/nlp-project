"""m3: the training loop of the course for the LSTM (S01 p9, S02 p13, TP04 Part D)."""

from __future__ import annotations

import copy
import logging
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from ledgar import data, metrics
from ledgar.config import Config
from ledgar.text import ClauseDataset, collate_fn

logger = logging.getLogger(__name__)


class LengthGroupedBatchSampler(Sampler[list[int]]):
    """Batches of clauses of similar lengths, reshuffled at every epoch.

    Shuffle, cut into mega-batches of `mega_batch_mult` batches, sort each mega-batch by length, cut into
    batches, then shuffle the order of the batches. The LSTM runs one step per position of the longest
    clause of a batch: grouping lengths removes most of the steps spent on short clauses (the same idea as
    `train_sampling_strategy="group_by_length"` for DistilBERT).
    """

    def __init__(self, lengths: list[int], batch_size: int, seed: int = 0, mega_batch_mult: int = 50):
        self.lengths = np.asarray(lengths)
        self.batch_size, self.mega_batch_mult = batch_size, mega_batch_mult
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        order = data.length_grouped_order(self.lengths, self.batch_size, self.rng, self.mega_batch_mult)
        batches = [order[i : i + self.batch_size].tolist() for i in range(0, len(order), self.batch_size)]
        for b in self.rng.permutation(len(batches)):
            yield batches[b]

    def __len__(self) -> int:
        return -(-len(self.lengths) // self.batch_size)


def make_loaders(
    train_ds: ClauseDataset, eval_sets: dict[str, ClauseDataset], cfg: Config
) -> tuple[DataLoader, dict[str, DataLoader]]:
    """Training batches of `lstm_batch_size` shuffled with the seed; evaluation batches of 256 (TP04 Ex 7)."""
    if cfg.group_by_length:
        sampler = LengthGroupedBatchSampler([len(x) for x in train_ds.ids], cfg.lstm_batch_size, cfg.seed)
        train = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn)
    else:
        generator = torch.Generator().manual_seed(cfg.seed)
        train = DataLoader(
            train_ds, batch_size=cfg.lstm_batch_size, shuffle=True, collate_fn=collate_fn, generator=generator
        )
    evals = {
        k: DataLoader(ds, batch_size=256, shuffle=False, collate_fn=collate_fn) for k, ds in eval_sets.items()
    }
    return train, evals


def train_one_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module, device: str
) -> float:
    """The five lines of S01 p9, plus gradient clipping at norm 1 (S02 p13). Returns the mean batch loss."""
    model.train()
    losses = []
    for ids, lengths, y in loader:
        ids, y = ids.to(device), y.to(device)
        loss = loss_fn(model(ids, lengths), y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y_true, y_pred, logits) over a loader, in the order of the dataset (TP04 Ex 12)."""
    model.eval()
    y_true, logits = [], []
    for ids, lengths, y in loader:
        logits.append(model(ids.to(device), lengths).float().cpu().numpy())
        y_true.append(y.numpy())
    logits_arr = np.concatenate(logits)
    return np.concatenate(y_true), logits_arr.argmax(axis=1), logits_arr


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Config,
    class_weights: np.ndarray | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], float]:
    """Adam + cross-entropy for `lstm_epochs` epochs; keeps the epoch with the best validation macro-F1.

    Choosing the epoch on validation is early stopping, a validation decision (S01 p12, TP04 cell 29).

    Returns:
        (model with the best weights, per-epoch history, training seconds).
    """
    device = cfg.device
    model.to(device)
    weight = (
        None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32, device=device)
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lstm_lr)
    history, best_f1, best_state = [], -1.0, None
    start = time.perf_counter()
    for epoch in range(1, cfg.lstm_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        y_true, y_pred, _ = predict(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **metrics.scores(y_true, y_pred, cfg.n_classes, prefix="val_"),
            "elapsed_s": round(time.perf_counter() - start, 1),
        }
        history.append(row)
        logger.info("lstm epoch %s", row)
        if row["val_macro_f1"] > best_f1:
            best_f1, best_state = row["val_macro_f1"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, history, time.perf_counter() - start

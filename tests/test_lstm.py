"""LSTM classifier against TP04 Ex 8-12 and S04 p14."""

import gzip
import math

import numpy as np
import pytest
import torch
from torch import nn

from ledgar.config import Config
from ledgar.models.lstm import LSTMClassifier, count_params, last_true_state, load_glove
from ledgar.text import ClauseDataset
from ledgar.train_lstm import LengthGroupedBatchSampler, fit, make_loaders, predict, train_one_epoch


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def test_count_params_course_value():
    assert count_params(20000, 50, 64, 4) == 1_029_956
    torch.manual_seed(0)
    assert n_params(LSTMClassifier(20000, 50, 64, 4)) == 1_029_956, "formula = PyTorch count"
    assert n_params(LSTMClassifier(500, 16, 8, 3, bidirectional=True)) == count_params(500, 16, 8, 3, True)


def test_pad_row_is_zero_and_logits_shape():
    torch.manual_seed(0)
    model = LSTMClassifier(50, 8, 6, 5)
    assert model.emb.weight[0].abs().sum() == 0, "padding_idx=0: the <pad> row is zero"
    ids = torch.tensor([[3, 4, 5], [6, 7, 0]])
    assert model(ids, torch.tensor([3, 2])).shape == (2, 5), "logits (B, C), no softmax"


@pytest.mark.parametrize("bidirectional", [False, True])
def test_packed_state_does_not_depend_on_padding(bidirectional):
    torch.manual_seed(0)
    model = LSTMClassifier(50, 8, 6, 5, bidirectional=bidirectional, dropout=0.0).eval()
    short = torch.tensor([[3, 4, 5]])
    padded = torch.tensor([[3, 4, 5, 0, 0, 0], [9, 8, 7, 6, 5, 4]])
    with torch.no_grad():
        alone = model(short, torch.tensor([3]))
        in_batch = model(padded, torch.tensor([3, 6]))[:1]
    assert torch.allclose(alone, in_batch, atol=1e-6), "the pads of the batch must not change the prediction"


def test_last_true_state_equals_packed_h_n():
    torch.manual_seed(0)
    lstm = nn.LSTM(4, 3, batch_first=True)
    x, lengths = torch.randn(2, 5, 4), torch.tensor([5, 2])
    out, _ = lstm(x)
    packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    _, (h_n, _) = lstm(packed)
    assert torch.allclose(last_true_state(out, lengths), h_n[-1], atol=1e-6)
    assert not torch.allclose(out[1, -1], h_n[-1][1]), "the last column is wrong for the shorter text"


def toy_data(n=128, seed=0):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 3, n)
    words = {0: "rent payment monthly", 1: "governed laws state", 2: "terminate notice party"}
    texts = [f"{words[y]} {words[y].split()[rng.integers(0, 3)]} the agreement" for y in labels]
    stoi = {
        w: i
        for i, w in enumerate(["<pad>", "<unk>", *sorted({t for v in words.values() for t in v.split()})])
    }
    return ClauseDataset(texts, labels, stoi, max_words=10), len(stoi)


def test_initial_loss_is_ln_c_and_training_reduces_it():
    torch.manual_seed(0)
    ds, V = toy_data()
    cfg = Config.from_env(device="cpu", n_classes=3, lstm_batch_size=16, lstm_epochs=3)
    train_loader, _ = make_loaders(ds, {}, cfg)
    model = LSTMClassifier(V, 8, 8, 3)
    optimizer, loss_fn = torch.optim.Adam(model.parameters(), lr=1e-2), nn.CrossEntropyLoss()
    first = train_one_epoch(model, train_loader, optimizer, loss_fn, "cpu")
    assert abs(first - math.log(3)) < 0.4, "first epoch close to ln(C)"
    assert train_one_epoch(model, train_loader, optimizer, loss_fn, "cpu") < first


def test_fit_keeps_best_epoch_and_predict_keeps_order():
    torch.manual_seed(0)
    ds, V = toy_data()
    cfg = Config.from_env(device="cpu", n_classes=3, lstm_batch_size=16, lstm_epochs=4, lstm_lr=1e-2)
    train_loader, evals = make_loaders(ds, {"val": ds}, cfg)
    model, history, _ = fit(LSTMClassifier(V, 8, 8, 3), train_loader, evals["val"], cfg)
    y_true, y_pred, logits = predict(model, evals["val"], "cpu")
    assert (y_true == ds.labels.numpy()).all(), "predict keeps the order of the dataset"
    best = max(h["val_macro_f1"] for h in history)
    assert len(history) == 4 and best > 0.9, "the toy task is learnt, and the best epoch is restored"
    assert logits.shape == (len(ds), 3)


def test_load_glove(tmp_path):
    path = tmp_path / "glove.txt.gz"
    with gzip.open(path, "wt") as f:
        f.write("3 2\nrent 0.5 -0.5\nlaw 1.0 2.0\nzebra 3.0 3.0\n")
    W, coverage = load_glove(path, ["<pad>", "<unk>", "rent", "law", "indemnify"], dim=2)
    assert W.shape == (5, 2) and (W[0] == 0).all(), "<pad> row stays zero"
    assert np.allclose(W[2], [0.5, -0.5]) and np.allclose(W[3], [1, 2])
    assert coverage == 2 / 3, "rent and law found, indemnify not"


def test_length_grouped_batches_cover_everything_once_per_epoch():
    lengths = np.random.default_rng(0).integers(1, 500, 1000)
    sampler = LengthGroupedBatchSampler(lengths.tolist(), batch_size=32, seed=0, mega_batch_mult=5)
    first, second = list(sampler), list(sampler)
    assert len(first) == len(sampler) == 32
    assert sorted(i for b in first for i in b) == list(range(1000)), "every clause once per epoch"
    assert first != second, "reshuffled at every epoch"
    spread = np.mean([np.ptp(lengths[b]) for b in first])
    assert spread < np.ptp(lengths) / 3, "clauses of a batch have similar lengths"

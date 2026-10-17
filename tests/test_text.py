"""Vocabulary, encoding and batching against the values of TP04 Ex 1-7."""

from collections import Counter

import torch

from ledgar.text import (
    PAD_ID,
    UNK_ID,
    ClauseDataset,
    build_vocab,
    collate_fn,
    count_words,
    encode,
    pad_share,
    tokenize,
    unk_rate,
)

ITOS = ["<pad>", "<unk>", "the", "match", "team", "wins", "stocks", "fall", "again", "market"]
STOI = {w: i for i, w in enumerate(ITOS)}


def test_tokenize_tp04():
    assert tokenize("Wall St. Bears Claw Back!") == ["wall", "st", "bears", "claw", "back"]
    assert tokenize("Section 2.1 of the Agreement") == ["section", "2", "1", "of", "the", "agreement"]
    assert tokenize("") == []


def test_build_vocab_tp04():
    itos, stoi = build_vocab(Counter({"b": 3, "a": 3, "c": 5, "d": 1}), min_freq=2)
    assert itos == ["<pad>", "<unk>", "c", "a", "b"], "pad, unk, then by decreasing count, ties alphabetical"
    assert stoi["<pad>"] == PAD_ID and stoi["<unk>"] == UNK_ID and stoi["b"] == 4


def test_count_words_on_train_only():
    counts = count_words(["The rent.", "the RENT is due"])
    assert counts == Counter({"the": 2, "rent": 2, "is": 1, "due": 1})


def test_encode_and_unk_rate():
    assert encode("the striker wins", STOI) == [2, UNK_ID, 5]
    assert encode("the team wins again", STOI, max_words=2) == [2, 4]
    assert unk_rate(["the striker wins", "stocks fall"], STOI) == 1 / 5


def test_collate_pads_to_the_batch_max():
    batch = [
        (torch.tensor(x), torch.tensor(y)) for x, y in [([2, 4, 5, 2, 3], 0), ([6, 7, 8], 1), ([2, 1, 5], 0)]
    ]
    ids, lengths, labels = collate_fn(batch)
    assert ids.tolist() == [[2, 4, 5, 2, 3], [6, 7, 8, 0, 0], [2, 1, 5, 0, 0]]
    assert lengths.tolist() == [5, 3, 3] and lengths.dtype == torch.long and labels.tolist() == [0, 1, 0]
    assert abs(pad_share(ids) - 4 / 15) < 1e-6


def test_dataset_empty_clause_becomes_unk():
    ds = ClauseDataset(["the team wins", "", "!!!"], [1, 2, 3], STOI, max_words=10)
    assert ds[0][0].tolist() == [2, 4, 5] and ds[1][0].tolist() == [UNK_ID] and ds[2][0].tolist() == [UNK_ID]
    assert ds[2][1].item() == 3 and ds[0][0].dtype == torch.long

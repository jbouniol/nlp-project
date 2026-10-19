"""m2: the retriever, its metrics and its diagnostics on toy data (no download)."""

import argparse
import json

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from ledgar import retrieval
from ledgar.config import Config
from ledgar.retrieval import Index


def test_config_from_args_accepts_a_script_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--choose", action="store_true")
    argv = ["--encoder", "minilm", "--classifier-run", "lora_s0", "--n-perturb", "7", "--device", "cpu"]
    cfg, args = Config.from_args(argv, parser=parser)
    assert args.encoder == "minilm" and not args.choose
    assert cfg.classifier_run == "lora_s0" and cfg.n_perturb == 7
    assert cfg.retrieval_k == 10 and cfg.hybrid_lambdas == (0.1, 0.3, 1.0, 3.0)


def test_fast_run_moves_the_embedding_cache(monkeypatch):
    monkeypatch.setenv("FAST_RUN", "1")
    cfg = Config.from_env(device="cpu")
    assert cfg.embeddings_dir == ".fast/embeddings" and cfg.n_perturb == 50


def test_precision_at_k_and_reciprocal_rank_on_toy_rankings():
    retrieved = np.array([[1, 1, 0], [0, 2, 2], [0, 0, 0]])
    y_q = np.array([1, 2, 2])
    hits = retrieved == y_q[:, None]
    p = retrieval.precision_at_k(hits, ks=(1, 3))
    assert p[1].tolist() == [1.0, 0.0, 0.0]
    assert p[3].tolist() == pytest.approx([2 / 3, 2 / 3, 0.0])
    assert retrieval.reciprocal_rank(hits).tolist() == [1.0, 0.5, 0.0]


def test_macro_by_type_averages_types_not_queries():
    values = np.array([1.0, 1.0, 1.0, 0.0])
    y_q = np.array([0, 0, 0, 1])
    assert retrieval.macro_by_type(values, y_q) == 0.5
    assert values.mean() == 0.75


def test_knn_vote_breaks_ties_by_the_nearest_neighbour():
    labels = np.array([[2, 5, 5, 2], [7, 3, 3, 3], [4, 1, 1, 4]])
    assert retrieval.knn_vote(labels).tolist() == [2, 3, 4]


def _toy_bank():
    E = torch.eye(4)
    labels = np.array([0, 0, 1, 1])
    return Index(E, labels, np.arange(4))


def test_search_returns_the_closest_bank_positions():
    idx = _toy_bank()
    q = torch.tensor([[0.9, 0.1, 0.0, 0.0], [0.0, 0.0, 0.2, 0.8]])
    scores, pos = idx.search(q, k=2)
    assert pos.tolist() == [[0, 1], [3, 2]]
    assert scores[0].tolist() == pytest.approx([0.9, 0.1])


def test_search_hard_filter_keeps_only_the_predicted_type():
    idx = _toy_bank()
    q = torch.tensor([[0.9, 0.1, 0.0, 0.0]])
    probs = np.array([[0.2, 0.8]])
    _, pos = idx.search(q, k=2, type_probs=probs, hard=True)
    assert sorted(pos[0].tolist()) == [2, 3]


def test_search_soft_prior_with_lambda_zero_equals_dense():
    idx = _toy_bank()
    q = torch.tensor([[0.3, 0.2, 0.4, 0.1]])
    probs = np.array([[0.9, 0.1]])
    _, dense = idx.search(q, k=4)
    _, soft0 = idx.search(q, k=4, type_probs=probs, lam=0.0)
    _, soft = idx.search(q, k=4, type_probs=probs, lam=1.0)
    assert dense.tolist() == soft0.tolist() == [[2, 0, 1, 3]]
    assert soft.tolist() == [[0, 1, 2, 3]], "a strong prior for type 0 moves its clauses first"


def test_search_exclude_self_masks_the_diagonal():
    idx = _toy_bank()
    _, pos = idx.search(idx.E, k=1, exclude_self=True)
    assert all(p != i for i, p in enumerate(pos[:, 0].tolist()))


def test_rank_of_source_counts_the_bank_clauses_above_it():
    idx = _toy_bank()
    q = torch.tensor([[0.9, 0.1, 0.0, 0.0]])
    assert idx.rank_of_source(q, source_scores=np.array([0.95])).tolist() == [1]
    assert idx.rank_of_source(q, source_scores=np.array([0.5])).tolist() == [2]
    probs = np.array([[0.1, 0.9]])
    ranks = idx.rank_of_source(q, np.array([0.5]), type_probs=probs, source_labels=np.array([1]), hard=True)
    assert ranks.tolist() == [1], "the hard filter removes the type-0 clauses above the source"


def test_sparse_index_gives_the_same_ranking():
    E = sp.csr_matrix(np.eye(4, dtype=np.float32))
    idx = Index(E, np.array([0, 0, 1, 1]), np.arange(4))
    q = sp.csr_matrix(np.array([[0.9, 0.1, 0.0, 0.0]], dtype=np.float32))
    _, pos = idx.search(q, k=2)
    assert pos.tolist() == [[0, 1]]
    assert idx.rank_of_source(q, np.array([0.5])).tolist() == [2]


def test_center_renormalises_the_rows():
    E = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]], dtype=np.float32)
    mean = E.mean(axis=0)
    C = retrieval.center(E, mean)
    assert np.allclose(np.linalg.norm(C, axis=1), 1.0)
    assert np.allclose(C, (E - mean) / np.linalg.norm(E - mean, axis=1, keepdims=True))


def test_cached_embeddings_reuse_the_cache_only_when_meta_matches(tmp_path):
    cfg = Config.from_env(device="cpu", embeddings_dir=str(tmp_path))
    calls = []

    def encode(texts, kind):
        calls.append(kind)
        return np.ones((len(texts), 3), dtype=np.float32)

    enc = retrieval.Encoder(retrieval.ENCODERS["minilm"], encode, 256)
    E1 = retrieval.cached_embeddings(enc, ["a", "b"], "train", "doc", cfg)
    E2 = retrieval.cached_embeddings(enc, ["a", "b"], "train", "doc", cfg)
    assert np.array_equal(E1, E2) and calls == ["doc"], "second call read from the cache"
    retrieval.cached_embeddings(enc, ["a", "b", "c"], "train", "doc", cfg)
    assert calls == ["doc", "doc"], "a different split size rebuilds the cache"
    meta = json.loads((tmp_path / "minilm" / "train.json").read_text())
    assert meta["n"] == 3 and meta["revision"] == retrieval.ENCODERS["minilm"].revision


def test_tfidf_is_never_cached(tmp_path):
    cfg = Config.from_env(device="cpu", embeddings_dir=str(tmp_path))
    enc = retrieval.Encoder(retrieval.ENCODERS["tfidf"], lambda t, k: sp.csr_matrix(np.eye(len(t))), None)
    assert sp.issparse(retrieval.cached_embeddings(enc, ["a"], "train", "doc", cfg))
    assert not list(tmp_path.iterdir())


def test_perturbation_views_are_deterministic_and_skip_short_clauses():
    text = "The first sentence is here. The second one follows it. The third closes the clause."
    rng = np.random.default_rng(0)
    assert retrieval.perturb(text, "head", rng) == "The first sentence is here. The second"
    assert retrieval.perturb(text, "tail", rng) == "one follows it. The third closes the clause."
    a = retrieval.perturb(text, "drop_sentence", np.random.default_rng(3))
    b = retrieval.perturb(text, "drop_sentence", np.random.default_rng(3))
    assert a == b and a.count(".") == 2 and a != text
    assert retrieval.perturb("One sentence. Two sentences.", "drop_sentence", rng) is None


def test_perturbation_sources_exclude_short_and_seen_clauses():
    texts = ["short clause", "x " * 60, "y " * 60, "z " * 60]
    seen = np.array([False, True, False, False])
    chosen = retrieval.perturbation_sources(texts, seen, min_words=40, n=10, seed=0)
    assert sorted(chosen.tolist()) == [2, 3]
    frame = retrieval.perturbation_queries(texts, chosen, seed=0)
    assert set(frame["view"]) == {"head", "tail"}, "no clause has 3 sentences"
    assert set(frame["source"]) == {2, 3} and len(frame) == 4


def test_constrained_queries_match_their_own_examples_and_name_ledgar_types():
    import re

    names = set(retrieval.LEDGAR_TYPES)
    assert len(retrieval.CONSTRAINED_QUERIES) >= 12
    for q in retrieval.CONSTRAINED_QUERIES:
        assert q.target in names, q.target
        if q.pattern is not None:
            assert re.search(q.pattern, q.example, flags=re.IGNORECASE), q.question


def test_constraint_precision_and_lift_on_a_toy_bank():
    q = retrieval.ConstrainedQuery(
        "notice in thirty days", "Notices", r"thirty \(30\) days", "within thirty (30) days"
    )
    bank_texts = ["thirty (30) days notice", "ten days notice", "any notice", "governed by Delaware law"]
    bank_labels = np.array([0, 0, 0, 1])
    names = ["Notices", "Governing Laws"]
    positions = np.array([[0, 1, 3]])
    row = retrieval.constraint_results([q], positions, bank_texts, bank_labels, names).iloc[0]
    assert row["type_p_at_k"] == pytest.approx(2 / 3)
    assert row["constraint_precision"] == pytest.approx(1 / 2), "among the 2 retrieved Notices, 1 matches"
    assert row["base_rate"] == pytest.approx(1 / 3) and row["lift"] == pytest.approx(1.5)


def test_effective_rank_and_anisotropy_on_known_matrices():
    eye = np.eye(6, dtype=np.float32)
    assert retrieval.effective_rank(eye) == pytest.approx(6.0)
    rank1 = np.tile(np.array([[0.6, 0.8, 0.0]], dtype=np.float32), (10, 1))
    assert retrieval.effective_rank(rank1) == pytest.approx(1.0)
    assert retrieval.anisotropy(eye, n_pairs=1000, seed=0) == pytest.approx(0.0)
    assert retrieval.anisotropy(rank1, n_pairs=1000, seed=0) == pytest.approx(1.0)


def test_hubness_is_skewed_when_one_clause_is_everybody_s_neighbour():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(200, 16)).astype(np.float32)
    E[1:] += 3 * E[0]  # every row leans towards row 0
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    idx = Index(torch.from_numpy(E), np.zeros(200, dtype=int), np.arange(200))
    skew, n_k = retrieval.hubness(idx, k=5)
    assert n_k.argmax() == 0 and n_k[0] > 3 * np.median(n_k) and skew > 1


def test_alignment_and_uniformity_on_a_toy_bank():
    E = np.eye(4, dtype=np.float32)
    labels = np.array([0, 0, 1, 1])
    assert retrieval.alignment(E, labels, n_pairs=100, seed=0) == pytest.approx(0.0)
    E2 = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32)
    assert retrieval.alignment(E2, labels, n_pairs=100, seed=0) == pytest.approx(1.0)
    assert retrieval.uniformity(E, n_pairs=1000, seed=0) == pytest.approx(-4.0)


def test_choose_applies_the_gate_then_the_best_p10_then_simplicity():
    rows = [
        {"run": "classifier", "centered": False, "variant": "dense", "p10": 0.95, "mrr": 0.40},
        {"run": "minilm", "centered": False, "variant": "hard", "p10": 0.90, "mrr": 0.80},
        {"run": "minilm", "centered": False, "variant": "soft:0.3", "p10": 0.88, "mrr": 0.90},
        {"run": "minilm", "centered": True, "variant": "dense", "p10": 0.88, "mrr": 0.91},
        {"run": "minilm", "centered": False, "variant": "dense", "p10": 0.88, "mrr": 0.92},
        {"run": "e5", "centered": False, "variant": "dense", "p10": 0.80, "mrr": 0.93},
    ]
    choice = retrieval.choose(rows, gate=0.95)
    assert choice["threshold"] == pytest.approx(0.95 * 0.93)
    assert (choice["run"], choice["centered"], choice["variant"]) == ("minilm", False, "dense")
    assert choice["n_gated"] == 4


def test_retrieval_table_reads_one_row_per_run(tmp_path):
    for name, p10 in (("minilm", 0.8), ("tfidf", 0.7)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text(
            json.dumps(
                {
                    "model": "retrieval",
                    "encoder": {"name": name},
                    "centered": False,
                    "encode_time_s": 1.0,
                    "val": {
                        "clause": {
                            "p_at_1_macro": 0.9,
                            "p_at_5_macro": 0.85,
                            "p_at_10_macro": p10,
                            "mrr": 0.95,
                            "knn_macro_f1": 0.8,
                        },
                        "perturbation": {"mrr": 0.9},
                        "type_names": {"p_at_10": 0.5},
                    },
                    "geometry": {"anisotropy": 0.1, "effective_rank": 100.0, "hubness_skew": 3.0},
                }
            )
        )
    table = retrieval.retrieval_table(tmp_path)
    assert table["run"].tolist() == ["minilm", "tfidf"] and table["p_at_10"].tolist() == [0.8, 0.7]
    assert "| minilm |" in retrieval.to_markdown(table) if hasattr(retrieval, "to_markdown") else True

"""End-to-end smoke run under FAST_RUN: grows with each milestone, must stay under a few minutes on CPU."""

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ledgar import data, metrics, runs
from ledgar.config import ROOT, Config, device_profile


def test_config_fast_run_overrides(monkeypatch):
    monkeypatch.setenv("FAST_RUN", "1")
    monkeypatch.setenv("SEED", "2")
    cfg = Config.from_env(epochs=2)
    assert cfg.fast_run and cfg.seed == 2
    assert cfg.max_length == 128 and cfg.n_train_per_class == 30, "FAST_RUN shrinks the data and sequences"
    assert cfg.epochs == 2, "an explicit override wins over the FAST_RUN value"
    assert cfg.device in {"cpu", "cuda", "mps"}


def test_config_from_args():
    cfg, args = Config.from_args(
        [
            "--model",
            "lora",
            "--lr-lora",
            "1e-3",
            "--lora-targets",
            "q_lin",
            "k_lin",
            "--lstm-class-weight",
            "none",
        ]
    )
    assert args.model == "lora" and cfg.lr_lora == 1e-3 and cfg.lstm_class_weight is None
    assert cfg.lora_targets == ("q_lin", "k_lin")


@pytest.mark.parametrize(
    ("bf16", "memory_gb", "expected"),
    [
        (True, 80, ("bf16", 32, 1, 256)),  # H100 / A100 80 GB
        (False, 16, ("fp16", 32, 1, 64)),  # T4
        (False, 12, ("fp16", 16, 2, 64)),
    ],
)
def test_device_profile_follows_the_gpu(monkeypatch, bf16, memory_gb, expected):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: bf16)
    properties = SimpleNamespace(total_memory=memory_gb * 1e9)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: properties)
    p = device_profile("cuda")
    assert (p["precision"], p["batch_size"], p["grad_accum"], p["eval_batch_size"]) == expected


def test_device_profile_keeps_the_effective_batch_and_explicit_values_win():
    for device in ("cpu", "mps"):
        p = device_profile(device)
        assert p["precision"] == "fp32" and p["batch_size"] * p["grad_accum"] == 32
    cfg = Config.from_env(device="cpu")
    assert (cfg.precision, cfg.batch_size, cfg.grad_accum, cfg.eval_batch_size) == ("fp32", 32, 1, 64)
    cfg = Config.from_env(device="cpu", batch_size=4, grad_accum=8)
    assert (cfg.batch_size, cfg.grad_accum) == (4, 8), "an explicit value wins over the device profile"


def test_save_run_roundtrip(tmp_path):
    cfg = Config.from_env(device="cpu")
    run_dir = tmp_path / "results" / "dummy_seed0"
    y = np.array([0, 1, 2, 1])
    runs.save_run(run_dir, cfg, {"model": "dummy", "val": {"macro_f1": 0.5}}, duration_s=1.23)
    runs.save_predictions(run_dir, "val", y, y, logits=np.eye(3)[y], label_names=["a", "b", "c"])
    with pytest.raises(FileNotFoundError):
        runs.results_table(tmp_path / "results")  # pred_test.npz is still missing
    runs.save_predictions(run_dir, "test", y, y[::-1])
    runs.save_run(run_dir, cfg, {"test": {"macro_f1": 0.25, "accuracy": 0.5}})
    table = runs.results_table(tmp_path / "results")
    assert table.loc[0, "val_macro_f1"] == 0.5 and table.loc[0, "test_macro_f1"] == 0.25, "metrics are merged"
    assert runs.load_predictions(run_dir, "val")["logits"].shape == (4, 3)
    assert runs.load_run(run_dir)["manifest"]["duration_s"] == 1.2
    assert "| dummy_seed0 |" in runs.to_markdown(table)


@pytest.mark.slow
def test_fast_run_data_and_floor(tmp_path):
    """FAST_RUN pipeline so far: per-class LEDGAR subset -> lazy floor -> a complete run directory."""
    cfg = Config.from_env(fast_run=True, device="cpu", out_dir=str(tmp_path))
    ledgar = data.load_ledgar(cfg)
    counts = data.class_counts(ledgar, cfg.n_classes)
    assert (counts["train"] > 0).all() and counts["train"].max() <= cfg.n_train_per_class, (
        "all 100 types kept"
    )
    assert (counts["validation"] > 0).sum() == 99, "Books has no validation clause"
    train_labels = np.array(ledgar["train"]["label"])
    for split in ("val", "test"):
        y = np.array(ledgar["validation" if split == "val" else "test"]["label"])
        y_pred = data.lazy_baseline(train_labels, len(y))
        assert metrics.macro_f1_score(y, y_pred, cfg.n_classes) < 0.05
        runs.save_predictions(tmp_path / "lazy", split, y, y_pred)
    runs.save_run(tmp_path / "lazy", cfg, {"model": "lazy", "val": {}, "test": {}})
    assert runs.missing_files(tmp_path / "lazy") == []


@pytest.mark.slow
def test_fast_run_tfidf_script(tmp_path):
    """m1 from the command line: scripts/train.py writes a complete run directory."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--model",
        "tfidf",
        "--fast-run",
        "--device",
        "cpu",
    ]
    subprocess.run([*cmd, "--out-dir", str(tmp_path)], check=True, cwd=ROOT)
    assert runs.missing_files(tmp_path / "tfidf") == []
    m = runs.load_run(tmp_path / "tfidf")["metrics"]
    assert m["val"]["macro_f1"] > 0.3 and m["trainable_params"] > 0, (
        "far above the lazy floor even on 30 clauses per type"
    )


@pytest.mark.slow
def test_fast_run_lstm_script(tmp_path):
    """m3 from the command line: a final run (val + test + weights) and a search run (validation only)."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--model",
        "lstm",
        "--fast-run",
        "--device",
        "cpu",
    ]
    subprocess.run([*cmd, "--out-dir", str(tmp_path)], check=True, cwd=ROOT)
    assert (
        runs.missing_files(tmp_path / "lstm_seed0") == []
        and (tmp_path / "lstm_seed0" / "model.pt").exists()
        and (tmp_path / "lstm_seed0" / "vocab.json").exists()
    )
    subprocess.run([*cmd, "--out-dir", str(tmp_path), "--search", "--no-glove"], check=True, cwd=ROOT)
    assert (tmp_path / "search" / "lstm_seed0" / "pred_val.npz").exists()
    assert not (tmp_path / "search" / "lstm_seed0" / "pred_test.npz").exists(), (
        "a search run never reads test"
    )


@pytest.mark.slow
def test_fast_run_lora_search_then_predict_test(tmp_path):
    """m4 on CPU: a LoRA search run (validation only), then its promotion with the test predicted once."""
    train = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--model",
        "lora",
        "--fast-run",
        "--device",
        "cpu",
    ]
    out = ["--out-dir", str(tmp_path), "--checkpoint-dir", str(tmp_path / "ckpt"), "--name", "lora_s0"]
    subprocess.run([*train, "--search", *out], check=True, cwd=ROOT)
    search_dir = tmp_path / "search" / "lora_s0"
    assert (search_dir / "adapter" / "adapter_model.safetensors").exists()
    assert not (search_dir / "pred_test.npz").exists() and not any((tmp_path / "ckpt").glob("*/checkpoint-*"))
    m = runs.load_run(search_dir)["metrics"]
    assert abs(m["initial_loss"] - np.log(100)) < 0.5, "random head: initial loss close to ln(100)"
    assert m["trainable_params"] == 147_456 + 768 * 768 + 768 + 768 * 100 + 100, "LoRA q/v + the whole head"
    promote = [
        sys.executable,
        str(ROOT / "scripts" / "predict_test.py"),
        "--name",
        "lora_s0",
        "--device",
        "cpu",
    ]
    subprocess.run([*promote, "--out-dir", str(tmp_path)], check=True, cwd=ROOT)
    final = runs.load_run(tmp_path / "lora_s0")["metrics"]
    assert runs.missing_files(tmp_path / "lora_s0") == [] and not search_dir.exists()
    assert final["val_reproduced"] == 1.0, "the saved adapter gives back the validation predictions"
    assert final["val"]["accuracy"] > 0.05 and final["test"]["accuracy"] > 0.05, (
        "predictions line up with clauses"
    )


def test_predictions_fall_back_to_the_csv(tmp_path):
    """pred_<split>.npz is not versioned: a fresh clone reads the same predictions from pred_<split>.csv."""
    y_true, y_pred = np.array([3, 1, 2, 1]), np.array([3, 1, 1, 1])
    runs.save_predictions(tmp_path, "test", y_true, y_pred, logits=np.eye(4)[y_pred], index=np.arange(10, 14))
    (tmp_path / "pred_test.npz").unlink()
    p = runs.load_predictions(tmp_path, "test")
    assert (
        (p["y_true"] == y_true).all()
        and (p["y_pred"] == y_pred).all()
        and (p["index"] == np.arange(10, 14)).all()
    )
    assert "logits" not in p


@pytest.mark.slow
def test_fast_run_retrieval_script(tmp_path):
    """m2 on CPU: two encoders on the FAST_RUN subset, the rule, then the test split read once by --final."""
    script = [sys.executable, str(ROOT / "scripts" / "retrieval.py"), "--fast-run", "--device", "cpu"]
    out = ["--out-dir", str(tmp_path), "--embeddings-dir", str(tmp_path / "emb")]
    subprocess.run([*script, "--encoder", "minilm", *out], check=True, cwd=ROOT)
    for name in ("minilm", "minilm+center"):
        run_dir = tmp_path / "retrieval" / name
        assert runs.missing_files(run_dir) == ["pred_test.csv"], "a validation run has everything but test"
        assert (run_dir / "topk_val.npz").exists() and (run_dir / "hubs.csv").exists()
        m = runs.load_run(run_dir)["metrics"]
        assert m["val"]["clause"]["p_at_10_macro"] > 0.05 and "test" not in m
        assert m["probs_source"] == "tfidf refit" and set(m["val"]["hybrid"]) >= {"hard", "soft:0.1"}
        assert 0 < m["val"]["perturbation"]["mrr"] <= 1 and len(m["val"]["constrained"]) >= 12
    assert (tmp_path / "emb" / "minilm" / "train.npy").exists(), "embeddings cached under --embeddings-dir"
    subprocess.run([*script, "--encoder", "tfidf", *out], check=True, cwd=ROOT)
    assert np.isnan(runs.load_run(tmp_path / "retrieval" / "tfidf")["metrics"]["geometry"]["effective_rank"])
    subprocess.run([*script, "--choose", *out], check=True, cwd=ROOT)
    choice = json.loads((tmp_path / "retrieval" / "choice.json").read_text())
    assert choice["run"] in {"minilm", "minilm+center", "tfidf"} and choice["n_candidates"] == 3 * 6
    subprocess.run([*script, "--final", *out], check=True, cwd=ROOT)
    final = runs.load_run(tmp_path / "retrieval" / choice["run"])["metrics"]["test"]
    assert set(final) >= {"dense", choice["variant"], "comparison"} or choice["variant"] == "dense"
    assert final["dense"]["clause"]["p_at_10_macro"] > 0.05 and "by_group" in final["dense"]
    assert final["read_at_utc"] and "git_sha" in final, "the one test read is dated and tied to a commit"
    assert runs.missing_files(tmp_path / "retrieval" / choice["run"]) == []
    assert (tmp_path / "retrieval" / choice["run"] / "topk_test.npz").exists()
    again = subprocess.run([*script, "--final", *out], cwd=ROOT, capture_output=True, text=True)
    assert again.returncode != 0, "the test split is predicted once"

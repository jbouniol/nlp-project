"""End-to-end smoke run under FAST_RUN: grows with each milestone, must stay under a few minutes on CPU."""

import subprocess
import sys

import numpy as np
import pytest

from ledgar import data, metrics, runs
from ledgar.config import ROOT, Config


def test_config_fast_run_overrides(monkeypatch):
    monkeypatch.setenv("FAST_RUN", "1")
    monkeypatch.setenv("SEED", "2")
    cfg = Config.from_env(epochs=2)
    assert cfg.fast_run and cfg.seed == 2
    assert cfg.max_length == 128 and cfg.n_train_per_class == 30, "FAST_RUN shrinks the data and sequences"
    assert cfg.epochs == 2, "an explicit override wins over the FAST_RUN value"
    assert cfg.device in {"cpu", "cuda", "mps"}


def test_config_from_args():
    cfg, args = Config.from_args(["--model", "lora", "--lr-lora", "1e-3", "--lora-targets", "q_lin", "k_lin"])
    assert args.model == "lora" and cfg.lr_lora == 1e-3
    assert cfg.lora_targets == ("q_lin", "k_lin")


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
        runs.missing_files(tmp_path / "lstm_seed0") == [] and (tmp_path / "lstm_seed0" / "model.pt").exists()
    )
    subprocess.run([*cmd, "--out-dir", str(tmp_path), "--search", "--bidirectional"], check=True, cwd=ROOT)
    assert (tmp_path / "search" / "lstm_seed0" / "pred_val.npz").exists()
    assert not (tmp_path / "search" / "lstm_seed0" / "pred_test.npz").exists(), (
        "a search run never reads test"
    )


@pytest.mark.slow
def test_fast_run_lora_script(tmp_path):
    """m4 from the command line (LoRA on CPU): complete run, adapter saved, predictions in dataset order."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train.py"),
        "--model",
        "lora",
        "--fast-run",
        "--device",
        "cpu",
    ]
    subprocess.run(
        [*cmd, "--out-dir", str(tmp_path), "--checkpoint-dir", str(tmp_path / "ckpt")], check=True, cwd=ROOT
    )
    run_dir = tmp_path / "lora_seed0"
    assert runs.missing_files(run_dir) == [] and (run_dir / "adapter" / "adapter_model.safetensors").exists()
    m = runs.load_run(run_dir)["metrics"]
    assert abs(m["initial_loss"] - np.log(100)) < 0.5, "random head: initial loss close to ln(100)"
    assert m["trainable_params"] == 147_456 + 768 * 768 + 768 + 768 * 100 + 100, "LoRA q/v + the whole head"
    pred = runs.load_predictions(run_dir, "val")
    assert m["val"]["accuracy"] == pytest.approx(np.mean(pred["y_true"] == pred["y_pred"]))
    assert m["val"]["accuracy"] > 0.05, "above chance: the predictions line up with the clauses"

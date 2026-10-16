"""End-to-end smoke run under FAST_RUN: grows with each milestone, must stay under a few minutes on CPU."""

import numpy as np
import pytest

from ledgar import data, metrics, runs
from ledgar.config import Config


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

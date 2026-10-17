"""Predict the test split once with a run chosen on validation, and promote it to a final run.

    python scripts/predict_test.py --name lora_lr5e-4_seed0          # results/search/<name> -> results/<name>

The run is first checked: its saved weights must give back its validation predictions. Then the test
split is read, once, and the run directory moves from results/search/ to results/.
"""

from __future__ import annotations

import argparse
import logging
import shutil

import numpy as np

from ledgar import data, metrics, runs
from ledgar.config import ROOT
from ledgar.predict import load_classifier, run_config

logger = logging.getLogger("predict_test")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--models-dir", default=None, help="where full fine-tuned weights were saved")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    src, dst = ROOT / args.out_dir / "search" / args.name, ROOT / args.out_dir / args.name
    if (dst / "pred_test.npz").exists():
        raise SystemExit(f"{dst} already has test predictions: the test split is predicted once")
    clf = load_classifier(src, device=args.device, models_dir=args.models_dir)
    cfg = run_config(src)
    ledgar = data.load_ledgar(cfg)

    saved = runs.load_predictions(src, "val")
    agreement = np.mean(clf.logits(ledgar["validation"]["text"]).argmax(axis=1) == saved["y_pred"])
    logger.info("validation predictions reproduced: %.4f", agreement)
    if agreement < 0.999:
        raise SystemExit(f"the loaded weights do not reproduce the validation predictions ({agreement:.4f})")

    shutil.copytree(src, dst, dirs_exist_ok=True)
    y_true = np.array(ledgar["test"]["label"])  # predict & save: the test split, used once
    logits = clf.logits(ledgar["test"]["text"])
    y_pred = logits.argmax(axis=1)
    index = ledgar["test"]["index"]
    runs.save_predictions(
        dst, "test", y_true, y_pred, logits=logits, index=index, label_names=clf.label_names
    )
    test = {"accuracy": metrics.accuracy(y_true, y_pred), "macro_f1": metrics.macro_f1_score(y_true, y_pred)}
    runs.save_run(dst, cfg, {"test": test, "val_reproduced": float(agreement)})
    shutil.rmtree(src)
    logger.info("test %s -> %s", test, dst)


if __name__ == "__main__":
    main()

"""Predict the test split once with a run chosen on validation, and promote it to a final run.

    python scripts/predict_test.py --name lora_lr5e-4_seed0          # results/search/<name> -> results/<name>

The run is first checked: its saved weights must give back its validation predictions (99.9 % of the clauses
that are not near-ties between two types). Then the test split is read, once, and the run directory moves
from results/search/ to results/.
"""

from __future__ import annotations

import argparse
import logging
import shutil

import numpy as np

from ledgar import data, runs
from ledgar.config import Config
from ledgar.predict import load_classifier, run_config

logger = logging.getLogger("predict_test")
TIE_MARGIN = 0.05


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", required=True)
    env, args = Config.from_args(argv, parser)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")

    src = env.path(env.out_dir, "search", args.name)
    dst = env.path(env.out_dir, args.name)
    if (dst / "pred_test.csv").exists() or (dst / "pred_test.npz").exists():
        raise SystemExit(f"{dst} already has test predictions: the test split is predicted once")
    models_dir = env.path(env.models_dir) if "models_dir" in vars(args) else None
    clf = load_classifier(src, device=env.device, models_dir=models_dir)
    cfg = run_config(src)
    ledgar = data.load_ledgar(cfg)

    saved = runs.load_predictions(src, "val")
    same = clf.logits(ledgar["validation"]["text"]).argmax(axis=1) == saved["y_pred"]
    # Near-ties between the two best types can flip between the fp16 evaluation of a GPU run and this fp32
    # one; saved logits are float16 (step ~0.01 around 8), so a margin under 0.05 is a tie, not an error.
    if "logits" in saved:
        top2 = np.sort(saved["logits"].astype(np.float32), axis=1)[:, -2:]
        clear = (top2[:, 1] - top2[:, 0]) > TIE_MARGIN
    else:  # no saved logits (pred_val.npz is not versioned): every clause counts
        clear = np.ones_like(same)
    agreement, agreement_clear = float(same.mean()), float(same[clear].mean())
    logger.info(
        "validation predictions reproduced: %.4f (%.4f outside near-ties)", agreement, agreement_clear
    )
    if agreement_clear < 0.999:
        raise SystemExit(
            f"the loaded weights do not reproduce the validation predictions ({agreement_clear:.4f})"
        )

    shutil.copytree(src, dst, dirs_exist_ok=True)
    y_true = np.array(ledgar["test"]["label"])  # predict & save: the test split, used once
    logits = clf.logits(ledgar["test"]["text"])
    test = runs.evaluate_and_save(
        dst,
        "test",
        y_true,
        logits.argmax(axis=1),
        cfg.n_classes,
        logits=logits,
        index=ledgar["test"]["index"],
        label_names=clf.label_names,
    )
    runs.save_run(
        dst, cfg, {"test": test, "val_reproduced": agreement, "val_reproduced_outside_ties": agreement_clear}
    )
    shutil.rmtree(src)
    logger.info("test %s -> %s", test, dst)


if __name__ == "__main__":
    main()

"""Train one model and write its run directory under results/.

python scripts/train.py --model tfidf                      # the configuration of Config
python scripts/train.py --model tfidf --logreg-c 3 --name tfidf_c3
FAST_RUN=1 python scripts/train.py --model tfidf           # quick check, written to .fast/results
"""

from __future__ import annotations

import json
import logging
import shutil
import time

import numpy as np
import torch

from ledgar import data, metrics, runs, text, train_bert, train_lstm, train_tfidf
from ledgar.config import Config, set_seed
from ledgar.models.lora import make_peft_model
from ledgar.models.lstm import LSTMClassifier, load_glove

logger = logging.getLogger("train")


def scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict[str, float]:
    return {
        "accuracy": metrics.accuracy(y_true, y_pred),
        "macro_f1": metrics.macro_f1_score(y_true, y_pred, n_classes),
    }


def run_tfidf(cfg: Config, ledgar, run_dir) -> None:
    labels = data.label_names(ledgar)
    model, seconds, converged = train_tfidf.fit_timed(
        train_tfidf.make_pipeline(cfg), ledgar["train"]["text"], np.array(ledgar["train"]["label"])
    )
    n_params = train_tfidf.count_parameters(model)
    result = {"model": "tfidf", "trainable_params": n_params, "converged": converged}
    # predict & save: validation, then the test split, used once
    for split, key in (("val", "validation"), ("test", "test")):
        y_true = np.array(ledgar[key]["label"])
        logits = model.decision_function(ledgar[key]["text"])
        y_pred = logits.argmax(axis=1)
        index = ledgar[key]["index"]
        runs.save_predictions(run_dir, split, y_true, y_pred, logits=logits, index=index, label_names=labels)
        result[split] = scores(y_true, y_pred, cfg.n_classes)
    runs.save_run(run_dir, cfg, result, duration_s=seconds)
    logger.info("tfidf: val %s | test %s | %.0f s", result["val"], result["test"], seconds)


def run_lstm(cfg: Config, ledgar, run_dir, search: bool) -> None:
    labels = data.label_names(ledgar)
    itos, stoi = text.build_vocab(text.count_words(ledgar["train"]["text"]), cfg.min_freq)  # train only
    splits = (
        {"train": "train", "val": "validation"}
        if search
        else {"train": "train", "val": "validation", "test": "test"}
    )
    sets = {
        k: text.ClauseDataset(ledgar[s]["text"], ledgar[s]["label"], stoi, cfg.max_words)
        for k, s in splits.items()
    }
    train_loader, evals = train_lstm.make_loaders(sets.pop("train"), sets, cfg)
    set_seed(cfg.seed)  # right before the model is built
    model = LSTMClassifier(len(itos), cfg.emb_dim, cfg.hidden, cfg.n_classes, cfg.bidirectional, cfg.dropout)
    result = {
        "model": "lstm",
        "vocabulary": len(itos),
        "unk_rate_val": text.unk_rate(ledgar["validation"]["text"], stoi),
    }
    if cfg.glove:
        assert cfg.emb_dim == 50, "GloVe vectors have 50 dimensions: pass --emb-dim 50"
        W, result["glove_coverage"] = load_glove(cfg.path(cfg.glove_path), itos, dim=50, seed=cfg.seed)
        with torch.no_grad():
            model.emb.weight.copy_(torch.from_numpy(W))
    train_counts = np.bincount(ledgar["train"]["label"], minlength=cfg.n_classes)
    weights = data.class_weights(train_counts, cfg.lstm_class_weight) if cfg.lstm_class_weight else None
    model, history, seconds = train_lstm.fit(model, train_loader, evals["val"], cfg, class_weights=weights)
    result["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result["best_epoch"] = max(history, key=lambda h: h["val_macro_f1"])["epoch"]
    for (
        split,
        loader,
    ) in evals.items():  # predict & save: validation, then (final runs only) the test split, once
        y_true, y_pred, logits = train_lstm.predict(model, loader, cfg.device)
        key = "validation" if split == "val" else "test"
        index = ledgar[key]["index"]
        runs.save_predictions(run_dir, split, y_true, y_pred, logits=logits, index=index, label_names=labels)
        result[split] = scores(y_true, y_pred, cfg.n_classes)
    runs.save_run(run_dir, cfg, result, log_history=history, duration_s=seconds)
    if not search:
        torch.save(model.state_dict(), run_dir / "model.pt")
        (run_dir / "vocab.json").write_text(json.dumps(itos) + "\n")
    logger.info("lstm: val %s | best epoch %d | %.0f s", result["val"], result["best_epoch"], seconds)


def run_bert(cfg: Config, ledgar, run_dir, name: str, lora: bool, search: bool) -> None:
    from transformers import DataCollatorWithPadding

    labels = data.label_names(ledgar)
    tokenizer = train_bert.load_tokenizer(cfg)
    keys = ["train", "validation"] + ([] if search else ["test"])
    tok = {k: train_bert.tokenize_split(ledgar[k], tokenizer, cfg.max_length) for k in keys}
    set_seed(cfg.seed)  # right before the model is built: the new head is random
    model = train_bert.make_model(cfg, labels)
    if lora:
        model = make_peft_model(model, cfg.lora_r, cfg.lora_alpha, cfg.lora_targets, cfg.lora_dropout)
    trainable, total = train_bert.count_parameters(model)
    first = DataCollatorWithPadding(tokenizer)([tok["train"][i] for i in range(cfg.batch_size)])
    result = {
        "model": "lora" if lora else "bert",
        "trainable_params": trainable,
        "total_params": total,
        "memory_gb_estimate": train_bert.training_memory_gb(total - trainable, trainable),
        "initial_loss": train_bert.initial_loss(model, {k: v.to(model.device) for k, v in first.items()}),
    }
    checkpoints = cfg.path(cfg.checkpoint_dir, name)
    args = train_bert.make_args(cfg, checkpoints, cfg.lr_lora if lora else cfg.lr_full)
    trainer = train_bert.build_trainer(model, args, tok["train"], tok["validation"], tokenizer)
    start = time.perf_counter()
    trainer.train(
        resume_from_checkpoint=any(checkpoints.glob("checkpoint-*")) or None
    )  # resumes a cut session
    seconds = time.perf_counter() - start
    result["global_step"] = trainer.state.global_step
    for split, key in (
        ("val", "validation"),
        ("test", "test"),
    ):  # predict & save: test used once, final runs only
        if key not in tok:
            continue
        logits = train_bert.predict_logits(trainer, tok[key])
        y_true, y_pred = np.array(ledgar[key]["label"]), logits.argmax(axis=1)
        index = ledgar[key]["index"]
        runs.save_predictions(run_dir, split, y_true, y_pred, logits=logits, index=index, label_names=labels)
        result[split] = scores(y_true, y_pred, cfg.n_classes)
    runs.save_run(run_dir, cfg, result, log_history=trainer.state.log_history, duration_s=seconds)
    (run_dir / "labels.json").write_text(json.dumps(labels) + "\n")
    if lora:
        trainer.model.save_pretrained(run_dir / "adapter")  # a few MB: versioned
    else:
        trainer.save_model(cfg.path(cfg.models_dir, name))  # 260 MB: not versioned, kept on Drive
    shutil.rmtree(checkpoints, ignore_errors=True)  # a full checkpoint with its optimizer is about 0.8 GB
    logger.info(
        "%s: val %s | %d steps | %.0f s", result["model"], result["val"], result["global_step"], seconds
    )


def main(argv: list[str] | None = None) -> None:
    cfg, args = Config.from_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    set_seed(cfg.seed)
    ledgar = data.load_ledgar(cfg)
    name = args.name or (args.model if args.model == "tfidf" else f"{args.model}_seed{cfg.seed}")
    run_dir = cfg.path(cfg.out_dir, "search", name) if args.search else cfg.path(cfg.out_dir, name)
    if args.model == "tfidf":
        run_tfidf(cfg, ledgar, run_dir)
    elif args.model == "lstm":
        run_lstm(cfg, ledgar, run_dir, args.search)
    elif args.model in ("bert", "lora"):
        run_bert(cfg, ledgar, run_dir, name, args.model == "lora", args.search)
    else:
        raise SystemExit(f"--model {args.model} is not implemented yet")
    logger.info("run written to %s", run_dir)


if __name__ == "__main__":
    main()

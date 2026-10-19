"""m4: fine-tuning a BERT-class encoder with the Hugging Face Trainer (S06 p7-11, TP06 Part B)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import SequentialSampler
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from ledgar import metrics
from ledgar.config import Config

logger = logging.getLogger(__name__)


def load_tokenizer(cfg: Config) -> PreTrainedTokenizerBase:
    """The model's own tokenizer: any other one is a silent bug (S03 p10)."""
    return AutoTokenizer.from_pretrained(cfg.model_name)


def tokenize_split(split: Dataset, tokenizer: PreTrainedTokenizerBase, max_length: int) -> Dataset:
    """WordPiece ids truncated to `max_length`, **not padded** (the collator pads per batch, TP06 Ex 6)."""

    def tokenize_batch(batch: dict[str, list]) -> dict[str, list]:
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return split.map(tokenize_batch, batched=True, remove_columns=["text"])


def make_model(cfg: Config, label_names: list[str]) -> torch.nn.Module:
    """Pretrained body + a **new, random** classification head of 100 outputs (S06 p7-8)."""
    return AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label_names),
        id2label=dict(enumerate(label_names)),
        label2id={name: i for i, name in enumerate(label_names)},
    )


def make_args(cfg: Config, output_dir: str | Path, lr: float) -> TrainingArguments:
    """Every hyperparameter of the loop (S06 p8-11, TP06 Ex 11), best epoch selected on validation.

    AdamW, weight decay 0.01, linear warmup over 10 % of the steps then linear decay, gradient clipping at 1,
    evaluation and checkpoint at each epoch, the checkpoint with the best validation macro-F1 reloaded at the
    end, batches of similar lengths, mixed precision and batch sizes of the device (`config.device_profile`).
    """
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=lr,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        warmup_steps=cfg.warmup_ratio,  # a float below 1 is a ratio of the training steps (transformers 5)
        lr_scheduler_type="linear",
        weight_decay=cfg.weight_decay,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        train_sampling_strategy="group_by_length" if cfg.group_by_length else "random",
        fp16=cfg.precision == "fp16",
        bf16=cfg.precision == "bf16",
        tf32=True if cfg.precision == "bf16" else None,  # bf16 GPUs (Ampere and later) all have TF32
        use_cpu=cfg.device == "cpu",
        logging_steps=50,
        report_to="none",
        seed=cfg.seed,
    )


class OrderedEvalTrainer(Trainer):
    """A Trainer whose evaluation and predictions keep the order of the dataset.

    With `train_sampling_strategy="group_by_length"`, transformers 5.19 also sorts the *evaluation* batches by
    length. Its own metrics stay right (labels are permuted with the predictions), but the predictions no
    longer line up with the clauses, which silently breaks every per-clause analysis and the paired tests.
    Training keeps the length-grouped batches; evaluation reads the dataset in order.
    """

    def _get_eval_sampler(self, eval_dataset: Dataset) -> SequentialSampler | None:
        return None if eval_dataset is None else SequentialSampler(eval_dataset)


def build_trainer(
    model: torch.nn.Module,
    args: TrainingArguments,
    train_tok: Dataset,
    val_tok: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Trainer:
    """Trainer = the training loop of course 1, with our own compute_metrics (accuracy, macro-F1)."""
    return OrderedEvalTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=metrics.compute_metrics,
    )


def predict_logits(trainer: Trainer, split_tok: Dataset) -> np.ndarray:
    """Logits (N, K) of a tokenized split, in its order (checked against its labels)."""
    out = trainer.predict(split_tok)
    if out.label_ids is not None and not np.array_equal(out.label_ids, np.asarray(split_tok["label"])):
        raise RuntimeError("predictions are not in the order of the dataset")
    return np.asarray(out.predictions, dtype=np.float32)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """(trainable, total) parameters (TP06 Ex 7)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


def training_memory_gb(n_frozen: int, n_trained: int, bytes_frozen: int = 4) -> float:
    """16 bytes per trained parameter (weight, gradient, Adam m and v in fp32), its weight only when frozen.

    S06 p16, TP06 Ex 17. Activations are not counted.
    """
    return (16 * n_trained + bytes_frozen * n_frozen) / 1e9


def initial_loss(model: torch.nn.Module, batch: dict[str, Any]) -> float:
    """Loss of the untrained model on one batch: about ln(100) = 4.61 with a random head (TP06 Ex 8)."""
    model.eval()
    inputs = {k: v for k, v in batch.items() if k != "index"}
    with torch.no_grad():
        return float(model(**inputs).loss)

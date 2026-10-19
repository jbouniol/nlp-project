"""Team interface: load a trained transformer classifier and label clauses.

    clf = load_classifier("results/lora_lr5e-4_seed0")
    labels, probs = clf.predict(["This Agreement shall be governed by the laws of Delaware."])
    clf.label_names[labels[0]]  # 'Governing Laws'

Used by the retriever (filter the clause bank by predicted type), the demo and the bias audit. The
clause is read with the model's own tokenizer and the training `max_length`.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerBase

from ledgar import metrics, train_bert
from ledgar.config import Config, device_profile, resolve_device

AUTOCAST_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


def length_sorted_batches(texts: Sequence[str], batch_size: int) -> list[np.ndarray]:
    """Index batches of texts sorted by word count, so each batch is padded to a similar length."""
    order = np.argsort([len(t.split()) for t in texts], kind="stable")
    return [order[i : i + batch_size] for i in range(0, len(order), batch_size)]


@dataclass
class Classifier:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    label_names: list[str]
    max_length: int
    device: str
    precision: str = "fp32"

    def autocast(self) -> contextlib.AbstractContextManager:
        """The mixed precision of the run on CUDA: the Trainer also evaluates under autocast, so the reloaded
        weights give back the saved validation predictions (the gate of scripts/predict_test.py)."""
        dtype = AUTOCAST_DTYPES.get(self.precision)
        if not str(self.device).startswith("cuda") or dtype is None:
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=dtype)

    def _tokenize(self, batch_texts: Sequence[str]):
        return self.tokenizer(
            list(batch_texts),
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

    @torch.no_grad()
    def logits(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        """Logits (N, 100), in the order of `texts`; batches of the device's evaluation size by default."""
        batch_size = batch_size or device_profile(self.device)["eval_batch_size"]
        self.model.eval()
        out = []
        for start in range(0, len(texts), batch_size):
            batch = self._tokenize(texts[start : start + batch_size])
            with self.autocast():
                out.append(self.model(**batch).logits.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.label_names)), dtype=np.float32)

    @torch.no_grad()
    def features(self, texts: Sequence[str], batch_size: int | None = None) -> np.ndarray:
        """Unit penultimate features relu(pre_classifier(h_CLS)) (N, 768), in the order of `texts`."""
        if not hasattr(self.model, "distilbert"):
            raise ValueError("not a DistilBERT run: no `distilbert` body to read features from")
        batch_size = batch_size or device_profile(self.device)["eval_batch_size"]
        self.model.eval()
        parts, order = [], []
        for idx in length_sorted_batches(texts, batch_size):
            batch = self._tokenize([texts[i] for i in idx])
            with self.autocast():
                h = self.model.distilbert(**batch).last_hidden_state[:, 0]
                h = torch.relu(self.model.pre_classifier(h))
            parts.append(torch.nn.functional.normalize(h.float(), dim=1).cpu().numpy())
            order.append(idx)
        if not parts:
            return np.zeros((0, self.model.pre_classifier.out_features), dtype=np.float32)
        E = np.concatenate(parts)
        out = np.empty_like(E)
        out[np.concatenate(order)] = E
        return out

    def predict(self, texts: Sequence[str], batch_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(label ids (N,), probabilities (N, 100))."""
        p = metrics.softmax(self.logits(texts, batch_size))
        return p.argmax(axis=1), p


def run_config(run_dir: str | Path) -> Config:
    """The Config a run was trained with (its config.json)."""
    return Config.from_dict(json.loads((Path(run_dir) / "config.json").read_text()))


def load_classifier(
    run_dir: str | Path, device: str = "auto", models_dir: str | Path | None = None
) -> Classifier:
    """Rebuild the classifier of a run directory.

    LoRA runs: the pretrained model + the adapter saved in `<run_dir>/adapter`. Full fine-tuning runs: the
    weights saved in `<models_dir>/<run name>` (default: the run's `models_dir`, not versioned).
    """
    run_dir = Path(run_dir)
    cfg = run_config(run_dir)
    kind = json.loads((run_dir / "metrics.json").read_text())["model"]
    names = json.loads((run_dir / "labels.json").read_text())
    if kind == "lora":
        from peft import PeftModel

        base = train_bert.make_model(cfg, names)
        model = PeftModel.from_pretrained(base, run_dir / "adapter").merge_and_unload()
    elif kind == "bert":
        model = AutoModelForSequenceClassification.from_pretrained(
            Path(models_dir) / run_dir.name if models_dir else cfg.path(cfg.models_dir, run_dir.name)
        )
    else:
        raise ValueError(f"{run_dir.name} is a {kind!r} run: only bert and lora runs can be loaded")
    device = resolve_device(device)
    tokenizer = train_bert.load_tokenizer(cfg)
    return Classifier(model.to(device), tokenizer, names, cfg.max_length, device, cfg.precision)

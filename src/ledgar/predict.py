"""Team interface: load a trained transformer classifier and label clauses.

    clf = load_classifier("results/lora_lr5e-4_seed0")
    labels, probs = clf.predict(["This Agreement shall be governed by the laws of Delaware."])
    clf.label_names[labels[0]]  # 'Governing Laws'

Used by the retriever (filter the clause bank by predicted type), the demo and the bias audit. The
clause is read with the model's own tokenizer and the training `max_length`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

from ledgar.config import Config, resolve_device


@dataclass
class Classifier:
    model: torch.nn.Module
    tokenizer: PreTrainedTokenizerBase
    label_names: list[str]
    max_length: int
    device: str

    @torch.no_grad()
    def logits(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Logits (N, 100), in the order of `texts`."""
        self.model.eval()
        out = []
        for start in range(0, len(texts), batch_size):
            batch = self.tokenizer(
                list(texts[start : start + batch_size]),
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            out.append(self.model(**batch).logits.float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.label_names)), dtype=np.float32)

    def predict(self, texts: Sequence[str], batch_size: int = 64) -> tuple[np.ndarray, np.ndarray]:
        """(label ids (N,), probabilities (N, 100))."""
        z = self.logits(texts, batch_size)
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        return p.argmax(axis=1), p


def run_config(run_dir: str | Path) -> Config:
    """The Config a run was trained with (its config.json)."""
    saved = json.loads((Path(run_dir) / "config.json").read_text())
    known = {f.name for f in fields(Config)}
    values = {k: tuple(v) if isinstance(v, list) else v for k, v in saved.items() if k in known}
    return Config(**values)


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
    if kind == "lora":
        from peft import PeftModel

        names = json.loads((run_dir / "labels.json").read_text())
        base = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=len(names),
            id2label=dict(enumerate(names)),
            label2id={n: i for i, n in enumerate(names)},
        )
        model = PeftModel.from_pretrained(base, run_dir / "adapter").merge_and_unload()
    elif kind == "bert":
        model = AutoModelForSequenceClassification.from_pretrained(
            Path(models_dir) / run_dir.name if models_dir else cfg.path(cfg.models_dir, run_dir.name)
        )
    else:
        raise ValueError(f"{run_dir.name} is a {kind!r} run: only bert and lora runs can be loaded")
    device = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    names = [model.config.id2label[i] for i in range(cfg.n_classes)]
    return Classifier(model.to(device), tokenizer, names, cfg.max_length, device)

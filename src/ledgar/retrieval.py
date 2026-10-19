"""m2: clause retrieval. Bank = the training clauses, exact cosine search, a type-filter hybrid with the
classifier of part 1, the evaluations and the geometry diagnostics (docs/DECISIONS.md, 2026-10-19)."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from ledgar.config import Config

logger = logging.getLogger(__name__)

Matrix = np.ndarray | sp.csr_matrix
EncodeFn = Callable[[Sequence[str], str], Matrix]
KINDS = ("doc", "clause", "nl")  # bank clause, pasted clause, lawyer's question


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    model: str
    revision: str | None = None
    doc_prefix: str = ""
    clause_query_prefix: str = ""
    nl_query_prefix: str = ""

    def prefix(self, kind: str) -> str:
        return {"doc": self.doc_prefix, "clause": self.clause_query_prefix, "nl": self.nl_query_prefix}[kind]


ENCODERS = {
    s.name: s
    for s in (
        EncoderSpec("tfidf", "tfidf"),
        EncoderSpec(
            "minilm", "sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
        ),
        EncoderSpec(
            "e5",
            "intfloat/e5-small-v2",
            "ffb93f3bd4047442299a41ebb6fa998a38507c52",
            doc_prefix="passage: ",
            clause_query_prefix="query: ",
            nl_query_prefix="query: ",
        ),
        EncoderSpec(
            "bge",
            "BAAI/bge-small-en-v1.5",
            "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            nl_query_prefix="Represent this sentence for searching relevant passages: ",
        ),
        EncoderSpec("distilbert", "distilbert-base-uncased", "12040accade4e8a0f71eabdb258fecc2e7e948be"),
        EncoderSpec(
            "legalbert", "nlpaueb/legal-bert-base-uncased", "15b570cbf88259610b082a167dacc190124f60f6"
        ),
    )
}


def encoder_spec(name: str) -> EncoderSpec:
    """`classifier:<run>` is the penultimate layer of a fine-tuned m4 run; the other names are in ENCODERS."""
    if name.startswith("classifier:"):
        return EncoderSpec(name, name.split(":", 1)[1])
    return ENCODERS[name]


@dataclass
class Encoder:
    spec: EncoderSpec
    encode: EncodeFn
    max_seq_length: int | None
    tokenizer: Any = None

    def truncated(self, texts: Sequence[str]) -> np.ndarray:
        """True for each text longer than the encoder's window (the TF-IDF encoder truncates nothing)."""
        if self.tokenizer is None or self.max_seq_length is None:
            return np.zeros(len(texts), dtype=bool)
        from ledgar.data import wordpiece_lengths

        return wordpiece_lengths(texts, self.tokenizer) > self.max_seq_length


def load_encoder(spec: EncoderSpec, cfg: Config, bank_texts: Sequence[str] | None = None) -> Encoder:
    """TF-IDF fitted on the bank, a sentence-transformer, or the classifier of an m4 run; unit rows always."""
    if spec.name == "tfidf":
        from ledgar.train_tfidf import make_vectorizer

        vectorizer = make_vectorizer(cfg).fit(bank_texts)
        return Encoder(spec, lambda texts, kind: vectorizer.transform(texts).astype(np.float32), None)
    if spec.name.startswith("classifier:"):
        return _classifier_encoder(spec, cfg)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec.model, revision=spec.revision, device=cfg.device)

    def encode(texts: Sequence[str], kind: str) -> np.ndarray:
        prefix = spec.prefix(kind)
        return model.encode(
            [prefix + t for t in texts],
            batch_size=cfg.encode_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

    return Encoder(spec, encode, model.max_seq_length, model.tokenizer)


def _classifier_encoder(spec: EncoderSpec, cfg: Config) -> Encoder:
    """relu(pre_classifier(h_CLS)) of a fine-tuned DistilBERT: the features the classification head reads."""
    from ledgar.predict import load_classifier

    clf = load_classifier(cfg.path(cfg.out_dir, spec.model), device=cfg.device)
    if not hasattr(clf.model, "distilbert"):
        raise ValueError(f"{spec.model} is not a DistilBERT run: no `distilbert` body to read features from")

    @torch.no_grad()
    def encode(texts: Sequence[str], kind: str) -> np.ndarray:
        out = []
        for start in range(0, len(texts), cfg.encode_batch_size):
            batch = clf.tokenizer(
                list(texts[start : start + cfg.encode_batch_size]),
                truncation=True,
                max_length=clf.max_length,
                padding=True,
                return_tensors="pt",
            ).to(clf.device)
            with clf.autocast():
                h = clf.model.distilbert(**batch).last_hidden_state[:, 0]
                h = torch.relu(clf.model.pre_classifier(h))
            out.append(torch.nn.functional.normalize(h.float(), dim=1).cpu().numpy())
        return np.concatenate(out)

    clf.model.eval()
    return Encoder(spec, encode, clf.max_length, clf.tokenizer)


def cached_embeddings(encoder: Encoder, texts: Sequence[str], split: str, kind: str, cfg: Config) -> Matrix:
    """Encode `texts` once per (encoder, split): <embeddings_dir>/<encoder>/<split>.npy, with a meta.json that
    must match the model, revision, size, kind and FAST_RUN flag, else it is rebuilt. TF-IDF: no cache."""
    if encoder.spec.name == "tfidf":
        return encoder.encode(texts, kind)
    folder = cfg.path(cfg.embeddings_dir, encoder.spec.name.replace(":", "_"))
    path, meta_path = folder / f"{split}.npy", folder / f"{split}.json"
    meta = {
        "model": encoder.spec.model,
        "revision": encoder.spec.revision,
        "n": len(texts),
        "kind": kind,
        "fast_run": cfg.fast_run,
    }
    if meta_path.exists() and path.exists():
        saved = json.loads(meta_path.read_text())
        if all(saved.get(k) == v for k, v in meta.items()):
            logger.info("embeddings of %s/%s read from %s", encoder.spec.name, split, path)
            return np.load(path)
    start = time.perf_counter()
    E = encoder.encode(texts, kind)
    folder.mkdir(parents=True, exist_ok=True)
    np.save(path, E)
    meta.update(
        {
            "d": int(E.shape[1]),
            "max_seq_length": encoder.max_seq_length,
            "seconds": round(time.perf_counter() - start, 1),
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    logger.info("embeddings of %s/%s: %s in %.0f s", encoder.spec.name, split, E.shape, meta["seconds"])
    return E


def center(E: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Subtract the bank mean and renormalise: the one-line fix for anisotropic encoders."""
    C = np.asarray(E, dtype=np.float32) - np.asarray(mean, dtype=np.float32)
    return C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)


class Index:
    """The clause bank: unit rows (dense torch tensor or sparse CSR), the type and official index of each."""

    def __init__(self, E: Matrix | torch.Tensor, labels: np.ndarray, index: np.ndarray, device: str = "cpu"):
        self.sparse = sp.issparse(E)
        self.E = (
            E.tocsr()
            if self.sparse
            else torch.as_tensor(np.asarray(E) if isinstance(E, np.ndarray) else E).to(device)
        )
        self.labels = np.asarray(labels)
        self.index = np.asarray(index)
        self.labels_t = torch.as_tensor(self.labels, device="cpu" if self.sparse else self.E.device)

    def __len__(self) -> int:
        return self.E.shape[0]

    def _scores(self, Q: Any) -> torch.Tensor:
        """Cosine of a block of unit queries with every bank row: (B, N)."""
        if self.sparse:
            return torch.from_numpy((Q @ self.E.T).toarray().astype(np.float32))
        Q = torch.as_tensor(np.asarray(Q) if isinstance(Q, np.ndarray) else Q).to(self.E.device, self.E.dtype)
        return Q @ self.E.T

    def _prior(
        self, probs: np.ndarray | None, lam: float | None, hard: bool, device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """(additive prior (B, N) or None, mask of rows to keep (B, N) or None) for the hybrid variants."""
        if probs is None or (lam is None and not hard):
            return None, None
        p = torch.as_tensor(np.asarray(probs, dtype=np.float32), device=device)
        prior = lam * torch.log(p.clamp_min(1e-6))[:, self.labels_t] if lam is not None else None
        keep = (self.labels_t[None, :] == p.argmax(dim=1)[:, None]) if hard else None
        return prior, keep

    def search(
        self,
        Q: Any,
        k: int,
        *,
        type_probs: np.ndarray | None = None,
        lam: float | None = None,
        hard: bool = False,
        exclude_self: bool = False,
        block: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k (scores, bank positions) of each query, in blocks; soft prior `+ lam log p(type)`, hard mask
        on the argmax type, and the diagonal removed when the queries are the bank itself."""
        scores, positions = [], []
        for start in range(0, Q.shape[0], block):
            s = self._scores(Q[start : start + block])
            probs = None if type_probs is None else type_probs[start : start + block]
            prior, keep = self._prior(probs, lam, hard, s.device)
            if prior is not None:
                s = s + prior
            if keep is not None:
                s = s.masked_fill(~keep, float("-inf"))
            if exclude_self:
                rows = torch.arange(s.shape[0], device=s.device)
                s[rows, rows + start] = float("-inf")
            top = torch.topk(s, min(k, s.shape[1]), dim=1)
            scores.append(top.values.float().cpu().numpy())
            positions.append(top.indices.cpu().numpy())
        return np.concatenate(scores), np.concatenate(positions)

    def rank_of_source(
        self,
        Q: Any,
        source_scores: np.ndarray,
        *,
        type_probs: np.ndarray | None = None,
        lam: float | None = None,
        hard: bool = False,
        source_labels: np.ndarray | None = None,
        block: int = 1024,
    ) -> np.ndarray:
        """Rank of each query's own source clause among bank + source: 1 + the bank rows scored above it. The
        source gets the same prior as a bank row of its type (and is masked out by a wrong hard filter)."""
        ranks = []
        for start in range(0, Q.shape[0], block):
            s = self._scores(Q[start : start + block])
            probs = None if type_probs is None else type_probs[start : start + block]
            prior, keep = self._prior(probs, lam, hard, s.device)
            src = torch.as_tensor(
                np.asarray(source_scores[start : start + block], dtype=np.float32), device=s.device
            )
            if prior is not None or keep is not None:
                p = torch.as_tensor(np.asarray(probs, dtype=np.float32), device=s.device)
                y = torch.as_tensor(np.asarray(source_labels[start : start + block]), device=s.device)
                if prior is not None:
                    s = s + prior
                    src = src + lam * torch.log(p.clamp_min(1e-6)).gather(1, y[:, None])[:, 0]
                if keep is not None:
                    s = s.masked_fill(~keep, float("-inf"))
                    src = src.masked_fill(p.argmax(dim=1) != y, float("-inf"))
            ranks.append(1 + (s > src[:, None]).sum(dim=1).cpu().numpy())
        return np.concatenate(ranks)


def precision_at_k(hits: np.ndarray, ks: Sequence[int] = (1, 5, 10)) -> dict[int, np.ndarray]:
    """Share of relevant rows among the first k retrieved, per query."""
    hits = np.asarray(hits, dtype=float)
    return {k: hits[:, :k].mean(axis=1) for k in ks}


def reciprocal_rank(hits: np.ndarray) -> np.ndarray:
    """1 / rank of the first relevant row within the retrieved list, 0 when there is none (MRR@k)."""
    hits = np.asarray(hits, dtype=bool)
    first = hits.argmax(axis=1)
    return np.where(hits.any(axis=1), 1.0 / (first + 1), 0.0)


def macro_by_type(values: np.ndarray, y_q: np.ndarray) -> float:
    """Mean over the clause types present among the queries of the per-type mean (the macro convention of
    `ledgar.metrics`: the same 99 types on validation whatever the retriever does)."""
    return float(pd.Series(np.asarray(values, dtype=float)).groupby(np.asarray(y_q)).mean().mean())


def knn_vote(labels: np.ndarray) -> np.ndarray:
    """Majority type among the k retrieved rows, ties broken towards the nearest rows (geometric bonus)."""
    labels = np.asarray(labels)
    m, k = labels.shape
    weights = 1 + 1e-3 * 0.5 ** np.arange(k)
    votes = np.zeros((m, labels.max() + 1))
    np.add.at(votes, (np.repeat(np.arange(m), k), labels.ravel()), np.tile(weights, m))
    return votes.argmax(axis=1)


def classifier_probs(
    cfg: Config, bank_texts: Sequence[str], bank_labels: np.ndarray, texts: Sequence[str]
) -> tuple[np.ndarray, str]:
    """Type probabilities (M, n_classes) of `texts` for the hybrid filter and their source: the m4 run
    `cfg.classifier_run`, or a TF-IDF + logistic regression refit on the bank (FAST_RUN and interim only)."""
    if cfg.classifier_run:
        from ledgar.predict import load_classifier

        clf = load_classifier(cfg.path(cfg.out_dir, cfg.classifier_run), device=cfg.device)
        return clf.predict(texts)[1], cfg.classifier_run
    from ledgar.train_tfidf import make_pipeline

    pipeline = make_pipeline(cfg).fit(bank_texts, np.asarray(bank_labels))
    probs = np.zeros((len(texts), cfg.n_classes), dtype=np.float32)
    probs[:, pipeline.classes_] = pipeline.predict_proba(texts)
    return probs, "tfidf refit"

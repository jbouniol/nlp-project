"""m2: clause retrieval. Bank = the training clauses, exact cosine search, a type-filter hybrid with the
classifier of part 1, the evaluations and the geometry diagnostics (docs/DECISIONS.md, 2026-10-19)."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
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


# --- within-type checks: self-retrieval under perturbation, constrained lawyer questions ---

VIEWS = ("head", "tail", "drop_sentence")
_SENTENCE_END = re.compile(r"(?<=\.)\s+")


def perturb(text: str, view: str, rng: np.random.Generator) -> str | None:
    """A degraded view of a clause: its first half, its second half, or one sentence removed (None when the
    clause has fewer than 3 sentences)."""
    words = text.split()
    if view == "head":
        return " ".join(words[: len(words) // 2])
    if view == "tail":
        return " ".join(words[len(words) // 2 :])
    sentences = _SENTENCE_END.split(text.strip())
    if len(sentences) < 3:
        return None
    drop = int(rng.integers(len(sentences)))
    return " ".join(s for i, s in enumerate(sentences) if i != drop)


def perturbation_sources(
    texts: Sequence[str], seen: np.ndarray, min_words: int, n: int, seed: int
) -> np.ndarray:
    """Positions of up to `n` clauses with at least `min_words` words and no copy in the bank, in order."""
    long_enough = np.array([len(t.split()) >= min_words for t in texts])
    eligible = np.flatnonzero(long_enough & ~np.asarray(seen))
    return np.sort(np.random.default_rng(seed).permutation(eligible)[:n])


def perturbation_queries(texts: Sequence[str], sources: np.ndarray, seed: int) -> pd.DataFrame:
    """One row per (source, view) with the perturbed text; views that do not apply are skipped."""
    rng = np.random.default_rng(seed)
    rows = []
    for pos in sources:
        for view in VIEWS:
            q = perturb(texts[pos], view, rng)
            if q:
                rows.append({"source": int(pos), "view": view, "text": q})
    return pd.DataFrame(rows, columns=["source", "view", "text"])


@dataclass(frozen=True)
class ConstrainedQuery:
    question: str
    target: str
    pattern: str | None
    example: str


# The 100 LEDGAR clause types (checked against the dataset's label names by scripts/retrieval.py).
LEDGAR_TYPES = (
    "Adjustments", "Agreements", "Amendments", "Anti-Corruption Laws", "Applicable Laws", "Approvals",
    "Arbitration", "Assignments", "Assigns", "Authority", "Authorizations", "Base Salary", "Benefits",
    "Binding Effects", "Books", "Brokers", "Capitalization", "Change In Control", "Closings",
    "Compliance With Laws", "Confidentiality", "Consent To Jurisdiction", "Consents", "Construction",
    "Cooperation", "Costs", "Counterparts", "Death", "Defined Terms", "Definitions", "Disability",
    "Disclosures", "Duties", "Effective Dates", "Effectiveness", "Employment", "Enforceability",
    "Enforcements", "Entire Agreements", "Erisa", "Existence", "Expenses", "Fees",
    "Financial Statements", "Forfeitures", "Further Assurances", "General", "Governing Laws",
    "Headings", "Indemnifications", "Indemnity", "Insurances", "Integration", "Intellectual Property",
    "Interests", "Interpretations", "Jurisdictions", "Liens", "Litigations", "Miscellaneous",
    "Modifications", "No Conflicts", "No Defaults", "No Waivers", "Non-Disparagement", "Notices",
    "Organizations", "Participations", "Payments", "Positions", "Powers", "Publicity", "Qualifications",
    "Records", "Releases", "Remedies", "Representations", "Sales", "Sanctions", "Severability",
    "Solvency", "Specific Performance", "Submission To Jurisdiction", "Subsidiaries", "Successors",
    "Survival", "Tax Withholdings", "Taxes", "Terminations", "Terms", "Titles",
    "Transactions With Affiliates", "Use Of Proceeds", "Vacations", "Venues", "Vesting",
    "Waiver Of Jury Trials", "Waivers", "Warranties", "Withholdings",
)  # fmt: skip

# Lawyer questions with a lexical, regex-checkable constraint; `example` is a phrase the pattern must match.
CONSTRAINED_QUERIES = (
    ConstrainedQuery(
        "termination clauses with a 30-day notice period",
        "Terminations",
        r"thirty \(30\) days|30 days",
        "upon thirty (30) days' prior written notice",
    ),
    ConstrainedQuery(
        "governing law clause choosing New York law",
        "Governing Laws",
        r"New York",
        "governed by the laws of the State of New York",
    ),
    ConstrainedQuery(
        "governing law clause choosing Delaware law",
        "Governing Laws",
        r"Delaware",
        "laws of the State of Delaware",
    ),
    ConstrainedQuery(
        "assignment clause requiring the prior written consent of the other party",
        "Assignments",
        r"prior written consent",
        "without the prior written consent of the Company",
    ),
    ConstrainedQuery(
        "agreement may be executed in counterparts delivered by electronic signature or facsimile",
        "Counterparts",
        r"electronic|facsimile|pdf",
        "delivered by facsimile or electronic transmission",
    ),
    ConstrainedQuery(
        "indemnification clause covering reasonable attorneys' fees",
        "Indemnifications",
        r"attorneys'? fees",
        "including reasonable attorneys' fees",
    ),
    ConstrainedQuery(
        "confidentiality obligations surviving for two years after termination",
        "Confidentiality",
        r"two \(2\) years|2 years|twenty-four \(24\) months",
        "for a period of two (2) years following termination",
    ),
    ConstrainedQuery(
        "notices sent by email or electronic mail are effective",
        "Notices",
        r"e-?mail|electronic mail",
        "by electronic mail (e-mail)",
    ),
    ConstrainedQuery(
        "arbitration under the rules of the American Arbitration Association",
        "Arbitration",
        r"American Arbitration Association|\bAAA\b",
        "the rules of the American Arbitration Association",
    ),
    ConstrainedQuery(
        "waiver of the right to a jury trial",
        "Waiver Of Jury Trials",
        r"jury",
        "waives any right to a trial by jury",
    ),
    ConstrainedQuery("severability clause", "Severability", None, ""),
    ConstrainedQuery(
        "entire agreement clause superseding all prior agreements",
        "Entire Agreements",
        r"supersed",
        "supersedes all prior agreements and understandings",
    ),
    ConstrainedQuery(
        "payment clause charging interest on late payments",
        "Payments",
        r"interest",
        "shall bear interest at the rate",
    ),
    ConstrainedQuery("non-disparagement clause", "Non-Disparagement", None, ""),
    ConstrainedQuery(
        "insurance clause requiring a minimum coverage amount in dollars",
        "Insurances",
        r"\$\s?[0-9][0-9,]*",
        "not less than $1,000,000 per occurrence",
    ),
    ConstrainedQuery(
        "withholding of taxes from payments to the employee",
        "Tax Withholdings",
        r"withh?old",
        "withhold from any amounts payable all federal, state and local taxes",
    ),
    ConstrainedQuery(
        "change in control triggering accelerated vesting",
        "Change In Control",
        r"vest",
        "shall become fully vested upon a Change in Control",
    ),
    ConstrainedQuery(
        "submission to the exclusive jurisdiction of the courts of New York",
        "Submission To Jurisdiction",
        r"New York",
        "the exclusive jurisdiction of the courts of the State of New York",
    ),
)


def constraint_results(
    queries: Sequence[ConstrainedQuery],
    positions: np.ndarray,
    bank_texts: Sequence[str],
    bank_labels: np.ndarray,
    label_names: Sequence[str],
) -> pd.DataFrame:
    """Per question: share of the top-k with the target type, share of those matching the constraint, the base
    rate of the constraint among the bank clauses of that type, and the lift (precision / base rate)."""
    bank_labels = np.asarray(bank_labels)
    rows = []
    for q, pos in zip(queries, positions, strict=True):
        target = list(label_names).index(q.target)
        of_type = pos[bank_labels[pos] == target]
        row: dict[str, Any] = {
            "question": q.question,
            "target": q.target,
            "pattern": q.pattern,
            "type_p_at_k": float(np.mean(bank_labels[pos] == target)),
            "constraint_precision": np.nan,
            "base_rate": np.nan,
            "lift": np.nan,
        }
        if q.pattern is not None:
            match = re.compile(q.pattern, flags=re.IGNORECASE)
            in_type = np.flatnonzero(bank_labels == target)
            row["base_rate"] = (
                float(np.mean([bool(match.search(bank_texts[i])) for i in in_type]))
                if len(in_type)
                else np.nan
            )
            if len(of_type):
                row["constraint_precision"] = float(
                    np.mean([bool(match.search(bank_texts[i])) for i in of_type])
                )
                row["lift"] = row["constraint_precision"] / row["base_rate"] if row["base_rate"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --- geometry of the bank embeddings ---


def _pair_cos(E: Matrix, i: np.ndarray, j: np.ndarray) -> np.ndarray:
    if sp.issparse(E):
        return np.asarray(E[i].multiply(E[j]).sum(axis=1)).ravel()
    E = np.asarray(E)
    return np.einsum("ij,ij->i", E[i], E[j])


def _random_pairs(n: int, n_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n - 1, n_pairs)
    return i, j + (j >= i)  # j != i


def anisotropy(E: Matrix, n_pairs: int = 100_000, seed: int = 0) -> float:
    """Mean cosine between random pairs of bank rows: 0 for an isotropic cloud, close to 1 in a narrow cone
    (Ethayarajh 2019)."""
    i, j = _random_pairs(E.shape[0], n_pairs, np.random.default_rng(seed))
    return float(_pair_cos(E, i, j).mean())


def effective_rank(E: Matrix) -> float:
    """exp(entropy of the normalised singular values) (Roy & Vetterli 2007); NaN for a sparse matrix."""
    if sp.issparse(E):
        return float("nan")
    E = np.asarray(E, dtype=np.float64)
    s = np.sqrt(np.clip(np.linalg.eigvalsh(E.T @ E), 0, None))
    p = s / s.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def hubness(index: Index, k: int = 10, block: int = 2048) -> tuple[float, np.ndarray]:
    """Skewness of N_k, the number of top-k lists (bank vs bank, self excluded) each clause appears in
    (Radovanovic et al. 2010), and N_k itself."""
    from scipy.stats import skew

    _, pos = index.search(index.E, k, exclude_self=True, block=block)
    n_k = np.bincount(pos.ravel(), minlength=len(index))
    return float(skew(n_k)), n_k


def alignment(E: Matrix, labels: np.ndarray, n_pairs: int = 10_000, seed: int = 0) -> float:
    """Mean cosine between random pairs of clauses of the same type (Wang & Isola 2020, higher is tighter)."""
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    groups = {y: np.flatnonzero(labels == y) for y in np.unique(labels)}
    groups = {y: g for y, g in groups.items() if len(g) > 1}
    anchors = np.concatenate(list(groups.values()))
    i = rng.choice(anchors, n_pairs)
    j = np.array([rng.choice(groups[labels[a]][groups[labels[a]] != a]) for a in i])
    return float(_pair_cos(E, i, j).mean())


def uniformity(E: Matrix, n_pairs: int = 100_000, seed: int = 0) -> float:
    """log E[exp(-2 ||u - v||^2)] over random pairs (Wang & Isola 2020; lower = more uniform)."""
    i, j = _random_pairs(E.shape[0], n_pairs, np.random.default_rng(seed))
    sq_dist = 2 - 2 * _pair_cos(E, i, j)
    return float(np.log(np.mean(np.exp(-2 * sq_dist))))


def geometry(
    index: Index, texts: Sequence[str], k: int = 10, seed: int = 0
) -> tuple[dict[str, float], pd.DataFrame]:
    """The diagnostics of the bank embeddings, and the 10 largest hubs with their type and a text snippet."""
    E = index.E if index.sparse else index.E.cpu().numpy()
    skew, n_k = hubness(index, k)
    top = np.argsort(-n_k, kind="stable")[:10]
    hubs = pd.DataFrame(
        {
            "position": top,
            "index": index.index[top],
            "label": index.labels[top],
            "n_k": n_k[top],
            "text": [texts[i][:120] for i in top],
        }
    )
    stats = {
        "anisotropy": anisotropy(E, seed=seed),
        "effective_rank": effective_rank(E),
        "hubness_skew": skew,
        "hubness_max_share": float(n_k.max() / (k * len(index))),
        "alignment": alignment(E, index.labels, seed=seed),
        "uniformity": uniformity(E, seed=seed),
    }
    return stats, hubs


# --- the pre-registered rule and the results table ---

_COMPLEXITY = {"dense": 0, "soft": 1, "hard": 2}


def choose(rows: Sequence[dict[str, Any]], gate: float = 0.95) -> dict[str, Any]:
    """Candidates = (run, centered, variant) rows with their validation macro P@10 and perturbation MRR. Gate:
    MRR >= gate x the best MRR; choice: best P@10 among gated rows, ties to fewer components."""
    best_mrr = max(r["mrr"] for r in rows)
    threshold = gate * best_mrr
    gated = [r for r in rows if r["mrr"] >= threshold]
    ranked = sorted(
        gated, key=lambda r: (-r["p10"], _COMPLEXITY[r["variant"].split(":")[0]], bool(r["centered"]))
    )
    return {
        **ranked[0],
        "threshold": threshold,
        "n_gated": len(gated),
        "n_candidates": len(rows),
        "gate": gate,
    }


def retrieval_table(results_dir: str | Path) -> pd.DataFrame:
    """One row per results/retrieval/*/metrics.json, built only from those files."""
    rows = []
    for path in sorted(Path(results_dir).glob("*/metrics.json")):
        m = json.loads(path.read_text())
        clause, geo = m["val"]["clause"], m.get("geometry", {})
        rows.append(
            {
                "run": path.parent.name,
                "encoder": m["encoder"]["name"],
                "centered": m.get("centered", False),
                "p_at_1": clause["p_at_1_macro"],
                "p_at_5": clause["p_at_5_macro"],
                "p_at_10": clause["p_at_10_macro"],
                "mrr": clause["mrr"],
                "knn_macro_f1": clause["knn_macro_f1"],
                "perturb_mrr": m["val"]["perturbation"]["mrr"],
                "type_names_p_at_10": m["val"]["type_names"]["p_at_10"],
                "anisotropy": geo.get("anisotropy"),
                "effective_rank": geo.get("effective_rank"),
                "hubness_skew": geo.get("hubness_skew"),
                "encode_time_s": m.get("encode_time_s"),
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_macro(
    a: np.ndarray, b: np.ndarray, y_q: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Paired bootstrap of macro(a) - macro(b) over the same resampled queries (as `stats.paired_bootstrap`),
    for per-query scores such as P@10; 95 % percentile interval."""
    a, b, y_q = map(np.asarray, (a, b, y_q))
    observed = macro_by_type(a, y_q) - macro_by_type(b, y_q)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for r in range(n_boot):
        idx = rng.integers(0, len(y_q), len(y_q))
        diffs[r] = macro_by_type(a[idx], y_q[idx]) - macro_by_type(b[idx], y_q[idx])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": float(observed),
        "low": float(low),
        "high": float(high),
        "n_boot": n_boot,
        "significant": bool(not low <= 0 <= high),
    }

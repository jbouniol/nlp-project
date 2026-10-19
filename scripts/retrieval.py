"""m2: index the clause bank with one encoder and evaluate it on validation; the rule; the test, once.

python scripts/retrieval.py --encoder minilm --classifier-run lora_lr5e-4_seed0  # results/retrieval/minilm
python scripts/retrieval.py --encoder classifier:lora_lr5e-4_seed0 --classifier-run lora_lr5e-4_seed0
python scripts/retrieval.py --choose                       # the rule -> results/retrieval/choice.json
python scripts/retrieval.py --final                        # the test split, read once
FAST_RUN=1 python scripts/retrieval.py --encoder minilm    # .fast/, TF-IDF probabilities
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ledgar import data, metrics, retrieval, runs
from ledgar.config import Config, set_seed
from ledgar.predict import run_config
from ledgar.retrieval import Index

logger = logging.getLogger("retrieval")
KS = (1, 5, 10)
GATE = 0.95


def row_cosine(A: Any, B: Any) -> np.ndarray:
    if sp.issparse(A):
        return np.asarray(A.multiply(B).sum(axis=1)).ravel()
    return np.einsum("ij,ij->i", np.asarray(A), np.asarray(B))


def variant_kwargs(variant: str) -> dict[str, Any]:
    """'dense' -> {}, 'soft:<lambda>' -> {'lam': ...}, 'hard' -> {'hard': True}."""
    if variant == "dense":
        return {}
    if variant == "hard":
        return {"hard": True}
    return {"lam": float(variant.split(":")[1])}


def clause_metrics(index: Index, Q: Any, y_q: np.ndarray, cfg: Config, **hybrid: Any) -> tuple:
    """Clause -> clause: P@k micro and macro over types, MRR, the k-NN vote scored as a classifier."""
    scores, pos = index.search(Q, cfg.retrieval_k, **hybrid)
    hits = index.labels[pos] == y_q[:, None]
    p = retrieval.precision_at_k(hits, KS)
    rr = retrieval.reciprocal_rank(hits)
    vote = retrieval.knn_vote(index.labels[pos])
    out: dict[str, Any] = {"n_queries": int(len(y_q))}
    for k in KS:
        out[f"p_at_{k}"] = float(p[k].mean())
        out[f"p_at_{k}_macro"] = retrieval.macro_by_type(p[k], y_q)
    out["mrr"] = float(rr.mean())
    out["mrr_macro"] = retrieval.macro_by_type(rr, y_q)
    out["knn_accuracy"] = metrics.accuracy(y_q, vote)
    out["knn_macro_f1"] = metrics.macro_f1_score(y_q, vote, cfg.n_classes)
    return out, scores, pos, vote, p[10]


def by_group(p10: np.ndarray, y_q: np.ndarray, groups: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for name, g in groups.items():
        for value in pd.unique(g):
            mask = g == value
            out[f"{name}={value}"] = {
                "n": int(mask.sum()),
                "p_at_10": float(p10[mask].mean()),
                "p_at_10_macro": retrieval.macro_by_type(p10[mask], y_q[mask]),
            }
    return out


def perturbation_metrics(
    index: Index, Qp: Any, source_scores: np.ndarray, frame: pd.DataFrame, y_src: np.ndarray, **hybrid: Any
) -> dict[str, Any]:
    ranks = index.rank_of_source(Qp, source_scores, source_labels=y_src, **hybrid)
    rr = 1.0 / ranks
    out: dict[str, Any] = {
        "n": int(len(ranks)),
        "mrr": float(rr.mean()),
        "hit_at_1": float((ranks == 1).mean()),
    }
    out["views"] = {}
    for view in retrieval.VIEWS:
        m = (frame["view"] == view).to_numpy()
        if m.any():
            out["views"][view] = {
                "n": int(m.sum()),
                "mrr": float(rr[m].mean()),
                "hit_at_1": float((ranks[m] == 1).mean()),
            }
    return out


def load_bank(cfg: Config):
    ledgar = data.load_ledgar(cfg)
    names = data.label_names(ledgar)
    if names != list(retrieval.LEDGAR_TYPES):
        raise SystemExit("retrieval.LEDGAR_TYPES differs from the dataset's label names")
    return ledgar, names


def split_arrays(ledgar, split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    return ledgar[split]["text"], np.array(ledgar[split]["label"]), np.array(ledgar[split]["index"])


def run_encoder(cfg: Config, name: str) -> None:
    spec = retrieval.encoder_spec(name)
    ledgar, names = load_bank(cfg)
    bank_texts, y_bank, i_bank = split_arrays(ledgar, "train")
    val_texts, y_val, i_val = split_arrays(ledgar, "validation")
    for suffix in ("", "+center"):
        path = cfg.path(cfg.out_dir, "retrieval", name + suffix, "metrics.json")
        if path.exists() and "test" in json.loads(path.read_text()):
            raise SystemExit(f"{name}{suffix} already has test metrics: a final run is not re-evaluated")
    set_seed(cfg.seed)
    encoder = retrieval.load_encoder(spec, cfg, bank_texts)
    start = time.perf_counter()
    E_bank = retrieval.cached_embeddings(encoder, bank_texts, "train", "doc", cfg)
    E_val = retrieval.cached_embeddings(encoder, val_texts, "validation", "clause", cfg)
    encode_time = time.perf_counter() - start  # about 0 on a cache hit

    train_counts = np.bincount(y_bank, minlength=cfg.n_classes)
    seen = data.seen_in_train(ledgar, "validation")
    truncated_val = encoder.truncated(val_texts)
    groups = {
        "band": data.frequency_band(train_counts)[y_val],
        "seen_in_train": seen,
        "truncated": truncated_val,
    }
    share_truncated_bank = float(encoder.truncated(bank_texts).mean())

    sources = retrieval.perturbation_sources(val_texts, seen, cfg.min_perturb_words, cfg.n_perturb, cfg.seed)
    frame = retrieval.perturbation_queries(val_texts, sources, cfg.seed)
    Qp = encoder.encode(frame["text"].tolist(), "clause")
    S = encoder.encode([val_texts[s] for s in sources], "doc")
    src_rows = pd.Index(sources).get_indexer(frame["source"])
    y_src = y_val[frame["source"].to_numpy()]
    Q_names = encoder.encode(names, "nl")
    Q_cq = encoder.encode([q.question for q in retrieval.CONSTRAINED_QUERIES], "nl")
    probs, probs_source = retrieval.classifier_probs(
        cfg, bank_texts, y_bank, list(val_texts) + frame["text"].tolist()
    )
    probs_val, probs_p = probs[: len(val_texts)], probs[len(val_texts) :]
    variants = [f"soft:{lam:g}" for lam in cfg.hybrid_lambdas] + ["hard"]

    for centered in (False,) if sp.issparse(E_bank) else (False, True):
        mats = (E_bank, E_val, Qp, S, Q_names, Q_cq)
        if centered:
            mean = E_bank.mean(axis=0)
            mats = tuple(retrieval.center(M, mean) for M in mats)
        Eb, Ev, Qp_, S_, Qn_, Qc_ = mats
        index = Index(Eb, y_bank, i_bank, device=cfg.device)
        src_scores = row_cosine(Qp_, S_[src_rows])
        result: dict[str, Any] = {
            "model": "retrieval",
            "encoder": {
                **dataclasses.asdict(spec),
                "max_seq_length": encoder.max_seq_length,
                "share_truncated_bank": share_truncated_bank,
                "share_truncated_val": float(truncated_val.mean()),
            },
            "centered": centered,
            "probs_source": probs_source,
            "encode_time_s": round(encode_time, 1),
            "val": {},
        }
        clause, scores, pos, vote, p10 = clause_metrics(index, Ev, y_val, cfg)
        result["val"]["clause"] = clause
        result["val"]["by_group"] = by_group(p10, y_val, groups)
        result["val"]["perturbation"] = perturbation_metrics(index, Qp_, src_scores, frame, y_src)
        _, pos_n = index.search(Qn_, cfg.retrieval_k)
        hits_n = index.labels[pos_n] == np.arange(len(names))[:, None]
        result["val"]["type_names"] = {
            "n": len(names),
            "p_at_10": float(hits_n.mean()),
            "mrr": float(retrieval.reciprocal_rank(hits_n).mean()),
        }
        _, pos_c = index.search(Qc_, cfg.retrieval_k)
        cq = retrieval.constraint_results(retrieval.CONSTRAINED_QUERIES, pos_c, bank_texts, y_bank, names)
        result["val"]["constrained"] = json.loads(cq.to_json(orient="records"))
        result["val"]["constrained_summary"] = {
            "n": len(cq),
            "type_p_at_10": float(cq["type_p_at_k"].mean()),
            "constraint_precision": float(cq["constraint_precision"].mean()),
            "lift": float(cq["lift"].mean()),
        }
        result["val"]["hybrid"] = {}
        for variant in variants:
            kw = variant_kwargs(variant)
            c, *_ = clause_metrics(index, Ev, y_val, cfg, type_probs=probs_val, **kw)
            pm = perturbation_metrics(index, Qp_, src_scores, frame, y_src, type_probs=probs_p, **kw)
            result["val"]["hybrid"][variant] = {"clause": c, "perturbation": pm}
        result["geometry"], hubs = retrieval.geometry(index, bank_texts, cfg.retrieval_k, cfg.seed)

        run_dir = cfg.path(cfg.out_dir, "retrieval", name + ("+center" if centered else ""))
        (run_dir / "metrics.json").unlink(missing_ok=True)  # a fresh validation run, nothing merged
        runs.save_run(run_dir, cfg, result, log_history=[], duration_s=encode_time)
        runs.save_predictions(run_dir, "val", y_val, vote, index=i_val, label_names=names)
        np.savez_compressed(
            run_dir / "topk_val.npz",
            index=i_val,
            positions=pos.astype(np.int32),
            bank_index=i_bank[pos].astype(np.int32),
            scores=scores.astype(np.float16),
        )
        hubs.to_csv(run_dir / "hubs.csv", index=False)
        logger.info(
            "%s%s: P@10 macro %.4f | perturbation MRR %.4f | kNN macro-F1 %.4f | anisotropy %.3f -> %s",
            name,
            "+center" if centered else "",
            clause["p_at_10_macro"],
            result["val"]["perturbation"]["mrr"],
            clause["knn_macro_f1"],
            result["geometry"]["anisotropy"],
            run_dir,
        )


def candidate_rows(root) -> tuple[list[dict[str, Any]], set[str], bool]:
    rows, sources, has_classifier = [], set(), False
    for path in sorted(root.glob("*/metrics.json")):
        m = json.loads(path.read_text())
        sources.add(m["probs_source"])
        has_classifier |= m["encoder"]["name"].startswith("classifier:")
        v = m["val"]
        rows.append(
            {
                "run": path.parent.name,
                "centered": m["centered"],
                "variant": "dense",
                "p10": v["clause"]["p_at_10_macro"],
                "mrr": v["perturbation"]["mrr"],
            }
        )
        for variant, h in v["hybrid"].items():
            rows.append(
                {
                    "run": path.parent.name,
                    "centered": m["centered"],
                    "variant": variant,
                    "p10": h["clause"]["p_at_10_macro"],
                    "mrr": h["perturbation"]["mrr"],
                }
            )
    return rows, sources, has_classifier


def choose(cfg: Config) -> None:
    """The pre-registered rule (README, DECISIONS.md 2026-10-19) over every run under results/retrieval/."""
    root = cfg.path(cfg.out_dir, "retrieval")
    out = root / "choice.json"
    if out.exists():
        raise SystemExit(f"{out} exists: the choice is made once")
    rows, sources, has_classifier = candidate_rows(root)
    if not rows:
        raise SystemExit(f"no run under {root}")
    if not has_classifier and not cfg.fast_run:
        raise SystemExit("--choose waits for a classifier:<run> encoder: the candidate set is pre-registered")
    if len(sources) > 1:
        raise SystemExit(
            f"mixed probability sources {sorted(sources)}: re-run the encoders with one --classifier-run"
        )
    choice = retrieval.choose(rows, GATE)
    choice.update(
        {"probs_source": sources.pop(), "date_utc": datetime.now(UTC).isoformat(timespec="seconds")}
    )
    choice["rows"] = rows
    out.write_text(json.dumps(choice, indent=2) + "\n")
    logger.info(
        "chosen: %s%s / %s (P@10 macro %.4f, perturbation MRR %.4f, gate %.4f, %d of %d kept) -> %s",
        choice["run"],
        "" if not choice["centered"] else " (centered)",
        choice["variant"],
        choice["p10"],
        choice["mrr"],
        choice["threshold"],
        choice["n_gated"],
        choice["n_candidates"],
        out,
    )


def final(cfg: Config) -> None:
    """The chosen configuration and the dense variant of the same run on the test split, read once."""
    root = cfg.path(cfg.out_dir, "retrieval")
    choice_path = root / "choice.json"
    if not choice_path.exists():
        raise SystemExit("run --choose first")
    choice = json.loads(choice_path.read_text())
    run_dir = root / choice["run"]
    m = runs.load_run(run_dir)["metrics"]
    if "test" in m:
        raise SystemExit(f"{run_dir} already has test metrics: the test split is predicted once")
    variant, centered, source = choice["variant"], m["centered"], m["probs_source"]
    run_cfg = run_config(run_dir)
    cfg = dataclasses.replace(
        run_cfg, device=cfg.device, classifier_run=None if source == "tfidf refit" else source
    )
    spec = retrieval.encoder_spec(m["encoder"]["name"])
    ledgar, names = load_bank(cfg)
    bank_texts, y_bank, i_bank = split_arrays(ledgar, "train")
    set_seed(cfg.seed)
    encoder = retrieval.load_encoder(spec, cfg, bank_texts)
    E_bank = retrieval.cached_embeddings(encoder, bank_texts, "train", "doc", cfg)
    test_texts, y_test, i_test = split_arrays(ledgar, "test")  # predict & save: the test split, used once
    E_test = encoder.encode(test_texts, "clause")
    if centered:
        mean = E_bank.mean(axis=0)
        E_bank, E_test = retrieval.center(E_bank, mean), retrieval.center(E_test, mean)
    index = Index(E_bank, y_bank, i_bank, device=cfg.device)
    train_counts = np.bincount(y_bank, minlength=cfg.n_classes)
    groups = {
        "band": data.frequency_band(train_counts)[y_test],
        "seen_in_train": data.seen_in_train(ledgar, "test"),
        "truncated": encoder.truncated(test_texts),
    }
    stamp = runs.make_manifest(run_cfg)  # the one test read, dated and tied to a commit
    test: dict[str, Any] = {
        "chosen_variant": variant,
        "read_at_utc": stamp["date_utc"],
        "git_sha": stamp["git_sha"],
    }
    clause, scores, pos, vote, p10 = clause_metrics(index, E_test, y_test, cfg)
    test["dense"] = {"clause": clause, "by_group": by_group(p10, y_test, groups)}
    if variant != "dense":
        probs, _ = retrieval.classifier_probs(cfg, bank_texts, y_bank, test_texts)
        c, scores, pos, vote, p10_v = clause_metrics(
            index, E_test, y_test, cfg, type_probs=probs, **variant_kwargs(variant)
        )
        test[variant] = {"clause": c, "by_group": by_group(p10_v, y_test, groups)}
        test["comparison"] = retrieval.paired_bootstrap_macro(p10_v, p10, y_test, cfg.n_boot, cfg.seed)
    runs.save_run(run_dir, run_cfg, {"test": test})
    runs.save_predictions(run_dir, "test", y_test, vote, index=i_test, label_names=names)
    np.savez_compressed(
        run_dir / "topk_test.npz",
        index=i_test,
        positions=pos.astype(np.int32),
        bank_index=i_bank[pos].astype(np.int32),
        scores=scores.astype(np.float16),
    )
    chosen = test[variant]["clause"]
    logger.info(
        "test: %s P@10 macro %.4f | dense %.4f -> %s",
        variant,
        chosen["p_at_10_macro"],
        clause["p_at_10_macro"],
        run_dir,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--encoder", help="tfidf, minilm, e5, bge, distilbert, legalbert or classifier:<run>")
    mode.add_argument("--choose", action="store_true", help="apply the pre-registered rule")
    mode.add_argument("--final", action="store_true", help="the chosen configuration on the test split, once")
    cfg, args = Config.from_args(argv, parser)
    if cfg.retrieval_k < max(KS):
        raise SystemExit(f"--retrieval-k must be at least {max(KS)}: P@{max(KS)} is reported")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    if args.encoder:
        run_encoder(cfg, args.encoder)
    elif args.choose:
        choose(cfg)
    else:
        final(cfg)


if __name__ == "__main__":
    main()

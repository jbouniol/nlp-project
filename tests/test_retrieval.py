"""m2: the retriever, its metrics and its diagnostics on toy data (no download)."""

import argparse

from ledgar.config import Config


def test_config_from_args_accepts_a_script_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default=None)
    parser.add_argument("--choose", action="store_true")
    argv = ["--encoder", "minilm", "--classifier-run", "lora_s0", "--n-perturb", "7", "--device", "cpu"]
    cfg, args = Config.from_args(argv, parser=parser)
    assert args.encoder == "minilm" and not args.choose
    assert cfg.classifier_run == "lora_s0" and cfg.n_perturb == 7
    assert cfg.retrieval_k == 10 and cfg.hybrid_lambdas == (0.1, 0.3, 1.0, 3.0)


def test_fast_run_moves_the_embedding_cache(monkeypatch):
    monkeypatch.setenv("FAST_RUN", "1")
    cfg = Config.from_env(device="cpu")
    assert cfg.embeddings_dir == ".fast/embeddings" and cfg.n_perturb == 50

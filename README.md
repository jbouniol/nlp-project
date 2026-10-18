# LEDGAR clause classifier

The classification part of a contract-review assistant: given one clause of a contract, predict its
type among the 100 clause types of [LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue)
(LexGLUE). Four models, from a cheap baseline to a fine-tuned transformer, evaluated with macro-F1
on the official held-out test set and compared with paired tests.

| Milestone | Model | Notebook |
|---|---|---|
| m1 | TF-IDF + logistic regression | `notebooks/02_baseline_tfidf.ipynb` |
| m3 | LSTM / BiLSTM (+ GloVe) | `notebooks/03_lstm_classifier.ipynb` |
| m4 | DistilBERT, full fine-tuning and LoRA | `colab/run_m4.ipynb`, `scripts/train_bert.py` |

> Work in progress: results, the pre-registered comparison are added milestone by milestone.

## Results

*Generated from `results/` by `make table`.*

| run | model | seed | trainable_params | train_time_s | val_macro_f1 | test_macro_f1 | test_accuracy |
|---|---|---|---|---|---|---|---|
| lazy | lazy | 0 | 0 | 0 | 0.0010 | 0.0011 | 0.0582 |
| lstm_seed0 | lstm | 0 | 1,344,684 | 2,055 | 0.7900 | 0.7799 | 0.8469 |
| lstm_seed1 | lstm | 1 | 1,344,684 | 1,721 | 0.7936 | 0.7859 | 0.8512 |
| lstm_seed2 | lstm | 2 | 1,344,684 | 2,097 | 0.7848 | 0.7749 | 0.8477 |
| tfidf | tfidf | 0 | 17,347,800 | 139 | 0.8265 | 0.8291 | 0.8771 |

## Pre-registered comparison

*Written and committed before any m4 model was trained (S06 p26: "the main comparison written in the README
before running it, then its paired test").*

**Question.** Full fine-tuning of DistilBERT trains 67 M parameters; LoRA trains 0.7 M (rank-8 adapters on the
query and value projections, plus the new head). Is the full model worth it?

**Hypothesis.** On the LEDGAR test set (10 000 clauses, seed 0), full fine-tuning of `distilbert-base-uncased`
has a higher macro-F1 than LoRA (r = 8, α = 16, `q_lin` and `v_lin`), by at most 1 point, and the difference
is not significant.

**Setting.** Same data, `max_length` 512 with length-grouped batches, 3 epochs, warmup 10 % then linear decay,
the epoch with the best validation macro-F1 kept; each method's learning rate chosen on validation
(full: 2e-5, 3e-5, 5e-5; LoRA: 3e-4, 5e-4, 1e-3). The test set is predicted once per model.

**Tests.** McNemar's test on the per-clause correctness (continuity correction; exact binomial test if
b + c < 25), and a paired bootstrap of the macro-F1 difference (1 000 resamples, seed 0, 95 % percentile
interval), at α = 5 %.

**Decision rule.** If the bootstrap interval contains 0 and McNemar's p ≥ 0.05, LoRA is the production
candidate; otherwise full fine-tuning is.

Secondary comparisons, same tests: the best transformer vs the TF-IDF baseline (is a transformer worth its
cost?), the transformer vs the LSTM, and LoRA on every linear layer vs LoRA on `q_lin`, `v_lin`.

## How to reproduce

```bash
make setup      # .venv with the pinned course environment, package installed in editable mode
make test       # unit tests
make smoke      # the whole pipeline on a per-class subset (FAST_RUN=1), then every notebook
```

GPU runs (m4) on Colab: open [`colab/run_m4.ipynb`](https://colab.research.google.com/github/jbouniol/nlp-project/blob/main/colab/run_m4.ipynb),
the run cards of every DistilBERT and LoRA run, in order.
The 10-epoch follow-up has its own runner,
[`colab/run_m4_10epochs.ipynb`](https://colab.research.google.com/github/jbouniol/nlp-project/blob/main/colab/run_m4_10epochs.ipynb). From the command line:

```bash
python scripts/train.py --model tfidf                                  # m1 (= make m1)
python scripts/search_lstm.py                                          # m3: one factor at a time, on validation
python scripts/train.py --model lora --search --lr-lora 5e-4 --seed 0  # m4: a validation-only run
python scripts/predict_test.py --name lora_lr5e-4_seed0                # the chosen run predicts the test split, once
```

## Repository map

```text
src/ledgar/     package: config, data, metrics, models, training, run bookkeeping
notebooks/      one notebook per milestone, readable top to bottom (outputs committed)
scripts/        the same pipeline without a notebook (train.py, search_lstm.py, predict_test.py, ...)
colab/          run cards for the GPU runs
results/        one directory per run: config, metrics, predictions, log, manifest
reports/        figures, results table
docs/           DECISIONS.md, the log of design choices
tests/          unit tests and the FAST_RUN smoke run
```

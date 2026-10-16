# LEDGAR clause classifier

The classification part of a contract-review assistant: given one clause of a contract, predict its
type among the 100 clause types of [LEDGAR](https://huggingface.co/datasets/coastalcph/lex_glue)
(LexGLUE). Four models, from a cheap baseline to a fine-tuned transformer, evaluated with macro-F1
on the official held-out test set and compared with paired tests.

| Milestone | Model | Notebook |
|---|---|---|
| m1 | TF-IDF + logistic regression | `notebooks/02_baseline_tfidf.ipynb` |
| m3 | LSTM / BiLSTM (+ GloVe) | `notebooks/03_lstm_classifier.ipynb` |
| m4 | DistilBERT, full fine-tuning and LoRA | `notebooks/04_finetune_distilbert.ipynb`, `notebooks/05_lora.ipynb` |

> Work in progress: results, the pre-registered comparison and the model card are added milestone by milestone.

## Results

*Generated from `results/` by `make table`.*

## How to reproduce

```bash
make setup      # .venv with the pinned course environment, package installed in editable mode
make test       # unit tests
make smoke      # the whole pipeline on a per-class subset (FAST_RUN=1), then every notebook
```

On Colab: `!pip install -r requirements.txt && pip install -e .`, then open a notebook from `notebooks/`.

## Repository map

```text
src/ledgar/     package: config, data, metrics, models, training, run bookkeeping
notebooks/      one notebook per milestone, readable top to bottom (outputs committed)
scripts/        the same pipeline without a notebook (train.py, evaluate.py, make_results_table.py)
results/        one directory per run: config, metrics, predictions, log, manifest
reports/        figures, results table, model card
docs/           DECISIONS.md, the log of design choices
tests/          unit tests and the FAST_RUN smoke run
```

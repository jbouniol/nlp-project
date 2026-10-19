# Model card — LEDGAR clause-type classifier

*All figures come from `notebooks/06_evaluation.ipynb` and the run folders under `results/`. Test scores were
read once, after every choice had been made on validation.*

## The model

| | |
|---|---|
| What it does | Reads one clause of a contract (English) and returns one of 100 clause types, with a probability per type |
| Model | `distilbert-base-uncased` (66 M parameters), every parameter fine-tuned, plus a new 768 → 768 → 100 head |
| Training | LEDGAR train split (60 000 clauses), 10 epochs, AdamW lr 5e-5 with 10 % warmup then linear decay, batch 32, clauses cut at 512 WordPiece tokens, best epoch kept on validation macro-F1 |
| Reference run | `results/bert_lr5e-5_ep10_seed0` (seeds 1 and 2 alongside); weights (260 MB) kept outside git, on the team Drive |
| Lighter alternative | LoRA (rank 8 on query and value, 815 k trained parameters): `results/lora_lr1e-3_ep10_seed0`, adapter of a few MB versioned in git; 1.2 points of macro-F1 below |
| Training cost | 11.6 minutes per run on one A100 (Colab); about 1.1 GB for weights, gradients and optimizer states |
| How to call it | `ledgar.predict.load_classifier(run_dir).predict(texts)` → type ids and probabilities |

## Intended use

A **first pass** in contract review: pre-tag every clause of an English commercial contract with its type, so
that a lawyer can jump to the clauses that matter (termination, indemnification, governing law…), filter the
firm's clause bank by type, or spot an expected clause that is missing. A person reviews every tag that drives a
decision.

## Evaluation (held-out test split, 10 000 clauses)

| | macro-F1 | accuracy |
|---|---|---|
| This model (mean ± std over 3 seeds) | 0.827 ± 0.004 | 0.880 |
| TF-IDF + logistic regression baseline | 0.829 | 0.877 |
| LoRA alternative (3 seeds) | 0.814 ± 0.001 | 0.875 |
| BiLSTM trained from scratch (3 seeds) | 0.780 ± 0.006 | 0.849 |

Paired tests on the same clauses (seed 0): **no significant difference with the TF-IDF baseline** (macro-F1
difference −0.003, 95 % bootstrap interval [−0.013, +0.007], McNemar p = 0.60); better than LoRA (+1.1 points,
interval [+0.002, +0.020]) and than the BiLSTM (+4.6 points, p < 0.001).

Where it is reliable and where it is not (seed 0, test):

| slice | clauses | macro-F1 |
|---|---|---|
| Frequent types (≥ 1 000 training clauses, 16 types) | 4 166 | 0.96 |
| Middle types | 5 183 | 0.87 |
| Rare types (< 300 training clauses, 22 types) | 651 | 0.76 |
| Clauses copied word for word from a training clause | 282 | 0.99 |
| All other clauses | 9 718 | 0.82 |
| Clauses longer than 512 tokens (their end is cut) | 175 | 0.87 |

Types under 0.5 F1: *Applicable Laws* (0.23), *Assigns* (0.40, only 4 test clauses), *Venues* (0.47). The most
confused pairs are near-synonyms: *Applicable* / *Governing Laws*, *Tax Withholdings* / *Withholdings*,
*Defined Terms* / *Definitions*, *No Waivers* / *Waivers*, *Amendments* / *Modifications*.

## Must not be used for

- **Deciding without a lawyer.** It labels; it does not interpret obligations, check validity or assess risk.
- **Contracts unlike its training data.** It learned from English contracts filed with the US SEC. It has not
  been evaluated on contracts in other languages (a French contract would be cut into fragments of its English
  vocabulary), on other legal systems, on consumer terms, or on scanned text with OCR errors.
- **Using its probability to skip review.** The model grows over-confident with training (its validation loss
  rises after epoch 4 while its accuracy still improves), and its errors are often made with probabilities above
  0.9. A high probability is not a guarantee.
- **Telling apart near-synonym types on its own.** *Applicable* / *Governing Laws*, the two withholding types, the
  definition types and the waiver types are where most errors are; for some of these pairs the labels themselves
  disagree (identical clauses are labelled both *Waivers* and *No Waivers*, or both *Applicable* and *Governing
  Laws*). Show the group, or ask a reviewer.
- **Clauses that carry several types.** It returns exactly one type per clause.
- **Rare types without checking.** On the 22 types with fewer than 300 training clauses, macro-F1 falls to 0.76.

## Data

LEDGAR from LexGLUE (`coastalcph/lex_glue`, configuration `ledgar`, revision `c23fdff1`): 100 types, official
splits of 60 000 / 10 000 / 10 000 clauses, very imbalanced (138 times more *Governing Laws* than *Books* in
training). *Books* has no validation clause; 282 test clauses (2.8 %) also appear in the training set; 6 texts
appear with two different labels. The dataset has no metadata on contract type, industry or parties, so
fairness across them cannot be measured from the labels alone: the team's bias audit derives these groups from
the text and uses `results/<run>/pred_test.csv`, whose `index` column points back to each clause.

## Reproduce

```bash
make setup && make test && make smoke
```

The GPU runs are in `colab/run_m4.ipynb` (registered 3-epoch runs) and `colab/run_m4_10epochs.ipynb` (this
model); every choice and the figure behind it is in `docs/DECISIONS.md`.

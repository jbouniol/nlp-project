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

Bias audit across inferred contract kinds, party roles, named jurisdictions and formatting (`data.bias_slices`,
`reports/bias_audit.md`, `reports/figures/bias_audit_slices.png`):

| dimension | slice | test clauses | train share | accuracy | DistilBERT macro-F1 | LoRA macro-F1 | TF-IDF macro-F1 | retrieval P@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Contract kind | `credit / loan` | 1 855 | 21.0 % | 0.894 | 0.818 | 0.847 | 0.826 | 0.794 |
| Contract kind | `M&A / purchase` | 637 | 8.8 % | 0.892 | 0.814 | 0.782 | 0.795 | 0.780 |
| Contract kind | `indenture / securities` | 452 | 5.6 % | 0.912 | 0.766 | 0.759 | 0.768 | 0.792 |
| Contract kind | `employment / equity` | 2 106 | 18.2 % | 0.861 | 0.732 | 0.741 | 0.763 | 0.684 |
| Contract kind | `lease / license` | 281 | 2.5 % | **0.758** | **0.657** | **0.620** | **0.649** | **0.576** |
| Contract kind | `general / unspecified` | 4 669 | 43.9 % | 0.883 | 0.786 | 0.766 | 0.782 | 0.763 |
| Party role | `borrower / lender` | 1 738 | 19.7 % | 0.901 | 0.837 | 0.855 | 0.834 | 0.800 |
| Party role | `holder / trustee / investor` | 368 | 4.3 % | 0.891 | 0.838 | 0.807 | 0.824 | 0.711 |
| Party role | `buyer / seller` | 511 | 7.1 % | 0.908 | 0.816 | 0.784 | 0.777 | 0.805 |
| Party role | `employee / participant` | 2 071 | 17.8 % | 0.857 | 0.713 | 0.723 | 0.751 | 0.682 |
| Party role | `landlord / tenant / licensor` | 188 | 1.6 % | **0.734** | **0.634** | **0.643** | **0.634** | **0.529** |
| Party role | `company / general` | 5 124 | 49.5 % | 0.882 | 0.790 | 0.774 | 0.794 | 0.764 |
| Jurisdiction | `New York` | 358 | 3.6 % | 0.899 | 0.801 | 0.706 | 0.835 | 0.812 |
| Jurisdiction | `California` | 65 | 0.9 % | 0.846 | **0.723** | 0.687 | **0.538** | 0.668 |
| Jurisdiction | `Delaware` | 188 | 1.8 % | 0.888 | 0.710 | 0.675 | 0.686 | 0.737 |
| Jurisdiction | `Texas / Nevada / Florida` | 119 | 1.4 % | 0.832 | 0.662 | 0.645 | 0.615 | 0.704 |
| Jurisdiction | `non-US / international` | 27 | 0.3 % | 0.963 | 0.943 | 0.867 | 0.872 | 0.741 |
| Jurisdiction | `none named` | 9 243 | 92.0 % | 0.878 | 0.819 | 0.812 | 0.821 | 0.748 |
| Formatting | `standard case` | 9 757 | 97.4 % | 0.878 | 0.826 | 0.813 | 0.827 | 0.747 |
| Formatting | `all-caps (>50% upper)` | 243 | 2.6 % | 0.885 | **0.517** | **0.585** | **0.519** | 0.814 |

Types under 0.5 F1: *Applicable Laws* (0.23), *Assigns* (0.40, only 4 test clauses), *Venues* (0.47). The most
confused pairs are near-synonyms: *Applicable* / *Governing Laws*, *Tax Withholdings* / *Withholdings*,
*Defined Terms* / *Definitions*, *No Waivers* / *Waivers*, *Amendments* / *Modifications*.

## Must not be used for

- **Deciding without a lawyer.** It labels; it does not interpret obligations, check validity or assess risk.
- **Contracts unlike its training data.** It learned from English contracts filed with the US SEC. It has not
  been evaluated on contracts in other languages (a French contract would be cut into fragments of its English
  vocabulary), on other legal systems, on consumer terms, or on scanned text with OCR errors.
- **Commercial leases, real-estate and IP licensing contracts without review.** Only 2.5 % of LEDGAR training
  clauses come from leases or licenses (1.6 % naming a *landlord*, *tenant* or *licensor*): accuracy drops from
  89–91 % on credit/loan, M&A and securities clauses to **75.8 %** on *lease / license* (macro-F1 **0.657**,
  retrieval P@10 **0.576**) and **73.4 %** on *landlord / tenant / licensor* (macro-F1 **0.634**, retrieval P@10
  **0.529**).
- **Conspicuous ALL-CAPS boilerplate outside standard waiver/warranty types.** While overall accuracy on ALL-CAPS
  clauses (> 50 % uppercase, 243 test clauses) is 88.5 %, macro-F1 across the 29 types that appear in ALL-CAPS
  drops to **0.517** because uncased WordPiece and lowercase TF-IDF erase casing and pull non-standard uppercase
  clauses toward *Waiver Of Jury Trials*, *Warranties* and *Governing Laws*.
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
appear with two different labels. Because the dataset provides no contract-level metadata on agreement family,
jurisdiction or party role, `ledgar.data.bias_slices` derives these four slices from the clause text and joins
them to `results/<run>/pred_test.csv` via `index` (`notebooks/06_evaluation.ipynb`, `reports/bias_audit.md`).

## Reproduce

```bash
make setup && make test && make smoke
```

The GPU runs are in `colab/run_m4.ipynb` (registered 3-epoch runs) and `colab/run_m4_10epochs.ipynb` (this
model); every choice and the figure behind it is in `docs/DECISIONS.md`.

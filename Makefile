PY ?= .venv/bin/python
SEED ?= 0
NB_OUT ?= /tmp/ledgar_nb
CODE := $(wildcard src tests scripts notebooks)  # only the folders that exist (git keeps no empty folder)

.PHONY: setup lint format test smoke notebooks table m1 m2 glove

setup:            ## create .venv and install the package with its pinned dependencies
	uv venv --python 3.14 .venv || python3 -m venv .venv
	uv pip install --python $(PY) -e ".[dev]" || $(PY) -m pip install -e ".[dev]"

lint:             ## ruff lint + format check (code and notebooks)
	$(PY) -m ruff check $(CODE)
	$(PY) -m ruff format --check $(CODE)

format:
	$(PY) -m ruff check --fix $(CODE)
	$(PY) -m ruff format $(CODE)

test:             ## unit tests (no download)
	$(PY) -m pytest -m "not slow"

smoke:            ## the whole pipeline on a per-class subset (FAST_RUN), then every notebook top to bottom
	test -f data/glove-wiki-gigaword-50.gz || $(MAKE) glove
	FAST_RUN=1 $(PY) -m pytest tests/test_smoke.py
	$(MAKE) notebooks

notebooks:        ## execute every notebook under FAST_RUN into $(NB_OUT) (committed outputs untouched)
	mkdir -p $(NB_OUT)
	for nb in notebooks/*.ipynb; do \
		[ -e "$$nb" ] || continue; \
		FAST_RUN=1 $(PY) -m jupyter nbconvert --to notebook --execute "$$nb" --output-dir $(NB_OUT) || exit 1; \
	done

m1:               ## TF-IDF + logistic regression with the configuration of Config -> results/tfidf
	$(PY) scripts/train.py --model tfidf

m2:               ## retrieval: every off-the-shelf encoder on validation (M2_ARGS="--classifier-run <run>")
	for e in tfidf minilm e5 bge distilbert legalbert; do \
		$(PY) scripts/retrieval.py --encoder $$e $(M2_ARGS) || exit 1; \
	done

glove:            ## GloVe 50-d (66 MB, 400 000 words) for the LSTM, from the gensim-data release on GitHub
	mkdir -p data
	curl -L --fail -o data/glove-wiki-gigaword-50.gz \
		https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-50/glove-wiki-gigaword-50.gz

table:            ## results/ -> reports/results_table.md
	$(PY) scripts/make_results_table.py

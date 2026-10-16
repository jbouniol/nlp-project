PY ?= .venv/bin/python
SEED ?= 0
NB_OUT ?= /tmp/ledgar_nb
CODE := $(wildcard src tests scripts notebooks)  # only the folders that exist (git keeps no empty folder)

.PHONY: setup lint format test smoke notebooks table

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
	FAST_RUN=1 $(PY) -m pytest tests/test_smoke.py
	$(MAKE) notebooks

notebooks:        ## execute every notebook under FAST_RUN into $(NB_OUT) (committed outputs untouched)
	mkdir -p $(NB_OUT)
	for nb in notebooks/*.ipynb; do \
		[ -e "$$nb" ] || continue; \
		FAST_RUN=1 $(PY) -m jupyter nbconvert --to notebook --execute "$$nb" --output-dir $(NB_OUT) || exit 1; \
	done

table:            ## results/ -> reports/results_table.md
	$(PY) scripts/make_results_table.py

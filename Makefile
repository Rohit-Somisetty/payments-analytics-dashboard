PYTHON ?= python
VENV ?= .venv
VENV_BIN := $(VENV)/bin
VENV_SCRIPTS := $(VENV)/Scripts
VENV_PY := $(VENV_BIN)/python
VENV_PY_WIN := $(VENV_SCRIPTS)/python

.PHONY: setup run clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PY) -m pip install --upgrade pip setuptools wheel || $(VENV_PY_WIN) -m pip install --upgrade pip setuptools wheel
	$(VENV_PY) -m pip install -r requirements.txt || $(VENV_PY_WIN) -m pip install -r requirements.txt

run:
	$(VENV_PY) etl/run_pipeline.py || $(VENV_PY_WIN) etl/run_pipeline.py

clean:
	rm -rf __pycache__ */__pycache__ *.duckdb data/processed/* data/marts/*

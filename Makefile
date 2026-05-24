.PHONY: install generate-data train-baseline train-pinn evaluate benchmark ablation test lint format

UV := uv
PYTHON := $(UV) run python
PYTEST := $(UV) run python -m pytest

install:
	$(UV) sync --all-extras

generate-data:
	$(PYTHON) -m data.generate_dataset

train-baseline:
	$(PYTHON) -m training.train_baseline

train-pinn:
	$(PYTHON) -m training.train_pinn

evaluate:
	$(PYTHON) -m evaluation.evaluate

benchmark:
	$(PYTHON) -m evaluation.benchmark

ablation:
	$(PYTHON) -m evaluation.ablation

test:
	$(PYTEST) tests/

lint:
	$(UV) run ruff check src tests
	$(UV) run black --check src tests
	$(UV) run mypy src

format:
	$(UV) run ruff check --fix src tests
	$(UV) run black src tests

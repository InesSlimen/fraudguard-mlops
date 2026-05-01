.PHONY: install test lint format clean tree

install:
	python -m venv .venv
	. .venv/bin/activate && python -m pip install -U pip
	. .venv/bin/activate && pip install -e .

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check .

format:
	. .venv/bin/activate && ruff format .
	. .venv/bin/activate && ruff check . --fix

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

tree:
	find . -maxdepth 4 -type d | sort

.PHONY: data-synthetic data-sample data-check
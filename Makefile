.PHONY: install test lint format clean tree

install:
	python3.11 -m venv .venv
	. .venv/bin/activate && python3.11 -m pip install -U pip
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

data-synthetic:
	. .venv/bin/activate && PYTHONPATH=. python3.11 scripts/create_synthetic_sample.py

data-sample:
	. .venv/bin/activate && PYTHONPATH=. python3.11 scripts/create_sample.py

data-check:
	. .venv/bin/activate && PYTHONPATH=. python3.11 scripts/check_data.py

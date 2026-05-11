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

.PHONY: data-validate

data-validate:
	. .venv/bin/activate && PYTHONPATH=. python3.11 scripts/validate_data.py

.PHONY: features-build features-check

features-build:
	. .venv/bin/activate && PYTHONPATH=. python3.11 products/fraudguard/features/build_features.py

features-check:
	. .venv/bin/activate && PYTHONPATH=. python3.11 -c "from products.fraudguard.features.build_features import write_feature_report; write_feature_report()"

.PHONY: train model-check

train:
	. .venv/bin/activate && PYTHONPATH=. python3.11 products/fraudguard/training/train.py

model-check:
	. .venv/bin/activate && PYTHONPATH=. python3.11 ci/quality-gates/check_metrics.py reports/model/metrics.json

.PHONY: serve smoke

serve:
	. .venv/bin/activate && PYTHONPATH=. uvicorn products.fraudguard.inference.app.main:app --host 0.0.0.0 --port 8000 --reload

smoke:
	bash scripts/smoke_test.sh
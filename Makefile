setup:
	pre-commit install

default: qa tests

qa:
	pre-commit run --all-files

tests:
	python -m pytest -vv --cov=. --cov-report=html

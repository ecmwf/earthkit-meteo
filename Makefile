# SPDX-FileCopyrightText: 2026 European Centre for Medium-Range Weather Forecasts (ECMWF)
# SPDX-License-Identifier: Apache-2.0

setup:
	pre-commit install

default: qa tests

qa:
	pre-commit run --all-files

tests:
	python -m pytest -vv --cov=. --cov-report=html

.PHONY: help install test lint format type-check clean all

# Default target
help:
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run all linting checks"
	@echo "  make format     - Auto-format code"
	@echo "  make type-check - Run mypy type checking"
	@echo "  make clean      - Remove cache files"
	@echo "  make all        - Run all checks (lint + type-check + test)"

install:
	python -m pip install -e ".[dev]"
	python -m pip install pre-commit
	pre-commit install

test:
	python -m pytest tests/ -v --cov=src/therapeutic_agent --cov-report=term-missing

lint:
	python -m isort --check-only src/ tests/
	python -m black --check src/ tests/
	python -m flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503

format:
	python -m isort src/ tests/
	python -m black src/ tests/

type-check:
	python -m mypy src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage

# Run all checks (what CI does)
all: lint type-check test
	@echo "All checks passed!"

# Quick check before commit
check: format lint type-check
	@echo "Ready to commit!"

.PHONY: lint format check install-dev clean help

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install-dev:  ## Install development dependencies
	pip install ruff

lint:  ## Run ruff linter (check only)
	ruff check .

lint-fix:  ## Run ruff linter with auto-fix
	ruff check --fix .

format:  ## Format code with ruff
	ruff format .

format-check:  ## Check code formatting without changes
	ruff format --check .

check: lint format-check  ## Run all checks (lint + format check)

fix: lint-fix format  ## Auto-fix linting issues and format code

clean:  ## Clean cache files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

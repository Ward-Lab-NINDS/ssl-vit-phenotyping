#!/usr/bin/env bash
set -euo pipefail

# Remove generated Python/build artifacts from Git's index while keeping local files.
# Run from the repository root after updating .gitignore.

git rm -r --cached --ignore-unmatch \
  __pycache__ \
  .pytest_cache \
  .ruff_cache \
  .mypy_cache \
  .venv \
  build \
  dist \
  *.egg-info \
  src/*.egg-info \
  src/**/__pycache__ \
  tests/**/__pycache__ \
  workflow/**/__pycache__

# Re-stage the intended source files after index cleanup.
git add .gitignore README.md docs src tests workflow .github data demo scripts pyproject.toml

echo "Index cleanup staged. Review with: git status"

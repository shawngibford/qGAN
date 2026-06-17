"""Pytest configuration for the Phase 13 regression suite.

The ``tests/`` package is greenfield (Phase 13 Wave 0). Tests import the
shared modules under ``revision.core`` directly. Pytest is normally launched
from the repo root via ``./qgan_env/bin/python -m pytest tests/ -q`` — this
conftest guarantees the repo root is on ``sys.path`` regardless of the
invocation cwd so ``import revision.core...`` always resolves.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Repo root = parent of this tests/ directory.
_REPO_ROOT = Path(__file__).resolve().parent.parent

# Prepend (not append) so the in-repo `revision` package always wins over any
# similarly named site-packages entry.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root (parent of ``tests/``)."""
    return _REPO_ROOT

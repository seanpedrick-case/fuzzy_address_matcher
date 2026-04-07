from __future__ import annotations

from pathlib import Path


def pytest_sessionstart(session) -> None:
    """
    Ensure folders that the matcher writes into exist in CI.

    The matching pipeline writes CSV outputs into `output/` by default. That directory
    exists locally for many developers but is not present on fresh CI runners.
    """
    repo_root = Path(__file__).resolve().parents[1]
    (repo_root / "output").mkdir(parents=True, exist_ok=True)

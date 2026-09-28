"""Test configuration: put the package on the path and gate the slow tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run the slow Monte-Carlo tests as well",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: slow Monte-Carlo test, run with --runslow")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow"):
        return
    skip = pytest.mark.skip(reason="slow Monte-Carlo test, run with --runslow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)

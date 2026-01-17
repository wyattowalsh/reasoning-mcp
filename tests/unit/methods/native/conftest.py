"""Shared fixtures for native reasoning method tests.

This module provides common fixtures used across all native method unit tests,
reducing code duplication and ensuring consistent test setup.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models import Session


@pytest.fixture
def session() -> Session:
    """Create a fresh session for testing.

    Returns:
        A started Session instance ready for method execution.
    """
    return Session().start()


@pytest.fixture
def fresh_session() -> Session:
    """Create a fresh unstarted session for testing.

    Returns:
        A new Session instance that hasn't been started.
    """
    return Session()

"""Pytest configuration for reliability tests."""

# This file prevents pytest from loading the parent conftest.py
# which has import issues not related to our circuit breaker tests

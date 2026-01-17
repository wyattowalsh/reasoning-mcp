"""End-to-end integration tests for verification."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.engine.executor import ExecutionContext
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.verification import VerificationStatus
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.verification.checkers import SelfConsistencyChecker
from reasoning_mcp.verification.engine import VerificationEngine
from reasoning_mcp.verification.extractors import RuleBasedExtractor


@pytest.fixture
def mock_execution_context() -> ExecutionContext:
    """Create a mock execution context for testing."""
    session = Session().start()
    registry = MethodRegistry()

    # Create execution context
    ctx = ExecutionContext(
        session=session,
        registry=registry,
        input_data={},
        variables={},
    )

    # Mock the FastMCP context for sampling capability
    mock_ctx = MagicMock()
    mock_ctx.sample = AsyncMock(return_value="AGREE")
    ctx.ctx = mock_ctx

    return ctx


@pytest.mark.integration
async def test_verification_e2e(mock_execution_context: ExecutionContext) -> None:
    """Test full verification flow from text to verification report."""
    # Sample text with claims
    text = """
    The population of France is approximately 67 million people.
    Paris is the capital city of France.
    The Eiffel Tower was built in 1889.
    France has a larger population than Germany.
    """

    # Create engine with extractors and checkers
    extractor = RuleBasedExtractor()
    checker = SelfConsistencyChecker()
    engine = VerificationEngine(extractor=extractor, checkers=[checker], ctx=mock_execution_context)

    # Run verification
    report = await engine.verify_text(text)

    # Verify report structure
    assert report is not None
    assert report.report_id is not None
    assert report.original_text == text

    # Verify claims were extracted
    assert len(report.claims) > 0

    # Verify results were generated
    assert len(report.results) > 0

    # Verify overall accuracy is in valid range
    assert 0.0 <= report.overall_accuracy <= 1.0

    # Verify each result has valid status
    for result in report.results:
        assert result.status in VerificationStatus


@pytest.mark.integration
async def test_verification_with_empty_text(
    mock_execution_context: ExecutionContext,
) -> None:
    """Test verification handles empty text gracefully."""
    extractor = RuleBasedExtractor()
    checker = SelfConsistencyChecker()
    engine = VerificationEngine(extractor=extractor, checkers=[checker], ctx=mock_execution_context)

    report = await engine.verify_text("")

    assert report is not None
    assert len(report.claims) == 0

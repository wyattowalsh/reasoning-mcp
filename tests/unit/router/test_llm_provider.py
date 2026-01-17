"""
Unit tests for LLM Provider integration.

Tests cover:
- LLMProvider initialization
- JSON parsing with markdown code blocks
- Helper functions for enum conversion
- Response model validation
- Graceful fallback on errors
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from reasoning_mcp.router.llm_provider import (
    LLMAnalysisResponse,
    LLMProvider,
    LLMRouteRecommendation,
    LLMSelectionResponse,
    parse_capabilities,
    parse_capability,
    parse_domain,
    parse_intent,
    parse_route_type,
)
from reasoning_mcp.router.models import (
    ProblemDomain,
    ProblemIntent,
    RequiredCapability,
    RouteType,
)

# ============================================================================
# LLMProvider Initialization Tests
# ============================================================================


@pytest.mark.unit
class TestLLMProviderInit:
    """Tests for LLMProvider initialization."""

    def test_init_without_context(self):
        """Test provider initializes without context."""
        provider = LLMProvider()
        assert provider._ctx is None
        assert provider.available is False

    def test_init_with_context(self):
        """Test provider initializes with context."""
        mock_ctx = MagicMock()
        provider = LLMProvider(ctx=mock_ctx)
        assert provider._ctx is mock_ctx
        assert provider.available is True

    def test_init_with_custom_params(self):
        """Test provider initializes with custom parameters."""
        provider = LLMProvider(timeout_ms=10000, max_retries=3)
        assert provider._timeout_ms == 10000
        assert provider._max_retries == 3

    def test_update_context(self):
        """Test updating context after initialization."""
        provider = LLMProvider()
        assert provider.available is False

        mock_ctx = MagicMock()
        provider.update_context(mock_ctx)
        assert provider._ctx is mock_ctx
        assert provider.available is True


# ============================================================================
# JSON Parsing Tests
# ============================================================================


@pytest.mark.unit
class TestJSONParsing:
    """Tests for JSON response parsing."""

    def test_parse_json_direct(self):
        """Test parsing direct JSON string."""
        provider = LLMProvider()
        result = provider._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_json_with_markdown_json_block(self):
        """Test parsing JSON wrapped in ```json block."""
        provider = LLMProvider()
        result = provider._parse_json_response('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_json_with_markdown_block(self):
        """Test parsing JSON wrapped in ``` block."""
        provider = LLMProvider()
        result = provider._parse_json_response('```\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_json_with_whitespace(self):
        """Test parsing JSON with surrounding whitespace."""
        provider = LLMProvider()
        result = provider._parse_json_response('  \n{"key": "value"}\n  ')
        assert result == {"key": "value"}

    def test_parse_json_invalid(self):
        """Test parsing invalid JSON returns None."""
        provider = LLMProvider()
        result = provider._parse_json_response("not valid json")
        assert result is None

    def test_parse_json_empty(self):
        """Test parsing empty string returns None."""
        provider = LLMProvider()
        result = provider._parse_json_response("")
        assert result is None

    def test_parse_json_none(self):
        """Test parsing None returns None."""
        provider = LLMProvider()
        result = provider._parse_json_response(None)
        assert result is None


# ============================================================================
# Helper Function Tests - parse_domain
# ============================================================================


@pytest.mark.unit
class TestParseDomain:
    """Tests for parse_domain helper function."""

    def test_parse_valid_domains(self):
        """Test parsing valid domain strings."""
        assert parse_domain("mathematical") == ProblemDomain.MATHEMATICAL
        assert parse_domain("code") == ProblemDomain.CODE
        assert parse_domain("ethical") == ProblemDomain.ETHICAL
        assert parse_domain("creative") == ProblemDomain.CREATIVE

    def test_parse_domain_case_insensitive(self):
        """Test domain parsing is case insensitive."""
        assert parse_domain("MATHEMATICAL") == ProblemDomain.MATHEMATICAL
        assert parse_domain("Mathematical") == ProblemDomain.MATHEMATICAL
        assert parse_domain("CODE") == ProblemDomain.CODE

    def test_parse_domain_with_whitespace(self):
        """Test domain parsing handles whitespace."""
        assert parse_domain("  mathematical  ") == ProblemDomain.MATHEMATICAL

    def test_parse_domain_fuzzy_matching(self):
        """Test domain fuzzy matching."""
        # Partial matches
        assert parse_domain("math") == ProblemDomain.MATHEMATICAL
        assert parse_domain("coding") == ProblemDomain.CODE

    def test_parse_domain_fallback_to_general(self):
        """Test unknown domain falls back to GENERAL."""
        assert parse_domain("unknown") == ProblemDomain.GENERAL
        assert parse_domain("") == ProblemDomain.GENERAL


# ============================================================================
# Helper Function Tests - parse_intent
# ============================================================================


@pytest.mark.unit
class TestParseIntent:
    """Tests for parse_intent helper function."""

    def test_parse_valid_intents(self):
        """Test parsing valid intent strings."""
        assert parse_intent("solve") == ProblemIntent.SOLVE
        assert parse_intent("analyze") == ProblemIntent.ANALYZE
        assert parse_intent("debug") == ProblemIntent.DEBUG

    def test_parse_intent_case_insensitive(self):
        """Test intent parsing is case insensitive."""
        assert parse_intent("SOLVE") == ProblemIntent.SOLVE
        assert parse_intent("Analyze") == ProblemIntent.ANALYZE

    def test_parse_intent_fallback_to_solve(self):
        """Test unknown intent falls back to SOLVE."""
        assert parse_intent("unknown") == ProblemIntent.SOLVE


# ============================================================================
# Helper Function Tests - parse_capability
# ============================================================================


@pytest.mark.unit
class TestParseCapability:
    """Tests for parse_capability helper function."""

    def test_parse_valid_capabilities(self):
        """Test parsing valid capability strings."""
        assert parse_capability("branching") == RequiredCapability.BRANCHING
        assert parse_capability("iteration") == RequiredCapability.ITERATION
        assert parse_capability("verification") == RequiredCapability.VERIFICATION

    def test_parse_capability_returns_none_for_unknown(self):
        """Test unknown capability returns None."""
        assert parse_capability("unknown") is None


@pytest.mark.unit
class TestParseCapabilities:
    """Tests for parse_capabilities helper function."""

    def test_parse_multiple_capabilities(self):
        """Test parsing multiple capabilities."""
        result = parse_capabilities(["branching", "iteration", "verification"])
        assert RequiredCapability.BRANCHING in result
        assert RequiredCapability.ITERATION in result
        assert RequiredCapability.VERIFICATION in result

    def test_parse_capabilities_skips_unknown(self):
        """Test unknown capabilities are skipped."""
        result = parse_capabilities(["branching", "unknown", "iteration"])
        assert len(result) == 2
        assert RequiredCapability.BRANCHING in result
        assert RequiredCapability.ITERATION in result

    def test_parse_capabilities_empty_list(self):
        """Test empty list returns empty frozenset."""
        result = parse_capabilities([])
        assert result == frozenset()


# ============================================================================
# Helper Function Tests - parse_route_type
# ============================================================================


@pytest.mark.unit
class TestParseRouteType:
    """Tests for parse_route_type helper function."""

    def test_parse_valid_route_types(self):
        """Test parsing valid route type strings."""
        assert parse_route_type("single_method") == RouteType.SINGLE_METHOD
        assert parse_route_type("pipeline_template") == RouteType.PIPELINE_TEMPLATE
        assert parse_route_type("method_ensemble") == RouteType.METHOD_ENSEMBLE

    def test_parse_route_type_case_insensitive(self):
        """Test route type parsing is case insensitive."""
        assert parse_route_type("SINGLE_METHOD") == RouteType.SINGLE_METHOD

    def test_parse_route_type_fallback(self):
        """Test unknown route type falls back to SINGLE_METHOD."""
        assert parse_route_type("unknown") == RouteType.SINGLE_METHOD


# ============================================================================
# Response Model Tests
# ============================================================================


@pytest.mark.unit
class TestResponseModels:
    """Tests for response model validation."""

    def test_analysis_response_model(self):
        """Test LLMAnalysisResponse model validation."""
        response = LLMAnalysisResponse(
            primary_domain="mathematical",
            intent="solve",
            complexity=7,
        )
        assert response.primary_domain == "mathematical"
        assert response.intent == "solve"
        assert response.complexity == 7
        assert response.secondary_domains == []
        assert response.capabilities == []

    def test_analysis_response_with_all_fields(self):
        """Test LLMAnalysisResponse with all fields."""
        response = LLMAnalysisResponse(
            primary_domain="code",
            secondary_domains=["analytical"],
            intent="debug",
            complexity=8,
            ambiguity=3,
            depth_required=7,
            breadth_required=4,
            capabilities=["verification", "iteration"],
            key_entities=["function", "bug"],
            keywords=["python", "error"],
            reasoning="Complex debugging task",
        )
        assert response.complexity == 8
        assert len(response.capabilities) == 2

    def test_route_recommendation_model(self):
        """Test LLMRouteRecommendation model."""
        rec = LLMRouteRecommendation(
            route_type="single_method",
            method_id="chain_of_thought",
            score=0.9,
            reasoning="Best for this problem",
        )
        assert rec.route_type == "single_method"
        assert rec.method_id == "chain_of_thought"
        assert rec.score == 0.9

    def test_selection_response_model(self):
        """Test LLMSelectionResponse model."""
        response = LLMSelectionResponse(
            recommendations=[
                LLMRouteRecommendation(
                    route_type="single_method",
                    method_id="mathematical_reasoning",
                    score=0.9,
                )
            ],
            confidence=0.85,
            analysis_summary="Math problem requiring formal reasoning",
        )
        assert len(response.recommendations) == 1
        assert response.confidence == 0.85


# ============================================================================
# LLMProvider Method Tests
# ============================================================================


@pytest.mark.unit
class TestLLMProviderMethods:
    """Tests for LLMProvider async methods."""

    async def test_analyze_problem_without_context(self):
        """Test analyze_problem returns None without context."""
        provider = LLMProvider()
        result = await provider.analyze_problem("What is 2+2?")
        assert result is None

    async def test_select_routes_without_context(self):
        """Test select_routes returns None without context."""
        provider = LLMProvider()
        result = await provider.select_routes(
            problem="What is 2+2?",
            analysis_summary="Math problem",
        )
        assert result is None

    async def test_analyze_problem_with_mocked_context(self):
        """Test analyze_problem with mocked MCP context."""
        # Create mock context with sample method
        mock_ctx = MagicMock()
        mock_response = MagicMock()
        mock_response.text = (
            '{"primary_domain": "mathematical", "intent": "solve", "complexity": 5}'
        )
        mock_ctx.sample = AsyncMock(return_value=mock_response)

        provider = LLMProvider(ctx=mock_ctx)
        result = await provider.analyze_problem("What is 2+2?")

        assert result is not None
        assert result.primary_domain == "mathematical"
        assert result.intent == "solve"
        assert result.complexity == 5

    async def test_select_routes_with_mocked_context(self):
        """Test select_routes with mocked MCP context."""
        mock_ctx = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """
        {
            "recommendations": [
                {
                    "route_type": "single_method",
                    "method_id": "mathematical_reasoning",
                    "score": 0.9,
                    "reasoning": "Best for math"
                }
            ],
            "confidence": 0.85,
            "analysis_summary": "Math problem"
        }
        """
        mock_ctx.sample = AsyncMock(return_value=mock_response)

        provider = LLMProvider(ctx=mock_ctx)
        result = await provider.select_routes(
            problem="What is 2+2?",
            analysis_summary="Math problem",
        )

        assert result is not None
        assert len(result.recommendations) == 1
        assert result.recommendations[0].method_id == "mathematical_reasoning"
        assert result.confidence == 0.85

    async def test_analyze_problem_handles_timeout(self):
        """Test analyze_problem handles timeout gracefully."""
        mock_ctx = MagicMock()
        mock_ctx.sample = AsyncMock(side_effect=TimeoutError("Timeout"))

        provider = LLMProvider(ctx=mock_ctx, max_retries=0)
        result = await provider.analyze_problem("What is 2+2?")

        assert result is None

    async def test_analyze_problem_handles_exception(self):
        """Test analyze_problem handles exceptions gracefully."""
        mock_ctx = MagicMock()
        mock_ctx.sample = AsyncMock(side_effect=Exception("Error"))

        provider = LLMProvider(ctx=mock_ctx, max_retries=0)
        result = await provider.analyze_problem("What is 2+2?")

        assert result is None

    async def test_analyze_problem_handles_invalid_json(self):
        """Test analyze_problem handles invalid JSON response."""
        mock_ctx = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "not valid json"
        mock_ctx.sample = AsyncMock(return_value=mock_response)

        provider = LLMProvider(ctx=mock_ctx)
        result = await provider.analyze_problem("What is 2+2?")

        assert result is None

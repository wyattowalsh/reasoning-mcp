"""LLM Provider for the Reasoning Router.

This module provides LLM-based analysis and route selection using
MCP sampling (ctx.sample()). It handles structured output parsing
and graceful fallback on errors.

Task 2.1: LLM Provider Integration (Phase 2)
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, ValidationError

from reasoning_mcp.router.models import (
    ProblemDomain,
    ProblemIntent,
    RequiredCapability,
    RouteType,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

logger = logging.getLogger(__name__)


# ============================================================================
# Response Models for Structured LLM Output
# ============================================================================


class LLMAnalysisResponse(BaseModel):
    """Structured response from LLM problem analysis."""

    primary_domain: str = Field(description="Primary problem domain")
    secondary_domains: list[str] = Field(default_factory=list)
    intent: str = Field(description="Primary intent")
    complexity: int = Field(ge=1, le=10, default=5)
    ambiguity: int = Field(ge=1, le=10, default=5)
    depth_required: int = Field(ge=1, le=10, default=5)
    breadth_required: int = Field(ge=1, le=10, default=5)
    capabilities: list[str] = Field(default_factory=list)
    key_entities: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    reasoning: str = Field(default="")


class LLMRouteRecommendation(BaseModel):
    """Single route recommendation from LLM."""

    route_type: str = Field(description="Type of route")
    method_id: str | None = Field(default=None)
    pipeline_id: str | None = Field(default=None)
    ensemble_methods: list[str] = Field(default_factory=list)
    score: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = Field(default="")


class LLMSelectionResponse(BaseModel):
    """Structured response from LLM route selection."""

    recommendations: list[LLMRouteRecommendation] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    analysis_summary: str = Field(default="")


# ============================================================================
# Prompts
# ============================================================================

ANALYSIS_SYSTEM_PROMPT = """You are an expert problem analyzer for a reasoning system.
Analyze the given problem and provide structured analysis.

Available domains: mathematical, code, ethical, creative, analytical, causal, decision, scientific, legal, medical, philosophical, general

Available intents: solve, analyze, evaluate, generate, explain, verify, optimize, debug, compare, synthesize

Available capabilities: branching, iteration, external_tools, verification, decomposition, synthesis, formal_logic, creativity, memory, multi_agent

Respond with ONLY valid JSON matching this exact schema (no markdown, no explanation):
{
    "primary_domain": "string (one of the available domains)",
    "secondary_domains": ["string", ...],
    "intent": "string (one of the available intents)",
    "complexity": 1-10,
    "ambiguity": 1-10,
    "depth_required": 1-10,
    "breadth_required": 1-10,
    "capabilities": ["string", ...],
    "key_entities": ["string", ...],
    "keywords": ["string", ...],
    "reasoning": "Brief explanation of the analysis"
}"""

SELECTION_SYSTEM_PROMPT = """You are a reasoning method selector for an AI reasoning system.
Given a problem analysis, recommend the best reasoning approach(es).

Available route types:
- single_method: Use a single reasoning method
- pipeline_template: Use a pre-defined pipeline template
- method_ensemble: Combine multiple methods in parallel

Available methods include: chain_of_thought, tree_of_thoughts, self_consistency,
react, reflexion, mathematical_reasoning, code_reasoning, ethical_reasoning,
lateral_thinking, step_back, self_refine, chain_of_verification, and more.

Available pipeline templates: verified_reasoning, iterative_improve, analyze_refine,
ethical_multi_view, math_proof, debug_code, creative_explore, scientific_method,
decompose_solve, multi_agent_debate, decision_matrix.

Respond with ONLY valid JSON matching this exact schema (no markdown, no explanation):
{
    "recommendations": [
        {
            "route_type": "single_method|pipeline_template|method_ensemble",
            "method_id": "method name (for single_method)",
            "pipeline_id": "template name (for pipeline_template)",
            "ensemble_methods": ["method1", "method2"] (for method_ensemble),
            "score": 0.0-1.0,
            "reasoning": "Why this approach"
        }
    ],
    "confidence": 0.0-1.0,
    "analysis_summary": "Brief summary of why these recommendations"
}"""


# ============================================================================
# LLM Provider Class
# ============================================================================


class LLMProvider:
    """Abstraction for LLM calls via MCP sampling.

    Provides structured output parsing and graceful error handling
    for router analysis and selection operations.
    """

    def __init__(
        self,
        ctx: Context[Any, Any, Any] | None = None,
        timeout_ms: int = 5000,
        max_retries: int = 1,
    ) -> None:
        """Initialize the LLM provider.

        Args:
            ctx: MCP context for sampling
            timeout_ms: Timeout for LLM calls in milliseconds
            max_retries: Maximum retries on failure
        """
        self._ctx = ctx
        self._timeout_ms = timeout_ms
        self._max_retries = max_retries

    @property
    def available(self) -> bool:
        """Check if LLM provider is available."""
        return self._ctx is not None

    def update_context(self, ctx: Context[Any, Any, Any]) -> None:
        """Update the MCP context.

        Args:
            ctx: New MCP context
        """
        self._ctx = ctx

    async def _sample(
        self,
        prompt: str,
        system: str,
        max_tokens: int = 1000,
    ) -> str | None:
        """Make an LLM sampling call.

        Args:
            prompt: User prompt
            system: System prompt
            max_tokens: Maximum response tokens

        Returns:
            Response text or None on error
        """
        if self._ctx is None:
            logger.warning("LLM provider called without context")
            return None

        for attempt in range(self._max_retries + 1):
            try:
                # Use MCP sampling via ctx.sample()
                response = await self._ctx.sample(
                    messages=[{"role": "user", "content": prompt}],
                    system_prompt=system,
                    max_tokens=max_tokens,
                )

                # Extract text from response
                if hasattr(response, "text"):
                    return str(response.text)
                elif hasattr(response, "content"):
                    # Handle different response formats
                    if isinstance(response.content, str):
                        return response.content
                    elif isinstance(response.content, list):
                        # Extract text from content blocks
                        for block in response.content:
                            if hasattr(block, "text"):
                                return str(block.text)
                return str(response)

            except TimeoutError:
                logger.warning(
                    "LLM sampling timeout (attempt %d/%d)",
                    attempt + 1,
                    self._max_retries + 1,
                )
            except Exception as e:
                logger.warning(
                    "LLM sampling error (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    str(e),
                )

        return None

    def _parse_json_response(self, text: str | None) -> dict[str, Any] | None:
        """Parse JSON from LLM response.

        Handles common LLM output issues like markdown wrapping.

        Args:
            text: Raw response text

        Returns:
            Parsed JSON dict or None on error
        """
        if not text:
            return None

        # Try direct parse first
        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass

        # Try stripping markdown code blocks
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            result = json.loads(cleaned)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM JSON response: %s", str(e))
            return None

    async def analyze_problem(self, problem: str) -> LLMAnalysisResponse | None:
        """Analyze a problem using LLM.

        Args:
            problem: The problem text to analyze

        Returns:
            Structured analysis response or None on error
        """
        prompt = f"Analyze this problem:\n\n{problem}"

        response_text = await self._sample(
            prompt=prompt,
            system=ANALYSIS_SYSTEM_PROMPT,
            max_tokens=500,
        )

        data = self._parse_json_response(response_text)
        if not data:
            return None

        try:
            return LLMAnalysisResponse.model_validate(data)
        except ValidationError as e:
            logger.warning("Failed to validate LLM analysis response: %s", str(e))
            return None

    async def select_routes(
        self,
        problem: str,
        analysis_summary: str,
        available_methods: list[str] | None = None,
        available_templates: list[str] | None = None,
    ) -> LLMSelectionResponse | None:
        """Select routes using LLM.

        Args:
            problem: The problem text
            analysis_summary: Summary of problem analysis
            available_methods: Optional list of available methods
            available_templates: Optional list of available templates

        Returns:
            Structured selection response or None on error
        """
        prompt_parts = [
            f"Problem: {problem}",
            f"\nAnalysis: {analysis_summary}",
        ]

        if available_methods:
            prompt_parts.append(f"\nAvailable methods: {', '.join(available_methods[:20])}")

        if available_templates:
            prompt_parts.append(f"\nAvailable templates: {', '.join(available_templates)}")

        prompt_parts.append("\nRecommend the best reasoning approach(es) for this problem.")

        prompt = "\n".join(prompt_parts)

        response_text = await self._sample(
            prompt=prompt,
            system=SELECTION_SYSTEM_PROMPT,
            max_tokens=800,
        )

        data = self._parse_json_response(response_text)
        if not data:
            return None

        try:
            return LLMSelectionResponse.model_validate(data)
        except ValidationError as e:
            logger.warning("Failed to validate LLM selection response: %s", str(e))
            return None


# ============================================================================
# Helper Functions for Converting LLM Responses to Router Models
# ============================================================================


def parse_domain(domain_str: str) -> ProblemDomain:
    """Parse domain string to enum, with fallback to GENERAL."""
    domain_str = domain_str.lower().strip()
    if not domain_str:
        return ProblemDomain.GENERAL
    try:
        return ProblemDomain(domain_str)
    except ValueError:
        # Try fuzzy matching - substring containment, prefix, or common stem
        for domain in ProblemDomain:
            val = domain.value
            # Check substring containment
            if val in domain_str or domain_str in val:
                return domain
            # Check if they share a common prefix (at least 3 chars)
            min_len = min(len(val), len(domain_str))
            if min_len >= 3 and val[:3] == domain_str[:3]:
                return domain
        return ProblemDomain.GENERAL


def parse_intent(intent_str: str) -> ProblemIntent:
    """Parse intent string to enum, with fallback to SOLVE."""
    intent_str = intent_str.lower().strip()
    try:
        return ProblemIntent(intent_str)
    except ValueError:
        # Try fuzzy matching
        for intent in ProblemIntent:
            if intent_str in intent.value or intent.value in intent_str:
                return intent
        return ProblemIntent.SOLVE


def parse_capability(cap_str: str) -> RequiredCapability | None:
    """Parse capability string to enum."""
    cap_str = cap_str.lower().strip()
    try:
        return RequiredCapability(cap_str)
    except ValueError:
        # Try fuzzy matching
        for cap in RequiredCapability:
            if cap_str in cap.value or cap.value in cap_str:
                return cap
        return None


def parse_route_type(type_str: str) -> RouteType:
    """Parse route type string to enum."""
    type_str = type_str.lower().strip()
    try:
        return RouteType(type_str)
    except ValueError:
        # Default to single method
        return RouteType.SINGLE_METHOD


def parse_capabilities(cap_strs: list[str]) -> frozenset[RequiredCapability]:
    """Parse capability strings to frozenset of enums."""
    capabilities = set()
    for cap_str in cap_strs:
        cap = parse_capability(cap_str)
        if cap is not None:
            capabilities.add(cap)
    return frozenset(capabilities)

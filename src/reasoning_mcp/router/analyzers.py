"""Problem analyzers for the Reasoning Router.

This module provides analyzers that classify problems by domain, intent,
complexity, and required capabilities.

Tier 1 (Fast): Regex patterns + embedding centroids (<5ms)
Tier 2 (Standard): ML classifiers (~20ms)
Tier 3 (Complex): LLM-based analysis (~200ms)

FastMCP v2.14+ Features:
- Response caching for expensive analyze() operations
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from reasoning_mcp.router.models import (
    INTENT_CAPABILITY_MAPPING,
    ProblemAnalysis,
    ProblemDomain,
    ProblemIntent,
    RequiredCapability,
    RouterTier,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import Context

    from reasoning_mcp.middleware import ResponseCacheMiddleware

logger = logging.getLogger(__name__)


class ProblemAnalyzer(ABC):
    """Abstract base class for problem analyzers."""

    @property
    @abstractmethod
    def tier(self) -> RouterTier:
        """Return the tier of this analyzer."""
        ...

    @abstractmethod
    async def analyze(self, problem: str) -> ProblemAnalysis:
        """Analyze a problem and return analysis result."""
        ...


# Domain detection patterns (regex-based)
# Note: Order and specificity matter - more distinctive patterns should have higher match counts
DOMAIN_PATTERNS: dict[ProblemDomain, list[re.Pattern[str]]] = {
    ProblemDomain.MATHEMATICAL: [
        re.compile(
            r"\b(calculate|compute|solve|equation|formula|prove|theorem|integral|derivative)\b",
            re.I,
        ),
        re.compile(r"\b(sum|product|factorial|probability|statistics|matrix|vector)\b", re.I),
        re.compile(r"\b\d+\s*[\+\-\*\/\^]\s*\d+\b"),  # Basic math expressions
        re.compile(r"\b(algebra|calculus|geometry|trigonometry)\b", re.I),
        re.compile(r"\b(positive integers?|real numbers?|complex numbers?)\b", re.I),
        re.compile(r"\b(equals?|equal to)\s*\d", re.I),  # "equals n(n+1)/2" patterns
    ],
    ProblemDomain.CODE: [
        # More specific patterns - require coding-specific context
        re.compile(r"\b(debug|bug|error|exception|stack.?trace|runtime)\b", re.I),
        re.compile(r"\b(function|class|method|module|package|library)\b", re.I),
        re.compile(r"\b(python|javascript|typescript|java|rust|go|c\+\+|ruby|php)\b", re.I),
        re.compile(r"\b(api|endpoint|database|sql|query|git|deploy|commit)\b", re.I),
        re.compile(r"\b(variable|loop|condition|array|object|string|boolean)\b", re.I),
        re.compile(r"\b(refactor|optimize|code.?review|pull.?request)\b", re.I),
        re.compile(r"```[\w]*\n"),  # Code blocks
        # Note: "implement" alone is too generic - could mean business implementation
        # Only match "implement" with code-specific context
        re.compile(r"\bimplement\s+(a\s+)?(function|class|method|algorithm|interface)\b", re.I),
    ],
    ProblemDomain.ETHICAL: [
        # Core ethical terms (high priority)
        re.compile(r"\b(ethical|ethics|moral|morality|immoral|unethical)\b", re.I),
        re.compile(r"\b(right|wrong|should|ought|fair|unfair|just|unjust)\b", re.I),
        # Privacy and consent terms (important for data ethics)
        re.compile(r"\b(privacy|consent|personal\s+data|user\s+data|data\s+collection)\b", re.I),
        re.compile(r"\b(collect(ing)?\s+(personal|user|private)?\s*data)\b", re.I),
        # Stakeholder and responsibility terms
        re.compile(r"\b(harm|benefit|stakeholder|responsibility|accountability)\b", re.I),
        # Decision-making ethical framing
        re.compile(r"\b(dilemma|tradeoff|trade.?off|principle|value|virtue)\b", re.I),
        # Ethical considerations in tech/business
        re.compile(r"\b(bias|discrimination|transparency|trust|security)\b", re.I),
        # "Should we" pattern often indicates ethical consideration
        re.compile(r"\bshould\s+we\b", re.I),
    ],
    ProblemDomain.CREATIVE: [
        re.compile(r"\b(create|design|invent|imagine|novel|innovative|brainstorm)\b", re.I),
        re.compile(r"\b(story|poem|art|music|game|brand|logo)\b", re.I),
        re.compile(r"\b(creative|original|unique|unconventional)\b", re.I),
    ],
    ProblemDomain.ANALYTICAL: [
        re.compile(r"\b(analyze|analyse|compare|evaluate|assess|examine|review)\b", re.I),
        re.compile(r"\b(pros|cons|advantage|disadvantage|strength|weakness)\b", re.I),
        re.compile(r"\b(breakdown|decompose|structure|framework)\b", re.I),
    ],
    ProblemDomain.CAUSAL: [
        re.compile(r"\b(cause|effect|why|because|reason|result|consequence)\b", re.I),
        re.compile(r"\b(lead to|due to|impact|influence|factor)\b", re.I),
        re.compile(r"\b(root cause|underlying|mechanism)\b", re.I),
    ],
    ProblemDomain.DECISION: [
        re.compile(r"\b(decide|decision|choice|option|alternative|tradeoff)\b", re.I),
        re.compile(r"\b(best|optimal|recommend|suggest|prefer)\b", re.I),
        re.compile(r"\b(criteria|priority|weight|rank)\b", re.I),
    ],
    ProblemDomain.SCIENTIFIC: [
        re.compile(r"\b(hypothesis|experiment|theory|evidence|data|research)\b", re.I),
        re.compile(r"\b(biology|chemistry|physics|neuroscience|ecology)\b", re.I),
        re.compile(r"\b(study|paper|finding|observation|conclusion)\b", re.I),
    ],
    ProblemDomain.LEGAL: [
        re.compile(r"\b(legal|law|court|contract|liability|compliance)\b", re.I),
        re.compile(r"\b(regulation|statute|precedent|jurisdiction)\b", re.I),
        re.compile(r"\b(plaintiff|defendant|judge|attorney|lawsuit)\b", re.I),
    ],
    ProblemDomain.MEDICAL: [
        re.compile(r"\b(medical|health|disease|symptom|treatment|diagnosis)\b", re.I),
        re.compile(r"\b(patient|doctor|medicine|drug|therapy)\b", re.I),
        re.compile(r"\b(clinical|hospital|surgery|prescription)\b", re.I),
    ],
    ProblemDomain.PHILOSOPHICAL: [
        re.compile(r"\b(philosophy|meaning|existence|consciousness|reality)\b", re.I),
        re.compile(r"\b(epistemology|ontology|metaphysics|ethics)\b", re.I),
        re.compile(r"\b(truth|knowledge|belief|mind|soul)\b", re.I),
    ],
}

# Intent detection patterns
INTENT_PATTERNS: dict[ProblemIntent, list[re.Pattern[str]]] = {
    ProblemIntent.SOLVE: [
        re.compile(r"\b(solve|find|calculate|determine|figure out)\b", re.I),
        re.compile(r"\bwhat is\b", re.I),
        re.compile(r"\bhow (do|can|would) (i|we|you)\b", re.I),
    ],
    ProblemIntent.ANALYZE: [
        re.compile(r"\b(analyze|analyse|break down|examine|study)\b", re.I),
        re.compile(r"\b(what are the|identify the)\b", re.I),
    ],
    ProblemIntent.EVALUATE: [
        re.compile(r"\b(evaluate|assess|judge|rate|review|critique)\b", re.I),
        re.compile(r"\b(is this|are these|does this)\s+(good|bad|correct|right)\b", re.I),
    ],
    ProblemIntent.GENERATE: [
        re.compile(r"\b(generate|create|write|produce|make|build)\b", re.I),
        re.compile(r"\b(come up with|think of|brainstorm)\b", re.I),
    ],
    ProblemIntent.EXPLAIN: [
        re.compile(r"\b(explain|describe|clarify|elaborate|tell me about)\b", re.I),
        re.compile(r"\bwhy (is|are|does|do)\b", re.I),
        re.compile(r"\bhow does\b", re.I),
    ],
    ProblemIntent.VERIFY: [
        re.compile(r"\b(verify|check|confirm|validate|test)\b", re.I),
        re.compile(r"\b(is (it|this) (true|correct|valid|right))\b", re.I),
    ],
    ProblemIntent.OPTIMIZE: [
        re.compile(r"\b(optimize|improve|enhance|refine|better)\b", re.I),
        re.compile(r"\b(more efficient|faster|cleaner)\b", re.I),
    ],
    ProblemIntent.DEBUG: [
        re.compile(r"\b(debug|fix|troubleshoot|diagnose)\b", re.I),
        re.compile(r"\b(error|bug|issue|problem|not working)\b", re.I),
        re.compile(r"\b(why (is|does) (it|this) (fail|crash|break))\b", re.I),
    ],
    ProblemIntent.COMPARE: [
        re.compile(r"\b(compare|contrast|versus|vs\.?|difference between)\b", re.I),
        re.compile(r"\b(which (is|are) better)\b", re.I),
        re.compile(r"\b(pros and cons)\b", re.I),
    ],
    ProblemIntent.SYNTHESIZE: [
        re.compile(r"\b(synthesize|combine|integrate|merge|unify)\b", re.I),
        re.compile(r"\b(bring together|consolidate)\b", re.I),
    ],
}

# Capability detection patterns
CAPABILITY_PATTERNS: dict[RequiredCapability, list[re.Pattern[str]]] = {
    RequiredCapability.BRANCHING: [
        re.compile(
            r"\b(multiple|several|various|different)\s+(options|approaches|paths|ways)\b", re.I
        ),
        re.compile(r"\b(explore|consider)\s+(alternatives|possibilities)\b", re.I),
    ],
    RequiredCapability.ITERATION: [
        re.compile(r"\b(iterate|refine|improve|evolve|revise)\b", re.I),
        re.compile(r"\b(step by step|incrementally|progressively)\b", re.I),
    ],
    RequiredCapability.EXTERNAL_TOOLS: [
        re.compile(r"\b(run|execute|test|deploy|fetch|call)\b", re.I),
        re.compile(r"\b(api|database|server|terminal|browser)\b", re.I),
    ],
    RequiredCapability.VERIFICATION: [
        re.compile(r"\b(verify|validate|check|confirm|prove)\b", re.I),
        re.compile(r"\b(make sure|ensure|double.?check)\b", re.I),
    ],
    RequiredCapability.DECOMPOSITION: [
        re.compile(r"\b(complex|complicated|multi.?part|multi.?step)\b", re.I),
        re.compile(r"\b(break down|decompose|divide)\b", re.I),
    ],
    RequiredCapability.SYNTHESIS: [
        re.compile(r"\b(synthesize|combine|integrate|merge)\b", re.I),
        re.compile(r"\b(multiple (sources|inputs|perspectives))\b", re.I),
    ],
    RequiredCapability.FORMAL_LOGIC: [
        re.compile(r"\b(prove|theorem|lemma|axiom|deduce)\b", re.I),
        re.compile(r"\b(logic|logical|if.?then|implies)\b", re.I),
    ],
    RequiredCapability.CREATIVITY: [
        re.compile(r"\b(creative|novel|innovative|original|unique)\b", re.I),
        re.compile(r"\b(outside the box|unconventional)\b", re.I),
    ],
    RequiredCapability.MEMORY: [
        re.compile(r"\b(remember|recall|based on (previous|earlier)|context)\b", re.I),
        re.compile(r"\b(history|past|before|earlier)\b", re.I),
    ],
    RequiredCapability.MULTI_AGENT: [
        re.compile(r"\b(debate|discuss|multiple perspectives|different viewpoints)\b", re.I),
        re.compile(r"\b(expert|specialist|role)\b", re.I),
    ],
}


def _count_pattern_matches(text: str, patterns: list[re.Pattern[str]]) -> int:
    """Count total matches across all patterns."""
    return sum(len(p.findall(text)) for p in patterns)


def _detect_domains(text: str) -> tuple[ProblemDomain, frozenset[ProblemDomain]]:
    """Detect primary and secondary domains from text."""
    scores: dict[ProblemDomain, int] = {}

    for domain, patterns in DOMAIN_PATTERNS.items():
        score = _count_pattern_matches(text, patterns)
        if score > 0:
            scores[domain] = score

    if not scores:
        return ProblemDomain.GENERAL, frozenset()

    sorted_domains = sorted(scores.items(), key=lambda x: -x[1])
    primary = sorted_domains[0][0]

    # Secondary domains are those with at least half the score of primary
    primary_score = sorted_domains[0][1]
    secondary = frozenset(
        domain for domain, score in sorted_domains[1:] if score >= primary_score * 0.5
    )

    return primary, secondary


def _detect_intent(text: str) -> ProblemIntent:
    """Detect the primary intent from text."""
    scores: dict[ProblemIntent, int] = {}

    for intent, patterns in INTENT_PATTERNS.items():
        score = _count_pattern_matches(text, patterns)
        if score > 0:
            scores[intent] = score

    if not scores:
        return ProblemIntent.SOLVE  # Default

    return max(scores.items(), key=lambda x: x[1])[0]


def _detect_capabilities(text: str, intent: ProblemIntent) -> frozenset[RequiredCapability]:
    """Detect required capabilities from text and intent."""
    capabilities: set[RequiredCapability] = set()

    # Add capabilities from patterns
    for capability, patterns in CAPABILITY_PATTERNS.items():
        if _count_pattern_matches(text, patterns) > 0:
            capabilities.add(capability)

    # Add capabilities from intent mapping
    capabilities.update(INTENT_CAPABILITY_MAPPING.get(intent, frozenset()))

    return frozenset(capabilities)


def _estimate_complexity(
    text: str,
    domain: ProblemDomain,
    intent: ProblemIntent,
    capabilities: frozenset[RequiredCapability],
) -> int:
    """Estimate problem complexity (1-10)."""
    complexity = 3  # Base complexity

    # Increase for text length
    word_count = len(text.split())
    if word_count > 50:
        complexity += 1
    if word_count > 150:
        complexity += 1
    if word_count > 300:
        complexity += 1

    # Increase for capabilities
    complexity += min(len(capabilities), 3)

    # Domain-specific adjustments
    if domain in (ProblemDomain.MATHEMATICAL, ProblemDomain.CODE, ProblemDomain.LEGAL):
        complexity += 1

    # Intent-specific adjustments
    if intent in (ProblemIntent.SYNTHESIZE, ProblemIntent.DEBUG, ProblemIntent.OPTIMIZE):
        complexity += 1

    return min(10, max(1, complexity))


def _extract_keywords(text: str) -> frozenset[str]:
    """Extract important keywords from text."""
    # Simple keyword extraction - extract capitalized words and technical terms
    words = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    # Also extract quoted terms
    quoted = re.findall(r'"([^"]+)"', text)
    return frozenset(words + quoted)


class FastProblemAnalyzer(ProblemAnalyzer):
    """Fast tier analyzer using regex patterns.

    Latency target: <5ms
    """

    def __init__(self, cache: ResponseCacheMiddleware | None = None) -> None:
        """Initialize with optional cache."""
        self._cache = cache

    @property
    def tier(self) -> RouterTier:
        return RouterTier.FAST

    async def analyze(self, problem: str) -> ProblemAnalysis:
        """Analyze a problem using regex patterns."""
        start = time.perf_counter()

        # Detect domains
        primary_domain, secondary_domains = _detect_domains(problem)

        # Detect intent
        intent = _detect_intent(problem)

        # Detect capabilities
        capabilities = _detect_capabilities(problem, intent)

        # Estimate complexity
        complexity = _estimate_complexity(problem, primary_domain, intent, capabilities)

        # Extract keywords
        keywords = _extract_keywords(problem)

        latency_ms = (time.perf_counter() - start) * 1000

        # Calculate confidence based on pattern matches
        total_patterns = sum(
            _count_pattern_matches(problem, patterns) for patterns in DOMAIN_PATTERNS.values()
        )
        confidence = min(0.9, 0.4 + (total_patterns * 0.05))

        return ProblemAnalysis(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            intent=intent,
            complexity=complexity,
            ambiguity=5,  # Default for fast tier
            depth_required=max(3, complexity - 2),
            breadth_required=5 if RequiredCapability.BRANCHING in capabilities else 3,
            capabilities=capabilities,
            keywords=keywords,
            entities=frozenset(),  # Entity extraction is for higher tiers
            confidence=confidence,
            analysis_latency_ms=latency_ms,
            analyzer_tier=RouterTier.FAST,
        )


class StandardProblemAnalyzer(ProblemAnalyzer):
    """Standard tier analyzer using ML classifiers.

    Latency target: ~20ms

    Note: Full ML implementation requires sentence-transformers.
    Currently falls back to enhanced regex with embedding-ready structure.
    """

    def __init__(
        self,
        embedding_provider: str = "local:all-MiniLM-L6-v2",
        cache: ResponseCacheMiddleware | None = None,
    ) -> None:
        """Initialize with embedding provider and optional cache."""
        self._embedding_provider = embedding_provider
        self._embeddings_loaded = False
        self._cache = cache
        # Don't pass cache to inner analyzer - we cache at StandardAnalyzer level if enabled
        self._fast_analyzer = FastProblemAnalyzer(cache=None)

    @property
    def tier(self) -> RouterTier:
        return RouterTier.STANDARD

    async def analyze(self, problem: str) -> ProblemAnalysis:
        """Analyze using ML classifiers (with fast fallback)."""
        start = time.perf_counter()

        # For now, use fast analyzer and enhance
        fast_result = await self._fast_analyzer.analyze(problem)

        # TODO: Add embedding-based classification when ML deps available
        # - Domain classifier
        # - Intent classifier
        # - Complexity regression

        latency_ms = (time.perf_counter() - start) * 1000

        return ProblemAnalysis(
            primary_domain=fast_result.primary_domain,
            secondary_domains=fast_result.secondary_domains,
            intent=fast_result.intent,
            complexity=fast_result.complexity,
            ambiguity=fast_result.ambiguity,
            depth_required=fast_result.depth_required,
            breadth_required=fast_result.breadth_required,
            capabilities=fast_result.capabilities,
            keywords=fast_result.keywords,
            entities=fast_result.entities,
            confidence=min(0.95, fast_result.confidence + 0.1),  # Slightly higher confidence
            analysis_latency_ms=latency_ms,
            analyzer_tier=RouterTier.STANDARD,
        )


class ComplexProblemAnalyzer(ProblemAnalyzer):
    """Complex tier analyzer using LLM.

    Latency target: ~200ms

    Uses MCP sampling to leverage the host LLM for deep analysis.
    Supports response caching for expensive operations (FastMCP v2.14+).
    """

    def __init__(
        self,
        ctx: Context[Any, Any, Any] | None = None,
        cache: ResponseCacheMiddleware | None = None,
        cache_ttl: int = 300,
    ) -> None:
        """Initialize with MCP context for sampling and optional cache.

        Args:
            ctx: Optional MCP context for LLM sampling
            cache: Optional response cache middleware
            cache_ttl: TTL in seconds for cached analysis (default 300 = 5 minutes)
        """
        self._ctx = ctx
        self._cache = cache
        self._cache_ttl = cache_ttl
        # Don't pass cache to inner analyzer - we cache at ComplexAnalyzer level
        self._standard_analyzer = StandardProblemAnalyzer(cache=None)

    def _make_cache_key(self, problem: str) -> str:
        """Generate a cache key for problem analysis.

        Args:
            problem: The problem text

        Returns:
            32-character hex digest cache key
        """
        problem_hash = hashlib.sha256(problem.encode()).hexdigest()[:16]
        key_data = f"analyze:complex:{problem_hash}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    @property
    def tier(self) -> RouterTier:
        return RouterTier.COMPLEX

    async def analyze(self, problem: str) -> ProblemAnalysis:
        """Analyze using LLM (with standard fallback if no context).

        Supports caching for expensive analysis operations.

        Args:
            problem: The problem text to analyze

        Returns:
            Problem analysis result
        """
        # Check cache first (FastMCP v2.14+ feature)
        if self._cache is not None:
            cache_key = self._make_cache_key(problem)
            cached = await self._cache.get(cache_key)
            if cached is not None and isinstance(cached, ProblemAnalysis):
                # Cast to satisfy mypy while preserving object identity for caching
                cached_analysis: ProblemAnalysis = cached
                return cached_analysis

        start = time.perf_counter()

        # Fall back to standard analyzer (LLM analysis TODO)
        if self._ctx is None:
            # No MCP context available - use standard analyzer
            result = await self._standard_analyzer.analyze(problem)
            latency_ms = (time.perf_counter() - start) * 1000
            analysis = ProblemAnalysis(
                primary_domain=result.primary_domain,
                secondary_domains=result.secondary_domains,
                intent=result.intent,
                complexity=result.complexity,
                ambiguity=result.ambiguity,
                depth_required=result.depth_required,
                breadth_required=result.breadth_required,
                capabilities=result.capabilities,
                keywords=result.keywords,
                entities=result.entities,
                confidence=result.confidence,
                analysis_latency_ms=latency_ms,
                analyzer_tier=RouterTier.COMPLEX,
            )
        else:
            # Use LLM-based analysis via MCP sampling
            from reasoning_mcp.router.llm_provider import (
                LLMProvider,
                parse_capabilities,
                parse_domain,
                parse_intent,
            )

            llm_provider = LLMProvider(ctx=self._ctx)
            llm_result = await llm_provider.analyze_problem(problem)

            if llm_result is not None:
                # Successfully got LLM analysis - convert to ProblemAnalysis
                latency_ms = (time.perf_counter() - start) * 1000

                # Parse LLM string responses to enums
                primary_domain = parse_domain(llm_result.primary_domain)
                secondary_domains = frozenset(parse_domain(d) for d in llm_result.secondary_domains)
                intent = parse_intent(llm_result.intent)
                capabilities = parse_capabilities(llm_result.capabilities)

                analysis = ProblemAnalysis(
                    primary_domain=primary_domain,
                    secondary_domains=secondary_domains,
                    intent=intent,
                    complexity=llm_result.complexity,
                    ambiguity=llm_result.ambiguity,
                    depth_required=llm_result.depth_required,
                    breadth_required=llm_result.breadth_required,
                    capabilities=capabilities,
                    keywords=frozenset(llm_result.keywords),
                    entities=frozenset(llm_result.key_entities),
                    confidence=0.95,  # High confidence for LLM analysis
                    analysis_latency_ms=latency_ms,
                    analyzer_tier=RouterTier.COMPLEX,
                )

                logger.debug(
                    "LLM analysis complete: domain=%s, intent=%s, complexity=%d",
                    primary_domain.value,
                    intent.value,
                    llm_result.complexity,
                )
            else:
                # LLM analysis failed - fall back to standard analyzer
                logger.warning("LLM analysis failed, falling back to standard analyzer")
                result = await self._standard_analyzer.analyze(problem)
                latency_ms = (time.perf_counter() - start) * 1000

                analysis = ProblemAnalysis(
                    primary_domain=result.primary_domain,
                    secondary_domains=result.secondary_domains,
                    intent=result.intent,
                    complexity=result.complexity,
                    ambiguity=result.ambiguity,
                    depth_required=result.depth_required,
                    breadth_required=result.breadth_required,
                    capabilities=result.capabilities,
                    keywords=result.keywords,
                    entities=result.entities,
                    confidence=result.confidence,
                    analysis_latency_ms=latency_ms,
                    analyzer_tier=RouterTier.COMPLEX,
                )

        # Cache the result (regardless of which path we took)
        if self._cache is not None:
            cache_key = self._make_cache_key(problem)
            await self._cache.set(cache_key, analysis, ttl=self._cache_ttl)

        return analysis


def get_analyzer(
    tier: RouterTier,
    ctx: Context[Any, Any, Any] | None = None,
    cache: ResponseCacheMiddleware | None = None,
    cache_ttl: int = 300,
) -> ProblemAnalyzer:
    """Get an analyzer for the specified tier.

    Args:
        tier: The routing tier to use
        ctx: Optional MCP context for LLM-based analysis
        cache: Optional response cache middleware
        cache_ttl: TTL in seconds for cached analysis (default 300)

    Returns:
        ProblemAnalyzer instance for the specified tier
    """
    if tier == RouterTier.FAST:
        return FastProblemAnalyzer(cache=cache)
    elif tier == RouterTier.STANDARD:
        return StandardProblemAnalyzer(cache=cache)
    elif tier == RouterTier.COMPLEX:
        return ComplexProblemAnalyzer(ctx=ctx, cache=cache, cache_ttl=cache_ttl)
    else:
        return FastProblemAnalyzer(cache=cache)  # Default fallback

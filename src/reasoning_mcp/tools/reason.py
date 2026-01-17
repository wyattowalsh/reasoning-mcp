"""Primary reasoning tool for reasoning-mcp.

This module provides the main `reason` tool for initiating and continuing
reasoning processes. The tool supports automatic method selection and
manual method specification.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pydantic import TypeAdapter

from reasoning_mcp.config import get_settings
from reasoning_mcp.debug.collector import TraceCollector
from reasoning_mcp.engine.executor import ExecutionContext
from reasoning_mcp.engine.registry import get_executor_for_stage
from reasoning_mcp.models.core import MethodIdentifier, ThoughtType
from reasoning_mcp.models.debug import TraceLevel
from reasoning_mcp.models.pipeline import (
    ConditionalPipeline,
    LoopPipeline,
    MethodStage,
    ParallelPipeline,
    Pipeline,
    PipelineTrace,
    SequencePipeline,
    SwitchPipeline,
)
from reasoning_mcp.models.session import Session
from reasoning_mcp.models.thought import ThoughtNode
from reasoning_mcp.models.tools import ReasonHints, ReasonOutput
from reasoning_mcp.registry import MethodRegistry
from reasoning_mcp.resources.template import get_available_templates, get_template
from reasoning_mcp.router import (
    ReasoningRouter,
    ResourceBudget,
    RouterTier,
    RouteType,
)
from reasoning_mcp.router.templates import get_template as get_router_template
from reasoning_mcp.selector import MethodSelector

if TYPE_CHECKING:
    # Note: Using mcp.server.fastmcp.Context for compatibility with ReasoningRouter
    # ExecutionContext expects fastmcp.server.Context which is runtime-compatible
    from mcp.server.fastmcp import Context

    from reasoning_mcp.models.cost import Budget
    from reasoning_mcp.sessions import SessionManager

_PIPELINE_ADAPTER: TypeAdapter[Pipeline] = TypeAdapter(Pipeline)

# Cached router instance (created per-request, but cached for efficiency)
_ROUTER_CACHE: ReasoningRouter | None = None
_ROUTER_LOCK: asyncio.Lock | None = None


def _get_router_lock() -> asyncio.Lock:
    """Get or create the router lock.

    This ensures we have a single lock for router cache access,
    created lazily to work with asyncio event loops.
    """
    global _ROUTER_LOCK
    if _ROUTER_LOCK is None:
        _ROUTER_LOCK = asyncio.Lock()
    return _ROUTER_LOCK


def _get_registry() -> MethodRegistry:
    """Get the registry from AppContext or create a fallback for testing.

    This function provides access to the method registry. In normal operation,
    it retrieves the registry from the AppContext set during server initialization.
    For testing scenarios where the server hasn't been started, it creates a
    standalone registry with native methods registered.

    Returns:
        MethodRegistry with native methods registered

    Note:
        Plugin-registered methods are only available when using the AppContext
        registry. The fallback registry only contains native methods.
    """
    try:
        from reasoning_mcp.server import get_app_context

        return get_app_context().registry
    except RuntimeError:
        # Fallback for testing or standalone usage
        # This creates an isolated registry without plugin methods
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(
            "AppContext not available, creating standalone registry. "
            "Plugin-registered methods will not be available."
        )
        registry = MethodRegistry()
        from reasoning_mcp.methods.native import register_all_native_methods

        register_all_native_methods(registry)
        return registry


def _get_session_manager() -> SessionManager:
    """Get the SessionManager from AppContext or create a fallback for testing.

    This function provides access to the session manager. In normal operation,
    it retrieves the session manager from the AppContext set during server
    initialization. For testing scenarios where the server hasn't been started,
    it creates a standalone session manager.

    Returns:
        SessionManager instance

    Note:
        The fallback session manager is isolated and won't share sessions
        with other components. Only use for testing.
    """
    try:
        from reasoning_mcp.server import get_app_context

        return get_app_context().session_manager
    except RuntimeError:
        # Fallback for testing or standalone usage
        import logging

        from reasoning_mcp.sessions import SessionManager

        logger = logging.getLogger(__name__)
        logger.warning(
            "AppContext not available, creating standalone SessionManager. "
            "Sessions will not be shared with other components."
        )
        return SessionManager()


async def _get_router(ctx: Context | None = None) -> ReasoningRouter:
    """Get or create a router instance.

    This function provides access to the reasoning router. It creates a new
    router using the registry from AppContext and caches it for efficiency.
    The router context can be updated if a new MCP context is provided.

    Args:
        ctx: Optional MCP context for LLM-based routing

    Returns:
        ReasoningRouter instance configured from settings
    """
    global _ROUTER_CACHE
    async with _get_router_lock():
        settings = get_settings()
        registry = _get_registry()

        if _ROUTER_CACHE is None:
            default_tier = RouterTier(settings.router_default_tier)
            _ROUTER_CACHE = ReasoningRouter(
                registry=registry,
                ctx=ctx,
                default_tier=default_tier,
                enable_ml_routing=settings.enable_ml_routing,
                enable_llm_routing=settings.enable_llm_routing,
            )
        elif ctx is not None:
            # Update context if provided (for LLM-based routing)
            _ROUTER_CACHE.update_context(ctx)

        return _ROUTER_CACHE


def _get_default_budget() -> ResourceBudget:
    """Get default resource budget from settings.

    Returns:
        ResourceBudget with defaults from configuration
    """
    settings = get_settings()
    return ResourceBudget(
        max_latency_ms=settings.router_default_max_latency_ms,
        max_tokens=settings.router_default_max_tokens,
        max_thoughts=settings.router_default_max_thoughts,
    )


def _parse_combo_spec(method: Any) -> tuple[Pipeline, dict[str, Any]] | None:
    """Parse combo specifications from the method argument.

    Supports:
    - "combo:<template_id>" for named templates
    - "combo:{...}" for inline pipeline JSON
    - "{...}" for inline pipeline JSON without prefix
    - dict/Pipeline objects passed directly
    """
    if isinstance(
        method,
        (
            MethodStage,
            SequencePipeline,
            ParallelPipeline,
            ConditionalPipeline,
            LoopPipeline,
            SwitchPipeline,
        ),
    ):
        return method, {"combo_source": "inline"}

    if isinstance(method, dict):
        pipeline = _PIPELINE_ADAPTER.validate_python(method)
        return pipeline, {"combo_source": "inline"}

    if isinstance(method, str):
        spec = method.strip()

        if spec.startswith("combo:"):
            combo_spec = spec[len("combo:") :].strip()
            if not combo_spec:
                raise ValueError("Combo specification cannot be empty")

            if combo_spec.startswith("{"):
                data = json.loads(combo_spec)
                pipeline = _PIPELINE_ADAPTER.validate_python(data)
                return pipeline, {"combo_source": "inline"}

            template = get_template(combo_spec)
            if template is None:
                available = get_available_templates()
                raise ValueError(
                    f"Combo template '{combo_spec}' not found. "
                    f"Available templates: {', '.join(available)}"
                )
            pipeline = _PIPELINE_ADAPTER.validate_python(template)
            return pipeline, {"combo_source": "template", "combo_id": combo_spec}

        if spec.startswith("{"):
            data = json.loads(spec)
            pipeline = _PIPELINE_ADAPTER.validate_python(data)
            return pipeline, {"combo_source": "inline"}

    return None


def _ensure_problem_metadata(thought: ThoughtNode, problem: str) -> ThoughtNode:
    """Attach the original problem to thought metadata if missing."""
    if "problem" in thought.metadata:
        return thought
    return thought.with_update(metadata={**thought.metadata, "problem": problem})


def _trim_thought_content(thought: ThoughtNode, max_len: int = 300) -> ThoughtNode:
    """Trim thought content to a maximum length for preview responses."""
    if len(thought.content) <= max_len:
        return thought
    trimmed = thought.content[: max_len - 3].rstrip() + "..."
    return thought.with_update(content=trimmed)


async def _execute_routed_pipeline(
    problem: str,
    pipeline: Pipeline,
    route_result: Any,
    hints: ReasonHints | None = None,
    routing_budget: ResourceBudget | None = None,
    ctx: Context | None = None,
    trace_collector: TraceCollector | None = None,
    root_span_id: str | None = None,
) -> ReasonOutput:
    """Execute a pipeline selected by the router.

    Args:
        problem: The problem being solved
        pipeline: The pipeline to execute
        route_result: Result from the router
        hints: Optional hints provided by the user
        routing_budget: Optional resource budget
        ctx: Optional FastMCP Context for sampling (v2.14+)
        trace_collector: Optional trace collector for capturing execution details
        root_span_id: Optional root span ID for tracing

    Returns:
        ReasonOutput with pipeline execution results
    """
    # Create session and execution context with initialized registry
    # Use trace collector's session ID if tracing is enabled
    if trace_collector:
        session = Session(id=trace_collector.session_id).start()
    else:
        session = Session().start()
    registry = _get_registry()
    start_time = datetime.now()
    context = ExecutionContext(
        session=session,
        registry=registry,
        input_data={"input": problem, "problem": problem},
        ctx=ctx,  # type: ignore[arg-type]  # FastMCP v2.14+ Context - cross-package compatibility
    )

    # Execute pipeline
    executor = get_executor_for_stage(pipeline)
    result = await executor.execute(context)
    end_time = datetime.now()

    trace = PipelineTrace(
        pipeline_id=pipeline.id,
        session_id=session.id,
        started_at=start_time,
        completed_at=end_time,
        status="completed" if result.success else "failed",
        root_trace=result.trace,
    )

    final_thoughts = session.get_recent_thoughts(n=1)
    if final_thoughts:
        thought = final_thoughts[0]
    else:
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="No thoughts were generated during pipeline execution.",
            confidence=0.5,
            step_number=1,
            depth=0,
            created_at=datetime.now(),
            metadata={"problem": problem},
        )

    thought = _ensure_problem_metadata(thought, problem)
    thought = _trim_thought_content(thought)

    route = route_result.primary_route
    metadata: dict[str, Any] = {
        "auto_selected": True,
        "hints_provided": hints is not None,
        "problem_length": len(problem),
        "routing": {
            "route_type": str(route.route_type),
            "router_tier": str(route.router_tier),
            "route_score": route.score,
            "route_confidence": route.confidence,
            "route_reasoning": route.reasoning,
            "routing_latency_ms": route_result.total_latency_ms,
            "pipeline_id": pipeline.id,
            "pipeline_type": str(pipeline.stage_type),
            "problem_analysis": {
                "domain": str(route_result.problem_analysis.primary_domain),
                "intent": str(route_result.problem_analysis.intent),
                "complexity": route_result.problem_analysis.complexity,
            },
        },
        "pipeline": {
            "pipeline_id": pipeline.id,
            "pipeline_type": str(pipeline.stage_type),
            "success": result.success,
        },
        "trace": trace.model_dump(mode="json"),
    }

    if hints:
        metadata["hints"] = {
            "domain": hints.domain,
            "complexity": hints.complexity,
            "preferred_methods": [str(m) for m in hints.prefer_methods],
            "avoided_methods": [str(m) for m in hints.avoid_methods],
        }

    # Finalize trace if enabled
    if trace_collector and root_span_id:
        from reasoning_mcp.models.debug import SpanStatus

        trace_collector.end_span(
            root_span_id,
            status=SpanStatus.COMPLETED if result.success else SpanStatus.FAILED,
        )
        metadata["trace_id"] = trace_collector.trace_id

    return ReasonOutput(
        session_id=session.id,
        thought=thought,
        method_used=thought.method_id,
        suggestions=[
            "Continue reasoning with session_continue",
            "Explore alternative approaches with session_branch",
            "Inspect current reasoning state with session_inspect",
        ],
        metadata=metadata,
    )


async def _execute_routed_ensemble(
    problem: str,
    route_result: Any,
    hints: ReasonHints | None = None,
    routing_budget: ResourceBudget | None = None,
    ctx: Context | None = None,
    trace_collector: TraceCollector | None = None,
    root_span_id: str | None = None,
) -> ReasonOutput:
    """Execute an ensemble of methods selected by the router.

    Args:
        problem: The problem being solved
        route_result: Result from the router containing ensemble configuration
        hints: Optional hints provided by the user
        routing_budget: Optional resource budget
        ctx: Optional FastMCP Context for sampling (v2.14+)
        trace_collector: Optional trace collector for capturing execution details
        root_span_id: Optional root span ID for tracing

    Returns:
        ReasonOutput with ensemble execution results
    """
    route = route_result.primary_route
    ensemble_methods = route.ensemble_methods
    ensemble_strategy = route.ensemble_strategy

    if not ensemble_methods:
        # Fallback if no methods specified
        ensemble_methods = ("chain_of_thought",)

    # Build a parallel pipeline from ensemble methods
    parallel_stages = [
        {
            "stage_type": "method",
            "method": method_id,
            "config": {},
        }
        for method_id in ensemble_methods
    ]

    # Create ensemble pipeline with aggregation
    ensemble_pipeline_def = {
        "id": f"ensemble_{uuid4().hex[:8]}",
        "stage_type": "sequence",
        "stages": [
            {
                "stage_type": "parallel",
                "stages": parallel_stages,
                "aggregation": "collect",
            },
            # Add self-consistency or voting based on strategy
            {
                "stage_type": "method",
                "method": "self_consistency" if ensemble_strategy == "vote" else "chain_of_thought",
                "config": {"aggregation": ensemble_strategy},
            },
        ],
    }

    pipeline = _PIPELINE_ADAPTER.validate_python(ensemble_pipeline_def)

    # Execute the ensemble pipeline
    # Use trace collector's session ID if tracing is enabled
    if trace_collector:
        session = Session(id=trace_collector.session_id).start()
    else:
        session = Session().start()
    registry = _get_registry()
    start_time = datetime.now()
    context = ExecutionContext(
        session=session,
        registry=registry,
        input_data={"input": problem, "problem": problem},
        ctx=ctx,  # type: ignore[arg-type]  # FastMCP v2.14+ - cross-package compatibility
    )

    executor = get_executor_for_stage(pipeline)
    result = await executor.execute(context)
    end_time = datetime.now()

    trace = PipelineTrace(
        pipeline_id=pipeline.id,
        session_id=session.id,
        started_at=start_time,
        completed_at=end_time,
        status="completed" if result.success else "failed",
        root_trace=result.trace,
    )

    final_thoughts = session.get_recent_thoughts(n=1)
    if final_thoughts:
        thought = final_thoughts[0]
    else:
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
            content="No thoughts were generated during ensemble execution.",
            confidence=0.5,
            step_number=1,
            depth=0,
            created_at=datetime.now(),
            metadata={"problem": problem},
        )

    thought = _ensure_problem_metadata(thought, problem)
    thought = _trim_thought_content(thought)

    metadata: dict[str, Any] = {
        "auto_selected": True,
        "hints_provided": hints is not None,
        "problem_length": len(problem),
        "sampling_available": ctx is not None,  # Track if sampling was available
        "routing": {
            "route_type": "method_ensemble",
            "router_tier": str(route.router_tier),
            "route_score": route.score,
            "route_confidence": route.confidence,
            "route_reasoning": route.reasoning,
            "routing_latency_ms": route_result.total_latency_ms,
            "problem_analysis": {
                "domain": str(route_result.problem_analysis.primary_domain),
                "intent": str(route_result.problem_analysis.intent),
                "complexity": route_result.problem_analysis.complexity,
            },
        },
        "ensemble": {
            "methods": list(ensemble_methods),
            "strategy": ensemble_strategy,
            "success": result.success,
        },
        "trace": trace.model_dump(mode="json"),
    }

    if hints:
        metadata["hints"] = {
            "domain": hints.domain,
            "complexity": hints.complexity,
            "preferred_methods": [str(m) for m in hints.prefer_methods],
            "avoided_methods": [str(m) for m in hints.avoid_methods],
        }

    # Finalize trace if enabled
    if trace_collector and root_span_id:
        from reasoning_mcp.models.debug import SpanStatus

        trace_collector.end_span(
            root_span_id,
            status=SpanStatus.COMPLETED if result.success else SpanStatus.FAILED,
        )
        metadata["trace_id"] = trace_collector.trace_id

    return ReasonOutput(
        session_id=session.id,
        thought=thought,
        method_used=thought.method_id,
        suggestions=[
            "Continue reasoning with session_continue",
            "Explore alternative approaches with session_branch",
            "Review ensemble results with session_inspect",
        ],
        metadata=metadata,
    )


async def reason(
    problem: str,
    method: str | dict[str, Any] | Pipeline | None = None,
    hints: ReasonHints | None = None,
    routing_budget: ResourceBudget | None = None,
    force_routing_tier: Literal["fast", "standard", "complex"] | None = None,
    ctx: Context | None = None,
    estimate_only: bool = False,
    budget: Budget | None = None,
    trace: bool = False,
    trace_level: TraceLevel = TraceLevel.STANDARD,
    verify: bool = False,
    verify_method: str = "chain_of_verification",
    ensemble: bool = False,
    ensemble_methods: list[str] | None = None,
) -> ReasonOutput:
    """Generate reasoning for a problem using specified or auto-selected method.

    This is the primary tool for initiating reasoning processes. It accepts a
    problem statement and either uses the specified method or automatically
    selects the most appropriate method based on problem analysis using the
    intelligent multi-tier router.

    The tool can:
    - Auto-select the best reasoning method via intelligent routing
    - Select pipeline templates for complex problems
    - Synthesize custom pipelines based on problem analysis
    - Create a new reasoning session if needed
    - Generate the initial thought for the problem
    - Provide suggestions for next steps

    Args:
        problem: The problem or question to reason about. Should be a clear
            statement of what needs to be analyzed or solved.
        method: Optional method identifier to use (e.g., "chain_of_thought",
            "ethical_reasoning") OR a combo spec:
            - "combo:<template_id>" (e.g., "combo:debate")
            - "combo:{...}" inline pipeline JSON
            - direct pipeline dict/object.
            If None, the best method/pipeline is automatically selected
            via intelligent routing.
        hints: Optional hints to guide method selection. Only used when method
            is None. Provides domain, complexity, and preference information
            to improve auto-selection.
        routing_budget: Optional resource budget constraints for routing.
            Controls max latency, tokens, and thoughts. Also allows specifying
            prefer_speed or prefer_quality flags.
        force_routing_tier: Force a specific routing tier:
            - "fast": <5ms, embedding + regex patterns
            - "standard": ~20ms, classifiers + matrix factorization
            - "complex": ~200ms, LLM-based analysis + pipeline synthesis
        ctx: Optional MCP context for LLM-based routing (Tier 3).
        estimate_only: If True, only estimate cost without executing reasoning.
        budget: Optional cost budget for reasoning execution.
        trace: Enable execution tracing. When True, creates a TraceCollector
            to capture execution details including spans, steps, and decisions.
        trace_level: Trace verbosity level (minimal, standard, detailed, verbose).
            Controls the amount of detail captured during execution.
        verify: Enable verification of reasoning output. When True, runs the
            VerificationEngine on the generated thought to extract and verify
            claims. Defaults to False.
        verify_method: The verification method to use when verify=True. Currently
            defaults to "chain_of_verification". This parameter is reserved for
            future use when multiple verification methods are supported.
        ensemble: Enable ensemble reasoning mode. When True, uses EnsembleReasoning
            method to combine multiple reasoning approaches for more robust results.
            Defaults to False.
        ensemble_methods: List of method names to include in the ensemble. If None
            and ensemble=True, uses default ensemble configuration (COT, Tree of
            Thoughts, Self-Reflection). Only used when ensemble=True.

    Returns:
        ReasonOutput containing:
        - session_id: UUID of the reasoning session (new or existing)
        - thought: The initial generated thought
        - method_used: The method that was applied
        - suggestions: Recommended next steps
        - metadata: Additional information about the reasoning process,
          including routing details when auto-selection was used

    Examples:
        Auto-select method for ethical problem:
        >>> output = await reason(
        ...     "Should we implement this feature that might compromise user privacy?"
        ... )
        >>> assert output.method_used == MethodIdentifier.ETHICAL_REASONING
        >>> assert output.thought.type == ThoughtType.INITIAL

        Use specific method:
        >>> output = await reason(
        ...     "Calculate the optimal solution to this optimization problem",
        ...     method="mathematical_reasoning"
        ... )
        >>> assert output.method_used == MethodIdentifier.MATHEMATICAL_REASONING

        Use hints for better auto-selection:
        >>> hints = ReasonHints(
        ...     domain="code",
        ...     complexity="high",
        ...     prefer_methods=[MethodIdentifier.CODE_REASONING]
        ... )
        >>> output = await reason(
        ...     "Debug this complex async race condition",
        ...     hints=hints
        ... )
        >>> assert output.method_used == MethodIdentifier.CODE_REASONING

        Use routing budget for speed:
        >>> budget = ResourceBudget(prefer_speed=True, max_latency_ms=5000)
        >>> output = await reason(
        ...     "Quick analysis needed",
        ...     routing_budget=budget
        ... )

        Force complex routing for thorough analysis:
        >>> output = await reason(
        ...     "Very complex ethical dilemma with many stakeholders",
        ...     force_routing_tier="complex"
        ... )

        Use ensemble reasoning with default methods:
        >>> output = await reason(
        ...     "Complex problem requiring multiple perspectives",
        ...     ensemble=True
        ... )
        >>> assert output.method_used == MethodIdentifier.ENSEMBLE_REASONING

        Use ensemble with custom methods:
        >>> output = await reason(
        ...     "Mathematical optimization problem",
        ...     ensemble=True,
        ...     ensemble_methods=["mathematical_reasoning", "chain_of_thought", "step_back"]
        ... )

    Raises:
        ValueError: If the specified method doesn't exist or is invalid
        RuntimeError: If reasoning engine encounters an error
    """
    # Create trace collector if tracing is enabled
    trace_collector: TraceCollector | None = None
    root_span_id: str | None = None
    if trace:
        # Create a session ID early so we can use it for the trace collector
        session_id_for_trace = str(uuid4())
        trace_collector = TraceCollector(session_id=session_id_for_trace)
        root_span_id = trace_collector.start_span(
            f"reason:{method or 'ensemble' if ensemble else 'auto'}",
            attributes={
                "query": problem[:100] if len(problem) > 100 else problem,
                "method": str(method) if method else ("ensemble" if ensemble else "auto"),
                "trace_level": str(trace_level),
                "ensemble": ensemble,
            },
        )

    # Handle ensemble reasoning mode
    if ensemble:
        from reasoning_mcp.methods.native.ensemble_reasoning import EnsembleReasoning
        from reasoning_mcp.models.ensemble import EnsembleConfig, EnsembleMember

        # Build ensemble config
        if ensemble_methods:
            # Use custom methods
            members = [
                EnsembleMember(method_name=method_name, weight=1.0)
                for method_name in ensemble_methods
            ]
            config = EnsembleConfig(members=members)
        else:
            # Use default config
            config = EnsembleReasoning.get_default_config()

        # Create session and execution context
        if trace_collector:
            session = Session(id=trace_collector.session_id).start()
        else:
            session = Session().start()
        registry = _get_registry()

        # Create execution context
        context_for_ensemble = ExecutionContext(
            session=session,
            registry=registry,
            input_data={"input": problem, "problem": problem},
            ctx=ctx,  # type: ignore[arg-type]  # Cross-package Context compatibility
        )

        # Execute ensemble reasoning
        ensemble_method = EnsembleReasoning(config=config)
        await ensemble_method.initialize()
        thought = await ensemble_method.execute(
            session=session,
            input_text=problem,
            execution_context=context_for_ensemble,
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "auto_selected": False,
            "ensemble_mode": True,
            "ensemble_methods": ensemble_methods or [m.method_name for m in config.members],
            "hints_provided": hints is not None,
            "problem_length": len(problem),
        }

        if hints:
            metadata["hints"] = {
                "domain": hints.domain,
                "complexity": hints.complexity,
                "preferred_methods": [str(m) for m in hints.prefer_methods],
                "avoided_methods": [str(m) for m in hints.avoid_methods],
            }

        # Finalize trace if enabled
        if trace_collector and root_span_id:
            from reasoning_mcp.models.debug import SpanStatus

            trace_collector.end_span(root_span_id, status=SpanStatus.COMPLETED)
            metadata["trace_id"] = trace_collector.trace_id

        return ReasonOutput(
            session_id=session.id,
            thought=thought,
            method_used=MethodIdentifier.ENSEMBLE_REASONING,
            suggestions=[
                "Continue reasoning with session_continue",
                "Explore alternative approaches with session_branch",
                "Review ensemble results with session_inspect",
            ],
            metadata=metadata,
        )

    combo = _parse_combo_spec(method) if method is not None else None
    if combo is not None:
        pipeline, combo_meta = combo

        # Create session and execution context with initialized registry
        # Use trace collector's session ID if tracing is enabled
        if trace_collector:
            session = Session(id=trace_collector.session_id).start()
        else:
            session = Session().start()
        registry = _get_registry()
        start_time = datetime.now()
        context = ExecutionContext(
            session=session,
            registry=registry,
            input_data={"input": problem, "problem": problem},
            ctx=ctx,  # type: ignore[arg-type]  # FastMCP v2.14+ - cross-package compatibility
        )

        # Execute pipeline
        executor = get_executor_for_stage(pipeline)
        result = await executor.execute(context)
        end_time = datetime.now()

        pipeline_trace = PipelineTrace(
            pipeline_id=pipeline.id,
            session_id=session.id,
            started_at=start_time,
            completed_at=end_time,
            status="completed" if result.success else "failed",
            root_trace=result.trace,
        )

        final_thoughts = session.get_recent_thoughts(n=1)
        if final_thoughts:
            thought = final_thoughts[0]
        else:
            thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.INITIAL,
                method_id=MethodIdentifier.CHAIN_OF_THOUGHT,
                content="No thoughts were generated during combo execution.",
                confidence=0.5,
                step_number=1,
                depth=0,
                created_at=datetime.now(),
                metadata={"problem": problem},
            )

        thought = _ensure_problem_metadata(thought, problem)
        thought = _trim_thought_content(thought)

        metadata = {
            "auto_selected": False,
            "hints_provided": hints is not None,
            "problem_length": len(problem),
            "sampling_available": ctx is not None,  # FastMCP v2.14+ sampling
            "combo": {
                **combo_meta,
                "pipeline_id": pipeline.id,
                "pipeline_type": str(pipeline.stage_type),
                "success": result.success,
            },
            "trace": pipeline_trace.model_dump(mode="json"),
        }
        if hints:
            metadata["hints"] = {
                "domain": hints.domain,
                "complexity": hints.complexity,
                "preferred_methods": [str(m) for m in hints.prefer_methods],
                "avoided_methods": [str(m) for m in hints.avoid_methods],
            }

        # Finalize trace if enabled
        if trace_collector and root_span_id:
            from reasoning_mcp.models.debug import SpanStatus

            trace_collector.end_span(
                root_span_id,
                status=SpanStatus.COMPLETED if result.success else SpanStatus.FAILED,
            )
            metadata["trace_id"] = trace_collector.trace_id

        return ReasonOutput(
            session_id=session.id,
            thought=thought,
            method_used=thought.method_id,
            suggestions=[
                "Continue reasoning with session_continue",
                "Explore alternative approaches with session_branch",
                "Inspect current reasoning state with session_inspect",
            ],
            metadata=metadata,
        )

    # Get settings and check if router is enabled
    settings = get_settings()

    # Get the global registry with native methods registered
    registry = _get_registry()

    # Determine which method to use
    if method is None:
        # Auto-select using the intelligent router
        if settings.enable_router:
            router = await _get_router(ctx)
            resource_budget = routing_budget or _get_default_budget()

            # Apply hints to budget if provided
            if hints:
                # If hints specify complexity, adjust budget accordingly
                if hints.complexity == "high":
                    resource_budget = ResourceBudget(
                        max_latency_ms=resource_budget.max_latency_ms,
                        max_tokens=resource_budget.max_tokens,
                        max_thoughts=resource_budget.max_thoughts,
                        max_branches=resource_budget.max_branches,
                        prefer_quality=True,
                    )
                elif hints.complexity == "low":
                    resource_budget = ResourceBudget(
                        max_latency_ms=resource_budget.max_latency_ms,
                        max_tokens=resource_budget.max_tokens,
                        max_thoughts=resource_budget.max_thoughts,
                        max_branches=resource_budget.max_branches,
                        prefer_speed=True,
                    )

            # Convert force_routing_tier to RouterTier
            force_tier = RouterTier(force_routing_tier) if force_routing_tier else None

            # Route the problem
            route_result = await router.route(problem, resource_budget, force_tier)
            route = route_result.primary_route

            # Handle different route types
            if route.route_type == RouteType.SINGLE_METHOD:
                # For SINGLE_METHOD routes, method_id is always set
                assert route.method_id is not None, "SINGLE_METHOD route must have method_id"
                method_identifier = MethodIdentifier(route.method_id)
                routing_metadata = {
                    "route_type": "single_method",
                    "router_tier": str(route.router_tier),
                    "route_score": route.score,
                    "route_confidence": route.confidence,
                    "route_reasoning": route.reasoning,
                    "routing_latency_ms": route_result.total_latency_ms,
                    "problem_analysis": {
                        "domain": str(route_result.problem_analysis.primary_domain),
                        "intent": str(route_result.problem_analysis.intent),
                        "complexity": route_result.problem_analysis.complexity,
                    },
                }
            elif route.route_type == RouteType.PIPELINE_TEMPLATE:
                # Execute as pipeline - pipeline_id is always set for PIPELINE_TEMPLATE routes
                assert route.pipeline_id is not None, (
                    "PIPELINE_TEMPLATE route must have pipeline_id"
                )
                template = get_router_template(route.pipeline_id)
                if template:
                    pipeline = _PIPELINE_ADAPTER.validate_python(template)
                    return await _execute_routed_pipeline(
                        problem=problem,
                        pipeline=pipeline,
                        route_result=route_result,
                        hints=hints,
                        routing_budget=routing_budget,
                        ctx=ctx,  # FastMCP v2.14+ Context for sampling
                        trace_collector=trace_collector,
                        root_span_id=root_span_id,
                    )
                else:
                    # Fallback to single method if template not found
                    method_identifier = MethodIdentifier.CHAIN_OF_THOUGHT
                    routing_metadata = {
                        "route_type": "pipeline_template_fallback",
                        "original_template": route.pipeline_id,
                        "fallback_reason": "Template not found",
                    }
            elif route.route_type == RouteType.SYNTHESIZED_PIPELINE:
                # Execute synthesized pipeline
                if route.pipeline_definition:
                    pipeline = _PIPELINE_ADAPTER.validate_python(route.pipeline_definition)
                    return await _execute_routed_pipeline(
                        problem=problem,
                        pipeline=pipeline,
                        route_result=route_result,
                        hints=hints,
                        routing_budget=routing_budget,
                        ctx=ctx,  # FastMCP v2.14+ Context for sampling
                        trace_collector=trace_collector,
                        root_span_id=root_span_id,
                    )
                else:
                    method_identifier = MethodIdentifier.CHAIN_OF_THOUGHT
                    routing_metadata = {
                        "route_type": "synthesized_pipeline_fallback",
                        "fallback_reason": "No pipeline definition",
                    }
            elif route.route_type == RouteType.METHOD_ENSEMBLE:
                # Execute ensemble (parallel methods with voting)
                return await _execute_routed_ensemble(
                    problem=problem,
                    route_result=route_result,
                    hints=hints,
                    routing_budget=routing_budget,
                    ctx=ctx,  # FastMCP v2.14+ Context for sampling
                    trace_collector=trace_collector,
                    root_span_id=root_span_id,
                )
            else:
                # Unknown route type, fallback
                method_identifier = MethodIdentifier.CHAIN_OF_THOUGHT
                routing_metadata = {
                    "route_type": "unknown_fallback",
                    "original_type": str(route.route_type),
                }
        else:
            # Router disabled, use legacy MethodSelector
            selector = MethodSelector(registry)

            # Convert hints to constraints if provided
            constraints = None
            if hints:
                from reasoning_mcp.selector import SelectionConstraint

                constraints = SelectionConstraint(
                    preferred_methods=frozenset(str(m) for m in hints.prefer_methods),
                    excluded_methods=frozenset(str(m) for m in hints.avoid_methods),
                )

            # Select best method
            selected_method = selector.select_best(
                problem=problem,
                constraints=constraints,
            )

            if selected_method is None:
                # Fallback to chain_of_thought if no good match
                selected_method = "chain_of_thought"
                method_identifier = MethodIdentifier.CHAIN_OF_THOUGHT
            else:
                method_identifier = MethodIdentifier(selected_method)

            routing_metadata = {
                "route_type": "legacy_selector",
                "router_enabled": False,
            }
    else:
        # Use specified method
        if not isinstance(method, str):
            raise ValueError("Method must be a string identifier or combo specification")
        method_identifier = MethodIdentifier(method)

        # Check if method is registered
        # If not, we still allow it to be used (generates placeholder thought)
        # This maintains backward compatibility during method implementation
        _method_meta = registry.get_metadata(str(method_identifier))
        # Note: Not all methods are implemented yet, so we don't raise an error
        # if the method isn't found - we just generate a placeholder thought

        routing_metadata = {
            "route_type": "explicit_method",
            "method_specified": True,
        }

    # Create a new session using SessionManager from AppContext
    session_manager = _get_session_manager()

    # Create session configuration based on trace collector session ID if available
    from reasoning_mcp.models.session import SessionConfig

    session_config = SessionConfig()
    try:
        session = await session_manager.create(config=session_config)
        # If trace collector has a specific session ID, update the session
        if trace_collector:
            # The trace collector already has its own session ID; we'll use the
            # session manager's session but include trace info in metadata
            pass
        session = session.start()
    except RuntimeError as e:
        # Max sessions limit reached - handle gracefully
        import logging

        logger = logging.getLogger(__name__)
        logger.warning(f"Could not create session via SessionManager: {e}")
        # Fall back to creating a standalone session
        if trace_collector:
            session = Session(id=trace_collector.session_id).start()
        else:
            session = Session().start()

    session_id = session.id

    # Generate initial thought by invoking the actual reasoning method
    reasoning_method = registry.get(str(method_identifier))

    if reasoning_method is not None:
        # Initialize the method if not already initialized
        if not registry.is_initialized(str(method_identifier)):
            try:
                await reasoning_method.initialize()
            except Exception as init_error:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(f"Failed to initialize method {method_identifier}: {init_error}")

        # Create ExecutionContext for the method
        execution_context = ExecutionContext(
            session=session,
            registry=registry,
            input_data={"input": problem, "problem": problem},
            ctx=ctx,  # type: ignore[arg-type]  # Cross-package Context compatibility
        )

        # Execute the reasoning method
        try:
            thought = await reasoning_method.execute(
                session=session,
                input_text=problem,
                execution_context=execution_context,
            )
            # Ensure problem metadata is attached
            thought = _ensure_problem_metadata(thought, problem)
            # Track that actual method was used
            routing_metadata["method_executed"] = True
        except Exception as exec_error:
            # Method execution failed - create a fallback thought with error info
            import logging

            logger = logging.getLogger(__name__)
            logger.error(
                f"Method {method_identifier} execution failed: {exec_error}",
                exc_info=True,
            )

            problem_preview = problem[:200]
            if len(problem) > 200:
                problem_preview += "..."

            thought = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.INITIAL,
                method_id=method_identifier,
                content=f"Initial analysis of the problem: {problem_preview}",
                confidence=0.5,
                step_number=1,
                depth=0,
                created_at=datetime.now(),
                metadata={
                    "problem": problem,
                    "auto_selected": method is None,
                    "execution_error": str(exec_error),
                },
            )
            routing_metadata["method_executed"] = False
            routing_metadata["execution_error"] = str(exec_error)
    else:
        # Method not registered - create a placeholder thought
        # This maintains backward compatibility during method implementation
        problem_preview = problem[:200]
        if len(problem) > 200:
            problem_preview += "..."

        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=method_identifier,
            content=f"Initial analysis of the problem: {problem_preview}",
            confidence=0.7,
            step_number=1,
            depth=0,
            created_at=datetime.now(),
            metadata={
                "problem": problem,
                "auto_selected": method is None,
            },
        )
        routing_metadata["method_executed"] = False
        routing_metadata["method_not_registered"] = True

    # Add thought to session
    session.add_thought(thought)

    # Update session in manager
    await session_manager.update(session_id, session)

    # Generate suggestions for next steps
    suggestions = [
        "Continue reasoning with session_continue",
        "Explore alternative approaches with session_branch",
        "Inspect current reasoning state with session_inspect",
    ]

    # Build metadata
    output_metadata: dict[str, Any] = {
        "auto_selected": method is None,
        "hints_provided": hints is not None,
        "problem_length": len(problem),
        "routing": routing_metadata,
    }

    if hints:
        output_metadata["hints"] = {
            "domain": hints.domain,
            "complexity": hints.complexity,
            "preferred_methods": [str(m) for m in hints.prefer_methods],
            "avoided_methods": [str(m) for m in hints.avoid_methods],
        }

    # Run verification if enabled
    if verify:
        try:
            from reasoning_mcp.verification.checkers import SelfConsistencyChecker
            from reasoning_mcp.verification.engine import VerificationEngine
            from reasoning_mcp.verification.extractors import RuleBasedExtractor

            # Create verification components
            extractor = RuleBasedExtractor()
            checker = SelfConsistencyChecker()

            # Create ExecutionContext for verification
            # Use trace collector's session ID if tracing is enabled
            if trace_collector:
                verification_session = Session(id=trace_collector.session_id).start()
            else:
                verification_session = Session().start()

            registry = _get_registry()
            verification_context = ExecutionContext(
                session=verification_session,
                registry=registry,
                input_data={"input": problem, "problem": problem},
                ctx=ctx,  # type: ignore[arg-type]  # Cross-package Context compatibility
            )

            # Create and run verification engine
            engine = VerificationEngine(
                extractor=extractor,
                checkers=[checker],
                ctx=verification_context,
            )

            # Verify the thought content
            verification_report = await engine.verify_text(thought.content)

            # Add verification data to metadata
            output_metadata["verification"] = {
                "overall_accuracy": verification_report.overall_accuracy,
                "claims_count": len(verification_report.claims),
                "verified_count": sum(
                    1 for r in verification_report.results if r.status.value == "verified"
                ),
                "flagged_count": len(verification_report.flagged_claims),
            }

        except Exception as e:
            # Don't block on verification failures - just log and continue
            import logging

            logger = logging.getLogger(__name__)
            logger.error(f"Verification failed: {e}", exc_info=True)
            output_metadata["verification_error"] = str(e)

    # Finalize trace if enabled
    if trace_collector and root_span_id:
        from reasoning_mcp.models.debug import SpanStatus

        trace_collector.end_span(root_span_id, status=SpanStatus.COMPLETED)
        output_metadata["trace_id"] = trace_collector.trace_id

    return ReasonOutput(
        session_id=session_id,
        thought=thought,
        method_used=method_identifier,
        suggestions=suggestions,
        metadata=output_metadata,
    )


__all__ = ["reason"]

"""Tool registration for FastMCP server.

This module provides the register_tools() function to expose all MCP tools
via the FastMCP server instance.

FastMCP v2.14+ Features Used:
- Context parameter for state management, sampling, and elicitation (v2.11+)
- Pydantic return types for automatic output schemas (v2.10+)
- Tool titles for human-readable display names (v2.10+)
- Meta parameters for tool categorization and metadata (v2.11+)
- Background task support with task=True decorator (v2.14+)
- ctx.sample() for LLM sampling in reasoning methods (v2.14+)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastmcp.server import Context
from pydantic import TypeAdapter

if TYPE_CHECKING:
    from fastmcp import FastMCP

from reasoning_mcp.models.tools import (
    BranchOutput,
    ComparisonResult,
    ComposeOutput,
    EvaluationReport,
    MergeOutput,
    MethodInfo,
    ReasonHints,
    ReasonOutput,
    Recommendation,
    SessionState,
    ThoughtOutput,
)

# Import health and cost tool types at module level for FastMCP type resolution
from reasoning_mcp.tools.cost import (
    CompareCostsInput,
    CompareCostsOutput,
    EstimateCostInput,
    EstimateCostOutput,
    GetBudgetStatusInput,
    GetBudgetStatusOutput,
    compare_costs_handler,
    estimate_cost_handler,
    get_budget_status_handler,
)
from reasoning_mcp.tools.health import (
    HealthCheckOutput,
    LivenessCheckOutput,
    ReadinessCheckOutput,
    health_check_handler,
    liveness_check_handler,
    readiness_check_handler,
)

# Tool metadata for categorization and discoverability
TOOL_META = {
    "reasoning": {"category": "reasoning", "priority": "high"},
    "methods": {"category": "discovery", "priority": "medium"},
    "session": {"category": "session", "priority": "medium"},
    "evaluation": {"category": "analysis", "priority": "low"},
}


def register_tools(mcp: FastMCP) -> None:
    """Register all MCP tools with the FastMCP server instance.

    This function registers the following tools:
    - reason: Primary reasoning tool (single method or combo pipeline)
    - reason_background: Background task version of reason
    - methods_list: List available reasoning methods
    - methods_recommend: Recommend methods for a problem
    - methods_compare: Compare methods for a problem
    - methods_compare_background: Background task version of methods_compare
    - session_continue: Continue reasoning in a session
    - session_continue_background: Background task version of session_continue
    - session_branch: Create a reasoning branch
    - session_inspect: Inspect session state
    - session_merge: Merge reasoning branches
    - compose: Execute a pipeline of reasoning methods
    - compose_background: Background task version of compose
    - evaluate: Evaluate session quality
    - ensemble_reason: Execute ensemble reasoning with multiple methods
    - list_voting_strategies: List available voting strategies for ensemble

    Args:
        mcp: The FastMCP server instance to register tools with
    """
    # Import tools here to avoid circular imports
    from reasoning_mcp.tools.compose import compose as compose_impl
    from reasoning_mcp.tools.ensemble import (
        ensemble_reason as ensemble_reason_impl,
    )
    from reasoning_mcp.tools.ensemble import (
        list_voting_strategies as list_voting_strategies_impl,
    )
    from reasoning_mcp.tools.evaluate import evaluate as evaluate_impl
    from reasoning_mcp.tools.methods import (
        methods_compare as methods_compare_impl,
    )
    from reasoning_mcp.tools.methods import (
        methods_list as methods_list_impl,
    )
    from reasoning_mcp.tools.methods import (
        methods_recommend as methods_recommend_impl,
    )
    from reasoning_mcp.tools.reason import reason as reason_impl
    from reasoning_mcp.tools.session import (
        session_branch as session_branch_impl,
    )
    from reasoning_mcp.tools.session import (
        session_continue as session_continue_impl,
    )
    from reasoning_mcp.tools.session import (
        session_inspect as session_inspect_impl,
    )
    from reasoning_mcp.tools.session import (
        session_merge as session_merge_impl,
    )

    # Register the primary reasoning tool with FastMCP v2 features
    # Note: Return type annotation provides output schema automatically (FastMCP v2.10+)
    @mcp.tool(
        title="Reason About Problem",
        description="Generate reasoning for a problem using specified or auto-selected method",
    )
    async def reason(
        ctx: Context,  # FastMCP injects automatically for sampling (v2.14+)
        problem: str,
        method: str | dict[str, Any] | None = None,
        hints: dict[str, Any] | None = None,
    ) -> ReasonOutput:
        """Generate reasoning for a problem using specified or auto-selected method.

        This is the primary tool for initiating reasoning processes. It accepts a
        problem statement and either uses the specified method or automatically
        selects the most appropriate method based on problem analysis.

        The tool supports:
        - Single method execution by ID (e.g., "chain_of_thought")
        - Combo pipeline execution via templates (e.g., "combo:debate")
        - Inline pipeline specification via dict
        - LLM sampling for advanced reasoning (when ctx available)

        Args:
            ctx: FastMCP Context (injected automatically) for sampling and state
            problem: The problem or question to reason about
            method: Optional method identifier or combo specification:
                - String like "chain_of_thought" for single method
                - String like "combo:debate" for template-based combo
                - Dict with pipeline specification for inline combo
                - None for auto-selection
            hints: Optional hints dict with keys like "domain", "complexity",
                "prefer_methods", "avoid_methods" to guide method selection

        Returns:
            ReasonOutput containing session_id, thought, method_used, suggestions, metadata
        """
        # Convert hints dict to ReasonHints if provided
        hints_obj = None
        if hints:
            from reasoning_mcp.models.core import MethodIdentifier

            prefer = [MethodIdentifier(m) for m in hints.get("prefer_methods", [])]
            avoid = [MethodIdentifier(m) for m in hints.get("avoid_methods", [])]
            hints_obj = ReasonHints(
                domain=hints.get("domain"),
                complexity=hints.get("complexity"),
                prefer_methods=prefer,
                avoid_methods=avoid,
                custom_hints=hints.get("custom_hints", {}),
            )

        return await reason_impl(problem, method=method, hints=hints_obj, ctx=ctx)  # type: ignore[arg-type]

    # Register background reasoning tool (FastMCP v2.14+ background task)
    @mcp.tool(
        task=True,
        title="Deep Reasoning (Background)",
        description=(
            "Long-running reasoning operation that runs in the background. "
            "Returns a task ID for polling."
        ),
    )
    async def reason_background(
        problem: str,
        method: str | dict[str, Any] | None = None,
        hints: dict[str, Any] | None = None,
    ) -> ReasonOutput:
        """Long-running reasoning operation as a background task.

        This tool is identical to `reason` but runs as a background task
        (FastMCP v2.14+ feature). It immediately returns a task ID that can
        be used to poll for results using the task_progress and task_result tools.

        Use this for:
        - Complex problems that may take a long time
        - When you want to continue other work while reasoning
        - Multi-method pipelines that execute sequentially

        Args:
            problem: The problem or question to reason about
            method: Optional method identifier or combo specification:
                - String like "chain_of_thought" for single method
                - String like "combo:debate" for template-based combo
                - Dict with pipeline specification for inline combo
                - None for auto-selection
            hints: Optional hints dict with keys like "domain", "complexity",
                "prefer_methods", "avoid_methods" to guide method selection

        Returns:
            ReasonOutput containing session_id, thought, method_used, suggestions, metadata
            (accessed via task_result after completion)
        """
        # Convert hints dict to ReasonHints if provided
        hints_obj = None
        if hints:
            from reasoning_mcp.models.core import MethodIdentifier

            prefer = [MethodIdentifier(m) for m in hints.get("prefer_methods", [])]
            avoid = [MethodIdentifier(m) for m in hints.get("avoid_methods", [])]
            hints_obj = ReasonHints(
                domain=hints.get("domain"),
                complexity=hints.get("complexity"),
                prefer_methods=prefer,
                avoid_methods=avoid,
                custom_hints=hints.get("custom_hints", {}),
            )

        return await reason_impl(problem, method=method, hints=hints_obj)

    # Register method discovery tools with FastMCP v2 features
    @mcp.tool(
        title="List Reasoning Methods",
        description="List available reasoning methods with optional filtering by category or tags",
    )
    def methods_list(
        category: str | None = None,
        tags: list[str] | None = None,
    ) -> list[MethodInfo]:
        """List available reasoning methods with optional filtering.

        Args:
            category: Optional category filter (e.g., "core", "high_value")
            tags: Optional list of tags to filter by (methods must have ALL tags)

        Returns:
            List of MethodInfo with id, name, description, category, parameters
        """
        return methods_list_impl(category=category, tags=tags)

    @mcp.tool(
        title="Recommend Reasoning Methods",
        description="Analyze a problem and recommend the most suitable reasoning methods",
    )
    def methods_recommend(
        problem: str,
        max_results: int = 3,
    ) -> list[Recommendation]:
        """Recommend reasoning methods for a specific problem.

        Analyzes the problem and returns the most suitable methods based on
        detected patterns, problem type, and complexity.

        Args:
            problem: The problem description to analyze
            max_results: Maximum number of recommendations (default 3)

        Returns:
            List of Recommendation with method_id, score, reason, confidence
        """
        return methods_recommend_impl(problem=problem, max_results=max_results)

    @mcp.tool(
        title="Compare Reasoning Methods",
        description="Compare multiple reasoning methods for a specific problem",
    )
    def methods_compare(
        methods: list[str],
        problem: str,
    ) -> ComparisonResult:
        """Compare multiple reasoning methods for a specific problem.

        Args:
            methods: List of method identifiers to compare
            problem: The problem to evaluate methods against

        Returns:
            ComparisonResult with winner, scores, and analysis
        """
        return methods_compare_impl(methods=methods, problem=problem)

    @mcp.tool(
        task=True,
        title="Compare Methods (Background)",
        description=(
            "Long-running method comparison that runs in the background. "
            "Returns a task ID for polling."
        ),
    )
    async def methods_compare_background(
        methods: list[str],
        problem: str,
    ) -> ComparisonResult:
        """Long-running method comparison as a background task.

        This tool is identical to `methods_compare` but runs as a background task
        (FastMCP v2.14+ feature). It immediately returns a task ID that can
        be used to poll for results using the task_progress and task_result tools.

        Use this for:
        - Comparing multiple methods that may take time to analyze
        - When you want to continue other work while comparison runs
        - Large-scale method evaluations

        Args:
            methods: List of method identifiers to compare
            problem: The problem to evaluate methods against

        Returns:
            ComparisonResult with winner, scores, and analysis
            (accessed via task_result after completion)
        """
        return methods_compare_impl(methods=methods, problem=problem)

    # Register session management tools with FastMCP v2 features
    @mcp.tool(
        title="Continue Reasoning Session",
        description="Continue reasoning in an existing session with optional guidance",
    )
    async def session_continue(
        session_id: str,
        guidance: str | None = None,
    ) -> ThoughtOutput:
        """Continue reasoning in an existing session.

        Args:
            session_id: UUID of the session to continue
            guidance: Optional guidance for the next reasoning step

        Returns:
            ThoughtOutput with id, content, type, confidence, step_number
        """
        return await session_continue_impl(session_id=session_id, guidance=guidance)

    @mcp.tool(
        task=True,
        title="Continue Session (Background)",
        description=(
            "Long-running session continuation that runs in the background. "
            "Returns a task ID for polling."
        ),
    )
    async def session_continue_background(
        session_id: str,
        guidance: str | None = None,
    ) -> ThoughtOutput:
        """Long-running session continuation as a background task.

        This tool is identical to `session_continue` but runs as a background task
        (FastMCP v2.14+ feature). It immediately returns a task ID that can
        be used to poll for results using the task_progress and task_result tools.

        Use this for:
        - Complex sessions that may take a long time to continue
        - When you want to continue other work while reasoning progresses
        - Multi-step reasoning that requires substantial computation

        Args:
            session_id: UUID of the session to continue
            guidance: Optional guidance for the next reasoning step

        Returns:
            ThoughtOutput with id, content, type, confidence, step_number
            (accessed via task_result after completion)
        """
        return await session_continue_impl(session_id=session_id, guidance=guidance)

    @mcp.tool(
        title="Branch Reasoning Session",
        description="Create a new branch in the reasoning session for exploring alternatives",
    )
    async def session_branch(
        session_id: str,
        branch_name: str,
        from_thought_id: str | None = None,
    ) -> BranchOutput:
        """Create a new branch in the reasoning session.

        Args:
            session_id: UUID of the session to branch
            branch_name: Human-readable name for the branch
            from_thought_id: Optional thought ID to branch from

        Returns:
            BranchOutput with branch_id, parent_thought_id, session_id, success
        """
        return await session_branch_impl(
            session_id=session_id,
            branch_name=branch_name,
            from_thought_id=from_thought_id,
        )

    @mcp.tool(
        title="Inspect Reasoning Session",
        description="Inspect the current state of a reasoning session",
    )
    async def session_inspect(
        session_id: str,
        include_graph: bool = False,
    ) -> SessionState:
        """Inspect the current state of a reasoning session.

        Args:
            session_id: UUID of the session to inspect
            include_graph: If True, includes graph visualization data

        Returns:
            SessionState with status, thought_count, branch_count, timestamps
        """
        return await session_inspect_impl(
            session_id=session_id,
            include_graph=include_graph,
        )

    @mcp.tool(
        title="Merge Reasoning Branches",
        description="Merge branches in a reasoning session using configurable strategies",
    )
    async def session_merge(
        session_id: str,
        source_branch: str,
        target_branch: str,
        strategy: str = "latest",
    ) -> MergeOutput:
        """Merge branches in a reasoning session.

        Args:
            session_id: UUID of the session containing the branches
            source_branch: Branch to merge from
            target_branch: Branch to merge into
            strategy: Merge strategy (latest, highest_confidence, synthesis, sequential)

        Returns:
            MergeOutput with merged_thought_id, source_branch_ids, success
        """
        return await session_merge_impl(
            session_id=session_id,
            source_branch=source_branch,
            target_branch=target_branch,
            strategy=strategy,
        )

    # Register pipeline composition tools with FastMCP v2 features
    @mcp.tool(
        title="Compose Reasoning Pipeline",
        description="Execute a pipeline of reasoning methods with various control flows",
    )
    async def compose(
        pipeline: dict[str, Any],
        input: str,
        session_id: str | None = None,
    ) -> ComposeOutput:
        """Execute a pipeline of reasoning methods.

        Args:
            pipeline: Pipeline specification as a dictionary
            input: The input text/problem to reason about
            session_id: Optional session ID to use

        Returns:
            ComposeOutput with session_id, pipeline_id, success, final_thoughts, trace
        """
        from reasoning_mcp.models.pipeline import Pipeline

        # Convert dict to Pipeline model using TypeAdapter for discriminated union
        pipeline_adapter: TypeAdapter[Pipeline] = TypeAdapter(Pipeline)
        pipeline_obj = pipeline_adapter.validate_python(pipeline)
        return await compose_impl(pipeline=pipeline_obj, input=input, session_id=session_id)

    @mcp.tool(
        task=True,
        title="Compose Reasoning (Background)",
        description=(
            "Long-running pipeline composition that runs in the background. "
            "Returns a task ID for polling."
        ),
    )
    async def compose_background(
        pipeline: dict[str, Any],
        input: str,
        session_id: str | None = None,
    ) -> ComposeOutput:
        """Long-running pipeline composition as a background task.

        This tool is identical to `compose` but runs as a background task
        (FastMCP v2.14+ feature). It immediately returns a task ID that can
        be used to poll for results using the task_progress and task_result tools.

        Use this for:
        - Complex pipelines with multiple reasoning methods
        - Long-running multi-step reasoning workflows
        - When you want to continue other work while the pipeline executes

        Args:
            pipeline: Pipeline specification as a dictionary
            input: The input text/problem to reason about
            session_id: Optional session ID to use

        Returns:
            ComposeOutput with session_id, pipeline_id, success, final_thoughts, trace
            (accessed via task_result after completion)
        """
        from reasoning_mcp.models.pipeline import Pipeline

        # Convert dict to Pipeline model using TypeAdapter for discriminated union
        pipeline_adapter: TypeAdapter[Pipeline] = TypeAdapter(Pipeline)
        pipeline_obj = pipeline_adapter.validate_python(pipeline)
        return await compose_impl(pipeline=pipeline_obj, input=input, session_id=session_id)

    # Register evaluation tool with FastMCP v2 features
    @mcp.tool(
        title="Evaluate Reasoning Session",
        description="Evaluate the quality and coherence of a reasoning session",
    )
    async def evaluate(
        session_id: str,
        criteria: list[str] | None = None,
    ) -> EvaluationReport:
        """Evaluate the quality of a reasoning session.

        Analyzes coherence, depth, and coverage of the reasoning process.

        Args:
            session_id: UUID of the session to evaluate
            criteria: Optional list of specific criteria to focus on

        Returns:
            EvaluationReport with scores, insights, and recommendations
        """
        return await evaluate_impl(session_id=session_id, criteria=criteria)

    # Register ensemble reasoning tools with FastMCP v2 features
    @mcp.tool(
        title="Ensemble Reasoning",
        description="Execute multiple reasoning methods in parallel and aggregate results",
    )
    async def ensemble_reason(
        query: str,
        methods: list[str] | None = None,
        strategy: str = "majority",
        weights: dict[str, float] | None = None,
        timeout_ms: int = 30000,
    ) -> dict[str, Any]:
        """Execute ensemble reasoning with multiple methods.

        Runs multiple reasoning methods in parallel and aggregates their results
        using the specified voting strategy. This provides more robust reasoning
        by combining diverse approaches.

        Args:
            query: The query to reason about
            methods: List of method names to include (uses defaults if None)
            strategy: Voting strategy for aggregation (majority, weighted, consensus,
                best_score, synthesis, ranked_choice, borda_count)
            weights: Optional weights per method name (only used with weighted strategy)
            timeout_ms: Timeout in milliseconds for ensemble execution

        Returns:
            EnsembleResult dict with final_answer, confidence, agreement_score,
            member_results, and voting_details
        """
        from reasoning_mcp.models.ensemble import VotingStrategy
        from reasoning_mcp.tools.ensemble import EnsembleToolInput

        # Convert strategy string to enum
        strategy_enum = VotingStrategy(strategy)

        input_data = EnsembleToolInput(
            query=query,
            methods=methods,
            strategy=strategy_enum,
            weights=weights,
            timeout_ms=timeout_ms,
        )

        result = await ensemble_reason_impl(input_data)
        # Convert to dict for MCP response
        return result.model_dump(mode="json")

    @mcp.tool(
        title="List Voting Strategies",
        description="List all available voting strategies for ensemble reasoning",
    )
    def list_voting_strategies() -> list[dict[str, str]]:
        """List all available voting strategies with descriptions.

        Returns a list of voting strategies that can be used with ensemble_reason,
        along with their descriptions and recommended use cases.

        Returns:
            List of strategy info dicts with 'name', 'value', 'description'
        """
        return list_voting_strategies_impl()

    # Register health check tools with FastMCP v2 features
    @mcp.tool(
        title="Health Check",
        description="Check server health status and component health",
    )
    async def health() -> HealthCheckOutput:
        """Perform comprehensive health check.

        Checks all server components (registry, session manager, persistence,
        middleware, cache) and returns their status.

        Returns:
            HealthCheckOutput with overall status, version, and component details
        """
        from reasoning_mcp.server import get_app_context

        ctx = get_app_context()
        return await health_check_handler(ctx)

    @mcp.tool(
        title="Readiness Check",
        description="Check if server is ready to accept requests",
    )
    async def ready() -> ReadinessCheckOutput:
        """Check if server is ready to accept requests.

        Verifies that all critical components are initialized and operational.
        Use this as a readiness probe in Kubernetes or similar environments.

        Returns:
            ReadinessCheckOutput with ready status and component readiness
        """
        from reasoning_mcp.server import get_app_context

        ctx = get_app_context()
        return await readiness_check_handler(ctx)

    @mcp.tool(
        title="Liveness Check",
        description="Check if server is alive",
    )
    async def alive() -> LivenessCheckOutput:
        """Check if server is alive.

        Simple liveness probe that returns quickly. Use this as a liveness
        probe in Kubernetes or similar environments.

        Returns:
            LivenessCheckOutput with alive status
        """
        from reasoning_mcp.server import get_app_context

        ctx = get_app_context()
        return await liveness_check_handler(ctx)

    # Register cost management tools with FastMCP v2 features
    @mcp.tool(
        title="Estimate Cost",
        description="Estimate the cost of a reasoning operation before execution",
    )
    def estimate_cost(
        method: str,
        input_text: str,
        model_id: str | None = None,
    ) -> EstimateCostOutput:
        """Estimate the cost of a reasoning operation.

        Provides token and cost estimates before executing an operation,
        allowing for better resource planning and budget management.

        Args:
            method: Reasoning method to estimate cost for (or "auto")
            input_text: Input text/prompt for the operation
            model_id: Model to use (defaults to configured model)

        Returns:
            EstimateCostOutput with token and cost estimates
        """
        from reasoning_mcp.cost.calculator import CostCalculator

        calculator = CostCalculator()
        input_data = EstimateCostInput(
            method=method,
            input_text=input_text,
            model_id=model_id,
        )
        return estimate_cost_handler(input_data, calculator)

    @mcp.tool(
        title="Get Budget Status",
        description="Get current budget usage and remaining budget",
    )
    def budget_status(
        session_id: str | None = None,
    ) -> GetBudgetStatusOutput:
        """Get current budget status.

        Returns the current budget usage including spent amount, tokens used,
        requests made, and remaining budget if configured.

        Args:
            session_id: Optional session ID to get status for

        Returns:
            GetBudgetStatusOutput with budget usage and remaining amounts
        """
        from reasoning_mcp.server import get_app_context

        ctx = get_app_context()

        # Get enforcer from shared state if configured
        enforcer = ctx.shared_state.get("budget_enforcer")

        input_data = GetBudgetStatusInput(session_id=session_id)
        return get_budget_status_handler(input_data, enforcer)

    @mcp.tool(
        title="Compare Method Costs",
        description="Compare the costs of different reasoning methods",
    )
    def compare_costs(
        input_text: str,
        methods: list[str] | None = None,
        model_id: str | None = None,
        top_n: int = 5,
    ) -> CompareCostsOutput:
        """Compare the costs of different reasoning methods.

        Estimates and ranks methods by cost for the given input, helping
        choose the most cost-effective approach.

        Args:
            input_text: Input text/prompt to estimate costs for
            methods: Methods to compare (defaults to all available)
            model_id: Model to use for estimation
            top_n: Number of results to return (default 5)

        Returns:
            CompareCostsOutput with ranked cost comparisons
        """
        from reasoning_mcp.cost.calculator import CostCalculator

        calculator = CostCalculator()
        input_data = CompareCostsInput(
            input_text=input_text,
            methods=methods,
            model_id=model_id,
            top_n=top_n,
        )
        return compare_costs_handler(input_data, calculator)


__all__ = ["register_tools"]

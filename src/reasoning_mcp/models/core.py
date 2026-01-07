"""Core enumerations for reasoning-mcp.

This module defines all core enums used throughout the reasoning-mcp framework,
including reasoning method identifiers, categories, thought types, session states,
and pipeline stage types.
"""

from enum import StrEnum


class MethodIdentifier(StrEnum):
    """Identifiers for all supported reasoning methods.

    This enum contains all 30 reasoning methods supported by reasoning-mcp,
    organized into five categories: Core, High-Value, Specialized, Advanced,
    and Holistic methods.
    """

    # Core Methods (5)
    SEQUENTIAL_THINKING = "sequential_thinking"
    """Basic step-by-step reasoning with explicit thought progression."""

    CHAIN_OF_THOUGHT = "chain_of_thought"
    """Classic step-by-step reasoning with intermediate steps shown."""

    TREE_OF_THOUGHTS = "tree_of_thoughts"
    """Explore multiple reasoning paths in a tree structure."""

    REACT = "react"
    """Reasoning and Acting - interleave thoughts with actions."""

    SELF_CONSISTENCY = "self_consistency"
    """Generate multiple reasoning paths and select the most consistent answer."""

    # High-Value Methods (5)
    ETHICAL_REASONING = "ethical_reasoning"
    """Structured ethical analysis with principles and stakeholder consideration."""

    CODE_REASONING = "code_reasoning"
    """Specialized reasoning for code analysis, debugging, and development."""

    DIALECTIC = "dialectic"
    """Thesis-antithesis-synthesis reasoning for balanced analysis."""

    SHANNON_THINKING = "shannon_thinking"
    """Information-theoretic reasoning with entropy and uncertainty analysis."""

    SELF_REFLECTION = "self_reflection"
    """Metacognitive reasoning with self-critique and improvement."""

    # Specialized Methods (10)
    GRAPH_OF_THOUGHTS = "graph_of_thoughts"
    """Graph-based reasoning with nodes and edges representing thought relationships."""

    MCTS = "mcts"
    """Monte Carlo Tree Search for decision-making with exploration-exploitation."""

    SKELETON_OF_THOUGHT = "skeleton_of_thought"
    """Create high-level skeleton first, then fill in details."""

    LEAST_TO_MOST = "least_to_most"
    """Break problem into subproblems, solve from simplest to most complex."""

    STEP_BACK = "step_back"
    """Step back to consider higher-level concepts before solving details."""

    SELF_ASK = "self_ask"
    """Decompose questions into subquestions and answer them iteratively."""

    DECOMPOSED_PROMPTING = "decomposed_prompting"
    """Break complex tasks into smaller, manageable subtasks."""

    MATHEMATICAL_REASONING = "mathematical_reasoning"
    """Formal mathematical reasoning with proofs and symbolic manipulation."""

    ABDUCTIVE = "abductive"
    """Inference to the best explanation from observations."""

    ANALOGICAL = "analogical"
    """Reasoning by analogy to similar problems or situations."""

    # Advanced Methods (5)
    CAUSAL_REASONING = "causal_reasoning"
    """Analyze cause-effect relationships and causal chains."""

    SOCRATIC = "socratic"
    """Question-driven reasoning to uncover assumptions and deepen understanding."""

    COUNTERFACTUAL = "counterfactual"
    """Explore alternative scenarios and what-if reasoning."""

    METACOGNITIVE = "metacognitive"
    """Thinking about thinking - analyze and optimize reasoning processes."""

    BEAM_SEARCH = "beam_search"
    """Maintain multiple promising reasoning paths simultaneously."""

    # Holistic Methods (5)
    LATERAL_THINKING = "lateral_thinking"
    """Creative, non-linear reasoning to find novel solutions."""

    LOTUS_WISDOM = "lotus_wisdom"
    """Inspired by Buddhist philosophy - layered insights and interconnected understanding."""

    ATOM_OF_THOUGHTS = "atom_of_thoughts"
    """Atomic, composable thought units that can be recombined."""

    CASCADE_THINKING = "cascade_thinking"
    """Progressive refinement through cascading reasoning stages."""

    CRASH = "crash"
    """Compact Reasoning And Self-correction Heuristic - iterative self-correction."""


class MethodCategory(StrEnum):
    """Categories for organizing reasoning methods.

    Methods are organized into five categories based on their complexity,
    use cases, and implementation requirements.
    """

    CORE = "core"
    """Essential foundational methods - simple, widely applicable, high ROI."""

    HIGH_VALUE = "high_value"
    """Valuable specialized methods for specific domains and use cases."""

    SPECIALIZED = "specialized"
    """Advanced methods for specific reasoning patterns and problem types."""

    ADVANCED = "advanced"
    """Sophisticated methods requiring complex implementation."""

    HOLISTIC = "holistic"
    """Creative and philosophical methods for novel problem-solving."""


class ThoughtType(StrEnum):
    """Types of thoughts in a reasoning process.

    Different types of thoughts serve different purposes in the reasoning flow,
    from initial analysis through to final conclusions.
    """

    INITIAL = "initial"
    """First thought in a reasoning chain - sets up the problem."""

    CONTINUATION = "continuation"
    """Continues the current line of reasoning."""

    BRANCH = "branch"
    """Creates a new branch in tree/graph-based reasoning."""

    REVISION = "revision"
    """Revises or corrects a previous thought."""

    SYNTHESIS = "synthesis"
    """Combines multiple thoughts or paths into a unified insight."""

    CONCLUSION = "conclusion"
    """Final thought that provides the answer or solution."""

    HYPOTHESIS = "hypothesis"
    """Proposes a potential explanation or solution to test."""

    VERIFICATION = "verification"
    """Checks or validates a hypothesis or previous thought."""

    OBSERVATION = "observation"
    """Records an observation or fact (used in ReAct and similar methods)."""

    ACTION = "action"
    """Represents an action taken (used in ReAct and similar methods)."""

    REASONING = "reasoning"
    """Explicit reasoning step analyzing information (used in ReAct)."""

    EXPLORATION = "exploration"
    """Exploratory thought for discovering new approaches (used in Lateral Thinking)."""

    INSIGHT = "insight"
    """Key insight or realization discovered during reasoning (used in Lotus Wisdom)."""


class SessionStatus(StrEnum):
    """Status of a reasoning session.

    Tracks the lifecycle of a reasoning session from creation
    through active processing to completion or termination.
    """

    CREATED = "created"
    """Session has been created but not yet started."""

    ACTIVE = "active"
    """Session is currently active and processing."""

    PAUSED = "paused"
    """Session is temporarily paused and can be resumed."""

    COMPLETED = "completed"
    """Session completed successfully."""

    FAILED = "failed"
    """Session failed due to an error."""

    CANCELLED = "cancelled"
    """Session was cancelled by the user or system."""


class PipelineStageType(StrEnum):
    """Types of stages in a reasoning pipeline.

    Pipelines are composed of different stage types that control the flow
    of reasoning methods and their execution order.
    """

    METHOD = "method"
    """Execute a single reasoning method."""

    SEQUENCE = "sequence"
    """Execute multiple stages in sequential order."""

    PARALLEL = "parallel"
    """Execute multiple stages in parallel."""

    CONDITIONAL = "conditional"
    """Execute stages based on conditional logic."""

    LOOP = "loop"
    """Repeat stages in a loop until a condition is met."""

    SWITCH = "switch"
    """Select one of multiple execution paths based on a value."""

"""Native reasoning method implementations.

This package contains all built-in reasoning methods provided by reasoning-mcp.
Methods are organized into waves based on implementation priority:

Wave 11.1 (Core): sequential, chain_of_thought, tree_of_thoughts, react,
                   self_consistency, ethical, code
Wave 11.2 (High-Value): dialectic, shannon, self_reflection, graph_of_thoughts,
                         mcts, skeleton, least_to_most, step_back
Wave 11.3 (Specialized): self_ask, decomposed, mathematical, abductive,
                          analogical, causal, socratic, counterfactual
Wave 11.4 (Advanced/Holistic): metacognitive, beam_search, lateral, lotus_wisdom,
                                atom_of_thoughts, cascade, crash
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reasoning_mcp.registry import MethodRegistry

logger = logging.getLogger(__name__)

# Import method classes and metadata with graceful fallback
# This allows the module to load even if some implementations are not yet complete

# Wave 11.1: Core Methods
try:
    from reasoning_mcp.methods.native.sequential import (
        SequentialThinking,
        SEQUENTIAL_METADATA,
    )
except ImportError as e:
    logger.debug(f"Sequential method not yet implemented: {e}")
    SequentialThinking = None  # type: ignore
    SEQUENTIAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.chain_of_thought import (
        ChainOfThought,
        CHAIN_OF_THOUGHT_METADATA,
    )
except ImportError as e:
    logger.debug(f"Chain of Thought method not yet implemented: {e}")
    ChainOfThought = None  # type: ignore
    CHAIN_OF_THOUGHT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.tree_of_thoughts import (
        TreeOfThoughts,
        TREE_OF_THOUGHTS_METADATA,
    )
except ImportError as e:
    logger.debug(f"Tree of Thoughts method not yet implemented: {e}")
    TreeOfThoughts = None  # type: ignore
    TREE_OF_THOUGHTS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.react import ReActMethod, REACT_METADATA
except ImportError as e:
    logger.debug(f"ReAct method not yet implemented: {e}")
    ReActMethod = None  # type: ignore
    REACT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.self_consistency import (
        SelfConsistency,
        SELF_CONSISTENCY_METADATA,
    )
except ImportError as e:
    logger.debug(f"Self-Consistency method not yet implemented: {e}")
    SelfConsistency = None  # type: ignore
    SELF_CONSISTENCY_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.ethical import (
        EthicalReasoning,
        ETHICAL_METADATA,
    )
except ImportError as e:
    logger.debug(f"Ethical Reasoning method not yet implemented: {e}")
    EthicalReasoning = None  # type: ignore
    ETHICAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.code import CodeReasoning, CODE_METADATA
except ImportError as e:
    logger.debug(f"Code Reasoning method not yet implemented: {e}")
    CodeReasoning = None  # type: ignore
    CODE_METADATA = None  # type: ignore

# Wave 11.2: High-Value Methods
try:
    from reasoning_mcp.methods.native.dialectic import (
        Dialectic,
        DIALECTIC_METADATA,
    )
except ImportError as e:
    logger.debug(f"Dialectic method not yet implemented: {e}")
    Dialectic = None  # type: ignore
    DIALECTIC_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.shannon import (
        ShannonThinking,
        SHANNON_METADATA,
    )
except ImportError as e:
    logger.debug(f"Shannon Thinking method not yet implemented: {e}")
    ShannonThinking = None  # type: ignore
    SHANNON_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.self_reflection import (
        SelfReflection,
        SELF_REFLECTION_METADATA,
    )
except ImportError as e:
    logger.debug(f"Self-Reflection method not yet implemented: {e}")
    SelfReflection = None  # type: ignore
    SELF_REFLECTION_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.graph_of_thoughts import (
        GraphOfThoughts,
        GRAPH_OF_THOUGHTS_METADATA,
    )
except ImportError as e:
    logger.debug(f"Graph of Thoughts method not yet implemented: {e}")
    GraphOfThoughts = None  # type: ignore
    GRAPH_OF_THOUGHTS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.mcts import MCTS, MCTS_METADATA
except ImportError as e:
    logger.debug(f"MCTS method not yet implemented: {e}")
    MCTS = None  # type: ignore
    MCTS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.skeleton import (
        SkeletonOfThought,
        SKELETON_METADATA,
    )
except ImportError as e:
    logger.debug(f"Skeleton of Thought method not yet implemented: {e}")
    SkeletonOfThought = None  # type: ignore
    SKELETON_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.least_to_most import (
        LeastToMost,
        LEAST_TO_MOST_METADATA,
    )
except ImportError as e:
    logger.debug(f"Least-to-Most method not yet implemented: {e}")
    LeastToMost = None  # type: ignore
    LEAST_TO_MOST_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.step_back import (
        StepBack,
        STEP_BACK_METADATA,
    )
except ImportError as e:
    logger.debug(f"Step-Back method not yet implemented: {e}")
    StepBack = None  # type: ignore
    STEP_BACK_METADATA = None  # type: ignore

# Wave 11.3: Specialized Methods
try:
    from reasoning_mcp.methods.native.self_ask import SelfAsk, SELF_ASK_METADATA
except ImportError as e:
    logger.debug(f"Self-Ask method not yet implemented: {e}")
    SelfAsk = None  # type: ignore
    SELF_ASK_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.decomposed import (
        DecomposedPrompting,
        DECOMPOSED_METADATA,
    )
except ImportError as e:
    logger.debug(f"Decomposed Prompting method not yet implemented: {e}")
    DecomposedPrompting = None  # type: ignore
    DECOMPOSED_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.mathematical import (
        MathematicalReasoning,
        MATHEMATICAL_METADATA,
    )
except ImportError as e:
    logger.debug(f"Mathematical Reasoning method not yet implemented: {e}")
    MathematicalReasoning = None  # type: ignore
    MATHEMATICAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.abductive import (
        Abductive,
        ABDUCTIVE_METADATA,
    )
except ImportError as e:
    logger.debug(f"Abductive method not yet implemented: {e}")
    Abductive = None  # type: ignore
    ABDUCTIVE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.analogical import (
        Analogical,
        ANALOGICAL_METADATA,
    )
except ImportError as e:
    logger.debug(f"Analogical method not yet implemented: {e}")
    Analogical = None  # type: ignore
    ANALOGICAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.causal import (
        CausalReasoning,
        CAUSAL_METADATA,
    )
except ImportError as e:
    logger.debug(f"Causal Reasoning method not yet implemented: {e}")
    CausalReasoning = None  # type: ignore
    CAUSAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.socratic import (
        Socratic,
        SOCRATIC_METADATA,
    )
except ImportError as e:
    logger.debug(f"Socratic method not yet implemented: {e}")
    Socratic = None  # type: ignore
    SOCRATIC_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.counterfactual import (
        Counterfactual,
        COUNTERFACTUAL_METADATA,
    )
except ImportError as e:
    logger.debug(f"Counterfactual method not yet implemented: {e}")
    Counterfactual = None  # type: ignore
    COUNTERFACTUAL_METADATA = None  # type: ignore

# Wave 11.4: Advanced & Holistic Methods
try:
    from reasoning_mcp.methods.native.metacognitive import (
        MetacognitiveMethod,
        METACOGNITIVE_METADATA,
    )
except ImportError as e:
    logger.debug(f"Metacognitive method not yet implemented: {e}")
    MetacognitiveMethod = None  # type: ignore
    METACOGNITIVE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.beam_search import (
        BeamSearchMethod,
        BEAM_SEARCH_METADATA,
    )
except ImportError as e:
    logger.debug(f"Beam Search method not yet implemented: {e}")
    BeamSearchMethod = None  # type: ignore
    BEAM_SEARCH_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.lateral import (
        LateralThinkingMethod,
        LATERAL_METADATA,
    )
except ImportError as e:
    logger.debug(f"Lateral Thinking method not yet implemented: {e}")
    LateralThinkingMethod = None  # type: ignore
    LATERAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.lotus_wisdom import (
        LotusWisdomMethod,
        LOTUS_WISDOM_METADATA,
    )
except ImportError as e:
    logger.debug(f"Lotus Wisdom method not yet implemented: {e}")
    LotusWisdomMethod = None  # type: ignore
    LOTUS_WISDOM_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.atom_of_thoughts import (
        AtomOfThoughtsMethod,
        ATOM_OF_THOUGHTS_METADATA,
    )
except ImportError as e:
    logger.debug(f"Atom of Thoughts method not yet implemented: {e}")
    AtomOfThoughtsMethod = None  # type: ignore
    ATOM_OF_THOUGHTS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.cascade import (
        CascadeThinkingMethod,
        CASCADE_METADATA,
    )
except ImportError as e:
    logger.debug(f"Cascade Thinking method not yet implemented: {e}")
    CascadeThinkingMethod = None  # type: ignore
    CASCADE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.crash import CRASHMethod, CRASH_METADATA
except ImportError as e:
    logger.debug(f"CRASH method not yet implemented: {e}")
    CRASHMethod = None  # type: ignore
    CRASH_METADATA = None  # type: ignore


# Consolidated lists of all method classes and metadata
ALL_NATIVE_METHODS = [
    # Wave 11.1: Core Methods
    SequentialThinking,
    ChainOfThought,
    TreeOfThoughts,
    ReActMethod,
    SelfConsistency,
    EthicalReasoning,
    CodeReasoning,
    # Wave 11.2: High-Value Methods
    Dialectic,
    ShannonThinking,
    SelfReflection,
    GraphOfThoughts,
    MCTS,
    SkeletonOfThought,
    LeastToMost,
    StepBack,
    # Wave 11.3: Specialized Methods
    SelfAsk,
    DecomposedPrompting,
    MathematicalReasoning,
    Abductive,
    Analogical,
    CausalReasoning,
    Socratic,
    Counterfactual,
    # Wave 11.4: Advanced & Holistic Methods
    MetacognitiveMethod,
    BeamSearchMethod,
    LateralThinkingMethod,
    LotusWisdomMethod,
    AtomOfThoughtsMethod,
    CascadeThinkingMethod,
    CRASHMethod,
]

ALL_NATIVE_METADATA = [
    # Wave 11.1: Core Methods
    SEQUENTIAL_METADATA,
    CHAIN_OF_THOUGHT_METADATA,
    TREE_OF_THOUGHTS_METADATA,
    REACT_METADATA,
    SELF_CONSISTENCY_METADATA,
    ETHICAL_METADATA,
    CODE_METADATA,
    # Wave 11.2: High-Value Methods
    DIALECTIC_METADATA,
    SHANNON_METADATA,
    SELF_REFLECTION_METADATA,
    GRAPH_OF_THOUGHTS_METADATA,
    MCTS_METADATA,
    SKELETON_METADATA,
    LEAST_TO_MOST_METADATA,
    STEP_BACK_METADATA,
    # Wave 11.3: Specialized Methods
    SELF_ASK_METADATA,
    DECOMPOSED_METADATA,
    MATHEMATICAL_METADATA,
    ABDUCTIVE_METADATA,
    ANALOGICAL_METADATA,
    CAUSAL_METADATA,
    SOCRATIC_METADATA,
    COUNTERFACTUAL_METADATA,
    # Wave 11.4: Advanced & Holistic Methods
    METACOGNITIVE_METADATA,
    BEAM_SEARCH_METADATA,
    LATERAL_METADATA,
    LOTUS_WISDOM_METADATA,
    ATOM_OF_THOUGHTS_METADATA,
    CASCADE_METADATA,
    CRASH_METADATA,
]


def register_all_native_methods(registry: MethodRegistry) -> dict[str, bool]:
    """Register all native reasoning methods with the registry.

    This function attempts to register all 30 native methods. If a method
    implementation is not yet available, it logs a warning and continues
    with the next method.

    Args:
        registry: The MethodRegistry instance to register methods with

    Returns:
        Dict mapping method identifiers to registration success status
        (True if registered, False if skipped due to missing implementation)

    Example:
        >>> from reasoning_mcp.registry import MethodRegistry
        >>> from reasoning_mcp.methods.native import register_all_native_methods
        >>> registry = MethodRegistry()
        >>> results = register_all_native_methods(registry)
        >>> print(f"Registered {sum(results.values())} of {len(results)} methods")
    """
    results: dict[str, bool] = {}

    # List of (class, metadata, identifier) tuples for all methods
    methods_to_register = [
        # Wave 11.1: Core Methods
        (SequentialThinking, SEQUENTIAL_METADATA, "sequential_thinking"),
        (ChainOfThought, CHAIN_OF_THOUGHT_METADATA, "chain_of_thought"),
        (TreeOfThoughts, TREE_OF_THOUGHTS_METADATA, "tree_of_thoughts"),
        (ReActMethod, REACT_METADATA, "react"),
        (SelfConsistency, SELF_CONSISTENCY_METADATA, "self_consistency"),
        (EthicalReasoning, ETHICAL_METADATA, "ethical_reasoning"),
        (CodeReasoning, CODE_METADATA, "code_reasoning"),
        # Wave 11.2: High-Value Methods
        (Dialectic, DIALECTIC_METADATA, "dialectic"),
        (ShannonThinking, SHANNON_METADATA, "shannon_thinking"),
        (SelfReflection, SELF_REFLECTION_METADATA, "self_reflection"),
        (GraphOfThoughts, GRAPH_OF_THOUGHTS_METADATA, "graph_of_thoughts"),
        (MCTS, MCTS_METADATA, "mcts"),
        (SkeletonOfThought, SKELETON_METADATA, "skeleton_of_thought"),
        (LeastToMost, LEAST_TO_MOST_METADATA, "least_to_most"),
        (StepBack, STEP_BACK_METADATA, "step_back"),
        # Wave 11.3: Specialized Methods
        (SelfAsk, SELF_ASK_METADATA, "self_ask"),
        (DecomposedPrompting, DECOMPOSED_METADATA, "decomposed_prompting"),
        (MathematicalReasoning, MATHEMATICAL_METADATA, "mathematical_reasoning"),
        (Abductive, ABDUCTIVE_METADATA, "abductive"),
        (Analogical, ANALOGICAL_METADATA, "analogical"),
        (CausalReasoning, CAUSAL_METADATA, "causal_reasoning"),
        (Socratic, SOCRATIC_METADATA, "socratic"),
        (Counterfactual, COUNTERFACTUAL_METADATA, "counterfactual"),
        # Wave 11.4: Advanced & Holistic Methods
        (MetacognitiveMethod, METACOGNITIVE_METADATA, "metacognitive"),
        (BeamSearchMethod, BEAM_SEARCH_METADATA, "beam_search"),
        (LateralThinkingMethod, LATERAL_METADATA, "lateral_thinking"),
        (LotusWisdomMethod, LOTUS_WISDOM_METADATA, "lotus_wisdom"),
        (AtomOfThoughtsMethod, ATOM_OF_THOUGHTS_METADATA, "atom_of_thoughts"),
        (CascadeThinkingMethod, CASCADE_METADATA, "cascade_thinking"),
        (CRASHMethod, CRASH_METADATA, "crash"),
    ]

    for method_class, metadata, identifier in methods_to_register:
        if method_class is None or metadata is None:
            logger.warning(
                f"Method '{identifier}' not yet implemented, skipping registration"
            )
            results[identifier] = False
            continue

        try:
            # Instantiate the method class
            method_instance = method_class()
            # Register with the registry
            registry.register(method_instance, metadata)
            logger.info(f"Successfully registered method: {identifier}")
            results[identifier] = True
        except Exception as e:
            logger.error(f"Failed to register method '{identifier}': {e}")
            results[identifier] = False

    # Log summary
    successful = sum(results.values())
    total = len(results)
    logger.info(
        f"Native method registration complete: {successful}/{total} methods registered"
    )

    return results


# Export all method classes (that are available)
__all__ = [
    # Core function
    "register_all_native_methods",
    # Wave 11.1: Core Methods
    "SequentialThinking",
    "SEQUENTIAL_METADATA",
    "ChainOfThought",
    "CHAIN_OF_THOUGHT_METADATA",
    "TreeOfThoughts",
    "TREE_OF_THOUGHTS_METADATA",
    "ReActMethod",
    "REACT_METADATA",
    "SelfConsistency",
    "SELF_CONSISTENCY_METADATA",
    "EthicalReasoning",
    "ETHICAL_METADATA",
    "CodeReasoning",
    "CODE_METADATA",
    # Wave 11.2: High-Value Methods
    "Dialectic",
    "DIALECTIC_METADATA",
    "ShannonThinking",
    "SHANNON_METADATA",
    "SelfReflection",
    "SELF_REFLECTION_METADATA",
    "GraphOfThoughts",
    "GRAPH_OF_THOUGHTS_METADATA",
    "MCTS",
    "MCTS_METADATA",
    "SkeletonOfThought",
    "SKELETON_METADATA",
    "LeastToMost",
    "LEAST_TO_MOST_METADATA",
    "StepBack",
    "STEP_BACK_METADATA",
    # Wave 11.3: Specialized Methods
    "SelfAsk",
    "SELF_ASK_METADATA",
    "DecomposedPrompting",
    "DECOMPOSED_METADATA",
    "MathematicalReasoning",
    "MATHEMATICAL_METADATA",
    "Abductive",
    "ABDUCTIVE_METADATA",
    "Analogical",
    "ANALOGICAL_METADATA",
    "CausalReasoning",
    "CAUSAL_METADATA",
    "Socratic",
    "SOCRATIC_METADATA",
    "Counterfactual",
    "COUNTERFACTUAL_METADATA",
    # Wave 11.4: Advanced & Holistic Methods
    "MetacognitiveMethod",
    "METACOGNITIVE_METADATA",
    "BeamSearchMethod",
    "BEAM_SEARCH_METADATA",
    "LateralThinkingMethod",
    "LATERAL_METADATA",
    "LotusWisdomMethod",
    "LOTUS_WISDOM_METADATA",
    "AtomOfThoughtsMethod",
    "ATOM_OF_THOUGHTS_METADATA",
    "CascadeThinkingMethod",
    "CASCADE_METADATA",
    "CRASHMethod",
    "CRASH_METADATA",
    # Helper lists
    "ALL_NATIVE_METHODS",
    "ALL_NATIVE_METADATA",
]

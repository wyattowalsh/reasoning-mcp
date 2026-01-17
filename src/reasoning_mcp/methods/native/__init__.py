"""Native reasoning method implementations.

This package contains all built-in reasoning methods provided by reasoning-mcp.
Methods are organized into waves based on implementation priority.

This module uses lazy loading to improve startup performance. Method classes
and metadata are only imported when actually accessed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from reasoning_mcp.registry import MethodRegistry

logger = structlog.get_logger(__name__)

# Mapping of class names to (module_name, class_name, metadata_name) tuples
# This enables lazy loading - modules are only imported when accessed
_METHOD_REGISTRY: dict[str, tuple[str, str, str]] = {
    # Wave 11.1: Core Methods
    "SequentialThinking": ("sequential", "SequentialThinking", "SEQUENTIAL_METADATA"),
    "ChainOfThought": ("chain_of_thought", "ChainOfThought", "CHAIN_OF_THOUGHT_METADATA"),
    "TreeOfThoughts": ("tree_of_thoughts", "TreeOfThoughts", "TREE_OF_THOUGHTS_METADATA"),
    "ReActMethod": ("react", "ReActMethod", "REACT_METADATA"),
    "SelfConsistency": ("self_consistency", "SelfConsistency", "SELF_CONSISTENCY_METADATA"),
    "EthicalReasoning": ("ethical", "EthicalReasoning", "ETHICAL_REASONING_METADATA"),
    "CodeReasoning": ("code", "CodeReasoning", "CODE_REASONING_METADATA"),
    # Wave 11.2: High-Value Methods
    "Dialectic": ("dialectic", "Dialectic", "DIALECTIC_METADATA"),
    "ShannonThinking": ("shannon", "ShannonThinking", "SHANNON_THINKING_METADATA"),
    "SelfReflection": ("self_reflection", "SelfReflection", "SELF_REFLECTION_METADATA"),
    "GraphOfThoughts": ("graph_of_thoughts", "GraphOfThoughts", "GRAPH_OF_THOUGHTS_METADATA"),
    "MCTS": ("mcts", "MCTS", "MCTS_METADATA"),
    "SkeletonOfThought": ("skeleton", "SkeletonOfThought", "SKELETON_OF_THOUGHT_METADATA"),
    "LeastToMost": ("least_to_most", "LeastToMost", "LEAST_TO_MOST_METADATA"),
    "StepBack": ("step_back", "StepBack", "STEP_BACK_METADATA"),
    # Wave 11.3: Specialized Methods
    "ChainOfVerification": ("chain_of_verification", "ChainOfVerification", "CHAIN_OF_VERIFICATION_METADATA"),
    "SelfRefine": ("self_refine", "SelfRefine", "SELF_REFINE_METADATA"),
    "PlanAndSolve": ("plan_and_solve", "PlanAndSolve", "PLAN_AND_SOLVE_METADATA"),
    "SelfAsk": ("self_ask", "SelfAsk", "SELF_ASK_METADATA"),
    "DecomposedPrompting": ("decomposed", "DecomposedPrompting", "DECOMPOSED_PROMPTING_METADATA"),
    "MathematicalReasoning": ("mathematical", "MathematicalReasoning", "MATHEMATICAL_REASONING_METADATA"),
    "Abductive": ("abductive", "Abductive", "ABDUCTIVE_METADATA"),
    "Analogical": ("analogical", "Analogical", "ANALOGICAL_METADATA"),
    "CausalReasoning": ("causal", "CausalReasoning", "CAUSAL_REASONING_METADATA"),
    "Socratic": ("socratic", "SocraticReasoning", "SOCRATIC_METADATA"),
    "Counterfactual": ("counterfactual", "Counterfactual", "COUNTERFACTUAL_METADATA"),
    # Wave 11.4: Advanced & Holistic Methods
    "MetacognitiveMethod": ("metacognitive", "MetacognitiveMethod", "METACOGNITIVE_METADATA"),
    "BeamSearchMethod": ("beam_search", "BeamSearchMethod", "BEAM_SEARCH_METADATA"),
    "LateralThinkingMethod": ("lateral", "LateralThinkingMethod", "LATERAL_THINKING_METADATA"),
    "LotusWisdomMethod": ("lotus_wisdom", "LotusWisdomMethod", "LOTUS_WISDOM_METADATA"),
    "AtomOfThoughtsMethod": ("atom_of_thoughts", "AtomOfThoughtsMethod", "ATOM_OF_THOUGHTS_METADATA"),
    "CascadeThinkingMethod": ("cascade", "CascadeThinkingMethod", "CASCADE_THINKING_METADATA"),
    "CRASHMethod": ("crash", "CRASHMethod", "CRASH_METADATA"),
    "HintOfThought": ("hint_of_thought", "HintOfThought", "HINT_OF_THOUGHT_METADATA"),
    # Wave 11.5: New Advanced Research Methods (2024-2026)
    "QuietStar": ("quiet_star", "QuietStar", "QUIET_STAR_METADATA"),
    "Reflexion": ("reflexion", "Reflexion", "REFLEXION_METADATA"),
    "DiagramOfThought": ("diagram_of_thought", "DiagramOfThought", "DIAGRAM_OF_THOUGHT_METADATA"),
    "MutualReasoning": ("mutual_reasoning", "MutualReasoning", "MUTUAL_REASONING_METADATA"),
    # Wave 12: Additional Research-Backed Methods
    "ProgramOfThoughts": ("program_of_thoughts", "ProgramOfThoughts", "PROGRAM_OF_THOUGHTS_METADATA"),
    "ThreadOfThought": ("thread_of_thought", "ThreadOfThought", "THREAD_OF_THOUGHT_METADATA"),
    "ContrastiveCoT": ("contrastive_cot", "ContrastiveCoT", "CONTRASTIVE_COT_METADATA"),
    "LogicOfThought": ("logic_of_thought", "LogicOfThought", "LOGIC_OF_THOUGHT_METADATA"),
    "CumulativeReasoning": ("cumulative_reasoning", "CumulativeReasoning", "CUMULATIVE_REASONING_METADATA"),
    "IndirectReasoning": ("indirect_reasoning", "IndirectReasoning", "INDIRECT_REASONING_METADATA"),
    "EverythingOfThoughts": ("everything_of_thoughts", "EverythingOfThoughts", "EVERYTHING_OF_THOUGHTS_METADATA"),
    "FocusedCot": ("focused_cot", "FocusedCot", "FOCUSED_COT_METADATA"),
    # Wave 13: 2025 Research Methods
    "TestTimeScaling": ("test_time_scaling", "TestTimeScaling", "TEST_TIME_SCALING_METADATA"),
    "KeyConceptThinking": ("key_concept_thinking", "KeyConceptThinking", "KEY_CONCEPT_THINKING_METADATA"),
    "SyzygyOfThoughts": ("syzygy_of_thoughts", "SyzygyOfThoughts", "SYZYGY_OF_THOUGHTS_METADATA"),
    "ThinkPRM": ("think_prm", "ThinkPRM", "THINK_PRM_METADATA"),
    "FilterSupervisor": ("filter_supervisor", "FilterSupervisor", "FILTER_SUPERVISOR_METADATA"),
    "SimpleTestTimeScaling": ("simple_test_time_scaling", "SimpleTestTimeScaling", "SIMPLE_TEST_TIME_SCALING_METADATA"),
    # Wave 14: High-Impact 2024-2025 Methods
    "BufferOfThoughts": ("buffer_of_thoughts", "BufferOfThoughts", "BUFFER_OF_THOUGHTS_METADATA"),
    "RStar": ("rstar", "RStar", "RSTAR_METADATA"),
    "SelfDiscover": ("self_discover", "SelfDiscover", "SELF_DISCOVER_METADATA"),
    "STaR": ("star", "STaR", "STAR_METADATA"),
    "BestOfN": ("best_of_n", "BestOfN", "BEST_OF_N_METADATA"),
    "OutcomeRewardModel": ("outcome_reward_model", "OutcomeRewardModel", "OUTCOME_REWARD_MODEL_METADATA"),
    "JourneyLearning": ("journey_learning", "JourneyLearning", "JOURNEY_LEARNING_METADATA"),
    "TwoStageGeneration": ("two_stage_generation", "TwoStageGeneration", "TWO_STAGE_GENERATION_METADATA"),
    # Wave 15: Foundational & Verification Methods
    "MultiAgentDebate": ("multi_agent_debate", "MultiAgentDebate", "MULTI_AGENT_DEBATE_METADATA"),
    "SelfVerification": ("self_verification", "SelfVerification", "SELF_VERIFICATION_METADATA"),
    "Sets": ("sets", "Sets", "SETS_METADATA"),
    "FaithfulCoT": ("faithful_cot", "FaithfulCoT", "FAITHFUL_COT_METADATA"),
    "ZeroShotCoT": ("zero_shot_cot", "ZeroShotCoT", "ZERO_SHOT_COT_METADATA"),
    "ActivePrompt": ("active_prompt", "ActivePrompt", "ACTIVE_PROMPT_METADATA"),
    "ComplexityBased": ("complexity_based", "ComplexityBased", "COMPLEXITY_BASED_METADATA"),
    "AutoCoT": ("auto_cot", "AutoCoT", "AUTO_COT_METADATA"),
    "IterativeRefinement": ("iterative_refinement", "IterativeRefinement", "ITERATIVE_REFINEMENT_METADATA"),
    # Wave 16: Advanced Reasoning & Retrieval Methods
    "GenPRM": ("gen_prm", "GenPRM", "GEN_PRM_METADATA"),
    "MetaCoT": ("meta_cot", "MetaCoT", "META_COT_METADATA"),
    "RetrievalAugmentedThoughts": ("retrieval_augmented_thoughts", "RetrievalAugmentedThoughts", "RETRIEVAL_AUGMENTED_THOUGHTS_METADATA"),
    "CoTRAG": ("cot_rag", "CoTRAG", "COT_RAG_METADATA"),
    "SCRAG": ("sc_rag", "SCRAG", "SC_RAG_METADATA"),
    "ThinkOnGraph": ("think_on_graph", "ThinkOnGraph", "THINK_ON_GRAPH_METADATA"),
    "LayeredCoT": ("layered_cot", "LayeredCoT", "LAYERED_COT_METADATA"),
    "CoTDecoding": ("cot_decoding", "CoTDecoding", "COT_DECODING_METADATA"),
    # Wave 17: Decomposition, Templates & Adaptive Compute Methods
    "BranchSolveMerge": ("branch_solve_merge", "BranchSolveMerge", "BRANCH_SOLVE_MERGE_METADATA"),
    "ProcessPreferenceModel": ("process_preference_model", "ProcessPreferenceModel", "PROCESS_PREFERENCE_MODEL_METADATA"),
    "RStarMath": ("rstar_math", "RStarMath", "RSTAR_MATH_METADATA"),
    "ReasoningPRM": ("reasoning_prm", "ReasoningPRM", "REASONING_PRM_METADATA"),
    "TypedThinker": ("typed_thinker", "TypedThinker", "TYPED_THINKER_METADATA"),
    "ComputeOptimalScaling": ("compute_optimal_scaling", "ComputeOptimalScaling", "COMPUTE_OPTIMAL_SCALING_METADATA"),
    "SuperCorrect": ("super_correct", "SuperCorrect", "SUPER_CORRECT_METADATA"),
    "ThoughtPreferenceOpt": ("thought_preference_opt", "ThoughtPreferenceOpt", "THOUGHT_PREFERENCE_OPT_METADATA"),
    # Wave 18: Verification, Planning & Agent Methods
    "ChainOfCode": ("chain_of_code", "ChainOfCode", "CHAIN_OF_CODE_METADATA"),
    "ReasoningViaPlanning": ("reasoning_via_planning", "ReasoningViaPlanning", "REASONING_VIA_PLANNING_METADATA"),
    "VStar": ("v_star", "VStar", "V_STAR_METADATA"),
    "GLoRe": ("glore", "GLoRe", "GLORE_METADATA"),
    "DiverseVerifier": ("diverse_verifier", "DiverseVerifier", "DIVERSE_VERIFIER_METADATA"),
    "Refiner": ("refiner", "Refiner", "REFINER_METADATA"),
    "Critic": ("critic", "Critic", "CRITIC_METADATA"),
    "Lats": ("lats", "Lats", "LATS_METADATA"),
    # Wave 19: Cutting-Edge 2025 Methods
    "Grpo": ("grpo", "Grpo", "GRPO_METADATA"),
    "CognitiveTools": ("cognitive_tools", "CognitiveTools", "COGNITIVE_TOOLS_METADATA"),
    "SSR": ("ssr", "SSR", "SSR_METADATA"),
    "AGoT": ("agot", "AGoT", "AGOT_METADATA"),
    "S2R": ("s2r", "S2R", "S2R_METADATA"),
    "TrainingFreeGrpo": ("training_free_grpo", "TrainingFreeGrpo", "TRAINING_FREE_GRPO_METADATA"),
    # Wave 20: Efficiency & Latent Reasoning Methods (2025)
    "ChainOfDraft": ("chain_of_draft", "ChainOfDraft", "CHAIN_OF_DRAFT_METADATA"),
    "HybridCot": ("hybrid_cot", "HybridCot", "HYBRID_COT_METADATA"),
    "Gar": ("gar", "Gar", "GAR_METADATA"),
    "Dro": ("dro", "Dro", "DRO_METADATA"),
    "MindEvolution": ("mind_evolution", "MindEvolution", "MIND_EVOLUTION_METADATA"),
    "HiddenCotDecoding": ("hidden_cot_decoding", "HiddenCotDecoding", "HIDDEN_COT_DECODING_METADATA"),
    "LightThinker": ("light_thinker", "LightThinker", "LIGHT_THINKER_METADATA"),
    "Spoc": ("spoc", "Spoc", "SPOC_METADATA"),
    # Ensemble & Meta-Methods
    "EnsembleReasoning": ("ensemble_reasoning", "EnsembleReasoning", "ENSEMBLE_REASONING_METADATA"),
}

# Mapping of metadata names to (module_name, metadata_name) tuples
_METADATA_REGISTRY: dict[str, tuple[str, str]] = {
    # Wave 11.1: Core Methods
    "SEQUENTIAL_METADATA": ("sequential", "SEQUENTIAL_METADATA"),
    "CHAIN_OF_THOUGHT_METADATA": ("chain_of_thought", "CHAIN_OF_THOUGHT_METADATA"),
    "TREE_OF_THOUGHTS_METADATA": ("tree_of_thoughts", "TREE_OF_THOUGHTS_METADATA"),
    "REACT_METADATA": ("react", "REACT_METADATA"),
    "SELF_CONSISTENCY_METADATA": ("self_consistency", "SELF_CONSISTENCY_METADATA"),
    "ETHICAL_METADATA": ("ethical", "ETHICAL_REASONING_METADATA"),
    "CODE_METADATA": ("code", "CODE_REASONING_METADATA"),
    # Wave 11.2: High-Value Methods
    "DIALECTIC_METADATA": ("dialectic", "DIALECTIC_METADATA"),
    "SHANNON_METADATA": ("shannon", "SHANNON_THINKING_METADATA"),
    "SELF_REFLECTION_METADATA": ("self_reflection", "SELF_REFLECTION_METADATA"),
    "GRAPH_OF_THOUGHTS_METADATA": ("graph_of_thoughts", "GRAPH_OF_THOUGHTS_METADATA"),
    "MCTS_METADATA": ("mcts", "MCTS_METADATA"),
    "SKELETON_METADATA": ("skeleton", "SKELETON_OF_THOUGHT_METADATA"),
    "LEAST_TO_MOST_METADATA": ("least_to_most", "LEAST_TO_MOST_METADATA"),
    "STEP_BACK_METADATA": ("step_back", "STEP_BACK_METADATA"),
    # Wave 11.3: Specialized Methods
    "CHAIN_OF_VERIFICATION_METADATA": ("chain_of_verification", "CHAIN_OF_VERIFICATION_METADATA"),
    "SELF_REFINE_METADATA": ("self_refine", "SELF_REFINE_METADATA"),
    "PLAN_AND_SOLVE_METADATA": ("plan_and_solve", "PLAN_AND_SOLVE_METADATA"),
    "SELF_ASK_METADATA": ("self_ask", "SELF_ASK_METADATA"),
    "DECOMPOSED_METADATA": ("decomposed", "DECOMPOSED_PROMPTING_METADATA"),
    "MATHEMATICAL_METADATA": ("mathematical", "MATHEMATICAL_REASONING_METADATA"),
    "ABDUCTIVE_METADATA": ("abductive", "ABDUCTIVE_METADATA"),
    "ANALOGICAL_METADATA": ("analogical", "ANALOGICAL_METADATA"),
    "CAUSAL_METADATA": ("causal", "CAUSAL_REASONING_METADATA"),
    "SOCRATIC_METADATA": ("socratic", "SOCRATIC_METADATA"),
    "COUNTERFACTUAL_METADATA": ("counterfactual", "COUNTERFACTUAL_METADATA"),
    # Wave 11.4: Advanced & Holistic Methods
    "METACOGNITIVE_METADATA": ("metacognitive", "METACOGNITIVE_METADATA"),
    "BEAM_SEARCH_METADATA": ("beam_search", "BEAM_SEARCH_METADATA"),
    "LATERAL_METADATA": ("lateral", "LATERAL_THINKING_METADATA"),
    "LOTUS_WISDOM_METADATA": ("lotus_wisdom", "LOTUS_WISDOM_METADATA"),
    "ATOM_OF_THOUGHTS_METADATA": ("atom_of_thoughts", "ATOM_OF_THOUGHTS_METADATA"),
    "CASCADE_METADATA": ("cascade", "CASCADE_THINKING_METADATA"),
    "CRASH_METADATA": ("crash", "CRASH_METADATA"),
    "HINT_OF_THOUGHT_METADATA": ("hint_of_thought", "HINT_OF_THOUGHT_METADATA"),
    # Wave 11.5: New Advanced Research Methods (2024-2026)
    "QUIET_STAR_METADATA": ("quiet_star", "QUIET_STAR_METADATA"),
    "REFLEXION_METADATA": ("reflexion", "REFLEXION_METADATA"),
    "DIAGRAM_OF_THOUGHT_METADATA": ("diagram_of_thought", "DIAGRAM_OF_THOUGHT_METADATA"),
    "MUTUAL_REASONING_METADATA": ("mutual_reasoning", "MUTUAL_REASONING_METADATA"),
    # Wave 12: Additional Research-Backed Methods
    "PROGRAM_OF_THOUGHTS_METADATA": ("program_of_thoughts", "PROGRAM_OF_THOUGHTS_METADATA"),
    "THREAD_OF_THOUGHT_METADATA": ("thread_of_thought", "THREAD_OF_THOUGHT_METADATA"),
    "CONTRASTIVE_COT_METADATA": ("contrastive_cot", "CONTRASTIVE_COT_METADATA"),
    "LOGIC_OF_THOUGHT_METADATA": ("logic_of_thought", "LOGIC_OF_THOUGHT_METADATA"),
    "CUMULATIVE_REASONING_METADATA": ("cumulative_reasoning", "CUMULATIVE_REASONING_METADATA"),
    "INDIRECT_REASONING_METADATA": ("indirect_reasoning", "INDIRECT_REASONING_METADATA"),
    "EVERYTHING_OF_THOUGHTS_METADATA": ("everything_of_thoughts", "EVERYTHING_OF_THOUGHTS_METADATA"),
    "FOCUSED_COT_METADATA": ("focused_cot", "FOCUSED_COT_METADATA"),
    # Wave 13: 2025 Research Methods
    "TEST_TIME_SCALING_METADATA": ("test_time_scaling", "TEST_TIME_SCALING_METADATA"),
    "KEY_CONCEPT_THINKING_METADATA": ("key_concept_thinking", "KEY_CONCEPT_THINKING_METADATA"),
    "SYZYGY_OF_THOUGHTS_METADATA": ("syzygy_of_thoughts", "SYZYGY_OF_THOUGHTS_METADATA"),
    "THINK_PRM_METADATA": ("think_prm", "THINK_PRM_METADATA"),
    "FILTER_SUPERVISOR_METADATA": ("filter_supervisor", "FILTER_SUPERVISOR_METADATA"),
    "SIMPLE_TEST_TIME_SCALING_METADATA": ("simple_test_time_scaling", "SIMPLE_TEST_TIME_SCALING_METADATA"),
    # Wave 14: High-Impact 2024-2025 Methods
    "BUFFER_OF_THOUGHTS_METADATA": ("buffer_of_thoughts", "BUFFER_OF_THOUGHTS_METADATA"),
    "RSTAR_METADATA": ("rstar", "RSTAR_METADATA"),
    "SELF_DISCOVER_METADATA": ("self_discover", "SELF_DISCOVER_METADATA"),
    "STAR_METADATA": ("star", "STAR_METADATA"),
    "BEST_OF_N_METADATA": ("best_of_n", "BEST_OF_N_METADATA"),
    "OUTCOME_REWARD_MODEL_METADATA": ("outcome_reward_model", "OUTCOME_REWARD_MODEL_METADATA"),
    "JOURNEY_LEARNING_METADATA": ("journey_learning", "JOURNEY_LEARNING_METADATA"),
    "TWO_STAGE_GENERATION_METADATA": ("two_stage_generation", "TWO_STAGE_GENERATION_METADATA"),
    # Wave 15: Foundational & Verification Methods
    "MULTI_AGENT_DEBATE_METADATA": ("multi_agent_debate", "MULTI_AGENT_DEBATE_METADATA"),
    "SELF_VERIFICATION_METADATA": ("self_verification", "SELF_VERIFICATION_METADATA"),
    "SETS_METADATA": ("sets", "SETS_METADATA"),
    "FAITHFUL_COT_METADATA": ("faithful_cot", "FAITHFUL_COT_METADATA"),
    "ZERO_SHOT_COT_METADATA": ("zero_shot_cot", "ZERO_SHOT_COT_METADATA"),
    "ACTIVE_PROMPT_METADATA": ("active_prompt", "ACTIVE_PROMPT_METADATA"),
    "COMPLEXITY_BASED_METADATA": ("complexity_based", "COMPLEXITY_BASED_METADATA"),
    "AUTO_COT_METADATA": ("auto_cot", "AUTO_COT_METADATA"),
    "ITERATIVE_REFINEMENT_METADATA": ("iterative_refinement", "ITERATIVE_REFINEMENT_METADATA"),
    # Wave 16: Advanced Reasoning & Retrieval Methods
    "GEN_PRM_METADATA": ("gen_prm", "GEN_PRM_METADATA"),
    "META_COT_METADATA": ("meta_cot", "META_COT_METADATA"),
    "RETRIEVAL_AUGMENTED_THOUGHTS_METADATA": ("retrieval_augmented_thoughts", "RETRIEVAL_AUGMENTED_THOUGHTS_METADATA"),
    "COT_RAG_METADATA": ("cot_rag", "COT_RAG_METADATA"),
    "SC_RAG_METADATA": ("sc_rag", "SC_RAG_METADATA"),
    "THINK_ON_GRAPH_METADATA": ("think_on_graph", "THINK_ON_GRAPH_METADATA"),
    "LAYERED_COT_METADATA": ("layered_cot", "LAYERED_COT_METADATA"),
    "COT_DECODING_METADATA": ("cot_decoding", "COT_DECODING_METADATA"),
    # Wave 17: Decomposition, Templates & Adaptive Compute Methods
    "BRANCH_SOLVE_MERGE_METADATA": ("branch_solve_merge", "BRANCH_SOLVE_MERGE_METADATA"),
    "PROCESS_PREFERENCE_MODEL_METADATA": ("process_preference_model", "PROCESS_PREFERENCE_MODEL_METADATA"),
    "RSTAR_MATH_METADATA": ("rstar_math", "RSTAR_MATH_METADATA"),
    "REASONING_PRM_METADATA": ("reasoning_prm", "REASONING_PRM_METADATA"),
    "TYPED_THINKER_METADATA": ("typed_thinker", "TYPED_THINKER_METADATA"),
    "COMPUTE_OPTIMAL_SCALING_METADATA": ("compute_optimal_scaling", "COMPUTE_OPTIMAL_SCALING_METADATA"),
    "SUPER_CORRECT_METADATA": ("super_correct", "SUPER_CORRECT_METADATA"),
    "THOUGHT_PREFERENCE_OPT_METADATA": ("thought_preference_opt", "THOUGHT_PREFERENCE_OPT_METADATA"),
    # Wave 18: Verification, Planning & Agent Methods
    "CHAIN_OF_CODE_METADATA": ("chain_of_code", "CHAIN_OF_CODE_METADATA"),
    "REASONING_VIA_PLANNING_METADATA": ("reasoning_via_planning", "REASONING_VIA_PLANNING_METADATA"),
    "V_STAR_METADATA": ("v_star", "V_STAR_METADATA"),
    "GLORE_METADATA": ("glore", "GLORE_METADATA"),
    "DIVERSE_VERIFIER_METADATA": ("diverse_verifier", "DIVERSE_VERIFIER_METADATA"),
    "REFINER_METADATA": ("refiner", "REFINER_METADATA"),
    "CRITIC_METADATA": ("critic", "CRITIC_METADATA"),
    "LATS_METADATA": ("lats", "LATS_METADATA"),
    # Wave 19: Cutting-Edge 2025 Methods
    "GRPO_METADATA": ("grpo", "GRPO_METADATA"),
    "COGNITIVE_TOOLS_METADATA": ("cognitive_tools", "COGNITIVE_TOOLS_METADATA"),
    "SSR_METADATA": ("ssr", "SSR_METADATA"),
    "AGOT_METADATA": ("agot", "AGOT_METADATA"),
    "S2R_METADATA": ("s2r", "S2R_METADATA"),
    "TRAINING_FREE_GRPO_METADATA": ("training_free_grpo", "TRAINING_FREE_GRPO_METADATA"),
    # Wave 20: Efficiency & Latent Reasoning Methods (2025)
    "CHAIN_OF_DRAFT_METADATA": ("chain_of_draft", "CHAIN_OF_DRAFT_METADATA"),
    "HYBRID_COT_METADATA": ("hybrid_cot", "HYBRID_COT_METADATA"),
    "GAR_METADATA": ("gar", "GAR_METADATA"),
    "DRO_METADATA": ("dro", "DRO_METADATA"),
    "MIND_EVOLUTION_METADATA": ("mind_evolution", "MIND_EVOLUTION_METADATA"),
    "HIDDEN_COT_DECODING_METADATA": ("hidden_cot_decoding", "HIDDEN_COT_DECODING_METADATA"),
    "LIGHT_THINKER_METADATA": ("light_thinker", "LIGHT_THINKER_METADATA"),
    "SPOC_METADATA": ("spoc", "SPOC_METADATA"),
    # Ensemble & Meta-Methods
    "ENSEMBLE_REASONING_METADATA": ("ensemble_reasoning", "ENSEMBLE_REASONING_METADATA"),
}

# Cache for loaded modules to avoid repeated imports
_loaded_modules: dict[str, Any] = {}
_loaded_classes: dict[str, Any] = {}
_loaded_metadata: dict[str, Any] = {}


def _load_module(module_name: str) -> Any:
    """Load a module by name, caching the result."""
    if module_name not in _loaded_modules:
        try:
            _loaded_modules[module_name] = importlib.import_module(
                f".{module_name}", __name__
            )
        except ImportError as e:
            logger.debug(f"Module {module_name} not yet implemented: {e}")
            _loaded_modules[module_name] = None
    return _loaded_modules[module_name]


def _get_class(name: str) -> Any:
    """Get a method class by name, loading the module if necessary."""
    if name not in _loaded_classes:
        if name not in _METHOD_REGISTRY:
            return None
        module_name, class_name, _ = _METHOD_REGISTRY[name]
        module = _load_module(module_name)
        if module is None:
            _loaded_classes[name] = None
        else:
            _loaded_classes[name] = getattr(module, class_name, None)
    return _loaded_classes[name]


def _get_metadata(name: str) -> Any:
    """Get metadata by name, loading the module if necessary."""
    if name not in _loaded_metadata:
        if name not in _METADATA_REGISTRY:
            return None
        module_name, metadata_name = _METADATA_REGISTRY[name]
        module = _load_module(module_name)
        if module is None:
            _loaded_metadata[name] = None
        else:
            _loaded_metadata[name] = getattr(module, metadata_name, None)
    return _loaded_metadata[name]


def __getattr__(name: str) -> Any:
    """Lazy loading for method classes and metadata.

    This function is called when an attribute is not found in the module.
    It enables importing method classes and metadata on-demand rather than
    at module load time, significantly improving startup performance.
    """
    # Check if it's a method class
    if name in _METHOD_REGISTRY:
        result = _get_class(name)
        if result is not None:
            return result
        raise AttributeError(f"Method class '{name}' could not be loaded")

    # Check if it's metadata
    if name in _METADATA_REGISTRY:
        result = _get_metadata(name)
        if result is not None:
            return result
        raise AttributeError(f"Metadata '{name}' could not be loaded")

    # Handle special attributes
    if name == "ALL_NATIVE_METHODS":
        return get_all_native_methods()
    if name == "ALL_NATIVE_METADATA":
        return get_all_native_metadata()

    raise AttributeError(f"module 'reasoning_mcp.methods.native' has no attribute '{name}'")


def get_all_native_methods() -> list[Any]:
    """Get all native method classes (loads all modules)."""
    methods = []
    for name in _METHOD_REGISTRY:
        cls = _get_class(name)
        if cls is not None:
            methods.append(cls)
    return methods


def get_all_native_metadata() -> list[Any]:
    """Get all native metadata objects (loads all modules)."""
    metadata = []
    for name in _METADATA_REGISTRY:
        meta = _get_metadata(name)
        if meta is not None:
            metadata.append(meta)
    return metadata


# Method registration info for register_all_native_methods
_REGISTRATION_INFO: list[tuple[str, str, str]] = [
    # (class_name, metadata_export_name, identifier)
    # Wave 11.1: Core Methods
    ("SequentialThinking", "SEQUENTIAL_METADATA", "sequential_thinking"),
    ("ChainOfThought", "CHAIN_OF_THOUGHT_METADATA", "chain_of_thought"),
    ("TreeOfThoughts", "TREE_OF_THOUGHTS_METADATA", "tree_of_thoughts"),
    ("ReActMethod", "REACT_METADATA", "react"),
    ("SelfConsistency", "SELF_CONSISTENCY_METADATA", "self_consistency"),
    ("EthicalReasoning", "ETHICAL_METADATA", "ethical_reasoning"),
    ("CodeReasoning", "CODE_METADATA", "code_reasoning"),
    # Wave 11.2: High-Value Methods
    ("Dialectic", "DIALECTIC_METADATA", "dialectic"),
    ("ShannonThinking", "SHANNON_METADATA", "shannon_thinking"),
    ("SelfReflection", "SELF_REFLECTION_METADATA", "self_reflection"),
    ("GraphOfThoughts", "GRAPH_OF_THOUGHTS_METADATA", "graph_of_thoughts"),
    ("MCTS", "MCTS_METADATA", "mcts"),
    ("SkeletonOfThought", "SKELETON_METADATA", "skeleton_of_thought"),
    ("LeastToMost", "LEAST_TO_MOST_METADATA", "least_to_most"),
    ("StepBack", "STEP_BACK_METADATA", "step_back"),
    # Wave 11.3: Specialized Methods
    ("ChainOfVerification", "CHAIN_OF_VERIFICATION_METADATA", "chain_of_verification"),
    ("SelfRefine", "SELF_REFINE_METADATA", "self_refine"),
    ("PlanAndSolve", "PLAN_AND_SOLVE_METADATA", "plan_and_solve"),
    ("SelfAsk", "SELF_ASK_METADATA", "self_ask"),
    ("DecomposedPrompting", "DECOMPOSED_METADATA", "decomposed_prompting"),
    ("MathematicalReasoning", "MATHEMATICAL_METADATA", "mathematical_reasoning"),
    ("Abductive", "ABDUCTIVE_METADATA", "abductive"),
    ("Analogical", "ANALOGICAL_METADATA", "analogical"),
    ("CausalReasoning", "CAUSAL_METADATA", "causal_reasoning"),
    ("Socratic", "SOCRATIC_METADATA", "socratic"),
    ("Counterfactual", "COUNTERFACTUAL_METADATA", "counterfactual"),
    # Wave 11.4: Advanced & Holistic Methods
    ("MetacognitiveMethod", "METACOGNITIVE_METADATA", "metacognitive"),
    ("BeamSearchMethod", "BEAM_SEARCH_METADATA", "beam_search"),
    ("LateralThinkingMethod", "LATERAL_METADATA", "lateral_thinking"),
    ("LotusWisdomMethod", "LOTUS_WISDOM_METADATA", "lotus_wisdom"),
    ("AtomOfThoughtsMethod", "ATOM_OF_THOUGHTS_METADATA", "atom_of_thoughts"),
    ("CascadeThinkingMethod", "CASCADE_METADATA", "cascade_thinking"),
    ("CRASHMethod", "CRASH_METADATA", "crash"),
    ("HintOfThought", "HINT_OF_THOUGHT_METADATA", "hint_of_thought"),
    # Wave 11.5: New Advanced Research Methods (2024-2026)
    ("QuietStar", "QUIET_STAR_METADATA", "quiet_star"),
    ("Reflexion", "REFLEXION_METADATA", "reflexion"),
    ("DiagramOfThought", "DIAGRAM_OF_THOUGHT_METADATA", "diagram_of_thought"),
    ("MutualReasoning", "MUTUAL_REASONING_METADATA", "mutual_reasoning"),
    # Wave 12: Additional Research-Backed Methods
    ("ProgramOfThoughts", "PROGRAM_OF_THOUGHTS_METADATA", "program_of_thoughts"),
    ("ThreadOfThought", "THREAD_OF_THOUGHT_METADATA", "thread_of_thought"),
    ("ContrastiveCoT", "CONTRASTIVE_COT_METADATA", "contrastive_cot"),
    ("LogicOfThought", "LOGIC_OF_THOUGHT_METADATA", "logic_of_thought"),
    ("CumulativeReasoning", "CUMULATIVE_REASONING_METADATA", "cumulative_reasoning"),
    ("IndirectReasoning", "INDIRECT_REASONING_METADATA", "indirect_reasoning"),
    ("EverythingOfThoughts", "EVERYTHING_OF_THOUGHTS_METADATA", "everything_of_thoughts"),
    ("FocusedCot", "FOCUSED_COT_METADATA", "focused_cot"),
    # Wave 13: 2025 Research Methods
    ("TestTimeScaling", "TEST_TIME_SCALING_METADATA", "test_time_scaling"),
    ("KeyConceptThinking", "KEY_CONCEPT_THINKING_METADATA", "key_concept_thinking"),
    ("SyzygyOfThoughts", "SYZYGY_OF_THOUGHTS_METADATA", "syzygy_of_thoughts"),
    ("ThinkPRM", "THINK_PRM_METADATA", "think_prm"),
    ("FilterSupervisor", "FILTER_SUPERVISOR_METADATA", "filter_supervisor"),
    ("SimpleTestTimeScaling", "SIMPLE_TEST_TIME_SCALING_METADATA", "simple_test_time_scaling"),
    # Wave 14: High-Impact 2024-2025 Methods
    ("BufferOfThoughts", "BUFFER_OF_THOUGHTS_METADATA", "buffer_of_thoughts"),
    ("RStar", "RSTAR_METADATA", "rstar"),
    ("SelfDiscover", "SELF_DISCOVER_METADATA", "self_discover"),
    ("STaR", "STAR_METADATA", "star"),
    ("BestOfN", "BEST_OF_N_METADATA", "best_of_n"),
    ("OutcomeRewardModel", "OUTCOME_REWARD_MODEL_METADATA", "outcome_reward_model"),
    ("JourneyLearning", "JOURNEY_LEARNING_METADATA", "journey_learning"),
    ("TwoStageGeneration", "TWO_STAGE_GENERATION_METADATA", "two_stage_generation"),
    # Wave 15: Foundational & Verification Methods
    ("MultiAgentDebate", "MULTI_AGENT_DEBATE_METADATA", "multi_agent_debate"),
    ("SelfVerification", "SELF_VERIFICATION_METADATA", "self_verification"),
    ("Sets", "SETS_METADATA", "sets"),
    ("FaithfulCoT", "FAITHFUL_COT_METADATA", "faithful_cot"),
    ("ZeroShotCoT", "ZERO_SHOT_COT_METADATA", "zero_shot_cot"),
    ("ActivePrompt", "ACTIVE_PROMPT_METADATA", "active_prompt"),
    ("ComplexityBased", "COMPLEXITY_BASED_METADATA", "complexity_based"),
    ("AutoCoT", "AUTO_COT_METADATA", "auto_cot"),
    ("IterativeRefinement", "ITERATIVE_REFINEMENT_METADATA", "iterative_refinement"),
    # Wave 16: Advanced Reasoning & Retrieval Methods
    ("GenPRM", "GEN_PRM_METADATA", "gen_prm"),
    ("MetaCoT", "META_COT_METADATA", "meta_cot"),
    ("RetrievalAugmentedThoughts", "RETRIEVAL_AUGMENTED_THOUGHTS_METADATA", "retrieval_augmented_thoughts"),
    ("CoTRAG", "COT_RAG_METADATA", "cot_rag"),
    ("SCRAG", "SC_RAG_METADATA", "sc_rag"),
    ("ThinkOnGraph", "THINK_ON_GRAPH_METADATA", "think_on_graph"),
    ("LayeredCoT", "LAYERED_COT_METADATA", "layered_cot"),
    ("CoTDecoding", "COT_DECODING_METADATA", "cot_decoding"),
    # Wave 17: Decomposition, Templates & Adaptive Compute Methods
    ("BranchSolveMerge", "BRANCH_SOLVE_MERGE_METADATA", "branch_solve_merge"),
    ("ProcessPreferenceModel", "PROCESS_PREFERENCE_MODEL_METADATA", "process_preference_model"),
    ("RStarMath", "RSTAR_MATH_METADATA", "rstar_math"),
    ("ReasoningPRM", "REASONING_PRM_METADATA", "reasoning_prm"),
    ("TypedThinker", "TYPED_THINKER_METADATA", "typed_thinker"),
    ("ComputeOptimalScaling", "COMPUTE_OPTIMAL_SCALING_METADATA", "compute_optimal_scaling"),
    ("SuperCorrect", "SUPER_CORRECT_METADATA", "super_correct"),
    ("ThoughtPreferenceOpt", "THOUGHT_PREFERENCE_OPT_METADATA", "thought_preference_opt"),
    # Wave 18: Verification, Planning & Agent Methods
    ("ChainOfCode", "CHAIN_OF_CODE_METADATA", "chain_of_code"),
    ("ReasoningViaPlanning", "REASONING_VIA_PLANNING_METADATA", "reasoning_via_planning"),
    ("VStar", "V_STAR_METADATA", "v_star"),
    ("GLoRe", "GLORE_METADATA", "glore"),
    ("DiverseVerifier", "DIVERSE_VERIFIER_METADATA", "diverse_verifier"),
    ("Refiner", "REFINER_METADATA", "refiner"),
    ("Critic", "CRITIC_METADATA", "critic"),
    ("Lats", "LATS_METADATA", "lats"),
    # Wave 19: Cutting-Edge 2025 Methods
    ("Grpo", "GRPO_METADATA", "grpo"),
    ("CognitiveTools", "COGNITIVE_TOOLS_METADATA", "cognitive_tools"),
    ("SSR", "SSR_METADATA", "ssr"),
    ("AGoT", "AGOT_METADATA", "agot"),
    ("S2R", "S2R_METADATA", "s2r"),
    ("TrainingFreeGrpo", "TRAINING_FREE_GRPO_METADATA", "training_free_grpo"),
    # Wave 20: Efficiency & Latent Reasoning Methods (2025)
    ("ChainOfDraft", "CHAIN_OF_DRAFT_METADATA", "chain_of_draft"),
    ("HybridCot", "HYBRID_COT_METADATA", "hybrid_cot"),
    ("Gar", "GAR_METADATA", "gar"),
    ("Dro", "DRO_METADATA", "dro"),
    ("MindEvolution", "MIND_EVOLUTION_METADATA", "mind_evolution"),
    ("HiddenCotDecoding", "HIDDEN_COT_DECODING_METADATA", "hidden_cot_decoding"),
    ("LightThinker", "LIGHT_THINKER_METADATA", "light_thinker"),
    ("Spoc", "SPOC_METADATA", "spoc"),
    # Ensemble & Meta-Methods
    ("EnsembleReasoning", "ENSEMBLE_REASONING_METADATA", "ensemble_reasoning"),
]


def register_all_native_methods(registry: MethodRegistry) -> dict[str, bool]:
    """Register all native reasoning methods with the registry.

    This function attempts to register all native methods. If a method
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

    for class_name, metadata_name, identifier in _REGISTRATION_INFO:
        method_class = _get_class(class_name)
        metadata = _get_metadata(metadata_name)

        if method_class is None or metadata is None:
            logger.warning(f"Method '{identifier}' not yet implemented, skipping registration")
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
    logger.info(f"Native method registration complete: {successful}/{total} methods registered")

    return results


# Export all method classes and metadata (available through lazy loading)
__all__ = [
    # Core function
    "register_all_native_methods",
    "get_all_native_methods",
    "get_all_native_metadata",
    # Helper lists (computed on-demand)
    "ALL_NATIVE_METHODS",
    "ALL_NATIVE_METADATA",
    # All method class names
    *_METHOD_REGISTRY.keys(),
    # All metadata names
    *_METADATA_REGISTRY.keys(),
]

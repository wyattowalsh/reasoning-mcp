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
        SEQUENTIAL_METADATA,
        SequentialThinking,
    )
except ImportError as e:
    logger.debug(f"Sequential method not yet implemented: {e}")
    SequentialThinking = None  # type: ignore
    SEQUENTIAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.chain_of_thought import (
        CHAIN_OF_THOUGHT_METADATA,
        ChainOfThought,
    )
except ImportError as e:
    logger.debug(f"Chain of Thought method not yet implemented: {e}")
    ChainOfThought = None  # type: ignore
    CHAIN_OF_THOUGHT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.tree_of_thoughts import (
        TREE_OF_THOUGHTS_METADATA,
        TreeOfThoughts,
    )
except ImportError as e:
    logger.debug(f"Tree of Thoughts method not yet implemented: {e}")
    TreeOfThoughts = None  # type: ignore
    TREE_OF_THOUGHTS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.react import REACT_METADATA, ReActMethod
except ImportError as e:
    logger.debug(f"ReAct method not yet implemented: {e}")
    ReActMethod = None  # type: ignore
    REACT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.self_consistency import (
        SELF_CONSISTENCY_METADATA,
        SelfConsistency,
    )
except ImportError as e:
    logger.debug(f"Self-Consistency method not yet implemented: {e}")
    SelfConsistency = None  # type: ignore
    SELF_CONSISTENCY_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.ethical import (
        ETHICAL_REASONING_METADATA as ETHICAL_METADATA,
    )
    from reasoning_mcp.methods.native.ethical import (
        EthicalReasoning,
    )
except ImportError as e:
    logger.debug(f"Ethical Reasoning method not yet implemented: {e}")
    EthicalReasoning = None  # type: ignore
    ETHICAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.code import (
        CODE_REASONING_METADATA as CODE_METADATA,
    )
    from reasoning_mcp.methods.native.code import (
        CodeReasoning,
    )
except ImportError as e:
    logger.debug(f"Code Reasoning method not yet implemented: {e}")
    CodeReasoning = None  # type: ignore
    CODE_METADATA = None  # type: ignore

# Wave 11.2: High-Value Methods
try:
    from reasoning_mcp.methods.native.dialectic import (
        DIALECTIC_METADATA,
        Dialectic,
    )
except ImportError as e:
    logger.debug(f"Dialectic method not yet implemented: {e}")
    Dialectic = None  # type: ignore
    DIALECTIC_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.shannon import (
        SHANNON_THINKING_METADATA as SHANNON_METADATA,
    )
    from reasoning_mcp.methods.native.shannon import (
        ShannonThinking,
    )
except ImportError as e:
    logger.debug(f"Shannon Thinking method not yet implemented: {e}")
    ShannonThinking = None  # type: ignore
    SHANNON_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.self_reflection import (
        SELF_REFLECTION_METADATA,
        SelfReflection,
    )
except ImportError as e:
    logger.debug(f"Self-Reflection method not yet implemented: {e}")
    SelfReflection = None  # type: ignore
    SELF_REFLECTION_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.graph_of_thoughts import (
        GRAPH_OF_THOUGHTS_METADATA,
        GraphOfThoughts,
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
        SKELETON_OF_THOUGHT_METADATA as SKELETON_METADATA,
    )
    from reasoning_mcp.methods.native.skeleton import (
        SkeletonOfThought,
    )
except ImportError as e:
    logger.debug(f"Skeleton of Thought method not yet implemented: {e}")
    SkeletonOfThought = None  # type: ignore
    SKELETON_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.least_to_most import (
        LEAST_TO_MOST_METADATA,
        LeastToMost,
    )
except ImportError as e:
    logger.debug(f"Least-to-Most method not yet implemented: {e}")
    LeastToMost = None  # type: ignore
    LEAST_TO_MOST_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.step_back import (
        STEP_BACK_METADATA,
        StepBack,
    )
except ImportError as e:
    logger.debug(f"Step-Back method not yet implemented: {e}")
    StepBack = None  # type: ignore
    STEP_BACK_METADATA = None  # type: ignore

# Wave 11.3: Specialized Methods
try:
    from reasoning_mcp.methods.native.chain_of_verification import (
        CHAIN_OF_VERIFICATION_METADATA,
        ChainOfVerification,
    )
except ImportError as e:
    logger.debug(f"Chain of Verification method not yet implemented: {e}")
    ChainOfVerification = None  # type: ignore
    CHAIN_OF_VERIFICATION_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.self_refine import (
        SELF_REFINE_METADATA,
        SelfRefine,
    )
except ImportError as e:
    logger.debug(f"Self-Refine method not yet implemented: {e}")
    SelfRefine = None  # type: ignore
    SELF_REFINE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.plan_and_solve import (
        PLAN_AND_SOLVE_METADATA,
        PlanAndSolve,
    )
except ImportError as e:
    logger.debug(f"Plan-and-Solve method not yet implemented: {e}")
    PlanAndSolve = None  # type: ignore
    PLAN_AND_SOLVE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.self_ask import SELF_ASK_METADATA, SelfAsk
except ImportError as e:
    logger.debug(f"Self-Ask method not yet implemented: {e}")
    SelfAsk = None  # type: ignore
    SELF_ASK_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.decomposed import (
        DECOMPOSED_PROMPTING_METADATA as DECOMPOSED_METADATA,
    )
    from reasoning_mcp.methods.native.decomposed import (
        DecomposedPrompting,
    )
except ImportError as e:
    logger.debug(f"Decomposed Prompting method not yet implemented: {e}")
    DecomposedPrompting = None  # type: ignore
    DECOMPOSED_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.mathematical import (
        MATHEMATICAL_REASONING_METADATA as MATHEMATICAL_METADATA,
    )
    from reasoning_mcp.methods.native.mathematical import (
        MathematicalReasoning,
    )
except ImportError as e:
    logger.debug(f"Mathematical Reasoning method not yet implemented: {e}")
    MathematicalReasoning = None  # type: ignore
    MATHEMATICAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.abductive import (
        ABDUCTIVE_METADATA,
        Abductive,
    )
except ImportError as e:
    logger.debug(f"Abductive method not yet implemented: {e}")
    Abductive = None  # type: ignore
    ABDUCTIVE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.analogical import (
        ANALOGICAL_METADATA,
        Analogical,
    )
except ImportError as e:
    logger.debug(f"Analogical method not yet implemented: {e}")
    Analogical = None  # type: ignore
    ANALOGICAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.causal import (
        CAUSAL_REASONING_METADATA as CAUSAL_METADATA,
    )
    from reasoning_mcp.methods.native.causal import (
        CausalReasoning,
    )
except ImportError as e:
    logger.debug(f"Causal Reasoning method not yet implemented: {e}")
    CausalReasoning = None  # type: ignore
    CAUSAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.socratic import (
        SOCRATIC_METADATA,
    )
    from reasoning_mcp.methods.native.socratic import (
        SocraticReasoning as Socratic,
    )
except ImportError as e:
    logger.debug(f"Socratic method not yet implemented: {e}")
    Socratic = None  # type: ignore
    SOCRATIC_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.counterfactual import (
        COUNTERFACTUAL_METADATA,
        Counterfactual,
    )
except ImportError as e:
    logger.debug(f"Counterfactual method not yet implemented: {e}")
    Counterfactual = None  # type: ignore
    COUNTERFACTUAL_METADATA = None  # type: ignore

# Wave 11.4: Advanced & Holistic Methods
try:
    from reasoning_mcp.methods.native.metacognitive import (
        METACOGNITIVE_METADATA,
        MetacognitiveMethod,
    )
except ImportError as e:
    logger.debug(f"Metacognitive method not yet implemented: {e}")
    MetacognitiveMethod = None  # type: ignore
    METACOGNITIVE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.beam_search import (
        BEAM_SEARCH_METADATA,
        BeamSearchMethod,
    )
except ImportError as e:
    logger.debug(f"Beam Search method not yet implemented: {e}")
    BeamSearchMethod = None  # type: ignore
    BEAM_SEARCH_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.lateral import (
        LATERAL_THINKING_METADATA as LATERAL_METADATA,
    )
    from reasoning_mcp.methods.native.lateral import (
        LateralThinkingMethod,
    )
except ImportError as e:
    logger.debug(f"Lateral Thinking method not yet implemented: {e}")
    LateralThinkingMethod = None  # type: ignore
    LATERAL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.lotus_wisdom import (
        LOTUS_WISDOM_METADATA,
        LotusWisdomMethod,
    )
except ImportError as e:
    logger.debug(f"Lotus Wisdom method not yet implemented: {e}")
    LotusWisdomMethod = None  # type: ignore
    LOTUS_WISDOM_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.atom_of_thoughts import (
        ATOM_OF_THOUGHTS_METADATA,
        AtomOfThoughtsMethod,
    )
except ImportError as e:
    logger.debug(f"Atom of Thoughts method not yet implemented: {e}")
    AtomOfThoughtsMethod = None  # type: ignore
    ATOM_OF_THOUGHTS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.cascade import (
        CASCADE_THINKING_METADATA as CASCADE_METADATA,
    )
    from reasoning_mcp.methods.native.cascade import (
        CascadeThinkingMethod,
    )
except ImportError as e:
    logger.debug(f"Cascade Thinking method not yet implemented: {e}")
    CascadeThinkingMethod = None  # type: ignore
    CASCADE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.crash import CRASH_METADATA, CRASHMethod
except ImportError as e:
    logger.debug(f"CRASH method not yet implemented: {e}")
    CRASHMethod = None  # type: ignore
    CRASH_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.hint_of_thought import (
        HINT_OF_THOUGHT_METADATA,
        HintOfThought,
    )
except ImportError as e:
    logger.debug(f"Hint of Thought method not yet implemented: {e}")
    HintOfThought = None  # type: ignore
    HINT_OF_THOUGHT_METADATA = None  # type: ignore

# Wave 11.5: New Advanced Research Methods (2024-2026)
try:
    from reasoning_mcp.methods.native.quiet_star import (
        QUIET_STAR_METADATA,
        QuietStar,
    )
except ImportError as e:
    logger.debug(f"Quiet-STaR method not yet implemented: {e}")
    QuietStar = None  # type: ignore
    QUIET_STAR_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.reflexion import (
        REFLEXION_METADATA,
        Reflexion,
    )
except ImportError as e:
    logger.debug(f"Reflexion method not yet implemented: {e}")
    Reflexion = None  # type: ignore
    REFLEXION_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.diagram_of_thought import (
        DIAGRAM_OF_THOUGHT_METADATA,
        DiagramOfThought,
    )
except ImportError as e:
    logger.debug(f"Diagram of Thought method not yet implemented: {e}")
    DiagramOfThought = None  # type: ignore
    DIAGRAM_OF_THOUGHT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.mutual_reasoning import (
        MUTUAL_REASONING_METADATA,
        MutualReasoning,
    )
except ImportError as e:
    logger.debug(f"Mutual Reasoning method not yet implemented: {e}")
    MutualReasoning = None  # type: ignore
    MUTUAL_REASONING_METADATA = None  # type: ignore

# Wave 12: Additional Research-Backed Methods
try:
    from reasoning_mcp.methods.native.program_of_thoughts import (
        PROGRAM_OF_THOUGHTS_METADATA,
        ProgramOfThoughts,
    )
except ImportError as e:
    logger.debug(f"Program of Thoughts method not yet implemented: {e}")
    ProgramOfThoughts = None  # type: ignore
    PROGRAM_OF_THOUGHTS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.thread_of_thought import (
        THREAD_OF_THOUGHT_METADATA,
        ThreadOfThought,
    )
except ImportError as e:
    logger.debug(f"Thread of Thought method not yet implemented: {e}")
    ThreadOfThought = None  # type: ignore
    THREAD_OF_THOUGHT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.contrastive_cot import (
        CONTRASTIVE_COT_METADATA,
        ContrastiveCoT,
    )
except ImportError as e:
    logger.debug(f"Contrastive CoT method not yet implemented: {e}")
    ContrastiveCoT = None  # type: ignore
    CONTRASTIVE_COT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.logic_of_thought import (
        LOGIC_OF_THOUGHT_METADATA,
        LogicOfThought,
    )
except ImportError as e:
    logger.debug(f"Logic of Thought method not yet implemented: {e}")
    LogicOfThought = None  # type: ignore
    LOGIC_OF_THOUGHT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.cumulative_reasoning import (
        CUMULATIVE_REASONING_METADATA,
        CumulativeReasoning,
    )
except ImportError as e:
    logger.debug(f"Cumulative Reasoning method not yet implemented: {e}")
    CumulativeReasoning = None  # type: ignore
    CUMULATIVE_REASONING_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.indirect_reasoning import (
        INDIRECT_REASONING_METADATA,
        IndirectReasoning,
    )
except ImportError as e:
    logger.debug(f"Indirect Reasoning method not yet implemented: {e}")
    IndirectReasoning = None  # type: ignore
    INDIRECT_REASONING_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.everything_of_thoughts import (
        EVERYTHING_OF_THOUGHTS_METADATA,
        EverythingOfThoughts,
    )
except ImportError as e:
    logger.debug(f"Everything of Thoughts method not yet implemented: {e}")
    EverythingOfThoughts = None  # type: ignore
    EVERYTHING_OF_THOUGHTS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.focused_cot import (
        FOCUSED_COT_METADATA,
        FocusedCot,
    )
except ImportError as e:
    logger.debug(f"Focused CoT method not yet implemented: {e}")
    FocusedCot = None  # type: ignore
    FOCUSED_COT_METADATA = None  # type: ignore

# Wave 13: 2025 Research Methods (Inference-Time Compute & Advanced Techniques)
try:
    from reasoning_mcp.methods.native.test_time_scaling import (
        TEST_TIME_SCALING_METADATA,
        TestTimeScaling,
    )
except ImportError as e:
    logger.debug(f"Test-Time Scaling method not yet implemented: {e}")
    TestTimeScaling = None  # type: ignore
    TEST_TIME_SCALING_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.key_concept_thinking import (
        KEY_CONCEPT_THINKING_METADATA,
        KeyConceptThinking,
    )
except ImportError as e:
    logger.debug(f"Key-Concept Thinking method not yet implemented: {e}")
    KeyConceptThinking = None  # type: ignore
    KEY_CONCEPT_THINKING_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.syzygy_of_thoughts import (
        SYZYGY_OF_THOUGHTS_METADATA,
        SyzygyOfThoughts,
    )
except ImportError as e:
    logger.debug(f"Syzygy of Thoughts method not yet implemented: {e}")
    SyzygyOfThoughts = None  # type: ignore
    SYZYGY_OF_THOUGHTS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.think_prm import (
        THINK_PRM_METADATA,
        ThinkPRM,
    )
except ImportError as e:
    logger.debug(f"Think-PRM method not yet implemented: {e}")
    ThinkPRM = None  # type: ignore
    THINK_PRM_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.filter_supervisor import (
        FILTER_SUPERVISOR_METADATA,
        FilterSupervisor,
    )
except ImportError as e:
    logger.debug(f"Filter Supervisor method not yet implemented: {e}")
    FilterSupervisor = None  # type: ignore
    FILTER_SUPERVISOR_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.simple_test_time_scaling import (
        SIMPLE_TEST_TIME_SCALING_METADATA,
        SimpleTestTimeScaling,
    )
except ImportError as e:
    logger.debug(f"Simple Test-Time Scaling method not yet implemented: {e}")
    SimpleTestTimeScaling = None  # type: ignore
    SIMPLE_TEST_TIME_SCALING_METADATA = None  # type: ignore

# Wave 14: High-Impact 2024-2025 Methods (Widely-Used Techniques)
try:
    from reasoning_mcp.methods.native.buffer_of_thoughts import (
        BUFFER_OF_THOUGHTS_METADATA,
        BufferOfThoughts,
    )
except ImportError as e:
    logger.debug(f"Buffer of Thoughts method not yet implemented: {e}")
    BufferOfThoughts = None  # type: ignore
    BUFFER_OF_THOUGHTS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.rstar import (
        RSTAR_METADATA,
        RStar,
    )
except ImportError as e:
    logger.debug(f"rStar method not yet implemented: {e}")
    RStar = None  # type: ignore
    RSTAR_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.self_discover import (
        SELF_DISCOVER_METADATA,
        SelfDiscover,
    )
except ImportError as e:
    logger.debug(f"Self-Discover method not yet implemented: {e}")
    SelfDiscover = None  # type: ignore
    SELF_DISCOVER_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.star import (
        STAR_METADATA,
        STaR,
    )
except ImportError as e:
    logger.debug(f"STaR method not yet implemented: {e}")
    STaR = None  # type: ignore
    STAR_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.best_of_n import (
        BEST_OF_N_METADATA,
        BestOfN,
    )
except ImportError as e:
    logger.debug(f"Best-of-N method not yet implemented: {e}")
    BestOfN = None  # type: ignore
    BEST_OF_N_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.outcome_reward_model import (
        OUTCOME_REWARD_MODEL_METADATA,
        OutcomeRewardModel,
    )
except ImportError as e:
    logger.debug(f"Outcome Reward Model method not yet implemented: {e}")
    OutcomeRewardModel = None  # type: ignore
    OUTCOME_REWARD_MODEL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.journey_learning import (
        JOURNEY_LEARNING_METADATA,
        JourneyLearning,
    )
except ImportError as e:
    logger.debug(f"Journey Learning method not yet implemented: {e}")
    JourneyLearning = None  # type: ignore
    JOURNEY_LEARNING_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.two_stage_generation import (
        TWO_STAGE_GENERATION_METADATA,
        TwoStageGeneration,
    )
except ImportError as e:
    logger.debug(f"Two-Stage Generation method not yet implemented: {e}")
    TwoStageGeneration = None  # type: ignore
    TWO_STAGE_GENERATION_METADATA = None  # type: ignore

# Wave 15: Foundational & Verification Methods
try:
    from reasoning_mcp.methods.native.multi_agent_debate import (
        MULTI_AGENT_DEBATE_METADATA,
        MultiAgentDebate,
    )
except ImportError as e:
    logger.debug(f"Multi-Agent Debate method not yet implemented: {e}")
    MultiAgentDebate = None  # type: ignore
    MULTI_AGENT_DEBATE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.self_verification import (
        SELF_VERIFICATION_METADATA,
        SelfVerification,
    )
except ImportError as e:
    logger.debug(f"Self-Verification method not yet implemented: {e}")
    SelfVerification = None  # type: ignore
    SELF_VERIFICATION_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.sets import (
        SETS_METADATA,
        Sets,
    )
except ImportError as e:
    logger.debug(f"SETS method not yet implemented: {e}")
    Sets = None  # type: ignore
    SETS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.faithful_cot import (
        FAITHFUL_COT_METADATA,
        FaithfulCoT,
    )
except ImportError as e:
    logger.debug(f"Faithful CoT method not yet implemented: {e}")
    FaithfulCoT = None  # type: ignore
    FAITHFUL_COT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.zero_shot_cot import (
        ZERO_SHOT_COT_METADATA,
        ZeroShotCoT,
    )
except ImportError as e:
    logger.debug(f"Zero-Shot CoT method not yet implemented: {e}")
    ZeroShotCoT = None  # type: ignore
    ZERO_SHOT_COT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.active_prompt import (
        ACTIVE_PROMPT_METADATA,
        ActivePrompt,
    )
except ImportError as e:
    logger.debug(f"Active Prompt method not yet implemented: {e}")
    ActivePrompt = None  # type: ignore
    ACTIVE_PROMPT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.complexity_based import (
        COMPLEXITY_BASED_METADATA,
        ComplexityBased,
    )
except ImportError as e:
    logger.debug(f"Complexity-Based method not yet implemented: {e}")
    ComplexityBased = None  # type: ignore
    COMPLEXITY_BASED_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.auto_cot import (
        AUTO_COT_METADATA,
        AutoCoT,
    )
except ImportError as e:
    logger.debug(f"Auto-CoT method not yet implemented: {e}")
    AutoCoT = None  # type: ignore
    AUTO_COT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.iterative_refinement import (
        ITERATIVE_REFINEMENT_METADATA,
        IterativeRefinement,
    )
except ImportError as e:
    logger.debug(f"Iterative Refinement method not yet implemented: {e}")
    IterativeRefinement = None  # type: ignore
    ITERATIVE_REFINEMENT_METADATA = None  # type: ignore

# Wave 16: Advanced Reasoning & Retrieval Methods
try:
    from reasoning_mcp.methods.native.gen_prm import (
        GEN_PRM_METADATA,
        GenPRM,
    )
except ImportError as e:
    logger.debug(f"GenPRM method not yet implemented: {e}")
    GenPRM = None  # type: ignore
    GEN_PRM_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.meta_cot import (
        META_COT_METADATA,
        MetaCoT,
    )
except ImportError as e:
    logger.debug(f"Meta-CoT method not yet implemented: {e}")
    MetaCoT = None  # type: ignore
    META_COT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.retrieval_augmented_thoughts import (
        RETRIEVAL_AUGMENTED_THOUGHTS_METADATA,
        RetrievalAugmentedThoughts,
    )
except ImportError as e:
    logger.debug(f"RAT method not yet implemented: {e}")
    RetrievalAugmentedThoughts = None  # type: ignore
    RETRIEVAL_AUGMENTED_THOUGHTS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.cot_rag import (
        COT_RAG_METADATA,
        CoTRAG,
    )
except ImportError as e:
    logger.debug(f"CoT-RAG method not yet implemented: {e}")
    CoTRAG = None  # type: ignore
    COT_RAG_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.sc_rag import (
        SC_RAG_METADATA,
        SCRAG,
    )
except ImportError as e:
    logger.debug(f"SC-RAG method not yet implemented: {e}")
    SCRAG = None  # type: ignore
    SC_RAG_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.think_on_graph import (
        THINK_ON_GRAPH_METADATA,
        ThinkOnGraph,
    )
except ImportError as e:
    logger.debug(f"Think-on-Graph method not yet implemented: {e}")
    ThinkOnGraph = None  # type: ignore
    THINK_ON_GRAPH_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.layered_cot import (
        LAYERED_COT_METADATA,
        LayeredCoT,
    )
except ImportError as e:
    logger.debug(f"Layered CoT method not yet implemented: {e}")
    LayeredCoT = None  # type: ignore
    LAYERED_COT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.cot_decoding import (
        COT_DECODING_METADATA,
        CoTDecoding,
    )
except ImportError as e:
    logger.debug(f"CoT Decoding method not yet implemented: {e}")
    CoTDecoding = None  # type: ignore
    COT_DECODING_METADATA = None  # type: ignore

# Wave 17: Decomposition, Templates & Adaptive Compute Methods
try:
    from reasoning_mcp.methods.native.branch_solve_merge import (
        BRANCH_SOLVE_MERGE_METADATA,
        BranchSolveMerge,
    )
except ImportError as e:
    logger.debug(f"Branch-Solve-Merge method not yet implemented: {e}")
    BranchSolveMerge = None  # type: ignore
    BRANCH_SOLVE_MERGE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.process_preference_model import (
        PROCESS_PREFERENCE_MODEL_METADATA,
        ProcessPreferenceModel,
    )
except ImportError as e:
    logger.debug(f"Process Preference Model method not yet implemented: {e}")
    ProcessPreferenceModel = None  # type: ignore
    PROCESS_PREFERENCE_MODEL_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.rstar_math import (
        RSTAR_MATH_METADATA,
        RStarMath,
    )
except ImportError as e:
    logger.debug(f"rStar-Math method not yet implemented: {e}")
    RStarMath = None  # type: ignore
    RSTAR_MATH_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.reasoning_prm import (
        REASONING_PRM_METADATA,
        ReasoningPRM,
    )
except ImportError as e:
    logger.debug(f"Reasoning-PRM method not yet implemented: {e}")
    ReasoningPRM = None  # type: ignore
    REASONING_PRM_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.typed_thinker import (
        TYPED_THINKER_METADATA,
        TypedThinker,
    )
except ImportError as e:
    logger.debug(f"TypedThinker method not yet implemented: {e}")
    TypedThinker = None  # type: ignore
    TYPED_THINKER_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.compute_optimal_scaling import (
        COMPUTE_OPTIMAL_SCALING_METADATA,
        ComputeOptimalScaling,
    )
except ImportError as e:
    logger.debug(f"Compute-Optimal Scaling method not yet implemented: {e}")
    ComputeOptimalScaling = None  # type: ignore
    COMPUTE_OPTIMAL_SCALING_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.super_correct import (
        SUPER_CORRECT_METADATA,
        SuperCorrect,
    )
except ImportError as e:
    logger.debug(f"SuperCorrect method not yet implemented: {e}")
    SuperCorrect = None  # type: ignore
    SUPER_CORRECT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.thought_preference_opt import (
        THOUGHT_PREFERENCE_OPT_METADATA,
        ThoughtPreferenceOpt,
    )
except ImportError as e:
    logger.debug(f"Thought Preference Optimization method not yet implemented: {e}")
    ThoughtPreferenceOpt = None  # type: ignore
    THOUGHT_PREFERENCE_OPT_METADATA = None  # type: ignore

# Wave 18: Verification, Planning & Agent Methods
try:
    from reasoning_mcp.methods.native.chain_of_code import (
        CHAIN_OF_CODE_METADATA,
        ChainOfCode,
    )
except ImportError as e:
    logger.debug(f"Chain of Code method not yet implemented: {e}")
    ChainOfCode = None  # type: ignore
    CHAIN_OF_CODE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.reasoning_via_planning import (
        REASONING_VIA_PLANNING_METADATA,
        ReasoningViaPlanning,
    )
except ImportError as e:
    logger.debug(f"Reasoning via Planning method not yet implemented: {e}")
    ReasoningViaPlanning = None  # type: ignore
    REASONING_VIA_PLANNING_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.v_star import (
        V_STAR_METADATA,
        VStar,
    )
except ImportError as e:
    logger.debug(f"V-STaR method not yet implemented: {e}")
    VStar = None  # type: ignore
    V_STAR_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.glore import (
        GLORE_METADATA,
        GLoRe,
    )
except ImportError as e:
    logger.debug(f"GLoRe method not yet implemented: {e}")
    GLoRe = None  # type: ignore
    GLORE_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.diverse_verifier import (
        DIVERSE_VERIFIER_METADATA,
        DiverseVerifier,
    )
except ImportError as e:
    logger.debug(f"DiVeRSe method not yet implemented: {e}")
    DiverseVerifier = None  # type: ignore
    DIVERSE_VERIFIER_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.refiner import (
        REFINER_METADATA,
        Refiner,
    )
except ImportError as e:
    logger.debug(f"REFINER method not yet implemented: {e}")
    Refiner = None  # type: ignore
    REFINER_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.critic import (
        CRITIC_METADATA,
        Critic,
    )
except ImportError as e:
    logger.debug(f"CRITIC method not yet implemented: {e}")
    Critic = None  # type: ignore
    CRITIC_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.lats import (
        LATS_METADATA,
        Lats,
    )
except ImportError as e:
    logger.debug(f"LATS method not yet implemented: {e}")
    Lats = None  # type: ignore
    LATS_METADATA = None  # type: ignore

# Wave 19: Cutting-Edge 2025 Methods
try:
    from reasoning_mcp.methods.native.grpo import (
        GRPO_METADATA,
        Grpo,
    )
except ImportError as e:
    logger.debug(f"GRPO method not yet implemented: {e}")
    Grpo = None  # type: ignore
    GRPO_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.cognitive_tools import (
        COGNITIVE_TOOLS_METADATA,
        CognitiveTools,
    )
except ImportError as e:
    logger.debug(f"Cognitive Tools method not yet implemented: {e}")
    CognitiveTools = None  # type: ignore
    COGNITIVE_TOOLS_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.ssr import (
        SSR,
        SSR_METADATA,
    )
except ImportError as e:
    logger.debug(f"SSR method not yet implemented: {e}")
    SSR = None  # type: ignore
    SSR_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.agot import (
        AGOT_METADATA,
        AGoT,
    )
except ImportError as e:
    logger.debug(f"AGoT method not yet implemented: {e}")
    AGoT = None  # type: ignore
    AGOT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.s2r import (
        S2R,
        S2R_METADATA,
    )
except ImportError as e:
    logger.debug(f"S2R method not yet implemented: {e}")
    S2R = None  # type: ignore
    S2R_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.training_free_grpo import (
        TRAINING_FREE_GRPO_METADATA,
        TrainingFreeGrpo,
    )
except ImportError as e:
    logger.debug(f"Training-Free GRPO method not yet implemented: {e}")
    TrainingFreeGrpo = None  # type: ignore
    TRAINING_FREE_GRPO_METADATA = None  # type: ignore

# Wave 20: Efficiency & Latent Reasoning Methods (2025)
try:
    from reasoning_mcp.methods.native.chain_of_draft import (
        CHAIN_OF_DRAFT_METADATA,
        ChainOfDraft,
    )
except ImportError as e:
    logger.debug(f"Chain of Draft method not yet implemented: {e}")
    ChainOfDraft = None  # type: ignore
    CHAIN_OF_DRAFT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.hybrid_cot import (
        HYBRID_COT_METADATA,
        HybridCot,
    )
except ImportError as e:
    logger.debug(f"HybridCoT method not yet implemented: {e}")
    HybridCot = None  # type: ignore
    HYBRID_COT_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.gar import (
        GAR_METADATA,
        Gar,
    )
except ImportError as e:
    logger.debug(f"GAR method not yet implemented: {e}")
    Gar = None  # type: ignore
    GAR_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.dro import (
        DRO_METADATA,
        Dro,
    )
except ImportError as e:
    logger.debug(f"DRO method not yet implemented: {e}")
    Dro = None  # type: ignore
    DRO_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.mind_evolution import (
        MIND_EVOLUTION_METADATA,
        MindEvolution,
    )
except ImportError as e:
    logger.debug(f"Mind Evolution method not yet implemented: {e}")
    MindEvolution = None  # type: ignore
    MIND_EVOLUTION_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.hidden_cot_decoding import (
        HIDDEN_COT_DECODING_METADATA,
        HiddenCotDecoding,
    )
except ImportError as e:
    logger.debug(f"Hidden CoT Decoding method not yet implemented: {e}")
    HiddenCotDecoding = None  # type: ignore
    HIDDEN_COT_DECODING_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.light_thinker import (
        LIGHT_THINKER_METADATA,
        LightThinker,
    )
except ImportError as e:
    logger.debug(f"LightThinker method not yet implemented: {e}")
    LightThinker = None  # type: ignore
    LIGHT_THINKER_METADATA = None  # type: ignore

try:
    from reasoning_mcp.methods.native.spoc import (
        SPOC_METADATA,
        Spoc,
    )
except ImportError as e:
    logger.debug(f"SPOC method not yet implemented: {e}")
    Spoc = None  # type: ignore
    SPOC_METADATA = None  # type: ignore


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
    ChainOfVerification,
    SelfRefine,
    PlanAndSolve,
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
    HintOfThought,
    # Wave 11.5: New Advanced Research Methods (2024-2026)
    QuietStar,
    Reflexion,
    DiagramOfThought,
    MutualReasoning,
    # Wave 12: Additional Research-Backed Methods
    ProgramOfThoughts,
    ThreadOfThought,
    ContrastiveCoT,
    LogicOfThought,
    CumulativeReasoning,
    IndirectReasoning,
    EverythingOfThoughts,
    FocusedCot,
    # Wave 13: 2025 Research Methods
    TestTimeScaling,
    KeyConceptThinking,
    SyzygyOfThoughts,
    ThinkPRM,
    FilterSupervisor,
    SimpleTestTimeScaling,
    # Wave 14: High-Impact 2024-2025 Methods
    BufferOfThoughts,
    RStar,
    SelfDiscover,
    STaR,
    BestOfN,
    OutcomeRewardModel,
    JourneyLearning,
    TwoStageGeneration,
    # Wave 15: Foundational & Verification Methods
    MultiAgentDebate,
    SelfVerification,
    Sets,
    FaithfulCoT,
    ZeroShotCoT,
    ActivePrompt,
    ComplexityBased,
    AutoCoT,
    IterativeRefinement,
    # Wave 16: Advanced Reasoning & Retrieval Methods
    GenPRM,
    MetaCoT,
    RetrievalAugmentedThoughts,
    CoTRAG,
    SCRAG,
    ThinkOnGraph,
    LayeredCoT,
    CoTDecoding,
    # Wave 17: Decomposition, Templates & Adaptive Compute Methods
    BranchSolveMerge,
    ProcessPreferenceModel,
    RStarMath,
    ReasoningPRM,
    TypedThinker,
    ComputeOptimalScaling,
    SuperCorrect,
    ThoughtPreferenceOpt,
    # Wave 18: Verification, Planning & Agent Methods
    ChainOfCode,
    ReasoningViaPlanning,
    VStar,
    GLoRe,
    DiverseVerifier,
    Refiner,
    Critic,
    Lats,
    # Wave 19: Cutting-Edge 2025 Methods
    Grpo,
    CognitiveTools,
    SSR,
    AGoT,
    S2R,
    TrainingFreeGrpo,
    # Wave 20: Efficiency & Latent Reasoning Methods (2025)
    ChainOfDraft,
    HybridCot,
    Gar,
    Dro,
    MindEvolution,
    HiddenCotDecoding,
    LightThinker,
    Spoc,
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
    CHAIN_OF_VERIFICATION_METADATA,
    SELF_REFINE_METADATA,
    PLAN_AND_SOLVE_METADATA,
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
    HINT_OF_THOUGHT_METADATA,
    # Wave 11.5: New Advanced Research Methods (2024-2026)
    QUIET_STAR_METADATA,
    REFLEXION_METADATA,
    DIAGRAM_OF_THOUGHT_METADATA,
    MUTUAL_REASONING_METADATA,
    # Wave 12: Additional Research-Backed Methods
    PROGRAM_OF_THOUGHTS_METADATA,
    THREAD_OF_THOUGHT_METADATA,
    CONTRASTIVE_COT_METADATA,
    LOGIC_OF_THOUGHT_METADATA,
    CUMULATIVE_REASONING_METADATA,
    INDIRECT_REASONING_METADATA,
    EVERYTHING_OF_THOUGHTS_METADATA,
    FOCUSED_COT_METADATA,
    # Wave 13: 2025 Research Methods
    TEST_TIME_SCALING_METADATA,
    KEY_CONCEPT_THINKING_METADATA,
    SYZYGY_OF_THOUGHTS_METADATA,
    THINK_PRM_METADATA,
    FILTER_SUPERVISOR_METADATA,
    SIMPLE_TEST_TIME_SCALING_METADATA,
    # Wave 14: High-Impact 2024-2025 Methods
    BUFFER_OF_THOUGHTS_METADATA,
    RSTAR_METADATA,
    SELF_DISCOVER_METADATA,
    STAR_METADATA,
    BEST_OF_N_METADATA,
    OUTCOME_REWARD_MODEL_METADATA,
    JOURNEY_LEARNING_METADATA,
    TWO_STAGE_GENERATION_METADATA,
    # Wave 15: Foundational & Verification Methods
    MULTI_AGENT_DEBATE_METADATA,
    SELF_VERIFICATION_METADATA,
    SETS_METADATA,
    FAITHFUL_COT_METADATA,
    ZERO_SHOT_COT_METADATA,
    ACTIVE_PROMPT_METADATA,
    COMPLEXITY_BASED_METADATA,
    AUTO_COT_METADATA,
    ITERATIVE_REFINEMENT_METADATA,
    # Wave 16: Advanced Reasoning & Retrieval Methods
    GEN_PRM_METADATA,
    META_COT_METADATA,
    RETRIEVAL_AUGMENTED_THOUGHTS_METADATA,
    COT_RAG_METADATA,
    SC_RAG_METADATA,
    THINK_ON_GRAPH_METADATA,
    LAYERED_COT_METADATA,
    COT_DECODING_METADATA,
    # Wave 17: Decomposition, Templates & Adaptive Compute Methods
    BRANCH_SOLVE_MERGE_METADATA,
    PROCESS_PREFERENCE_MODEL_METADATA,
    RSTAR_MATH_METADATA,
    REASONING_PRM_METADATA,
    TYPED_THINKER_METADATA,
    COMPUTE_OPTIMAL_SCALING_METADATA,
    SUPER_CORRECT_METADATA,
    THOUGHT_PREFERENCE_OPT_METADATA,
    # Wave 18: Verification, Planning & Agent Methods
    CHAIN_OF_CODE_METADATA,
    REASONING_VIA_PLANNING_METADATA,
    V_STAR_METADATA,
    GLORE_METADATA,
    DIVERSE_VERIFIER_METADATA,
    REFINER_METADATA,
    CRITIC_METADATA,
    LATS_METADATA,
    # Wave 19: Cutting-Edge 2025 Methods
    GRPO_METADATA,
    COGNITIVE_TOOLS_METADATA,
    SSR_METADATA,
    S2R_METADATA,
    AGOT_METADATA,
    TRAINING_FREE_GRPO_METADATA,
    # Wave 20: Efficiency & Latent Reasoning Methods (2025)
    CHAIN_OF_DRAFT_METADATA,
    HYBRID_COT_METADATA,
    GAR_METADATA,
    DRO_METADATA,
    MIND_EVOLUTION_METADATA,
    HIDDEN_COT_DECODING_METADATA,
    LIGHT_THINKER_METADATA,
    SPOC_METADATA,
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
        (ChainOfVerification, CHAIN_OF_VERIFICATION_METADATA, "chain_of_verification"),
        (SelfRefine, SELF_REFINE_METADATA, "self_refine"),
        (PlanAndSolve, PLAN_AND_SOLVE_METADATA, "plan_and_solve"),
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
        (HintOfThought, HINT_OF_THOUGHT_METADATA, "hint_of_thought"),
        # Wave 11.5: New Advanced Research Methods (2024-2026)
        (QuietStar, QUIET_STAR_METADATA, "quiet_star"),
        (Reflexion, REFLEXION_METADATA, "reflexion"),
        (DiagramOfThought, DIAGRAM_OF_THOUGHT_METADATA, "diagram_of_thought"),
        (MutualReasoning, MUTUAL_REASONING_METADATA, "mutual_reasoning"),
        # Wave 12: Additional Research-Backed Methods
        (ProgramOfThoughts, PROGRAM_OF_THOUGHTS_METADATA, "program_of_thoughts"),
        (ThreadOfThought, THREAD_OF_THOUGHT_METADATA, "thread_of_thought"),
        (ContrastiveCoT, CONTRASTIVE_COT_METADATA, "contrastive_cot"),
        (LogicOfThought, LOGIC_OF_THOUGHT_METADATA, "logic_of_thought"),
        (CumulativeReasoning, CUMULATIVE_REASONING_METADATA, "cumulative_reasoning"),
        (IndirectReasoning, INDIRECT_REASONING_METADATA, "indirect_reasoning"),
        (EverythingOfThoughts, EVERYTHING_OF_THOUGHTS_METADATA, "everything_of_thoughts"),
        (FocusedCot, FOCUSED_COT_METADATA, "focused_cot"),
        # Wave 13: 2025 Research Methods
        (TestTimeScaling, TEST_TIME_SCALING_METADATA, "test_time_scaling"),
        (KeyConceptThinking, KEY_CONCEPT_THINKING_METADATA, "key_concept_thinking"),
        (SyzygyOfThoughts, SYZYGY_OF_THOUGHTS_METADATA, "syzygy_of_thoughts"),
        (ThinkPRM, THINK_PRM_METADATA, "think_prm"),
        (FilterSupervisor, FILTER_SUPERVISOR_METADATA, "filter_supervisor"),
        (SimpleTestTimeScaling, SIMPLE_TEST_TIME_SCALING_METADATA, "simple_test_time_scaling"),
        # Wave 14: High-Impact 2024-2025 Methods
        (BufferOfThoughts, BUFFER_OF_THOUGHTS_METADATA, "buffer_of_thoughts"),
        (RStar, RSTAR_METADATA, "rstar"),
        (SelfDiscover, SELF_DISCOVER_METADATA, "self_discover"),
        (STaR, STAR_METADATA, "star"),
        (BestOfN, BEST_OF_N_METADATA, "best_of_n"),
        (OutcomeRewardModel, OUTCOME_REWARD_MODEL_METADATA, "outcome_reward_model"),
        (JourneyLearning, JOURNEY_LEARNING_METADATA, "journey_learning"),
        (TwoStageGeneration, TWO_STAGE_GENERATION_METADATA, "two_stage_generation"),
        # Wave 15: Foundational & Verification Methods
        (MultiAgentDebate, MULTI_AGENT_DEBATE_METADATA, "multi_agent_debate"),
        (SelfVerification, SELF_VERIFICATION_METADATA, "self_verification"),
        (Sets, SETS_METADATA, "sets"),
        (FaithfulCoT, FAITHFUL_COT_METADATA, "faithful_cot"),
        (ZeroShotCoT, ZERO_SHOT_COT_METADATA, "zero_shot_cot"),
        (ActivePrompt, ACTIVE_PROMPT_METADATA, "active_prompt"),
        (ComplexityBased, COMPLEXITY_BASED_METADATA, "complexity_based"),
        (AutoCoT, AUTO_COT_METADATA, "auto_cot"),
        (IterativeRefinement, ITERATIVE_REFINEMENT_METADATA, "iterative_refinement"),
        # Wave 16: Advanced Reasoning & Retrieval Methods
        (GenPRM, GEN_PRM_METADATA, "gen_prm"),
        (MetaCoT, META_COT_METADATA, "meta_cot"),
        (
            RetrievalAugmentedThoughts,
            RETRIEVAL_AUGMENTED_THOUGHTS_METADATA,
            "retrieval_augmented_thoughts",
        ),
        (CoTRAG, COT_RAG_METADATA, "cot_rag"),
        (SCRAG, SC_RAG_METADATA, "sc_rag"),
        (ThinkOnGraph, THINK_ON_GRAPH_METADATA, "think_on_graph"),
        (LayeredCoT, LAYERED_COT_METADATA, "layered_cot"),
        (CoTDecoding, COT_DECODING_METADATA, "cot_decoding"),
        # Wave 17: Decomposition, Templates & Adaptive Compute Methods
        (BranchSolveMerge, BRANCH_SOLVE_MERGE_METADATA, "branch_solve_merge"),
        (ProcessPreferenceModel, PROCESS_PREFERENCE_MODEL_METADATA, "process_preference_model"),
        (RStarMath, RSTAR_MATH_METADATA, "rstar_math"),
        (ReasoningPRM, REASONING_PRM_METADATA, "reasoning_prm"),
        (TypedThinker, TYPED_THINKER_METADATA, "typed_thinker"),
        (ComputeOptimalScaling, COMPUTE_OPTIMAL_SCALING_METADATA, "compute_optimal_scaling"),
        (SuperCorrect, SUPER_CORRECT_METADATA, "super_correct"),
        (ThoughtPreferenceOpt, THOUGHT_PREFERENCE_OPT_METADATA, "thought_preference_opt"),
        # Wave 18: Verification, Planning & Agent Methods
        (ChainOfCode, CHAIN_OF_CODE_METADATA, "chain_of_code"),
        (ReasoningViaPlanning, REASONING_VIA_PLANNING_METADATA, "reasoning_via_planning"),
        (VStar, V_STAR_METADATA, "v_star"),
        (GLoRe, GLORE_METADATA, "glore"),
        (DiverseVerifier, DIVERSE_VERIFIER_METADATA, "diverse_verifier"),
        (Refiner, REFINER_METADATA, "refiner"),
        (Critic, CRITIC_METADATA, "critic"),
        (Lats, LATS_METADATA, "lats"),
        # Wave 19: Cutting-Edge 2025 Methods
        (Grpo, GRPO_METADATA, "grpo"),
        (CognitiveTools, COGNITIVE_TOOLS_METADATA, "cognitive_tools"),
        (SSR, SSR_METADATA, "ssr"),
        (AGoT, AGOT_METADATA, "agot"),
        (S2R, S2R_METADATA, "s2r"),
        (TrainingFreeGrpo, TRAINING_FREE_GRPO_METADATA, "training_free_grpo"),
        # Wave 20: Efficiency & Latent Reasoning Methods (2025)
        (ChainOfDraft, CHAIN_OF_DRAFT_METADATA, "chain_of_draft"),
        (HybridCot, HYBRID_COT_METADATA, "hybrid_cot"),
        (Gar, GAR_METADATA, "gar"),
        (Dro, DRO_METADATA, "dro"),
        (MindEvolution, MIND_EVOLUTION_METADATA, "mind_evolution"),
        (HiddenCotDecoding, HIDDEN_COT_DECODING_METADATA, "hidden_cot_decoding"),
        (LightThinker, LIGHT_THINKER_METADATA, "light_thinker"),
        (Spoc, SPOC_METADATA, "spoc"),
    ]

    for method_class, metadata, identifier in methods_to_register:
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
    "ChainOfVerification",
    "CHAIN_OF_VERIFICATION_METADATA",
    "SelfRefine",
    "SELF_REFINE_METADATA",
    "PlanAndSolve",
    "PLAN_AND_SOLVE_METADATA",
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
    "HintOfThought",
    "HINT_OF_THOUGHT_METADATA",
    # Wave 11.5: New Advanced Research Methods (2024-2026)
    "QuietStar",
    "QUIET_STAR_METADATA",
    "Reflexion",
    "REFLEXION_METADATA",
    "DiagramOfThought",
    "DIAGRAM_OF_THOUGHT_METADATA",
    "MutualReasoning",
    "MUTUAL_REASONING_METADATA",
    # Wave 12: Additional Research-Backed Methods
    "ProgramOfThoughts",
    "PROGRAM_OF_THOUGHTS_METADATA",
    "ThreadOfThought",
    "THREAD_OF_THOUGHT_METADATA",
    "ContrastiveCoT",
    "CONTRASTIVE_COT_METADATA",
    "LogicOfThought",
    "LOGIC_OF_THOUGHT_METADATA",
    "CumulativeReasoning",
    "CUMULATIVE_REASONING_METADATA",
    "IndirectReasoning",
    "INDIRECT_REASONING_METADATA",
    "EverythingOfThoughts",
    "EVERYTHING_OF_THOUGHTS_METADATA",
    "FocusedCot",
    "FOCUSED_COT_METADATA",
    # Wave 13: 2025 Research Methods
    "TestTimeScaling",
    "TEST_TIME_SCALING_METADATA",
    "KeyConceptThinking",
    "KEY_CONCEPT_THINKING_METADATA",
    "SyzygyOfThoughts",
    "SYZYGY_OF_THOUGHTS_METADATA",
    "ThinkPRM",
    "THINK_PRM_METADATA",
    "FilterSupervisor",
    "FILTER_SUPERVISOR_METADATA",
    "SimpleTestTimeScaling",
    "SIMPLE_TEST_TIME_SCALING_METADATA",
    # Wave 14: High-Impact 2024-2025 Methods
    "BufferOfThoughts",
    "BUFFER_OF_THOUGHTS_METADATA",
    "RStar",
    "RSTAR_METADATA",
    "SelfDiscover",
    "SELF_DISCOVER_METADATA",
    "STaR",
    "STAR_METADATA",
    "BestOfN",
    "BEST_OF_N_METADATA",
    "OutcomeRewardModel",
    "OUTCOME_REWARD_MODEL_METADATA",
    "JourneyLearning",
    "JOURNEY_LEARNING_METADATA",
    "TwoStageGeneration",
    "TWO_STAGE_GENERATION_METADATA",
    # Wave 15: Foundational & Verification Methods
    "MultiAgentDebate",
    "MULTI_AGENT_DEBATE_METADATA",
    "SelfVerification",
    "SELF_VERIFICATION_METADATA",
    "Sets",
    "SETS_METADATA",
    "FaithfulCoT",
    "FAITHFUL_COT_METADATA",
    "ZeroShotCoT",
    "ZERO_SHOT_COT_METADATA",
    "ActivePrompt",
    "ACTIVE_PROMPT_METADATA",
    "ComplexityBased",
    "COMPLEXITY_BASED_METADATA",
    "AutoCoT",
    "AUTO_COT_METADATA",
    "IterativeRefinement",
    "ITERATIVE_REFINEMENT_METADATA",
    # Wave 16: Advanced Reasoning & Retrieval Methods
    "GenPRM",
    "GEN_PRM_METADATA",
    "MetaCoT",
    "META_COT_METADATA",
    "RetrievalAugmentedThoughts",
    "RETRIEVAL_AUGMENTED_THOUGHTS_METADATA",
    "CoTRAG",
    "COT_RAG_METADATA",
    "SCRAG",
    "SC_RAG_METADATA",
    "ThinkOnGraph",
    "THINK_ON_GRAPH_METADATA",
    "LayeredCoT",
    "LAYERED_COT_METADATA",
    "CoTDecoding",
    "COT_DECODING_METADATA",
    # Wave 17: Decomposition, Templates & Adaptive Compute Methods
    "BranchSolveMerge",
    "BRANCH_SOLVE_MERGE_METADATA",
    "ProcessPreferenceModel",
    "PROCESS_PREFERENCE_MODEL_METADATA",
    "RStarMath",
    "RSTAR_MATH_METADATA",
    "ReasoningPRM",
    "REASONING_PRM_METADATA",
    "TypedThinker",
    "TYPED_THINKER_METADATA",
    "ComputeOptimalScaling",
    "COMPUTE_OPTIMAL_SCALING_METADATA",
    "SuperCorrect",
    "SUPER_CORRECT_METADATA",
    "ThoughtPreferenceOpt",
    "THOUGHT_PREFERENCE_OPT_METADATA",
    # Wave 18: Verification, Planning & Agent Methods
    "ChainOfCode",
    "CHAIN_OF_CODE_METADATA",
    "ReasoningViaPlanning",
    "REASONING_VIA_PLANNING_METADATA",
    "VStar",
    "V_STAR_METADATA",
    "GLoRe",
    "GLORE_METADATA",
    "DiverseVerifier",
    "DIVERSE_VERIFIER_METADATA",
    "Refiner",
    "REFINER_METADATA",
    "Critic",
    "CRITIC_METADATA",
    "Lats",
    "LATS_METADATA",
    # Wave 19: Cutting-Edge 2025 Methods
    "Grpo",
    "GRPO_METADATA",
    "CognitiveTools",
    "COGNITIVE_TOOLS_METADATA",
    "SSR",
    "SSR_METADATA",
    "AGoT",
    "AGOT_METADATA",
    "S2R",
    "S2R_METADATA",
    "TrainingFreeGrpo",
    "TRAINING_FREE_GRPO_METADATA",
    # Wave 20: Efficiency & Latent Reasoning Methods (2025)
    "ChainOfDraft",
    "CHAIN_OF_DRAFT_METADATA",
    "HybridCot",
    "HYBRID_COT_METADATA",
    "Gar",
    "GAR_METADATA",
    "Dro",
    "DRO_METADATA",
    "MindEvolution",
    "MIND_EVOLUTION_METADATA",
    "HiddenCotDecoding",
    "HIDDEN_COT_DECODING_METADATA",
    "LightThinker",
    "LIGHT_THINKER_METADATA",
    "Spoc",
    "SPOC_METADATA",
    # Helper lists
    "ALL_NATIVE_METHODS",
    "ALL_NATIVE_METADATA",
]

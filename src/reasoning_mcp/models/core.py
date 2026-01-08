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

    # Specialized Methods (10 + 9 new = 19)
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

    # New Specialized Methods (2024-2026 Research)
    CHAIN_OF_VERIFICATION = "chain_of_verification"
    """Verify reasoning through explicit verification questions and independent answers."""

    PROGRAM_OF_THOUGHTS = "program_of_thoughts"
    """Generate executable code to solve problems, then interpret results."""

    THREAD_OF_THOUGHT = "thread_of_thought"
    """Process long contexts by segmenting and reasoning incrementally."""

    SELF_REFINE = "self_refine"
    """Iterative self-improvement through generate-feedback-refine cycles."""

    CONTRASTIVE_COT = "contrastive_cot"
    """Contrast correct and incorrect reasoning to improve accuracy."""

    LOGIC_OF_THOUGHT = "logic_of_thought"
    """Formal logic-based reasoning with premises, inferences, and conclusions."""

    CUMULATIVE_REASONING = "cumulative_reasoning"
    """Accumulate verified propositions in a DAG structure."""

    PLAN_AND_SOLVE = "plan_and_solve"
    """Explicit planning phase before solving, with step decomposition."""

    INDIRECT_REASONING = "indirect_reasoning"
    """Proof by contradiction - assume negation, derive contradiction."""

    # Advanced Methods (5 + 4 new = 9)
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

    # New Advanced Methods (2024-2026 Research)
    REFLEXION = "reflexion"
    """Self-reflection with episodic memory for learning from past attempts."""

    MUTUAL_REASONING = "mutual_reasoning"
    """Mutual reasoning (rStar) with discriminator-guided MCTS exploration."""

    DIAGRAM_OF_THOUGHT = "diagram_of_thought"
    """DAG-based reasoning with proposer-critic-summarizer roles."""

    QUIET_STAR = "quiet_star"
    """Internal rationale generation before producing outputs."""

    # Holistic Methods (5 + 2 new = 7)
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

    # New Holistic Methods (2024-2026 Research)
    EVERYTHING_OF_THOUGHTS = "everything_of_thoughts"
    """Meta-framework dynamically switching between chain/tree/graph structures."""

    HINT_OF_THOUGHT = "hint_of_thought"
    """Zero-shot reasoning with structural hints and pseudocode guidance."""

    # 2025 Research Methods (Inference-Time Compute & Advanced Techniques)
    TEST_TIME_SCALING = "test_time_scaling"
    """Scale inference-time compute with extended thinking and search (DeepSeek-R1, o1/o3 style)."""

    KEY_CONCEPT_THINKING = "key_concept_thinking"
    """Extract and reason with key domain concepts before solving (Zheng et al. 2025)."""

    SYZYGY_OF_THOUGHTS = "syzygy_of_thoughts"
    """Combinatorial symbolic reasoning through alignment of complementary perspectives."""

    THINK_PRM = "think_prm"
    """Process Reward Model - score and guide reasoning with learned process rewards."""

    FILTER_SUPERVISOR = "filter_supervisor"
    """Filter-based supervision with self-correction (FS-C pattern)."""

    SIMPLE_TEST_TIME_SCALING = "simple_test_time_scaling"
    """Budget-aware test-time scaling with wait tokens (s1 by Muennighoff et al. 2025)."""

    # 2024-2025 High-Impact Methods (Widely-Used Techniques)
    BUFFER_OF_THOUGHTS = "buffer_of_thoughts"
    """Cache reusable thought templates for efficient multi-step reasoning (NeurIPS 2024)."""

    RSTAR = "rstar"
    """Self-play mutual reasoning with discriminator-guided code execution (Microsoft 2024)."""

    SELF_DISCOVER = "self_discover"
    """Discover task-specific reasoning structures before solving (Zhou et al. 2024)."""

    STAR = "star"
    """Self-Taught Reasoner - bootstrap reasoning from rationale examples (Zelikman et al. 2022)."""

    BEST_OF_N = "best_of_n"
    """Sample N reasoning paths and select best via reward model or verifier."""

    OUTCOME_REWARD_MODEL = "outcome_reward_model"
    """Verify solutions using outcome-based reward scoring (ORM pattern)."""

    JOURNEY_LEARNING = "journey_learning"
    """Learn from the reasoning journey, not just the final answer."""

    TWO_STAGE_GENERATION = "two_stage_generation"
    """Think-then-answer: extended thinking followed by concise summary (R1/o1 style)."""

    # Wave 15: Foundational & Verification Methods
    MULTI_AGENT_DEBATE = "multi_agent_debate"
    """Multiple agents debate to improve reasoning accuracy (Du et al. 2023)."""

    SELF_VERIFICATION = "self_verification"
    """Forward reasoning + backward verification for error correction (Weng et al. 2022)."""

    FAITHFUL_COT = "faithful_cot"
    """Ensures reasoning chain faithfully supports the final answer (Lyu et al. 2023)."""

    ZERO_SHOT_COT = "zero_shot_cot"
    """Simple 'Let's think step by step' trigger for reasoning (Kojima et al. 2022)."""

    ACTIVE_PROMPT = "active_prompt"
    """Selects examples based on uncertainty for better demonstrations (Diao et al. 2023)."""

    COMPLEXITY_BASED = "complexity_based"
    """Uses complex examples for better reasoning performance (Fu et al. 2023)."""

    AUTO_COT = "auto_cot"
    """Automatically generates chain-of-thought examples (Zhang et al. 2022)."""

    ITERATIVE_REFINEMENT = "iterative_refinement"
    """Multiple passes to progressively refine and improve answers."""

    # Wave 16: Advanced Reasoning & Retrieval Methods
    GEN_PRM = "gen_prm"
    """Generative Process Reward Model for test-time compute scaling (Zhao et al. 2025)."""

    META_COT = "meta_cot"
    """Meta Chain-of-Thought - learning how to think with meta-reasoning (2025)."""

    RETRIEVAL_AUGMENTED_THOUGHTS = "retrieval_augmented_thoughts"
    """RAT - combines RAG with CoT for grounded reasoning (2024)."""

    COT_RAG = "cot_rag"
    """Integrates knowledge graphs, RAG, and CoT with pseudo-program prompting (Li et al. 2025)."""

    SC_RAG = "sc_rag"
    """Self-Corrective RAG with evidence extraction and CoT self-correction (2024)."""

    THINK_ON_GRAPH = "think_on_graph"
    """ToG - iterative beam search on knowledge graphs for reasoning (Sun et al. 2024)."""

    LAYERED_COT = "layered_cot"
    """Multi-pass reasoning with layer-by-layer review and adjustment (2025)."""

    COT_DECODING = "cot_decoding"
    """Elicits CoT reasoning through decoding without explicit prompting (Wang et al. 2024)."""

    # Wave 17: Decomposition, Templates & Adaptive Compute Methods
    BRANCH_SOLVE_MERGE = "branch_solve_merge"
    """Decompose task into parallel sub-tasks, solve independently, merge solutions (Saha et al. 2024)."""

    PROCESS_PREFERENCE_MODEL = "process_preference_model"
    """PPM - step-level preference scoring avoiding naive annotation (rStar-Math 2025)."""

    RSTAR_MATH = "rstar_math"
    """MCTS-based self-evolution with code-augmented CoT for math reasoning (Guan et al. 2025)."""

    REASONING_PRM = "reasoning_prm"
    """R-PRM - reasoning-driven process reward with self-evolution and inference scaling (EMNLP 2025)."""

    TYPED_THINKER = "typed_thinker"
    """Diversify reasoning by categorizing into deductive/inductive/abductive/analogical types (Wang et al. 2024)."""

    COMPUTE_OPTIMAL_SCALING = "compute_optimal_scaling"
    """Adaptive test-time compute allocation based on problem difficulty (Snell et al. 2024)."""

    SUPER_CORRECT = "super_correct"
    """Hierarchical thought templates + cross-model DPO for self-correction (Yang et al. 2024)."""

    THOUGHT_PREFERENCE_OPT = "thought_preference_opt"
    """TPO - train LLMs to generate internal thoughts before responses (Wu et al. 2024)."""

    # Wave 18: Verification, Planning & Agent Methods
    CHAIN_OF_CODE = "chain_of_code"
    """LM-augmented code emulator for semantic sub-tasks in reasoning (Li et al. 2024, ICML Oral)."""

    REASONING_VIA_PLANNING = "reasoning_via_planning"
    """RAP - LLM as world model with MCTS for strategic exploration (Hao et al. 2023, EMNLP)."""

    V_STAR = "v_star"
    """Training verifiers for self-taught reasoners with DPO on correctness (Hosseini et al. 2024, COLM)."""

    GLORE = "glore"
    """Global and Local Refinements with stepwise ORM for process supervision (Havrilla et al. 2024, ICML)."""

    DIVERSE_VERIFIER = "diverse_verifier"
    """DiVeRSe - Diverse Verifier on Reasoning Steps with multiple samples (Li et al. 2023)."""

    REFINER = "refiner"
    """Generator-critic feedback loop on intermediate reasoning representations (Paul et al. 2024, EACL)."""

    CRITIC = "critic"
    """Tool-interactive critiquing for self-correction with external tools (Gou et al. 2024, ICLR)."""

    LATS = "lats"
    """Language Agent Tree Search - unifies reasoning, acting, and planning (Zhou et al. 2024, ICML)."""

    # Wave 19: Cutting-Edge 2025 Methods
    GRPO = "grpo"
    """Group Relative Policy Optimization - critic-free RL with group-level comparisons (DeepSeek-R1, Shao et al. 2024)."""

    COGNITIVE_TOOLS = "cognitive_tools"
    """Modular cognitive operations (analogical, deductive, abductive, inductive) in agentic framework (Ebouky et al. 2025, NeurIPS)."""

    S2R = "s2r"
    """Self-verification and Self-correction with Reinforcement Learning (arXiv Feb 2025)."""

    SETS = "sets"
    """Self-Verification and Self-Correction combined - verify then correct iteratively (Chen et al. Jan 2025)."""

    SSR = "ssr"
    """Socratic Self-Refine - Socratic questioning to drive iterative self-refinement (2025)."""

    FOCUSED_COT = "focused_cot"
    """Focused Chain-of-Thought - condition-first reasoning that filters irrelevant information (Xu et al. 2025)."""

    AGOT = "agot"
    """Adaptive Graph of Thoughts - dynamically adapting graph structure during reasoning (2025)."""

    TRAINING_FREE_GRPO = "training_free_grpo"
    """Training-Free GRPO - inference-time optimization with GRPO benefits without training (Cai et al. Oct 2025)."""

    # Wave 20: Efficiency & Latent Reasoning Methods (2025)
    CHAIN_OF_DRAFT = "chain_of_draft"
    """Chain of Draft - minimal 5-word steps for 76% latency reduction (Xu et al. 2025, Zoom)."""

    HYBRID_COT = "hybrid_cot"
    """HybridCoT - interleave latent and text reasoning for efficiency (ICLR 2026)."""

    GAR = "gar"
    """Generator-Adversarial Reasoning with trainable discriminator (Xi et al. 2025)."""

    DRO = "dro"
    """Direct Reasoning Optimization - LLMs self-reward and self-refine (arXiv 2025)."""

    MIND_EVOLUTION = "mind_evolution"
    """Mind Evolution - genetic algorithm-based population search reasoning (2025)."""

    HIDDEN_COT_DECODING = "hidden_cot_decoding"
    """Hidden CoT Decoding - efficient CoT without explicit tokens (Wang et al. 2025)."""

    LIGHT_THINKER = "light_thinker"
    """LightThinker - gist token compression for efficient reasoning (2025)."""

    SPOC = "spoc"
    """SPOC - Spontaneous Self-Correction without external feedback (2025)."""


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


class RouterIdentifier(StrEnum):
    """Identifiers for all supported reasoning routers.

    Routers dynamically select and allocate compute resources across
    reasoning methods based on query complexity and task requirements.
    """

    AUTO_THINK = "auto_think"
    """Auto-Think - adaptive CoT activation via classifier (Agarwal et al. 2025)."""

    SELF_BUDGETER = "self_budgeter"
    """SelfBudgeter - token allocation optimization based on problem difficulty."""

    THINK_SWITCHER = "think_switcher"
    """ThinkSwitcher - Fast/Normal/Slow mode selection for compute efficiency."""

    ROUTER_R1 = "router_r1"
    """Router-R1 - RL-based multi-round routing with learned policies."""

    GRAPH_ROUTER = "graph_router"
    """GraphRouter - graph-based model routing for complex task decomposition."""

    BEST_ROUTE = "best_route"
    """Best-Route - optimal test-time compute allocation across methods."""

    MAS_ROUTER = "mas_router"
    """MasRouter - multi-agent system routing for distributed reasoning."""

    RAG_ROUTER = "rag_router"
    """RAGRouter - retrieval-aware routing for knowledge-intensive tasks."""


class VerifierIdentifier(StrEnum):
    """Identifiers for all supported reasoning verifiers (PRMs).

    Verifiers score and validate reasoning steps, providing process rewards
    to guide search and improve answer quality.
    """

    THINK_PRM = "think_prm"
    """ThinkPRM - generative CoT verifier with 8K labels (Qwang et al. 2025)."""

    GEN_PRM = "gen_prm"
    """GenPRM - generative process rewards with explicit CoT verification."""

    R_PRM = "r_prm"
    """R-PRM - reasoning-driven process rewards with self-evolution."""

    VERSA_PRM = "versa_prm"
    """VersaPRM - versatile multi-domain process reward model."""

    RRM = "rrm"
    """RRM - Reward Reasoning Models with deliberative reward scoring."""

    GAR_DISCRIMINATOR = "gar_discriminator"
    """GAR-Discriminator - adversarial discriminator for reasoning verification."""

    OR_PRM = "or_prm"
    """OR-PRM - outcome-aware process rewards combining step and outcome signals."""


class EnsemblerIdentifier(StrEnum):
    """Identifiers for all supported reasoning ensemblers.

    Ensemblers combine multiple models or reasoning paths to improve
    accuracy and robustness through diverse aggregation strategies.
    """

    DER = "der"
    """DER - Dynamic Ensemble Reasoning as MDP (Shen et al. 2024)."""

    MOA = "moa"
    """MoA - Mixture of Agents with layered model collaboration."""

    SLM_MUX = "slm_mux"
    """SLM-MUX - small model orchestration for efficient ensemble."""

    MULTI_AGENT_VERIFICATION = "multi_agent_verification"
    """Multi-Agent Verification - independent cross-verification ensemble."""

    EMA_FUSION = "ema_fusion"
    """EMAFusion - self-optimizing LLM integration with adaptive weights."""

    MODEL_SWITCH = "model_switch"
    """ModelSwitch - multi-LLM repeated sampling with dynamic selection."""

    TRAINING_FREE_ORCHESTRATION = "training_free_orchestration"
    """Training-Free Orchestration - central controller routing without training."""

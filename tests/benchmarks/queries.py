"""100-query benchmark suite for Reasoning Router evaluation.

This module defines the benchmark suite stratified by domain as specified
in Phase 1.5.2.2 of the router specification.

Each query includes:
- Expected primary domain
- Expected intent
- Expected complexity (1-10)
- Recommended method(s)
- Recommended pipeline (if applicable)

Domain distribution (100 queries total):
- Mathematical: 15 queries
- Code: 15 queries
- Ethical: 10 queries
- Creative: 10 queries
- Analytical: 10 queries
- Causal: 10 queries
- Decision: 10 queries
- Scientific: 10 queries
- General: 10 queries

Task 1.5.2.2: Define 100-query benchmark suite (stratified by domain)
"""

from __future__ import annotations

from reasoning_mcp.router.evaluation import BenchmarkQuery
from reasoning_mcp.router.models import ProblemDomain, ProblemIntent

# =============================================================================
# MATHEMATICAL QUERIES (15)
# =============================================================================

MATH_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        query="Prove that √2 is irrational",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=7,
        recommended_methods=["mathematical_reasoning", "logic_of_thought"],
        recommended_pipeline="math_proof",
        tags=["proof", "number-theory"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="What is the derivative of sin(x)cos(x)?",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=4,
        recommended_methods=["mathematical_reasoning", "chain_of_thought"],
        tags=["calculus", "derivatives"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Find the eigenvalues of the matrix [[1,2],[3,4]]",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=6,
        recommended_methods=["mathematical_reasoning"],
        tags=["linear-algebra", "matrices"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Calculate the integral of e^(-x²) from 0 to infinity",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=8,
        recommended_methods=["mathematical_reasoning", "step_back"],
        recommended_pipeline="math_proof",
        tags=["calculus", "integration", "special-functions"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="How many ways can you arrange 5 books on a shelf?",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=3,
        recommended_methods=["chain_of_thought", "mathematical_reasoning"],
        tags=["combinatorics", "permutations"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Prove by induction that the sum of first n natural numbers is n(n+1)/2",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=5,
        recommended_methods=["mathematical_reasoning", "logic_of_thought"],
        recommended_pipeline="math_proof",
        tags=["proof", "induction"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Solve the differential equation dy/dx = y with initial condition y(0) = 1",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=5,
        recommended_methods=["mathematical_reasoning"],
        tags=["differential-equations"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Find the limit of (sin x)/x as x approaches 0",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=4,
        recommended_methods=["mathematical_reasoning", "chain_of_thought"],
        tags=["calculus", "limits"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="What is the probability of rolling a sum of 7 with two dice?",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=3,
        recommended_methods=["chain_of_thought", "mathematical_reasoning"],
        tags=["probability"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Prove that there are infinitely many prime numbers",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=6,
        recommended_methods=["mathematical_reasoning", "logic_of_thought"],
        recommended_pipeline="math_proof",
        tags=["proof", "number-theory", "primes"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Find all solutions to x² + 4x + 3 = 0",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=2,
        recommended_methods=["mathematical_reasoning", "chain_of_thought"],
        tags=["algebra", "quadratic"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Calculate the volume of a sphere with radius 5",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=2,
        recommended_methods=["chain_of_thought"],
        tags=["geometry"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Determine if the series sum(1/n²) from n=1 to infinity converges",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=6,
        recommended_methods=["mathematical_reasoning", "step_back"],
        tags=["calculus", "series", "convergence"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Find the Taylor series expansion of e^x around x=0",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=5,
        recommended_methods=["mathematical_reasoning"],
        tags=["calculus", "series"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Prove that the function f(x) = x² is continuous at x = 2",
        expected_domain=ProblemDomain.MATHEMATICAL,
        expected_intent=ProblemIntent.VERIFY,
        expected_complexity=5,
        recommended_methods=["mathematical_reasoning", "logic_of_thought"],
        tags=["analysis", "continuity"],
        difficulty="medium",
    ),
]

# =============================================================================
# CODE QUERIES (15)
# =============================================================================

CODE_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        query="Debug this Python function that has an off-by-one error in the loop",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.DEBUG,
        expected_complexity=5,
        recommended_methods=["code_reasoning", "react"],
        recommended_pipeline="debug_code",
        tags=["debugging", "python", "loops"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Refactor this code to use async/await instead of callbacks",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.OPTIMIZE,
        expected_complexity=6,
        recommended_methods=["code_reasoning", "self_refine"],
        tags=["refactoring", "async"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Explain what this recursive function does and its time complexity",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=5,
        recommended_methods=["code_reasoning", "chain_of_thought"],
        tags=["analysis", "recursion", "complexity"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Write a function to find the longest palindromic substring",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=7,
        recommended_methods=["code_reasoning", "chain_of_thought"],
        tags=["algorithms", "strings", "dynamic-programming"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Find the memory leak in this C++ code",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.DEBUG,
        expected_complexity=7,
        recommended_methods=["code_reasoning", "react"],
        recommended_pipeline="debug_code",
        tags=["debugging", "memory", "cpp"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Identify potential SQL injection vulnerabilities in this code",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=6,
        recommended_methods=["code_reasoning", "chain_of_thought"],
        tags=["security", "sql", "vulnerabilities"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Design a REST API endpoint for user authentication",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=6,
        recommended_methods=["code_reasoning", "chain_of_thought"],
        tags=["api-design", "authentication"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Write unit tests for this sorting function",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=4,
        recommended_methods=["code_reasoning"],
        tags=["testing", "unit-tests"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Optimize this O(n²) algorithm to O(n log n)",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.OPTIMIZE,
        expected_complexity=7,
        recommended_methods=["code_reasoning", "step_back"],
        tags=["optimization", "algorithms", "complexity"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Implement a binary search tree with insert, delete, and search operations",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=5,
        recommended_methods=["code_reasoning", "chain_of_thought"],
        tags=["data-structures", "trees"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Convert this synchronous function to use Python async/await",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.OPTIMIZE,
        expected_complexity=5,
        recommended_methods=["code_reasoning"],
        tags=["async", "python", "refactoring"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query=(
            "What does the following regex pattern match: "
            "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"
        ),
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=4,
        recommended_methods=["code_reasoning", "chain_of_thought"],
        tags=["regex", "patterns"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Implement a thread-safe singleton pattern in Java",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=5,
        recommended_methods=["code_reasoning"],
        tags=["design-patterns", "concurrency", "java"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Debug why this React component is re-rendering infinitely",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.DEBUG,
        expected_complexity=6,
        recommended_methods=["code_reasoning", "react"],
        recommended_pipeline="debug_code",
        tags=["debugging", "react", "performance"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Write a function to detect cycles in a linked list",
        expected_domain=ProblemDomain.CODE,
        expected_intent=ProblemIntent.SOLVE,
        expected_complexity=5,
        recommended_methods=["code_reasoning", "chain_of_thought"],
        tags=["algorithms", "linked-list"],
        difficulty="medium",
    ),
]

# =============================================================================
# ETHICAL QUERIES (10)
# =============================================================================

ETHICAL_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        query="Is it ethical to lie to protect someone's feelings?",
        expected_domain=ProblemDomain.ETHICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=6,
        recommended_methods=["ethical_reasoning", "dialectic"],
        recommended_pipeline="ethical_multi_view",
        tags=["deception", "harm", "relationships"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Should autonomous vehicles prioritize passenger or pedestrian safety?",
        expected_domain=ProblemDomain.ETHICAL,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=8,
        recommended_methods=["ethical_reasoning", "multi_agent_debate"],
        recommended_pipeline="ethical_multi_view",
        tags=["trolley-problem", "ai", "autonomy"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="What are the ethical implications of using AI in hiring decisions?",
        expected_domain=ProblemDomain.ETHICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=7,
        recommended_methods=["ethical_reasoning", "socratic"],
        recommended_pipeline="ethical_multi_view",
        tags=["ai", "fairness", "employment"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Is it ethical for companies to collect user data for personalized advertising?",
        expected_domain=ProblemDomain.ETHICAL,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=6,
        recommended_methods=["ethical_reasoning", "dialectic"],
        recommended_pipeline="ethical_multi_view",
        tags=["privacy", "consent", "business"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Should genetic editing of human embryos be allowed to prevent disease?",
        expected_domain=ProblemDomain.ETHICAL,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=8,
        recommended_methods=["ethical_reasoning", "multi_agent_debate"],
        recommended_pipeline="ethical_multi_view",
        tags=["genetics", "medicine", "consent"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Is civil disobedience ever morally justified?",
        expected_domain=ProblemDomain.ETHICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=7,
        recommended_methods=["ethical_reasoning", "socratic"],
        recommended_pipeline="ethical_multi_view",
        tags=["justice", "law", "protest"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="What ethical obligations do wealthy nations have toward climate change?",
        expected_domain=ProblemDomain.ETHICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=7,
        recommended_methods=["ethical_reasoning", "dialectic"],
        tags=["environment", "justice", "responsibility"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Is it ethical to use animals for medical research?",
        expected_domain=ProblemDomain.ETHICAL,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=6,
        recommended_methods=["ethical_reasoning", "multi_agent_debate"],
        recommended_pipeline="ethical_multi_view",
        tags=["animals", "research", "medicine"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Should social media platforms be held responsible for misinformation?",
        expected_domain=ProblemDomain.ETHICAL,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=6,
        recommended_methods=["ethical_reasoning", "dialectic"],
        tags=["free-speech", "responsibility", "technology"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Is it ethical for parents to track their children's location?",
        expected_domain=ProblemDomain.ETHICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=5,
        recommended_methods=["ethical_reasoning", "socratic"],
        tags=["privacy", "parenting", "autonomy"],
        difficulty="medium",
    ),
]

# =============================================================================
# CREATIVE QUERIES (10)
# =============================================================================

CREATIVE_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        query="Generate 5 novel uses for a paperclip beyond holding papers",
        expected_domain=ProblemDomain.CREATIVE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=3,
        recommended_methods=["lateral_thinking", "tree_of_thoughts"],
        recommended_pipeline="creative_explore",
        tags=["brainstorming", "divergent-thinking"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Design a new board game that teaches programming concepts to children",
        expected_domain=ProblemDomain.CREATIVE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=7,
        recommended_methods=["lateral_thinking", "tree_of_thoughts"],
        recommended_pipeline="creative_explore",
        tags=["game-design", "education"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Write a haiku about artificial intelligence",
        expected_domain=ProblemDomain.CREATIVE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=3,
        recommended_methods=["lateral_thinking"],
        tags=["poetry", "writing"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Propose innovative solutions to reduce plastic waste in oceans",
        expected_domain=ProblemDomain.CREATIVE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=6,
        recommended_methods=["lateral_thinking", "tree_of_thoughts"],
        recommended_pipeline="creative_explore",
        tags=["environment", "innovation"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Create a unique marketing campaign for a sustainable fashion brand",
        expected_domain=ProblemDomain.CREATIVE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=6,
        recommended_methods=["lateral_thinking", "counterfactual"],
        tags=["marketing", "business"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Imagine what cities might look like in 100 years",
        expected_domain=ProblemDomain.CREATIVE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=5,
        recommended_methods=["lateral_thinking", "counterfactual"],
        tags=["futurism", "imagination"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Design a new emoji that represents the feeling of nostalgia",
        expected_domain=ProblemDomain.CREATIVE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=4,
        recommended_methods=["lateral_thinking"],
        tags=["design", "emotions"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Create a plot outline for a mystery novel set in a virtual reality world",
        expected_domain=ProblemDomain.CREATIVE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=6,
        recommended_methods=["lateral_thinking", "tree_of_thoughts"],
        tags=["writing", "storytelling"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Invent a new sport that can be played in zero gravity",
        expected_domain=ProblemDomain.CREATIVE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=5,
        recommended_methods=["lateral_thinking", "counterfactual"],
        tags=["sports", "innovation"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Design an app feature that helps people form new habits",
        expected_domain=ProblemDomain.CREATIVE,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=5,
        recommended_methods=["lateral_thinking", "tree_of_thoughts"],
        tags=["product-design", "psychology"],
        difficulty="medium",
    ),
]

# =============================================================================
# ANALYTICAL QUERIES (10)
# =============================================================================

ANALYTICAL_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        query="Compare the trade-offs between SQL and NoSQL databases",
        expected_domain=ProblemDomain.ANALYTICAL,
        expected_intent=ProblemIntent.COMPARE,
        expected_complexity=6,
        recommended_methods=["chain_of_thought", "self_consistency"],
        tags=["databases", "architecture"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Analyze the strengths and weaknesses of the agile methodology",
        expected_domain=ProblemDomain.ANALYTICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=5,
        recommended_methods=["chain_of_thought", "dialectic"],
        tags=["methodology", "software"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Break down the factors that contributed to the 2008 financial crisis",
        expected_domain=ProblemDomain.ANALYTICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=8,
        recommended_methods=["chain_of_thought", "causal_reasoning"],
        tags=["economics", "history"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Compare microservices vs monolithic architecture for a startup",
        expected_domain=ProblemDomain.ANALYTICAL,
        expected_intent=ProblemIntent.COMPARE,
        expected_complexity=6,
        recommended_methods=["chain_of_thought", "self_consistency"],
        tags=["architecture", "startup"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Analyze why some startups fail while others succeed",
        expected_domain=ProblemDomain.ANALYTICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=7,
        recommended_methods=["chain_of_thought", "causal_reasoning"],
        tags=["business", "success-factors"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query=(
            "Compare the effectiveness of different machine learning algorithms "
            "for image classification"
        ),
        expected_domain=ProblemDomain.ANALYTICAL,
        expected_intent=ProblemIntent.COMPARE,
        expected_complexity=7,
        recommended_methods=["chain_of_thought", "self_consistency"],
        tags=["machine-learning", "comparison"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Analyze the impact of remote work on team productivity",
        expected_domain=ProblemDomain.ANALYTICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=5,
        recommended_methods=["chain_of_thought"],
        tags=["work", "productivity"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Break down the components of a successful product launch",
        expected_domain=ProblemDomain.ANALYTICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=5,
        recommended_methods=["chain_of_thought", "decomposed"],
        tags=["product", "business"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Compare static vs dynamic typing in programming languages",
        expected_domain=ProblemDomain.ANALYTICAL,
        expected_intent=ProblemIntent.COMPARE,
        expected_complexity=5,
        recommended_methods=["chain_of_thought", "dialectic"],
        tags=["programming", "types"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Analyze the factors that make a programming language popular",
        expected_domain=ProblemDomain.ANALYTICAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=6,
        recommended_methods=["chain_of_thought", "causal_reasoning"],
        tags=["programming", "trends"],
        difficulty="medium",
    ),
]

# =============================================================================
# CAUSAL QUERIES (10)
# =============================================================================

CAUSAL_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        query="What factors contribute to climate change?",
        expected_domain=ProblemDomain.CAUSAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=6,
        recommended_methods=["causal_reasoning", "chain_of_thought"],
        tags=["climate", "environment"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Why do software projects often run over budget and schedule?",
        expected_domain=ProblemDomain.CAUSAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=6,
        recommended_methods=["causal_reasoning", "counterfactual"],
        tags=["software", "project-management"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="What would happen if we removed all social media platforms?",
        expected_domain=ProblemDomain.CAUSAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=6,
        recommended_methods=["counterfactual", "causal_reasoning"],
        tags=["counterfactual", "social-media"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="What caused the decline of the Roman Empire?",
        expected_domain=ProblemDomain.CAUSAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=8,
        recommended_methods=["causal_reasoning", "chain_of_thought"],
        tags=["history", "civilization"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="How does sleep deprivation affect cognitive performance?",
        expected_domain=ProblemDomain.CAUSAL,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=5,
        recommended_methods=["causal_reasoning"],
        tags=["health", "cognition"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="What would happen if interest rates were raised by 5%?",
        expected_domain=ProblemDomain.CAUSAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=7,
        recommended_methods=["counterfactual", "causal_reasoning"],
        tags=["economics", "counterfactual"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Why do some habits stick while others don't?",
        expected_domain=ProblemDomain.CAUSAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=5,
        recommended_methods=["causal_reasoning"],
        tags=["psychology", "behavior"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="What causes inflation in an economy?",
        expected_domain=ProblemDomain.CAUSAL,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=6,
        recommended_methods=["causal_reasoning", "chain_of_thought"],
        tags=["economics", "inflation"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Why do some companies successfully pivot while others fail?",
        expected_domain=ProblemDomain.CAUSAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=6,
        recommended_methods=["causal_reasoning", "counterfactual"],
        tags=["business", "strategy"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="What would happen if Moore's Law suddenly stopped?",
        expected_domain=ProblemDomain.CAUSAL,
        expected_intent=ProblemIntent.ANALYZE,
        expected_complexity=7,
        recommended_methods=["counterfactual", "causal_reasoning"],
        tags=["technology", "counterfactual"],
        difficulty="hard",
    ),
]

# =============================================================================
# DECISION QUERIES (10)
# =============================================================================

DECISION_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        query=(
            "Should I accept job offer A with higher salary "
            "or job offer B with better work-life balance?"
        ),
        expected_domain=ProblemDomain.DECISION,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=6,
        recommended_methods=["tree_of_thoughts", "mcts"],
        recommended_pipeline="decision_matrix",
        tags=["career", "tradeoffs"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Which programming language should I learn first as a beginner?",
        expected_domain=ProblemDomain.DECISION,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=4,
        recommended_methods=["tree_of_thoughts", "chain_of_thought"],
        tags=["programming", "learning"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Should our company build or buy this software component?",
        expected_domain=ProblemDomain.DECISION,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=7,
        recommended_methods=["tree_of_thoughts", "mcts"],
        recommended_pipeline="decision_matrix",
        tags=["business", "software", "strategy"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Which cloud provider should we choose for our startup?",
        expected_domain=ProblemDomain.DECISION,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=6,
        recommended_methods=["tree_of_thoughts", "chain_of_thought"],
        recommended_pipeline="decision_matrix",
        tags=["technology", "infrastructure"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Should I pursue a PhD or start working after my Master's?",
        expected_domain=ProblemDomain.DECISION,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=6,
        recommended_methods=["tree_of_thoughts", "dialectic"],
        tags=["career", "education"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Which investment strategy is best for long-term wealth building?",
        expected_domain=ProblemDomain.DECISION,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=7,
        recommended_methods=["tree_of_thoughts", "mcts"],
        tags=["finance", "investment"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Should we use React, Vue, or Angular for our new web application?",
        expected_domain=ProblemDomain.DECISION,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=5,
        recommended_methods=["tree_of_thoughts", "chain_of_thought"],
        tags=["technology", "frontend"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Should I rent or buy a home in my current situation?",
        expected_domain=ProblemDomain.DECISION,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=6,
        recommended_methods=["tree_of_thoughts", "mcts"],
        recommended_pipeline="decision_matrix",
        tags=["finance", "housing"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Which database technology should we use for our real-time application?",
        expected_domain=ProblemDomain.DECISION,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=6,
        recommended_methods=["tree_of_thoughts", "chain_of_thought"],
        tags=["technology", "databases"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Should our team work in sprints or use continuous flow?",
        expected_domain=ProblemDomain.DECISION,
        expected_intent=ProblemIntent.EVALUATE,
        expected_complexity=5,
        recommended_methods=["tree_of_thoughts", "dialectic"],
        tags=["methodology", "agile"],
        difficulty="medium",
    ),
]

# =============================================================================
# SCIENTIFIC QUERIES (10)
# =============================================================================

SCIENTIFIC_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        query="Explain the mechanism of mRNA vaccines",
        expected_domain=ProblemDomain.SCIENTIFIC,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=6,
        recommended_methods=["step_back", "chain_of_thought"],
        recommended_pipeline="scientific_method",
        tags=["biology", "medicine", "vaccines"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="How does quantum entanglement work?",
        expected_domain=ProblemDomain.SCIENTIFIC,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=8,
        recommended_methods=["step_back", "mathematical_reasoning"],
        tags=["physics", "quantum"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="Design an experiment to test if a new drug reduces inflammation",
        expected_domain=ProblemDomain.SCIENTIFIC,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=7,
        recommended_methods=["step_back", "chain_of_thought"],
        recommended_pipeline="scientific_method",
        tags=["biology", "experiment-design"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="What causes antibiotic resistance in bacteria?",
        expected_domain=ProblemDomain.SCIENTIFIC,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=6,
        recommended_methods=["step_back", "causal_reasoning"],
        tags=["biology", "medicine"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Explain how CRISPR gene editing works",
        expected_domain=ProblemDomain.SCIENTIFIC,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=7,
        recommended_methods=["step_back", "chain_of_thought"],
        tags=["biology", "genetics"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="What evidence supports the Big Bang theory?",
        expected_domain=ProblemDomain.SCIENTIFIC,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=7,
        recommended_methods=["step_back", "chain_of_thought"],
        tags=["physics", "cosmology"],
        difficulty="hard",
    ),
    BenchmarkQuery(
        query="How do neural networks learn from data?",
        expected_domain=ProblemDomain.SCIENTIFIC,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=6,
        recommended_methods=["step_back", "mathematical_reasoning"],
        tags=["computer-science", "machine-learning"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="What causes ocean acidification and its effects?",
        expected_domain=ProblemDomain.SCIENTIFIC,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=5,
        recommended_methods=["step_back", "causal_reasoning"],
        tags=["environment", "chemistry"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="Explain the process of photosynthesis",
        expected_domain=ProblemDomain.SCIENTIFIC,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=5,
        recommended_methods=["step_back", "chain_of_thought"],
        tags=["biology", "biochemistry"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="How does the immune system identify and fight pathogens?",
        expected_domain=ProblemDomain.SCIENTIFIC,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=6,
        recommended_methods=["step_back", "chain_of_thought"],
        tags=["biology", "immunology"],
        difficulty="medium",
    ),
]

# =============================================================================
# GENERAL QUERIES (10)
# =============================================================================

GENERAL_QUERIES: list[BenchmarkQuery] = [
    BenchmarkQuery(
        query="Summarize the key points of this article about remote work",
        expected_domain=ProblemDomain.GENERAL,
        expected_intent=ProblemIntent.SYNTHESIZE,
        expected_complexity=3,
        recommended_methods=["chain_of_thought", "sequential_thinking"],
        tags=["summarization"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="What are the best practices for conducting a job interview?",
        expected_domain=ProblemDomain.GENERAL,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=4,
        recommended_methods=["chain_of_thought"],
        tags=["career", "hiring"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="Help me plan a two-week trip to Japan",
        expected_domain=ProblemDomain.GENERAL,
        expected_intent=ProblemIntent.GENERATE,
        expected_complexity=5,
        recommended_methods=["chain_of_thought", "tree_of_thoughts"],
        tags=["travel", "planning"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="What should I consider when buying a laptop?",
        expected_domain=ProblemDomain.GENERAL,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=4,
        recommended_methods=["chain_of_thought"],
        tags=["technology", "purchasing"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="How can I improve my public speaking skills?",
        expected_domain=ProblemDomain.GENERAL,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=4,
        recommended_methods=["chain_of_thought", "self_refine"],
        tags=["skills", "communication"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="What are some effective strategies for time management?",
        expected_domain=ProblemDomain.GENERAL,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=4,
        recommended_methods=["chain_of_thought"],
        tags=["productivity", "self-improvement"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="How do I prepare for a marathon?",
        expected_domain=ProblemDomain.GENERAL,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=5,
        recommended_methods=["chain_of_thought", "sequential_thinking"],
        tags=["fitness", "planning"],
        difficulty="medium",
    ),
    BenchmarkQuery(
        query="What makes a good book club discussion?",
        expected_domain=ProblemDomain.GENERAL,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=3,
        recommended_methods=["chain_of_thought"],
        tags=["social", "reading"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="How can I reduce my carbon footprint?",
        expected_domain=ProblemDomain.GENERAL,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=4,
        recommended_methods=["chain_of_thought"],
        tags=["environment", "lifestyle"],
        difficulty="easy",
    ),
    BenchmarkQuery(
        query="What are the key elements of effective team communication?",
        expected_domain=ProblemDomain.GENERAL,
        expected_intent=ProblemIntent.EXPLAIN,
        expected_complexity=4,
        recommended_methods=["chain_of_thought"],
        tags=["teamwork", "communication"],
        difficulty="easy",
    ),
]

# =============================================================================
# COMBINED BENCHMARK SUITE
# =============================================================================

ALL_BENCHMARK_QUERIES: list[BenchmarkQuery] = (
    MATH_QUERIES
    + CODE_QUERIES
    + ETHICAL_QUERIES
    + CREATIVE_QUERIES
    + ANALYTICAL_QUERIES
    + CAUSAL_QUERIES
    + DECISION_QUERIES
    + SCIENTIFIC_QUERIES
    + GENERAL_QUERIES
)

# Verify count
assert len(ALL_BENCHMARK_QUERIES) == 100, f"Expected 100 queries, got {len(ALL_BENCHMARK_QUERIES)}"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_queries_by_domain(domain: ProblemDomain) -> list[BenchmarkQuery]:
    """Get all benchmark queries for a specific domain.

    Args:
        domain: The problem domain to filter by

    Returns:
        List of benchmark queries for that domain
    """
    return [q for q in ALL_BENCHMARK_QUERIES if q.expected_domain == domain]


def get_queries_by_difficulty(difficulty: str) -> list[BenchmarkQuery]:
    """Get all benchmark queries of a specific difficulty.

    Args:
        difficulty: "easy", "medium", or "hard"

    Returns:
        List of benchmark queries at that difficulty
    """
    return [q for q in ALL_BENCHMARK_QUERIES if q.difficulty == difficulty]


def get_queries_by_tag(tag: str) -> list[BenchmarkQuery]:
    """Get all benchmark queries with a specific tag.

    Args:
        tag: The tag to filter by

    Returns:
        List of benchmark queries with that tag
    """
    return [q for q in ALL_BENCHMARK_QUERIES if tag in q.tags]


def get_benchmark_statistics() -> dict:
    """Get statistics about the benchmark suite.

    Returns:
        Dictionary with counts by domain, difficulty, etc.
    """
    return {
        "total_queries": len(ALL_BENCHMARK_QUERIES),
        "by_domain": {domain.value: len(get_queries_by_domain(domain)) for domain in ProblemDomain},
        "by_difficulty": {
            diff: len(get_queries_by_difficulty(diff)) for diff in ["easy", "medium", "hard"]
        },
        "avg_complexity": (
            sum(q.expected_complexity for q in ALL_BENCHMARK_QUERIES) / len(ALL_BENCHMARK_QUERIES)
        ),
    }

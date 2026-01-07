"""Lotus Wisdom reasoning method implementation.

This module implements the Lotus Wisdom method - a holistic approach that examines
problems through 5 wisdom domains inspired by the lotus flower's layers: Technical,
Emotional, Ethical, Practical, and Intuitive perspectives.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from reasoning_mcp.methods.base import MethodMetadata
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtGraph, ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.models.session import Session


# Metadata for Lotus Wisdom method
LOTUS_WISDOM_METADATA = MethodMetadata(
    identifier=MethodIdentifier.LOTUS_WISDOM,
    name="Lotus Wisdom",
    description="5-domain holistic analysis examining technical, emotional, ethical, practical, and intuitive aspects",
    category=MethodCategory.HOLISTIC,
    tags=frozenset({
        "holistic",
        "wisdom",
        "multi-domain",
        "balanced",
        "integrative",
        "comprehensive",
    }),
    complexity=7,  # High complexity due to multi-domain analysis
    supports_branching=True,  # Each domain is a branch
    supports_revision=True,  # Can revise domain analyses
    requires_context=False,  # No special context needed
    min_thoughts=7,  # Center + 5 domains + synthesis
    max_thoughts=0,  # No limit (can have deeper analysis in each domain)
    avg_tokens_per_thought=500,  # Moderate token usage per domain
    best_for=(
        "complex life decisions",
        "organizational decisions",
        "situations requiring wisdom",
        "balancing multiple concerns",
        "holistic problem-solving",
        "multi-stakeholder decisions",
    ),
    not_recommended_for=(
        "purely technical problems",
        "time-critical decisions",
        "simple factual queries",
        "narrow domain-specific issues",
    ),
)


class LotusWisdomMethod:
    """Lotus Wisdom reasoning method implementation.

    This method examines problems through 5 wisdom domains, like the petals
    of a lotus flower surrounding a central core. Each domain provides a
    unique perspective, and together they create a comprehensive, balanced
    understanding.

    The 5 Wisdom Domains (Petals):
    1. TECHNICAL: Logical analysis, feasibility, implementation details
    2. EMOTIONAL: Emotional intelligence, human impact, feelings
    3. ETHICAL: Moral considerations, values, principles
    4. PRACTICAL: Real-world constraints, resources, pragmatism
    5. INTUITIVE: Pattern recognition, gut feelings, holistic sensing

    Structure:
    - Center: Core problem understanding and essence
    - Petals: Five domain analyses branching from center
    - Synthesis: Integration of all domains into wisdom

    The method tracks domain balance to ensure no perspective is neglected.

    Examples:
        Initialize and execute:
        >>> method = LotusWisdomMethod()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> graph = await method.execute(
        ...     session=session,
        ...     problem="Should we adopt a 4-day work week?"
        ... )
        >>> assert graph.node_count == 7  # Center + 5 domains + synthesis

        Continue reasoning with feedback:
        >>> updated_graph = await method.continue_reasoning(
        ...     session=session,
        ...     graph=graph,
        ...     feedback="Explore the technical domain more deeply"
        ... )
    """

    # Domain definitions
    DOMAINS = {
        "TECHNICAL": {
            "name": "Technical",
            "description": "Logical/technical analysis, feasibility, implementation",
            "focus": [
                "Logical structure and coherence",
                "Technical feasibility and constraints",
                "Implementation requirements and steps",
                "Data, facts, and measurable aspects",
                "Systematic analysis and reasoning",
            ],
        },
        "EMOTIONAL": {
            "name": "Emotional",
            "description": "Emotional intelligence, human impact, feelings",
            "focus": [
                "Emotional impact on individuals and groups",
                "Human feelings and psychological effects",
                "Empathy and emotional intelligence",
                "Morale, motivation, and well-being",
                "Interpersonal dynamics and reactions",
            ],
        },
        "ETHICAL": {
            "name": "Ethical",
            "description": "Moral considerations, values, principles",
            "focus": [
                "Moral principles and ethical frameworks",
                "Values and what matters most",
                "Right and wrong, fairness and justice",
                "Integrity and alignment with principles",
                "Broader societal and moral implications",
            ],
        },
        "PRACTICAL": {
            "name": "Practical",
            "description": "Real-world constraints, resources, pragmatism",
            "focus": [
                "Real-world constraints and limitations",
                "Resource availability (time, money, people)",
                "Pragmatic considerations and trade-offs",
                "Feasibility in current circumstances",
                "Short-term and long-term practicality",
            ],
        },
        "INTUITIVE": {
            "name": "Intuitive",
            "description": "Pattern recognition, gut feelings, holistic sensing",
            "focus": [
                "Pattern recognition and analogies",
                "Gut feelings and instinctive responses",
                "Holistic sensing beyond logic",
                "Unconscious wisdom and experience",
                "What feels right or wrong intuitively",
            ],
        },
    }

    def __init__(self) -> None:
        """Initialize the Lotus Wisdom method."""
        self._initialized = False
        self._step_counter = 0
        self._domain_balance: dict[str, int] = {
            domain: 0 for domain in self.DOMAINS
        }

    @property
    def identifier(self) -> str:
        """Get the method identifier.

        Returns:
            The MethodIdentifier enum value as a string
        """
        return str(MethodIdentifier.LOTUS_WISDOM)

    @property
    def name(self) -> str:
        """Get the human-readable method name.

        Returns:
            The method name
        """
        return LOTUS_WISDOM_METADATA.name

    @property
    def description(self) -> str:
        """Get the method description.

        Returns:
            A brief description of the method
        """
        return LOTUS_WISDOM_METADATA.description

    @property
    def category(self) -> str:
        """Get the method category.

        Returns:
            The MethodCategory enum value as a string
        """
        return str(MethodCategory.HOLISTIC)

    async def initialize(self) -> None:
        """Initialize the method.

        This method prepares the Lotus Wisdom method for execution.
        Resets internal state for multi-domain holistic analysis.

        Examples:
            >>> method = LotusWisdomMethod()
            >>> await method.initialize()
            >>> assert method._initialized is True
            >>> assert method._step_counter == 0
        """
        self._initialized = True
        self._step_counter = 0
        self._domain_balance = {domain: 0 for domain in self.DOMAINS}

    async def execute(
        self,
        session: Session,
        problem: str,
    ) -> ThoughtGraph:
        """Execute the Lotus Wisdom method.

        This method creates a complete lotus structure:
        1. Center thought: Understanding the problem essence
        2. Five domain petals: Analyzing from each perspective
        3. Synthesis: Integrating all domains into wisdom

        Args:
            session: The current reasoning session
            problem: The problem or question to analyze

        Returns:
            A ThoughtGraph containing the complete lotus structure

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = LotusWisdomMethod()
            >>> await method.initialize()
            >>> graph = await method.execute(
            ...     session=session,
            ...     problem="Should we expand internationally?"
            ... )
            >>> assert graph.node_count == 7  # Center + 5 domains + synthesis
            >>> assert graph.root_id is not None
        """
        if not self._initialized:
            raise RuntimeError(
                "LotusWisdomMethod must be initialized before execution"
            )

        # Reset state for new execution
        self._step_counter = 0
        self._domain_balance = {domain: 0 for domain in self.DOMAINS}

        # Create the thought graph for this lotus
        graph = ThoughtGraph()

        # Step 1: Create the center (problem essence)
        self._step_counter += 1
        center = self._create_center(problem)
        graph.add_thought(center)
        session.add_thought(center)
        session.current_method = MethodIdentifier.LOTUS_WISDOM

        # Steps 2-6: Create the five domain petals
        domain_thoughts = []
        for domain_key in self.DOMAINS:
            self._step_counter += 1
            domain_thought = self._create_domain_petal(
                problem=problem,
                domain_key=domain_key,
                center_id=center.id,
            )
            graph.add_thought(domain_thought)
            session.add_thought(domain_thought)
            domain_thoughts.append(domain_thought)
            self._domain_balance[domain_key] += 1

        # Step 7: Create the synthesis (lotus bloom)
        self._step_counter += 1
        synthesis = self._create_synthesis(
            problem=problem,
            center=center,
            domain_thoughts=domain_thoughts,
        )
        graph.add_thought(synthesis)
        session.add_thought(synthesis)

        return graph

    async def continue_reasoning(
        self,
        session: Session,
        graph: ThoughtGraph,
        feedback: str | None = None,
    ) -> ThoughtGraph:
        """Continue reasoning from a previous thought graph.

        This method can deepen analysis in specific domains, explore
        new perspectives, or refine the synthesis based on feedback.

        Args:
            session: The current reasoning session
            graph: The thought graph to continue from
            feedback: Optional feedback for continuation direction

        Returns:
            The updated ThoughtGraph with additional reasoning

        Raises:
            RuntimeError: If the method has not been initialized

        Examples:
            >>> session = Session().start()
            >>> method = LotusWisdomMethod()
            >>> await method.initialize()
            >>> graph = await method.execute(session, "Problem")
            >>> updated = await method.continue_reasoning(
            ...     session=session,
            ...     graph=graph,
            ...     feedback="Explore emotional aspects more deeply"
            ... )
            >>> assert updated.node_count > graph.node_count
        """
        if not self._initialized:
            raise RuntimeError(
                "LotusWisdomMethod must be initialized before continuation"
            )

        # Determine which domain to deepen or if we need new synthesis
        domain_to_deepen = self._determine_continuation_focus(feedback, graph)

        if domain_to_deepen:
            # Deepen analysis in specific domain
            self._step_counter += 1
            deeper_thought = self._deepen_domain_analysis(
                graph=graph,
                domain_key=domain_to_deepen,
                feedback=feedback,
            )
            graph.add_thought(deeper_thought)
            session.add_thought(deeper_thought)
            self._domain_balance[domain_to_deepen] += 1

            # Create new synthesis incorporating deeper insight
            self._step_counter += 1
            new_synthesis = self._create_refined_synthesis(
                graph=graph,
                new_insight=deeper_thought,
                feedback=feedback,
            )
            graph.add_thought(new_synthesis)
            session.add_thought(new_synthesis)

        else:
            # General continuation - add evaluative thought
            self._step_counter += 1
            evaluation = self._create_evaluation(
                graph=graph,
                feedback=feedback,
            )
            graph.add_thought(evaluation)
            session.add_thought(evaluation)

        return graph

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        For Lotus Wisdom, this checks initialization status.

        Returns:
            True if the method is initialized and healthy

        Examples:
            >>> method = LotusWisdomMethod()
            >>> assert await method.health_check() is False
            >>> await method.initialize()
            >>> assert await method.health_check() is True
        """
        return self._initialized

    def _create_center(self, problem: str) -> ThoughtNode:
        """Create the center thought representing problem essence.

        Args:
            problem: The problem to analyze

        Returns:
            ThoughtNode representing the center of the lotus
        """
        content = self._generate_center_content(problem)

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.LOTUS_WISDOM,
            content=content,
            summary="Problem essence and core understanding",
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,  # Initial moderate confidence
            metadata={
                "phase": "center",
                "structure": "lotus_core",
                "problem": problem,
                "domains_pending": list(self.DOMAINS.keys()),
            },
        )

    def _create_domain_petal(
        self,
        problem: str,
        domain_key: str,
        center_id: str,
    ) -> ThoughtNode:
        """Create a domain petal thought.

        Args:
            problem: The problem being analyzed
            domain_key: Which domain (TECHNICAL, EMOTIONAL, etc.)
            center_id: ID of the center thought

        Returns:
            ThoughtNode representing one domain petal
        """
        domain_info = self.DOMAINS[domain_key]
        content = self._generate_domain_content(problem, domain_key, domain_info)

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=MethodIdentifier.LOTUS_WISDOM,
            content=content,
            summary=f"{domain_info['name']} perspective analysis",
            parent_id=center_id,
            branch_id=f"domain_{domain_key.lower()}",
            step_number=self._step_counter,
            depth=1,
            confidence=0.7,  # Good confidence in domain analysis
            metadata={
                "phase": "domain_analysis",
                "domain": domain_key,
                "domain_name": domain_info["name"],
                "structure": "lotus_petal",
            },
        )

    def _create_synthesis(
        self,
        problem: str,
        center: ThoughtNode,
        domain_thoughts: list[ThoughtNode],
    ) -> ThoughtNode:
        """Create the synthesis thought integrating all domains.

        Args:
            problem: The problem being analyzed
            center: The center thought
            domain_thoughts: List of all domain petal thoughts

        Returns:
            ThoughtNode representing the synthesis
        """
        content = self._generate_synthesis_content(
            problem, center, domain_thoughts
        )

        # Synthesis is a child of the center, synthesizing all branches
        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.LOTUS_WISDOM,
            content=content,
            summary="Integrated wisdom from all domains",
            parent_id=center.id,
            step_number=self._step_counter,
            depth=2,
            confidence=0.85,  # Higher confidence from integration
            quality_score=0.9,  # High quality holistic analysis
            metadata={
                "phase": "synthesis",
                "structure": "lotus_bloom",
                "domains_synthesized": list(self.DOMAINS.keys()),
                "balance_check": self._check_domain_balance(),
            },
        )

    def _deepen_domain_analysis(
        self,
        graph: ThoughtGraph,
        domain_key: str,
        feedback: str | None,
    ) -> ThoughtNode:
        """Create a deeper analysis thought for a specific domain.

        Args:
            graph: The current thought graph
            domain_key: Which domain to deepen
            feedback: Optional feedback guiding the deepening

        Returns:
            ThoughtNode with deeper domain analysis
        """
        domain_info = self.DOMAINS[domain_key]

        # Find the original domain petal thought
        original_petal = None
        for node in graph.nodes.values():
            if node.metadata.get("domain") == domain_key:
                original_petal = node
                break

        parent_id = original_petal.id if original_petal else graph.root_id

        content = self._generate_deeper_domain_content(
            domain_key, domain_info, feedback
        )

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.LOTUS_WISDOM,
            content=content,
            summary=f"Deeper {domain_info['name']} analysis",
            parent_id=parent_id,
            branch_id=f"domain_{domain_key.lower()}",
            step_number=self._step_counter,
            depth=(original_petal.depth + 1) if original_petal else 2,
            confidence=0.75,
            metadata={
                "phase": "deepening",
                "domain": domain_key,
                "domain_name": domain_info["name"],
                "feedback": feedback or "",
            },
        )

    def _create_refined_synthesis(
        self,
        graph: ThoughtGraph,
        new_insight: ThoughtNode,
        feedback: str | None,
    ) -> ThoughtNode:
        """Create a refined synthesis incorporating new insights.

        Args:
            graph: The current thought graph
            new_insight: The new insight to incorporate
            feedback: Optional feedback

        Returns:
            ThoughtNode representing refined synthesis
        """
        # Find previous synthesis
        previous_synthesis = None
        for node in graph.nodes.values():
            if node.metadata.get("phase") == "synthesis":
                previous_synthesis = node

        parent_id = previous_synthesis.id if previous_synthesis else graph.root_id
        depth = (previous_synthesis.depth + 1) if previous_synthesis else 3

        content = self._generate_refined_synthesis_content(
            new_insight, feedback
        )

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=MethodIdentifier.LOTUS_WISDOM,
            content=content,
            summary="Refined integrated wisdom",
            parent_id=parent_id,
            step_number=self._step_counter,
            depth=depth,
            confidence=0.87,
            quality_score=0.92,
            metadata={
                "phase": "refined_synthesis",
                "incorporated_insight_id": new_insight.id,
                "balance_check": self._check_domain_balance(),
            },
        )

    def _create_evaluation(
        self,
        graph: ThoughtGraph,
        feedback: str | None,
    ) -> ThoughtNode:
        """Create an evaluation thought assessing the analysis.

        Args:
            graph: The current thought graph
            feedback: Optional feedback

        Returns:
            ThoughtNode with evaluation
        """
        # Find the latest synthesis or last node
        latest_synthesis = None
        for node in graph.nodes.values():
            if "synthesis" in node.metadata.get("phase", ""):
                latest_synthesis = node

        parent_id = latest_synthesis.id if latest_synthesis else graph.root_id
        depth = (latest_synthesis.depth + 1) if latest_synthesis else 3

        content = self._generate_evaluation_content(graph, feedback)

        return ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INSIGHT,
            method_id=MethodIdentifier.LOTUS_WISDOM,
            content=content,
            summary="Evaluation and insights",
            parent_id=parent_id,
            step_number=self._step_counter,
            depth=depth,
            confidence=0.8,
            metadata={
                "phase": "evaluation",
                "feedback": feedback or "",
                "balance_check": self._check_domain_balance(),
            },
        )

    def _determine_continuation_focus(
        self,
        feedback: str | None,
        graph: ThoughtGraph,
    ) -> str | None:
        """Determine which domain to focus on for continuation.

        Args:
            feedback: Optional feedback string
            graph: Current thought graph

        Returns:
            Domain key to focus on, or None for general continuation
        """
        if not feedback:
            # Check for least analyzed domain
            min_count = min(self._domain_balance.values())
            for domain, count in self._domain_balance.items():
                if count == min_count:
                    return domain
            return None

        feedback_lower = feedback.lower()

        # Check for domain keywords in feedback
        domain_keywords = {
            "TECHNICAL": ["technical", "logic", "feasibility", "implementation"],
            "EMOTIONAL": ["emotional", "feeling", "human", "impact", "empathy"],
            "ETHICAL": ["ethical", "moral", "values", "principles", "right"],
            "PRACTICAL": ["practical", "pragmatic", "resource", "constraint"],
            "INTUITIVE": ["intuitive", "gut", "pattern", "instinct", "holistic"],
        }

        for domain, keywords in domain_keywords.items():
            if any(keyword in feedback_lower for keyword in keywords):
                return domain

        return None

    def _check_domain_balance(self) -> dict[str, int]:
        """Check the balance of domain analyses.

        Returns:
            Dictionary of domain analysis counts
        """
        return dict(self._domain_balance)

    # Content generation methods
    # In a full implementation, these would use an LLM to generate actual content

    def _generate_center_content(self, problem: str) -> str:
        """Generate content for the center thought."""
        return (
            f"Step {self._step_counter}: CENTER - Understanding the Essence\n\n"
            f"Problem: {problem}\n\n"
            f"At the center of the lotus, we find the essential nature of this question. "
            f"Before examining from multiple perspectives, let us understand the core:\n\n"
            f"- What is the fundamental question being asked?\n"
            f"- What are the key elements and relationships?\n"
            f"- What makes this problem significant?\n"
            f"- What wisdom is needed here?\n\n"
            f"From this center, we will unfold five petals of wisdom - Technical, "
            f"Emotional, Ethical, Practical, and Intuitive - each offering a unique "
            f"lens through which to view this question."
        )

    def _generate_domain_content(
        self, problem: str, domain_key: str, domain_info: dict
    ) -> str:
        """Generate content for a domain petal thought."""
        focus_items = "\n".join(f"- {item}" for item in domain_info["focus"])

        return (
            f"Step {self._step_counter}: {domain_info['name'].upper()} DOMAIN\n\n"
            f"Examining this problem through the {domain_info['name']} lens:\n"
            f"{domain_info['description']}\n\n"
            f"Key considerations in this domain:\n{focus_items}\n\n"
            f"This domain reveals aspects of the problem that require {domain_info['name'].lower()} "
            f"wisdom and understanding. It provides one essential perspective that must be "
            f"integrated with the other domains for complete wisdom."
        )

    def _generate_synthesis_content(
        self,
        problem: str,
        center: ThoughtNode,
        domain_thoughts: list[ThoughtNode],
    ) -> str:
        """Generate content for the synthesis thought."""
        domain_names = [
            self.DOMAINS[dt.metadata["domain"]]["name"]
            for dt in domain_thoughts
        ]

        return (
            f"Step {self._step_counter}: SYNTHESIS - The Lotus Blooms\n\n"
            f"Having examined this question through {len(domain_thoughts)} wisdom domains "
            f"({', '.join(domain_names)}), the lotus now blooms with integrated understanding:\n\n"
            f"**Integrated Wisdom:**\n"
            f"The complete picture emerges when we weave together insights from all domains. "
            f"No single perspective holds all truth - wisdom arises from their integration.\n\n"
            f"**Balance Assessment:**\n"
            f"Domain balance: {self._check_domain_balance()}\n\n"
            f"**Holistic Recommendation:**\n"
            f"Drawing from technical feasibility, emotional awareness, ethical grounding, "
            f"practical constraints, and intuitive sensing, the wisest path forward "
            f"honors all these dimensions. True wisdom holds space for complexity and "
            f"acknowledges that different domains may suggest different directions - "
            f"the art is in the skillful integration.\n\n"
            f"**Key Insights:**\n"
            f"- Technical analysis ensures sound reasoning and implementation\n"
            f"- Emotional intelligence honors human dimensions and impacts\n"
            f"- Ethical framework maintains moral integrity and values\n"
            f"- Practical wisdom grounds ideas in reality and resources\n"
            f"- Intuitive understanding accesses patterns beyond logic\n\n"
            f"Like a lotus flower rising from muddy water to bloom in clarity, "
            f"this multi-domain approach transforms complex problems into integrated wisdom."
        )

    def _generate_deeper_domain_content(
        self, domain_key: str, domain_info: dict, feedback: str | None
    ) -> str:
        """Generate content for deeper domain analysis."""
        feedback_text = f"\n\nFocusing on: {feedback}" if feedback else ""

        return (
            f"Step {self._step_counter}: Deepening {domain_info['name'].upper()} Analysis\n\n"
            f"Exploring the {domain_info['name']} domain with greater depth and nuance:{feedback_text}\n\n"
            f"This deeper examination reveals additional layers within the {domain_info['name'].lower()} "
            f"perspective:\n\n"
            f"- Subtle aspects not visible in initial analysis\n"
            f"- Edge cases and boundary conditions\n"
            f"- Interconnections with other domains\n"
            f"- Greater specificity and detail\n\n"
            f"By diving deeper into this domain, we strengthen our overall understanding "
            f"and enhance the quality of our integrated wisdom."
        )

    def _generate_refined_synthesis_content(
        self, new_insight: ThoughtNode, feedback: str | None
    ) -> str:
        """Generate content for refined synthesis."""
        domain_name = self.DOMAINS[
            new_insight.metadata.get("domain", "TECHNICAL")
        ]["name"]

        return (
            f"Step {self._step_counter}: REFINED SYNTHESIS\n\n"
            f"Incorporating new insights from deeper {domain_name} analysis, "
            f"the synthesis evolves to a more complete understanding:\n\n"
            f"The refined wisdom now includes:\n"
            f"- Enhanced understanding from {domain_name} domain\n"
            f"- Stronger integration across all perspectives\n"
            f"- More nuanced appreciation of complexity\n"
            f"- Deeper coherence and clarity\n\n"
            f"As each petal of the lotus strengthens, the flower itself becomes more "
            f"beautiful and complete. This refined synthesis represents a higher-order "
            f"integration of multi-domain wisdom."
        )

    def _generate_evaluation_content(
        self, graph: ThoughtGraph, feedback: str | None
    ) -> str:
        """Generate content for evaluation thought."""
        balance = self._check_domain_balance()
        balanced = all(count > 0 for count in balance.values())

        return (
            f"Step {self._step_counter}: EVALUATION AND INSIGHTS\n\n"
            f"Reflecting on this lotus wisdom analysis:\n\n"
            f"**Domain Coverage:**\n"
            f"- All five domains explored: {'Yes' if balanced else 'Incomplete'}\n"
            f"- Analysis depth by domain: {balance}\n\n"
            f"**Quality of Integration:**\n"
            f"The synthesis successfully weaves together multiple perspectives, "
            f"creating a holistic understanding that honors complexity while "
            f"seeking clarity.\n\n"
            f"**Key Insights:**\n"
            f"- Multi-domain analysis reveals aspects invisible to single perspectives\n"
            f"- Balance across domains strengthens overall wisdom\n"
            f"- Integration is more valuable than simple aggregation\n"
            f"- Wisdom emerges from holding multiple truths simultaneously\n\n"
            f"The lotus wisdom approach demonstrates that comprehensive understanding "
            f"requires technical rigor, emotional intelligence, ethical grounding, "
            f"practical wisdom, and intuitive sensing working in harmony."
        )

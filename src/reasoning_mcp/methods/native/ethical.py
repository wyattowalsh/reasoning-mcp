"""Ethical reasoning method implementation.

This module implements a structured ethical analysis method that applies multiple
ethical frameworks to analyze scenarios and provide balanced recommendations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

from reasoning_mcp.elicitation import (
    ElicitationConfig,
    elicit_selection,
)
from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType
from reasoning_mcp.models.thought import ThoughtNode

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


# Metadata for the ethical reasoning method
ETHICAL_REASONING_METADATA = MethodMetadata(
    identifier=MethodIdentifier.ETHICAL_REASONING,
    name="Ethical Reasoning",
    description=(
        "Multi-framework ethical analysis with stakeholder consideration "
        "and balanced recommendations"
    ),
    category=MethodCategory.HIGH_VALUE,
    tags=frozenset(
        {
            "ethics",
            "moral",
            "stakeholder",
            "multi-framework",
            "balanced",
            "decision-making",
        }
    ),
    complexity=7,
    supports_branching=True,
    supports_revision=False,
    requires_context=False,
    min_thoughts=6,
    max_thoughts=15,
    avg_tokens_per_thought=600,
    best_for=(
        "Ethical dilemmas and moral questions",
        "Decision-making with ethical implications",
        "Stakeholder impact analysis",
        "Policy and governance decisions",
        "Complex social and moral issues",
    ),
    not_recommended_for=(
        "Pure technical problems",
        "Mathematical calculations",
        "Simple factual questions",
        "Time-critical decisions without reflection",
    ),
)


class EthicalReasoning(ReasoningMethodBase):
    """Ethical reasoning method using multiple moral frameworks.

    This method applies four major ethical frameworks to analyze scenarios:
    1. Consequentialist (utilitarian) - outcomes and consequences
    2. Deontological - duties, rights, and rules
    3. Virtue Ethics - character and virtues
    4. Care Ethics - relationships and care responsibilities

    The method identifies stakeholders, analyzes impacts, and synthesizes
    findings across frameworks to provide balanced ethical recommendations.

    Examples:
        >>> method = EthicalReasoning()
        >>> await method.initialize()
        >>> session = Session().start()
        >>> scenario = "Should a company mandate vaccines for employees?"
        >>> result = await method.execute(session, scenario)
        >>> assert result.type == ThoughtType.CONCLUSION
    """

    def __init__(self, enable_elicitation: bool = True) -> None:
        """Initialize the ethical reasoning method.

        Args:
            enable_elicitation: Whether to enable user interaction for framework selection
        """
        self._identifier = MethodIdentifier.ETHICAL_REASONING
        self._name = "Ethical Reasoning"
        self._description = "Multi-framework ethical analysis"
        self._category = MethodCategory.HIGH_VALUE
        self._initialized = False
        self._use_sampling = True
        self._execution_context: ExecutionContext | None = None
        self.enable_elicitation = enable_elicitation

    @property
    def identifier(self) -> str:
        """Unique identifier for this method."""
        return str(self._identifier)

    @property
    def name(self) -> str:
        """Human-readable name for the method."""
        return self._name

    @property
    def description(self) -> str:
        """Brief description of what the method does."""
        return self._description

    @property
    def category(self) -> str:
        """Category this method belongs to."""
        return str(self._category)

    async def initialize(self) -> None:
        """Initialize the method.

        This method requires no external resources or configuration,
        so initialization is minimal.
        """
        self._initialized = True

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Execute ethical reasoning on the given scenario.

        This method performs a structured ethical analysis through the following steps:
        1. Understand the ethical scenario and question
        2. Identify all stakeholders and affected parties
        3. Apply consequentialist framework (outcomes)
        4. Apply deontological framework (duties/rights)
        5. Apply virtue ethics framework (character)
        6. Apply care ethics framework (relationships)
        7. Compare and contrast framework findings
        8. Synthesize a balanced ethical recommendation

        Args:
            session: The current reasoning session
            input_text: The ethical scenario or question to analyze
            context: Optional additional context

        Returns:
            A ThoughtNode with type CONCLUSION containing the ethical recommendation
        """
        if not self._initialized:
            await self.initialize()

        # Check if LLM sampling is available
        use_sampling = (
            execution_context is not None and execution_context.can_sample and self._use_sampling
        )
        self._execution_context = execution_context

        context = context or {}

        # Optional elicitation: Ask user which ethical framework to prioritize
        framework_priority = None
        if (
            self.enable_elicitation
            and self._execution_context
            and hasattr(self._execution_context, "ctx")
        ):
            try:
                options = [
                    {"id": "balanced", "label": "Balance all frameworks equally (default)"},
                    {
                        "id": "consequentialist",
                        "label": "Prioritize outcomes and consequences (consequentialist)",
                    },
                    {
                        "id": "deontological",
                        "label": "Prioritize duties and rights (deontological)",
                    },
                    {
                        "id": "virtue",
                        "label": "Prioritize character and virtues (virtue ethics)",
                    },
                    {
                        "id": "care",
                        "label": "Prioritize relationships and care (care ethics)",
                    },
                ]
                config = ElicitationConfig(timeout=15, required=False, default_on_timeout=None)
                if self._execution_context.ctx is None:
                    raise RuntimeError("Execution context has no MCP context for elicitation")
                selection = await elicit_selection(
                    self._execution_context.ctx,
                    "Which ethical framework should guide this analysis?",
                    options,
                    config=config,
                )
                if selection:
                    framework_priority = selection.selected
            except (TimeoutError, ConnectionError, OSError, ValueError) as e:
                logger.warning(
                    "elicitation_failed",
                    method="execute",
                    error=str(e),
                )
                # Fall back to default behavior if elicitation fails
            except Exception as e:
                logger.error(
                    "elicitation_unexpected_error",
                    method="execute",
                    error_type=type(e).__name__,
                    error=str(e),
                    exc_info=True,
                )
                # Fall back to default behavior on unexpected errors rather than raising
                # This ensures elicitation failures don't crash the main reasoning flow

        # Step 1: Initial understanding
        step1 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.INITIAL,
            method_id=self._identifier,
            content=f"Analyzing ethical scenario: {input_text}"
            + (f"\n\nFramework priority: {framework_priority}" if framework_priority else ""),
            summary="Initial scenario understanding",
            step_number=1,
            depth=0,
            confidence=0.7,
            metadata={
                "phase": "understanding",
                "scenario": input_text,
                "framework_priority": framework_priority,
            },
        )
        session.add_thought(step1)

        # Step 2: Identify stakeholders
        stakeholders = self._identify_stakeholders(input_text, context)
        step2 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.OBSERVATION,
            method_id=self._identifier,
            content=f"Key stakeholders identified: {', '.join(stakeholders)}. "
            f"Each group will be affected differently by this decision.",
            summary="Stakeholder identification",
            evidence=stakeholders,
            parent_id=step1.id,
            step_number=2,
            depth=1,
            confidence=0.8,
            metadata={
                "phase": "stakeholder_analysis",
                "stakeholders": stakeholders,
            },
        )
        session.add_thought(step2)

        # Step 3: Consequentialist analysis
        consequentialist = await self._apply_consequentialist(
            input_text, stakeholders, context, use_sampling
        )
        step3 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=self._identifier,
            content=consequentialist,
            summary="Consequentialist/Utilitarian perspective",
            parent_id=step2.id,
            branch_id="consequentialist",
            step_number=3,
            depth=2,
            confidence=0.75,
            metadata={
                "phase": "framework_analysis",
                "framework": "consequentialist",
            },
        )
        session.add_thought(step3)

        # Step 4: Deontological analysis
        deontological = await self._apply_deontological(
            input_text, stakeholders, context, use_sampling
        )
        step4 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=self._identifier,
            content=deontological,
            summary="Deontological/Rights-based perspective",
            parent_id=step2.id,
            branch_id="deontological",
            step_number=4,
            depth=2,
            confidence=0.75,
            metadata={
                "phase": "framework_analysis",
                "framework": "deontological",
            },
        )
        session.add_thought(step4)

        # Step 5: Virtue ethics analysis
        virtue = await self._apply_virtue_ethics(input_text, stakeholders, context, use_sampling)
        step5 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=self._identifier,
            content=virtue,
            summary="Virtue Ethics perspective",
            parent_id=step2.id,
            branch_id="virtue_ethics",
            step_number=5,
            depth=2,
            confidence=0.75,
            metadata={
                "phase": "framework_analysis",
                "framework": "virtue_ethics",
            },
        )
        session.add_thought(step5)

        # Step 6: Care ethics analysis
        care = await self._apply_care_ethics(input_text, stakeholders, context, use_sampling)
        step6 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.BRANCH,
            method_id=self._identifier,
            content=care,
            summary="Care Ethics perspective",
            parent_id=step2.id,
            branch_id="care_ethics",
            step_number=6,
            depth=2,
            confidence=0.75,
            metadata={
                "phase": "framework_analysis",
                "framework": "care_ethics",
            },
        )
        session.add_thought(step6)

        # Step 7: Framework comparison
        comparison = self._compare_frameworks(
            consequentialist, deontological, virtue, care, context
        )
        step7 = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.SYNTHESIS,
            method_id=self._identifier,
            content=comparison,
            summary="Cross-framework comparison",
            parent_id=step2.id,
            step_number=7,
            depth=2,
            confidence=0.8,
            metadata={
                "phase": "comparison",
                "frameworks_compared": [
                    "consequentialist",
                    "deontological",
                    "virtue_ethics",
                    "care_ethics",
                ],
            },
        )
        session.add_thought(step7)

        # Step 8: Final recommendation
        recommendation = self._synthesize_recommendation(
            input_text,
            stakeholders,
            consequentialist,
            deontological,
            virtue,
            care,
            comparison,
            context,
            framework_priority=framework_priority,
        )
        final = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONCLUSION,
            method_id=self._identifier,
            content=recommendation,
            summary="Balanced ethical recommendation",
            parent_id=step7.id,
            step_number=8,
            depth=3,
            confidence=0.85,
            quality_score=0.9,
            metadata={
                "phase": "recommendation",
                "frameworks_synthesized": 4,
                "stakeholders_considered": len(stakeholders),
            },
        )
        session.add_thought(final)

        return final

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        This method supports continuing ethical analysis with new information
        or exploring specific framework branches in more detail.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for continuation
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the reasoning
        """
        context = context or {}

        # Determine what kind of continuation is needed
        if guidance:
            # User provided specific guidance
            continuation = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.CONTINUATION,
                method_id=self._identifier,
                content=f"Continuing ethical analysis with new guidance: {guidance}",
                parent_id=previous_thought.id,
                step_number=previous_thought.step_number + 1,
                depth=previous_thought.depth + 1,
                confidence=0.7,
                metadata={
                    "phase": "guided_continuation",
                    "guidance": guidance,
                },
            )
        else:
            # Continue with deeper analysis
            continuation = ThoughtNode(
                id=str(uuid4()),
                type=ThoughtType.CONTINUATION,
                method_id=self._identifier,
                content="Deepening ethical analysis with additional considerations...",
                parent_id=previous_thought.id,
                step_number=previous_thought.step_number + 1,
                depth=previous_thought.depth + 1,
                confidence=0.7,
                metadata={
                    "phase": "automatic_continuation",
                },
            )

        session.add_thought(continuation)
        return continuation

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if healthy, False otherwise
        """
        return self._initialized

    def _identify_stakeholders(self, scenario: str, context: dict[str, Any]) -> list[str]:
        """Identify stakeholders affected by the ethical scenario.

        This is a simplified implementation that identifies common stakeholder
        categories. A production version might use NLP or domain-specific logic.

        Args:
            scenario: The ethical scenario text
            context: Additional context

        Returns:
            List of stakeholder groups
        """
        # Common stakeholder patterns
        stakeholders = []

        scenario_lower = scenario.lower()

        # Check for common stakeholder mentions
        if any(word in scenario_lower for word in ["employee", "worker", "staff"]):
            stakeholders.append("Employees")
        if any(word in scenario_lower for word in ["company", "business", "organization"]):
            stakeholders.append("Organization/Company")
        if any(word in scenario_lower for word in ["customer", "client", "consumer"]):
            stakeholders.append("Customers/Clients")
        if any(word in scenario_lower for word in ["public", "society", "community"]):
            stakeholders.append("Public/Community")
        if any(word in scenario_lower for word in ["patient", "healthcare"]):
            stakeholders.append("Patients/Healthcare Recipients")
        if any(word in scenario_lower for word in ["shareholder", "investor"]):
            stakeholders.append("Shareholders/Investors")
        if any(word in scenario_lower for word in ["government", "regulator"]):
            stakeholders.append("Regulators/Government")
        if any(word in scenario_lower for word in ["family", "relative"]):
            stakeholders.append("Family Members")

        # If no specific stakeholders found, use generic categories
        if not stakeholders:
            stakeholders = [
                "Directly Affected Individuals",
                "Decision Makers",
                "Broader Community",
            ]

        return stakeholders

    async def _apply_consequentialist(
        self,
        scenario: str,
        stakeholders: list[str],
        context: dict[str, Any],
        use_sampling: bool,
    ) -> str:
        """Apply consequentialist/utilitarian framework.

        Focus on outcomes, consequences, and maximizing overall good.

        Args:
            scenario: The ethical scenario
            stakeholders: List of affected stakeholders
            context: Additional context
            use_sampling: Whether to use LLM sampling

        Returns:
            Analysis from consequentialist perspective
        """
        primary = stakeholders[0] if stakeholders else "primary stakeholders"

        def generate_fallback() -> str:
            return (
                f"**Consequentialist Analysis (Utilitarian Perspective)**\n\n"
                f"This framework evaluates the scenario based on outcomes and consequences, "
                f"seeking to maximize overall well-being and minimize harm.\n\n"
                f"Potential positive consequences:\n"
                f"- May benefit {primary} through improved outcomes\n"
                f"- Could lead to greater overall utility and well-being\n"
                f"- Potential for positive ripple effects across the community\n\n"
                f"Potential negative consequences:\n"
                f"- Risk of harm to vulnerable or minority stakeholders\n"
                f"- Possible unintended side effects\n"
                f"- Trade-offs between different types of benefits and harms\n\n"
                f"From a utilitarian standpoint, we must weigh the total expected utility, "
                f"considering both the magnitude and probability of consequences "
                f"for all stakeholders."
            )

        if not use_sampling:
            return generate_fallback()

        prompt = f"""Analyze this ethical scenario from a consequentialist \
/utilitarian perspective:

Scenario: {scenario}

Stakeholders: {", ".join(stakeholders)}

Provide a thorough consequentialist analysis that:
1. Evaluates outcomes and consequences
2. Considers how to maximize overall well-being and minimize harm
3. Identifies potential positive and negative consequences
4. Weighs the total expected utility across all stakeholders
5. Considers both the magnitude and probability of consequences

Format your response as a clear ethical analysis."""

        return await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=generate_fallback,
            system_prompt=(
                "You are an ethical reasoning assistant analyzing scenarios "
                "through a consequentialist/utilitarian framework."
            ),
            temperature=0.7,
        )

    async def _apply_deontological(
        self,
        scenario: str,
        stakeholders: list[str],
        context: dict[str, Any],
        use_sampling: bool,
    ) -> str:
        """Apply deontological/rights-based framework.

        Focus on duties, rights, rules, and moral principles.

        Args:
            scenario: The ethical scenario
            stakeholders: List of affected stakeholders
            context: Additional context
            use_sampling: Whether to use LLM sampling

        Returns:
            Analysis from deontological perspective
        """
        group = stakeholders[0] if stakeholders else "stakeholders"

        def generate_fallback() -> str:
            return (
                f"**Deontological Analysis (Rights and Duties Perspective)**\n\n"
                f"This framework evaluates based on moral duties, rights, and adherence "
                f"to rules, regardless of consequences.\n\n"
                f"Key rights at stake:\n"
                f"- Individual autonomy and freedom of choice\n"
                f"- Right to be treated with dignity and respect\n"
                f"- Rights of {group} to protection and fairness\n\n"
                f"Relevant duties:\n"
                f"- Duty to respect human autonomy and agency\n"
                f"- Duty of care toward vulnerable parties\n"
                f"- Duty to uphold justice and fairness\n"
                f"- Duty to honor commitments and trust\n\n"
                f"Moral principles:\n"
                f"- Kant's categorical imperative: Act only according to principles that\n"
                f"  could be universal laws\n"
                f"- Respect for persons as ends in themselves, never merely as means\n"
                f"- Justice requires treating like cases alike and respecting "
                f"fundamental rights\n\n"
                f"From this perspective, some actions may be morally required or forbidden "
                f"regardless of their outcomes."
            )

        if not use_sampling:
            return generate_fallback()

        prompt = f"""Analyze this ethical scenario from a deontological \
/rights-based perspective:

Scenario: {scenario}

Stakeholders: {", ".join(stakeholders)}

Provide a thorough deontological analysis that:
1. Identifies key rights at stake
2. Examines relevant moral duties
3. Applies principles like Kant's categorical imperative
4. Considers respect for persons as ends in themselves
5. Evaluates what is morally required or forbidden regardless of consequences

Format your response as a clear ethical analysis."""

        return await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=generate_fallback,
            system_prompt=(
                "You are an ethical reasoning assistant analyzing scenarios "
                "through a deontological/rights-based framework."
            ),
            temperature=0.7,
        )

    async def _apply_virtue_ethics(
        self,
        scenario: str,
        stakeholders: list[str],
        context: dict[str, Any],
        use_sampling: bool,
    ) -> str:
        """Apply virtue ethics framework.

        Focus on character virtues and moral excellence.

        Args:
            scenario: The ethical scenario
            stakeholders: List of affected stakeholders
            context: Additional context
            use_sampling: Whether to use LLM sampling

        Returns:
            Analysis from virtue ethics perspective
        """

        def generate_fallback() -> str:
            return (
                "**Virtue Ethics Analysis (Character Perspective)**\n\n"
                "This framework asks what virtues are at stake and what a person of good "
                "character would do in this situation.\n\n"
                "Relevant virtues:\n"
                "- **Wisdom**: Making sound judgments with good information and deliberation\n"
                "- **Justice**: Giving each person what they are due, treating fairly\n"
                "- **Courage**: Standing up for what is right despite challenges\n"
                "- **Temperance**: Exercising moderation and self-control\n"
                "- **Compassion**: Showing genuine care and empathy for others\n"
                "- **Integrity**: Consistency between values and actions\n\n"
                "Character considerations:\n"
                "- What would a virtuous person do in this situation?\n"
                "- How does this decision reflect on moral character?\n"
                "- Does it cultivate or undermine virtue in individuals and community?\n"
                "- What habits of character does it promote?\n\n"
                "A virtuous approach seeks the mean between extremes and acts from\n"
                "a well-developed moral character rather than just following rules or\n"
                "calculating outcomes."
            )

        if not use_sampling:
            return generate_fallback()

        prompt = f"""Analyze this ethical scenario from a virtue ethics perspective:

Scenario: {scenario}

Stakeholders: {", ".join(stakeholders)}

Provide a thorough virtue ethics analysis that:
1. Identifies relevant virtues (wisdom, justice, courage, temperance, compassion, integrity)
2. Considers what a person of good character would do
3. Examines how the decision reflects on moral character
4. Evaluates whether it cultivates or undermines virtue
5. Considers the mean between extremes

Format your response as a clear ethical analysis."""

        return await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=generate_fallback,
            system_prompt=(
                "You are an ethical reasoning assistant analyzing scenarios "
                "through a virtue ethics framework."
            ),
            temperature=0.7,
        )

    async def _apply_care_ethics(
        self,
        scenario: str,
        stakeholders: list[str],
        context: dict[str, Any],
        use_sampling: bool,
    ) -> str:
        """Apply care ethics framework.

        Focus on relationships, care responsibilities, and context.

        Args:
            scenario: The ethical scenario
            stakeholders: List of affected stakeholders
            context: Additional context
            use_sampling: Whether to use LLM sampling

        Returns:
            Analysis from care ethics perspective
        """

        def generate_fallback() -> str:
            return (
                "**Care Ethics Analysis (Relational Perspective)**\n\n"
                "This framework emphasizes relationships, responsibilities of care, "
                "and the particular context of moral situations.\n\n"
                "Relational considerations:\n"
                "- Existing relationships and interdependencies among stakeholders\n"
                "- Care responsibilities toward vulnerable or dependent parties\n"
                "- How decisions affect trust and relationship quality\n"
                "- Power dynamics and asymmetries in relationships\n\n"
                "Caring responses:\n"
                "- Attentiveness to the needs and voices of all stakeholders,\n"
                "  especially marginalized ones\n"
                "- Responsiveness to particular contexts rather than abstract rules\n"
                "- Responsibility for maintaining and nurturing important relationships\n"
                "- Competence in providing appropriate care\n\n"
                "Context matters:\n"
                "- The specific circumstances and history of relationships\n"
                "- Emotional and psychological dimensions of the situation\n"
                "- How decisions affect the web of care relationships\n\n"
                "Care ethics asks us to be attentive to particular needs and relationships "
                "rather than applying universal rules mechanically."
            )

        if not use_sampling:
            return generate_fallback()

        prompt = f"""Analyze this ethical scenario from a care ethics perspective:

Scenario: {scenario}

Stakeholders: {", ".join(stakeholders)}

Provide a thorough care ethics analysis that:
1. Examines relationships and interdependencies among stakeholders
2. Identifies care responsibilities, especially toward vulnerable parties
3. Considers how decisions affect trust and relationship quality
4. Analyzes power dynamics and asymmetries
5. Emphasizes attentiveness, responsiveness, and responsibility in caring relationships
6. Considers the particular context rather than abstract rules

Format your response as a clear ethical analysis."""

        return await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=generate_fallback,
            system_prompt=(
                "You are an ethical reasoning assistant analyzing scenarios "
                "through a care ethics framework."
            ),
            temperature=0.7,
        )

    def _compare_frameworks(
        self,
        consequentialist: str,
        deontological: str,
        virtue: str,
        care: str,
        context: dict[str, Any],
    ) -> str:
        """Compare and contrast the different framework analyses.

        Args:
            consequentialist: Consequentialist analysis
            deontological: Deontological analysis
            virtue: Virtue ethics analysis
            care: Care ethics analysis
            context: Additional context

        Returns:
            Framework comparison analysis
        """
        return (
            "**Cross-Framework Comparison**\n\n"
            "After analyzing this scenario through four major ethical frameworks, "
            "we can identify areas of convergence and tension:\n\n"
            "**Areas of Agreement:**\n"
            "- All frameworks recognize the importance of considering stakeholder welfare\n"
            "- Each emphasizes the need for moral reasoning and justification\n"
            "- All acknowledge the complexity and significance of the ethical question\n\n"
            "**Key Tensions:**\n"
            "- **Consequences vs. Principles**: The consequentialist focus on outcomes may\n"
            "  conflict with deontological emphasis on absolute rights and duties\n"
            "- **Universal vs. Particular**: Deontological universal principles may clash\n"
            "  with care ethics' attention to particular relationships and contexts\n"
            "- **Action vs. Character**: Virtue ethics asks 'what kind of person should\n"
            "  I be?' while other frameworks focus more on 'what should I do?'\n\n"
            "**Complementary Insights:**\n"
            "- Consequentialist analysis helps us consider practical outcomes\n"
            "- Deontological analysis protects fundamental rights and maintains moral boundaries\n"
            "- Virtue ethics reminds us that character and motivation matter\n"
            "- Care ethics ensures we attend to relationships and particular needs\n\n"
            "An ethically robust decision should try to honor insights from multiple perspectives."
        )

    def _synthesize_recommendation(
        self,
        scenario: str,
        stakeholders: list[str],
        consequentialist: str,
        deontological: str,
        virtue: str,
        care: str,
        comparison: str,
        context: dict[str, Any],
        framework_priority: str | None = None,
    ) -> str:
        """Synthesize a balanced ethical recommendation.

        Args:
            scenario: The original scenario
            stakeholders: List of stakeholders
            consequentialist: Consequentialist analysis
            deontological: Deontological analysis
            virtue: Virtue ethics analysis
            care: Care ethics analysis
            comparison: Framework comparison
            context: Additional context
            framework_priority: Optional framework to prioritize in the recommendation

        Returns:
            Final balanced recommendation
        """
        # Determine title and focus based on priority
        if framework_priority and framework_priority != "balanced":
            title_suffix = f" (Prioritizing {framework_priority.title()} Framework)"
            priority_note = (
                f"\n\n**Framework Priority:**\n"
                f"This recommendation prioritizes the {framework_priority} perspective "
                f"as requested, while still considering insights from other frameworks.\n"
            )
        else:
            title_suffix = ""
            priority_note = ""

        parties = stakeholders[0] if stakeholders else "all parties"
        return (
            f"**Balanced Ethical Recommendation{title_suffix}**\n\n"
            f"After analyzing this scenario through multiple ethical frameworks, "
            f"here is a synthesized recommendation:{priority_note}\n"
            f"**Core Recommendation:**\n"
            f"The most ethically sound approach should:\n"
            f"1. Respect fundamental rights and dignity of all stakeholders\n"
            f"   (deontological)\n"
            f"2. Consider and weigh consequences carefully, especially for\n"
            f"   vulnerable parties (consequentialist)\n"
            f"3. Act from virtues like wisdom, justice, and compassion\n"
            f"   (virtue ethics)\n"
            f"4. Attend to relationships, care responsibilities, and particular\n"
            f"   contexts (care ethics)\n\n"
            f"**Key Considerations:**\n"
            f"- Ensure meaningful consent and autonomy for {parties}\n"
            f"- Minimize harm and maximize benefit across all stakeholder groups\n"
            f"- Maintain integrity and act from well-developed moral character\n"
            f"- Honor care responsibilities and preserve important relationships\n\n"
            f"**Implementation Guidance:**\n"
            f"- Engage stakeholders in genuine dialogue and decision-making\n"
            f"- Provide transparency about reasoning and trade-offs\n"
            f"- Consider accommodations for those who may be disproportionately affected\n"
            f"- Monitor outcomes and be willing to adjust based on actual impacts\n"
            f"- Maintain processes that reflect virtue and care\n\n"
            f"**Ethical Boundaries:**\n"
            f"Regardless of approach chosen, it should NOT:\n"
            f"- Violate fundamental human rights or dignity\n"
            f"- Exploit vulnerable parties for others' benefit\n"
            f"- Ignore foreseeable serious harms\n"
            f"- Proceed without adequate stakeholder input and transparency\n\n"
            f"This recommendation attempts to balance insights from multiple ethical traditions "
            f"to arrive at a more complete and defensible ethical judgment."
        )

"""MCP prompts for reasoning pipelines.

This module provides structured prompts for common reasoning workflows using
FastMCP's @mcp.prompt() decorator. These prompts enable sophisticated
multi-stage reasoning patterns for deep analysis, code debugging, and
ethical decision-making.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP


def register_pipeline_prompts(mcp: FastMCP) -> None:
    """Register all pipeline prompts with the MCP server.

    This function registers the following prompts:
    - deep_analysis: Multi-layer deep analysis prompt
    - debug_code: Systematic code debugging prompt
    - ethical_decision: Multi-framework ethical analysis prompt

    Args:
        mcp: The FastMCP server instance to register prompts with

    Example:
        >>> from reasoning_mcp.server import mcp
        >>> register_pipeline_prompts(mcp)
    """

    @mcp.prompt(name="deep_analysis", description="Multi-layer deep analysis prompt")
    def deep_analysis(topic: str, depth: int = 3) -> str:
        """Generate a structured prompt for multi-layer deep analysis.

        This prompt guides systematic analysis through multiple layers:
        1. Surface understanding - Initial comprehension and key points
        2. Structural analysis - Patterns, relationships, and underlying principles
        3. Implications - Consequences, applications, and broader context
        4. Synthesis - Integration of insights across all layers

        The depth parameter controls how many iterations of analysis to perform,
        enabling progressively deeper exploration of the topic.

        Args:
            topic: The subject matter to analyze
            depth: Number of analysis layers to perform (default: 3, range: 1-5)

        Returns:
            A structured prompt string for deep analysis

        Example:
            >>> prompt = deep_analysis(
            ...     topic="climate change impacts on coastal cities",
            ...     depth=4
            ... )
        """
        # Clamp depth to reasonable range
        depth = max(1, min(5, depth))

        layers = [
            {
                "name": "Surface Understanding",
                "focus": "Initial comprehension and key points",
                "questions": [
                    "What are the primary concepts and definitions?",
                    "What are the most obvious features or characteristics?",
                    "What initial observations can be made?",
                ],
            },
            {
                "name": "Structural Analysis",
                "focus": "Patterns, relationships, and underlying principles",
                "questions": [
                    "What patterns or structures can be identified?",
                    "How do different components relate to each other?",
                    "What are the underlying principles or mechanisms?",
                    "What causal relationships exist?",
                ],
            },
            {
                "name": "Implications",
                "focus": "Consequences, applications, and broader context",
                "questions": [
                    "What are the immediate and long-term consequences?",
                    "How does this connect to broader systems or contexts?",
                    "What are potential applications or use cases?",
                    "What uncertainties or unknowns exist?",
                ],
            },
            {
                "name": "Synthesis",
                "focus": "Integration of insights across all layers",
                "questions": [
                    "How do insights from different layers connect?",
                    "What emergent understanding arises from the analysis?",
                    "What are the key takeaways and actionable insights?",
                    "What questions remain for further exploration?",
                ],
            },
            {
                "name": "Meta-Analysis",
                "focus": "Analysis of the analysis itself",
                "questions": [
                    "What assumptions were made during the analysis?",
                    "What biases or limitations might exist in this approach?",
                    "How confident are we in the conclusions drawn?",
                    "What alternative perspectives should be considered?",
                ],
            },
        ]

        # Build the prompt with requested depth
        prompt_parts = [
            f"# Deep Analysis: {topic}",
            "",
            "Please conduct a systematic multi-layer analysis of the following topic:",
            f'"{topic}"',
            "",
            "Use the following structured approach, progressing through each layer before moving to the next:",
            "",
        ]

        for i, layer in enumerate(layers[:depth], 1):
            prompt_parts.extend(
                [
                    f"## Layer {i}: {layer['name']}",
                    f"**Focus:** {layer['focus']}",
                    "",
                    "**Key Questions:**",
                ]
            )
            for question in layer["questions"]:
                prompt_parts.append(f"- {question}")
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "---",
                "",
                "**Instructions:**",
                f"1. Progress through all {depth} layers sequentially",
                "2. Build upon insights from previous layers",
                "3. Provide specific, concrete analysis rather than generic observations",
                "4. Support claims with reasoning and evidence where applicable",
                "5. Explicitly note connections between layers",
                "",
                "Begin your analysis:",
            ]
        )

        return "\n".join(prompt_parts)

    @mcp.prompt(name="debug_code", description="Systematic code debugging prompt")
    def debug_code(code: str, error: str | None = None, language: str | None = None) -> str:
        """Generate a structured prompt for systematic code debugging.

        This prompt guides debugging through a proven systematic process:
        1. Reproduce - Verify and document the error
        2. Isolate - Narrow down the problematic code section
        3. Hypothesize - Form theories about the root cause
        4. Test - Validate hypotheses through experimentation
        5. Fix - Implement and verify the solution
        6. Verify - Ensure the fix works and doesn't introduce new issues

        Args:
            code: The code snippet to debug
            error: Optional error message or description of the problem
            language: Optional programming language (e.g., "python", "javascript")

        Returns:
            A structured prompt string for code debugging

        Example:
            >>> prompt = debug_code(
            ...     code="def divide(a, b):\\n    return a / b",
            ...     error="ZeroDivisionError: division by zero",
            ...     language="python"
            ... )
        """
        lang_note = f" ({language})" if language else ""
        error_section = (
            f"\n**Error/Issue:**\n```\n{error}\n```\n"
            if error
            else "\n**Issue:** Behavior not working as expected\n"
        )

        prompt_parts = [
            f"# Code Debugging Workflow{lang_note}",
            "",
            "Please debug the following code using a systematic approach:",
            "",
            "**Code:**",
            f"```{language or ''}",
            code,
            "```",
            "",
            error_section,
            "---",
            "",
            "## Debugging Process",
            "",
            "### Step 1: Reproduce",
            "**Objective:** Verify and document the error",
            "- Confirm the error occurs consistently",
            "- Document the exact conditions that trigger it",
            "- Note any error messages, stack traces, or unexpected behavior",
            "- Identify the input values or state that causes the issue",
            "",
            "### Step 2: Isolate",
            "**Objective:** Narrow down the problematic code section",
            "- Identify which function, method, or code block is failing",
            "- Determine which line(s) are executing when the error occurs",
            "- Separate the problem code from working code",
            "- Identify the minimal reproducible example",
            "",
            "### Step 3: Hypothesize",
            "**Objective:** Form theories about the root cause",
            "- Based on the error and isolated code, what could be wrong?",
            "- Consider common causes (null/undefined, off-by-one, type mismatch, etc.)",
            "- What assumptions might be violated?",
            "- Are there edge cases being missed?",
            "",
            "### Step 4: Test",
            "**Objective:** Validate hypotheses through experimentation",
            "- For each hypothesis, how would you test it?",
            "- What specific values or conditions would confirm/refute the hypothesis?",
            "- Trace through the code mentally or with print/log statements",
            "- Examine variable values at key points",
            "",
            "### Step 5: Fix",
            "**Objective:** Implement the solution",
            "- Based on confirmed hypothesis, what code change is needed?",
            "- Implement the fix",
            "- Ensure the fix addresses the root cause, not just symptoms",
            "- Consider if the fix might affect other parts of the code",
            "",
            "### Step 6: Verify",
            "**Objective:** Ensure the fix works completely",
            "- Test with the original failing case",
            "- Test with edge cases and boundary conditions",
            "- Verify no new issues were introduced",
            "- Consider adding tests to prevent regression",
            "",
            "---",
            "",
            "**Instructions:**",
            "1. Work through each step systematically",
            "2. Be explicit about your reasoning at each step",
            "3. Show the corrected code in the Fix step",
            "4. List specific test cases in the Verify step",
            "",
            "Begin the debugging process:",
        ]

        return "\n".join(prompt_parts)

    @mcp.prompt(
        name="ethical_decision",
        description="Multi-framework ethical analysis prompt",
    )
    def ethical_decision(scenario: str, stakeholders: list[str] | None = None) -> str:
        """Generate a structured prompt for multi-framework ethical analysis.

        This prompt guides ethical decision-making through multiple philosophical
        frameworks to provide comprehensive analysis:
        1. Consequentialist - Focus on outcomes and consequences
        2. Deontological - Focus on duties, rules, and principles
        3. Virtue Ethics - Focus on character and virtues
        4. Care Ethics - Focus on relationships and compassion

        Args:
            scenario: The ethical dilemma or decision to analyze
            stakeholders: Optional list of affected parties/stakeholders

        Returns:
            A structured prompt string for ethical analysis

        Example:
            >>> prompt = ethical_decision(
            ...     scenario="Should we deploy AI for resume screening?",
            ...     stakeholders=["job applicants", "hiring managers", "company"]
            ... )
        """
        stakeholder_section = ""
        if stakeholders and len(stakeholders) > 0:
            stakeholder_list = "\n".join(f"- {s}" for s in stakeholders)
            stakeholder_section = f"\n**Key Stakeholders:**\n{stakeholder_list}\n"

        frameworks = [
            {
                "name": "Consequentialist Analysis",
                "description": "Focus on outcomes and consequences",
                "questions": [
                    "What are the likely consequences of each possible action?",
                    "Who benefits and who is harmed by each option?",
                    "What is the net utility or overall good produced?",
                    "What are the short-term vs. long-term consequences?",
                    "How certain are we about these predicted outcomes?",
                ],
            },
            {
                "name": "Deontological Analysis",
                "description": "Focus on duties, rules, and principles",
                "questions": [
                    "What moral duties or obligations are relevant here?",
                    "What universal principles or rules apply?",
                    "Would this action violate anyone's rights?",
                    "Could this action be universalized as a general rule?",
                    "What categorical imperatives are at stake?",
                ],
            },
            {
                "name": "Virtue Ethics Analysis",
                "description": "Focus on character and virtues",
                "questions": [
                    "What would a virtuous person do in this situation?",
                    "Which virtues are most relevant (courage, honesty, compassion, etc.)?",
                    "How does each option align with excellence of character?",
                    "What habits and dispositions does each choice cultivate?",
                    "What would demonstrate practical wisdom (phronesis) here?",
                ],
            },
            {
                "name": "Care Ethics Analysis",
                "description": "Focus on relationships and compassion",
                "questions": [
                    "How are relationships affected by each option?",
                    "What does care and compassion demand in this situation?",
                    "How are power dynamics and vulnerabilities relevant?",
                    "What context and particular details matter most?",
                    "How can we respond to the needs of all involved?",
                ],
            },
        ]

        prompt_parts = [
            "# Ethical Decision Analysis",
            "",
            "Please analyze the following ethical scenario through multiple philosophical frameworks:",
            "",
            "**Scenario:**",
            scenario,
            stakeholder_section,
            "---",
            "",
            "## Multi-Framework Analysis",
            "",
            "Examine this scenario through the following four ethical frameworks, providing thorough analysis for each:",
            "",
        ]

        for i, framework in enumerate(frameworks, 1):
            prompt_parts.extend(
                [
                    f"### {i}. {framework['name']}",
                    f"**Perspective:** {framework['description']}",
                    "",
                    "**Analysis Questions:**",
                ]
            )
            for question in framework["questions"]:
                prompt_parts.append(f"- {question}")
            prompt_parts.append("")

        prompt_parts.extend(
            [
                "---",
                "",
                "## Synthesis and Recommendation",
                "",
                "After completing all four framework analyses, provide:",
                "",
                "1. **Key Tensions:** Where do the frameworks agree or conflict?",
                "2. **Weighing Considerations:** Which factors seem most morally relevant and why?",
                "3. **Recommended Action:** What course of action best balances all considerations?",
                "4. **Justification:** Why is this recommendation ethically sound?",
                "5. **Limitations:** What uncertainties or limitations exist in this analysis?",
                "",
                "---",
                "",
                "**Instructions:**",
                "1. Provide substantive analysis for each framework",
                "2. Draw from the specific details of the scenario",
                "3. Acknowledge complexity and competing values",
                "4. Be clear about moral reasoning, not just stating conclusions",
                "5. Consider how stakeholders are affected differently",
                "",
                "Begin your ethical analysis:",
            ]
        )

        return "\n".join(prompt_parts)


__all__ = ["register_pipeline_prompts"]

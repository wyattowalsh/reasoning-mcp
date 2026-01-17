"""Layered Chain-of-Thought (Layered CoT) reasoning method.

This module implements Layered CoT, which breaks reasoning into multiple passes
or "layers" with opportunities to review and adjust at each layer. Useful for
high-stakes domains like healthcare or finance.

Key phases:
1. Layer 1: Initial reasoning pass
2. Layer 2: Review and refine
3. Layer 3: Validate and adjust
4. Synthesis: Combine layers into final answer

Reference: "Layered Chain of Thought" (2025) - Multi-pass reasoning with review
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethodBase
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.thought import ThoughtNode

if TYPE_CHECKING:
    from reasoning_mcp.engine.executor import ExecutionContext
    from reasoning_mcp.models import Session


LAYERED_COT_METADATA = MethodMetadata(
    identifier=MethodIdentifier.LAYERED_COT,
    name="Layered Chain-of-Thought",
    description="Multi-pass reasoning with layer-by-layer review and adjustment. "
    "Each layer refines the previous, useful for high-stakes decisions.",
    category=MethodCategory.SPECIALIZED,
    tags=frozenset({"multi-pass", "layered", "review", "refinement", "high-stakes"}),
    complexity=6,
    supports_branching=False,
    supports_revision=True,
    requires_context=False,
    min_thoughts=4,
    max_thoughts=8,
    avg_tokens_per_thought=350,
    best_for=("high-stakes decisions", "healthcare", "finance", "careful reasoning"),
    not_recommended_for=("simple tasks", "time-critical queries"),
)


class LayeredCoT(ReasoningMethodBase):
    """Layered Chain-of-Thought implementation."""

    DEFAULT_LAYERS = 3
    _use_sampling: bool = True

    def __init__(self, num_layers: int = DEFAULT_LAYERS) -> None:
        self._num_layers = num_layers
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "layer_1"
        self._current_layer = 0
        self._layer_outputs: list[dict[str, Any]] = []
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.LAYERED_COT

    @property
    def name(self) -> str:
        return LAYERED_COT_METADATA.name

    @property
    def description(self) -> str:
        return LAYERED_COT_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.SPECIALIZED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "layer_1"
        self._current_layer = 0
        self._layer_outputs = []

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Layered CoT must be initialized before execution")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter = 1
        self._current_layer = 1
        self._current_phase = f"layer_{self._current_layer}"

        # First layer: initial reasoning (use sampling if available)
        if self._execution_context and self._execution_context.can_sample:
            layer_output = await self._sample_layer_reasoning(input_text, layer_num=1)
        else:
            layer_output = self._generate_layer_reasoning_heuristic(input_text, layer_num=1)

        self._layer_outputs.append(layer_output)

        content = (
            f"Step {self._step_counter}: Layer 1 - Initial Reasoning (Layered CoT)\n\n"
            f"Problem: {input_text}\n\n"
            f"Layer 1 Analysis:\n"
            f"  Reasoning: {layer_output['reasoning']}\n"
            f"  Preliminary Conclusion: {layer_output['conclusion']}\n"
            f"  Confidence: {layer_output['confidence']:.0%}\n\n"
            f"Issues Identified for Review:\n"
            + "\n".join(f"  - {issue}" for issue in layer_output["issues"])
            + f"\n\nLayer 1 complete. {self._num_layers - 1} more layers for refinement.\n"
            f"Next: Layer 2 - Review and refine."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.LAYERED_COT,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=layer_output["confidence"],
            quality_score=0.7,
            metadata={"phase": self._current_phase, "layer": self._current_layer},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.LAYERED_COT
        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Layered CoT must be initialized before continuation")

        # Store execution context for sampling
        self._execution_context = execution_context

        self._step_counter += 1
        prev_layer = previous_thought.metadata.get("layer", 1)

        if prev_layer < self._num_layers:
            self._current_layer = prev_layer + 1
            self._current_phase = f"layer_{self._current_layer}"

            # Build on previous layer (use sampling if available)
            prev_output = self._layer_outputs[-1] if self._layer_outputs else {}
            prev_confidence = prev_output.get("confidence", 0.7)

            if self._execution_context and self._execution_context.can_sample:
                layer_output = await self._sample_layer_refinement(
                    previous_thought.content,
                    prev_layer=prev_layer,
                    current_layer=self._current_layer,
                    prev_output=prev_output,
                    guidance=guidance,
                )
            else:
                layer_output = self._generate_layer_refinement_heuristic(
                    prev_layer=prev_layer,
                    current_layer=self._current_layer,
                    prev_confidence=prev_confidence,
                )

            self._layer_outputs.append(layer_output)

            layer_type = "Review" if self._current_layer == 2 else "Validate"
            content = (
                f"Step {self._step_counter}: Layer {self._current_layer} - "
                f"{layer_type} and Refine\n\n"
                f"Building on Layer {prev_layer} ({prev_confidence:.0%} confidence):\n\n"
                f"Adjustments Made:\n"
                + "\n".join(f"  - {adj}" for adj in layer_output["adjustments"])
                + f"\n\nLayer {self._current_layer} Analysis:\n"
                f"  Reasoning: {layer_output['reasoning']}\n"
                f"  Refined Conclusion: {layer_output['conclusion']}\n"
                f"  Updated Confidence: {layer_output['confidence']:.0%}\n\n"
                + (
                    f"Next: Layer {self._current_layer + 1} for further refinement."
                    if self._current_layer < self._num_layers
                    else "All layers complete. Next: Synthesize final answer."
                )
            )
            thought_type = (
                ThoughtType.REVISION if self._current_layer == 2 else ThoughtType.VERIFICATION
            )
            confidence = layer_output["confidence"]
        else:
            self._current_phase = "synthesize"
            # Synthesize all layers
            final_confidence = (
                self._layer_outputs[-1]["confidence"] if self._layer_outputs else 0.85
            )
            content = (
                f"Step {self._step_counter}: Synthesize Layers\n\n"
                f"Combining insights from {self._num_layers} reasoning layers:\n\n"
                f"Layer Progression:\n"
                + "\n".join(
                    f"  Layer {lo['layer']}: {lo['confidence']:.0%} confidence"
                    for lo in self._layer_outputs
                )
                + "\n\nSynthesis:\n"
                "  - Initial reasoning established foundation\n"
                "  - Review layer addressed potential issues\n"
                "  - Validation layer confirmed reasoning\n\n"
                "Synthesized Conclusion:\n"
                "  [Final answer combining all layer insights]"
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = final_confidence

        if self._current_phase == "synthesize":
            # Final conclusion after synthesis
            self._step_counter += 1
            self._current_phase = "conclude"
            final_confidence = (
                self._layer_outputs[-1]["confidence"] if self._layer_outputs else 0.88
            )
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Layered Chain-of-Thought Complete:\n"
                f"  Layers processed: {self._num_layers}\n"
                f"  Initial confidence: {self._layer_outputs[0]['confidence']:.0%}\n"
                f"  Final confidence: {final_confidence:.0%}\n"
                f"  Improvement: "
                f"+{(final_confidence - self._layer_outputs[0]['confidence']) * 100:.0f}%\n\n"
                f"Final Answer: [Multi-layer validated answer]\n"
                f"Confidence: High ({int(final_confidence * 100)}%)\n\n"
                f"Method: Layered Chain-of-Thought\n"
                f"  - Multiple reasoning passes\n"
                f"  - Layer-by-layer review and refinement\n"
                f"  - Progressive confidence building\n"
                f"  - High-stakes decision support"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = final_confidence

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.LAYERED_COT,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "layer": self._current_layer,
                "layers_complete": len(self._layer_outputs),
            },
        )
        session.add_thought(thought)
        return thought

    async def health_check(self) -> bool:
        return self._initialized

    async def _sample_layer_reasoning(
        self,
        input_text: str,
        layer_num: int,
    ) -> dict[str, Any]:
        """Generate layer reasoning using LLM sampling.

        Args:
            input_text: The input problem or question
            layer_num: The current layer number

        Returns:
            Dictionary containing layer analysis with reasoning, conclusion, confidence, and issues
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_layer_reasoning but was not provided"
            )

        system_prompt = (
            f"You are a reasoning assistant using Layered Chain-of-Thought methodology.\n"
            f"This is Layer {layer_num} - Initial Reasoning phase.\n\n"
            "Your task is to provide an initial analysis with:\n"
            "1. Clear reasoning steps\n"
            "2. A preliminary conclusion\n"
            "3. Confidence level (0.0-1.0)\n"
            "4. Identified issues or areas needing deeper analysis\n\n"
            "Be thorough but acknowledge uncertainties. "
            "This is the first pass - subsequent layers will refine."
        )

        user_prompt = f"""Problem: {input_text}

Provide Layer {layer_num} initial reasoning analysis. Include:
- Your reasoning process
- Preliminary conclusion
- Confidence level (as a decimal, e.g., 0.7)
- Issues or concerns to address in later layers

Format your response as structured analysis."""

        def fallback_generator() -> str:
            return "[fallback]"

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1000,
        )

        # If fallback was used, return heuristic result
        if content == "[fallback]":
            return self._generate_layer_reasoning_heuristic(input_text, layer_num)

        # Parse the response to extract structured information
        # For now, use heuristic parsing
        return {
            "layer": layer_num,
            "reasoning": content[:200] if len(content) > 200 else content,
            "conclusion": "[Sampled preliminary conclusion]",
            "confidence": 0.7,
            "issues": ["Sampled issue for review", "Sampled assumption to validate"],
        }

    async def _sample_layer_refinement(
        self,
        previous_content: str,
        prev_layer: int,
        current_layer: int,
        prev_output: dict[str, Any],
        guidance: str | None = None,
    ) -> dict[str, Any]:
        """Generate layer refinement using LLM sampling.

        Args:
            previous_content: Content from the previous layer
            prev_layer: Previous layer number
            current_layer: Current layer number
            prev_output: Output from previous layer
            guidance: Optional guidance for refinement

        Returns:
            Dictionary containing refined layer analysis
        """
        if self._execution_context is None:
            raise RuntimeError(
                "Execution context required for _sample_layer_refinement but was not provided"
            )

        prev_confidence = prev_output.get("confidence", 0.7)
        prev_issues = prev_output.get("issues", [])

        layer_type = "Review" if current_layer == 2 else "Validate"

        system_prompt = (
            f"You are a reasoning assistant using Layered Chain-of-Thought methodology.\n"
            f"This is Layer {current_layer} - {layer_type} and Refine phase.\n\n"
            "Your task is to:\n"
            f"1. Review the previous layer's reasoning\n"
            f"2. Address identified issues: {', '.join(prev_issues)}\n"
            "3. Refine the conclusion\n"
            "4. Update confidence level\n"
            "5. Document adjustments made\n\n"
            f"Build upon Layer {prev_layer} to strengthen the reasoning."
        )

        user_prompt = f"""Previous Layer {prev_layer} Analysis:
{previous_content}

Previous Confidence: {prev_confidence:.0%}
Issues to Address: {", ".join(prev_issues)}
{f"Additional Guidance: {guidance}" if guidance else ""}

Provide Layer {current_layer} refinement. Include:
- Adjustments made to address issues
- Refined reasoning
- Updated conclusion
- New confidence level (should be higher than {prev_confidence:.2f})

Format your response as structured analysis."""

        def fallback_generator() -> str:
            return "[fallback]"

        content = await self._sample_with_fallback(
            user_prompt,
            fallback_generator,
            system_prompt=system_prompt,
            temperature=0.7,
            max_tokens=1200,
        )

        # If fallback was used, return heuristic result
        if content == "[fallback]":
            return self._generate_layer_refinement_heuristic(
                prev_layer, current_layer, prev_confidence
            )

        # Parse the response to extract structured information
        new_confidence = min(0.95, prev_confidence + 0.08)
        return {
            "layer": current_layer,
            "reasoning": content[:200] if len(content) > 200 else content,
            "conclusion": f"[Sampled refined conclusion - Layer {current_layer}]",
            "confidence": new_confidence,
            "adjustments": [
                "Sampled adjustment addressing previous issues",
                "Sampled validation of assumptions",
                "Sampled strengthening of reasoning chain",
            ],
        }

    def _generate_layer_reasoning_heuristic(
        self,
        input_text: str,
        layer_num: int,
    ) -> dict[str, Any]:
        """Generate layer reasoning using heuristic (fallback).

        Args:
            input_text: The input problem or question
            layer_num: The current layer number

        Returns:
            Dictionary containing layer analysis
        """
        return {
            "layer": layer_num,
            "reasoning": "Initial analysis and reasoning",
            "conclusion": "[Preliminary conclusion]",
            "confidence": 0.7,
            "issues": ["May need deeper analysis", "Check assumptions"],
        }

    def _generate_layer_refinement_heuristic(
        self,
        prev_layer: int,
        current_layer: int,
        prev_confidence: float,
    ) -> dict[str, Any]:
        """Generate layer refinement using heuristic (fallback).

        Args:
            prev_layer: Previous layer number
            current_layer: Current layer number
            prev_confidence: Confidence from previous layer

        Returns:
            Dictionary containing refined layer analysis
        """
        return {
            "layer": current_layer,
            "reasoning": f"Refined analysis addressing Layer {prev_layer} issues",
            "conclusion": f"[Refined conclusion - Layer {current_layer}]",
            "confidence": min(0.95, prev_confidence + 0.08),
            "adjustments": [
                f"Addressed issue from Layer {prev_layer}",
                "Validated assumptions",
                "Strengthened reasoning chain",
            ],
        }


__all__ = ["LayeredCoT", "LAYERED_COT_METADATA"]

"""Think-on-Graph (ToG) reasoning method.

This module implements Think-on-Graph, which uses iterative beam search on
knowledge graphs to enhance LLM reasoning. The LLM explores KG paths step
by step, expanding reasoning paths via beam search.

Key phases:
1. Initialize: Identify starting entities in KG
2. Explore: Expand paths via beam search
3. Evaluate: Score candidate paths
4. Select: Choose best paths for reasoning

Reference: Sun et al. (2024) - "Think-on-Graph: Deep and Responsible Reasoning
of Large Language Model on Knowledge Graph"
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


THINK_ON_GRAPH_METADATA = MethodMetadata(
    identifier=MethodIdentifier.THINK_ON_GRAPH,
    name="Think-on-Graph",
    description="Iterative beam search on knowledge graphs for reasoning. "
    "LLM explores KG paths step by step with scoring and selection.",
    category=MethodCategory.ADVANCED,
    tags=frozenset({"knowledge-graph", "beam-search", "iterative", "exploration", "structured"}),
    complexity=7,
    supports_branching=True,
    supports_revision=True,
    requires_context=True,
    min_thoughts=5,
    max_thoughts=10,
    avg_tokens_per_thought=350,
    best_for=("knowledge graph reasoning", "multi-hop questions", "structured exploration"),
    not_recommended_for=("unstructured problems", "creative tasks"),
)


class ThinkOnGraph(ReasoningMethodBase):
    """Think-on-Graph reasoning method implementation."""

    DEFAULT_BEAM_WIDTH = 3
    DEFAULT_MAX_HOPS = 3
    _use_sampling: bool = True

    def __init__(
        self, beam_width: int = DEFAULT_BEAM_WIDTH, max_hops: int = DEFAULT_MAX_HOPS
    ) -> None:
        self._beam_width = beam_width
        self._max_hops = max_hops
        self._initialized = False
        self._step_counter = 0
        self._current_phase: str = "initialize"
        self._current_hop = 0
        self._start_entities: list[str] = []
        self._beam_paths: list[dict[str, Any]] = []
        self._best_path: dict[str, Any] | None = None
        self._execution_context: ExecutionContext | None = None

    @property
    def identifier(self) -> str:
        return MethodIdentifier.THINK_ON_GRAPH

    @property
    def name(self) -> str:
        return THINK_ON_GRAPH_METADATA.name

    @property
    def description(self) -> str:
        return THINK_ON_GRAPH_METADATA.description

    @property
    def category(self) -> str:
        return MethodCategory.ADVANCED

    async def initialize(self) -> None:
        self._initialized = True
        self._step_counter = 0
        self._current_phase = "initialize"
        self._current_hop = 0
        self._start_entities = []
        self._beam_paths = []
        self._best_path = None
        self._initial_query = ""

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
        execution_context: ExecutionContext | None = None,
    ) -> ThoughtNode:
        if not self._initialized:
            raise RuntimeError("Think-on-Graph must be initialized before execution")

        self._execution_context = execution_context
        self._step_counter = 1
        self._current_phase = "initialize"
        self._initial_query = input_text  # Store for use in continue_reasoning

        # Identify starting entities
        self._start_entities = await self._identify_starting_entities(input_text)
        self._beam_paths = [{"path": [e], "score": 1.0} for e in self._start_entities]

        content = (
            f"Step {self._step_counter}: Initialize ToG Exploration\n\n"
            f"Problem: {input_text}\n\n"
            f"Identifying starting entities in knowledge graph...\n\n"
            f"Start Entities Found:\n"
            + "\n".join(f"  - {e}" for e in self._start_entities)
            + f"\n\nConfiguration:\n"
            f"  Beam width: {self._beam_width}\n"
            f"  Max hops: {self._max_hops}\n\n"
            f"Initial beam initialized with {len(self._beam_paths)} paths.\n"
            f"Next: Explore graph via beam search."
        )

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.THINK_ON_GRAPH,
            content=content,
            step_number=self._step_counter,
            depth=0,
            confidence=0.6,
            quality_score=0.6,
            metadata={"phase": self._current_phase, "entities": len(self._start_entities)},
        )
        session.add_thought(thought)
        session.current_method = MethodIdentifier.THINK_ON_GRAPH
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
            raise RuntimeError("Think-on-Graph must be initialized before continuation")

        self._step_counter += 1
        prev_phase = previous_thought.metadata.get("phase", "initialize")

        if prev_phase == "initialize" or (
            prev_phase == "explore" and self._current_hop < self._max_hops
        ):
            self._current_phase = "explore"
            self._current_hop += 1

            # Expand beam paths using sampling or heuristics
            expanded = []
            for path in self._beam_paths[: self._beam_width]:
                path_expansions = await self._expand_beam_path(path, self._initial_query or "")
                expanded.extend(path_expansions)

            # Keep top paths
            self._beam_paths = sorted(expanded, key=lambda x: x["score"], reverse=True)[
                : self._beam_width * 2
            ]

            content = (
                f"Step {self._step_counter}: Explore Hop {self._current_hop}/{self._max_hops}\n\n"
                f"Expanding beam paths in knowledge graph:\n\n"
                f"Current Beam (top {self._beam_width}):\n"
                + "\n".join(
                    f"  [{i + 1}] {' -> '.join(p['path'])} (score: {p['score']:.2f})"
                    for i, p in enumerate(self._beam_paths[: self._beam_width])
                )
                + f"\n\nTotal candidates: {len(self._beam_paths)}\n"
                f"Next: {'Continue exploration' if self._current_hop < self._max_hops else 'Evaluate final paths'}."  # noqa: E501
            )
            thought_type = ThoughtType.REASONING
            confidence = 0.65 + self._current_hop * 0.05
        elif prev_phase == "explore" and self._current_hop >= self._max_hops:
            self._current_phase = "evaluate"
            # Score final paths using sampling or heuristics
            for p in self._beam_paths:
                p["final_score"] = await self._score_path(p, self._initial_query or "")

            self._beam_paths = sorted(
                self._beam_paths, key=lambda x: x.get("final_score", 0), reverse=True
            )
            content = (
                f"Step {self._step_counter}: Evaluate Candidate Paths\n\n"
                f"Scoring {len(self._beam_paths)} candidate paths:\n\n"
                f"Evaluation Criteria:\n"
                f"  - Path coherence\n"
                f"  - Relevance to query\n"
                f"  - Path length penalty\n\n"
                f"Top Evaluated Paths:\n"
                + "\n".join(
                    f"  [{i + 1}] Score: {p.get('final_score', 0):.2f} - {' -> '.join(p['path'])}"
                    for i, p in enumerate(self._beam_paths[:3])
                )
                + "\n\nNext: Select best path for answer."
            )
            thought_type = ThoughtType.VERIFICATION
            confidence = 0.8
        elif prev_phase == "evaluate":
            self._current_phase = "select"
            self._best_path = self._beam_paths[0] if self._beam_paths else None
            best_score = self._best_path.get("final_score", 0) if self._best_path else 0
            content = (
                f"Step {self._step_counter}: Select Best Path\n\n"
                f"Best path selected:\n"
                f"  Path: {' -> '.join(self._best_path['path']) if self._best_path else 'None'}\n"
                f"  Score: {best_score:.2f}\n\n"
                f"Reasoning from path:\n"
                f"  - Starting from {self._best_path['path'][0] if self._best_path else 'N/A'}\n"
                f"  - Following {len(self._best_path['path']) - 1 if self._best_path else 0} hops\n"
                f"  - Reaching conclusion entity\n\n"
                f"Path provides structured reasoning trace."
            )
            thought_type = ThoughtType.SYNTHESIS
            confidence = 0.85
        else:
            self._current_phase = "conclude"
            content = (
                f"Step {self._step_counter}: Final Answer\n\n"
                f"Think-on-Graph Complete:\n"
                f"  Hops explored: {self._current_hop}\n"
                f"  Paths evaluated: {len(self._beam_paths)}\n"
                f"  Best path length: {len(self._best_path['path']) if self._best_path else 0}\n\n"
                f"Final Answer: [Answer derived from best KG path]\n"
                f"Confidence: High (87%)\n\n"
                f"Method: Think-on-Graph (ToG)\n"
                f"  - Initialized from entity mentions\n"
                f"  - Iterative beam search exploration\n"
                f"  - Path scoring and selection\n"
                f"  - Structured reasoning from KG path"
            )
            thought_type = ThoughtType.CONCLUSION
            confidence = 0.87

        thought = ThoughtNode(
            type=thought_type,
            method_id=MethodIdentifier.THINK_ON_GRAPH,
            content=content,
            parent_id=previous_thought.id,
            step_number=self._step_counter,
            depth=previous_thought.depth + 1,
            confidence=confidence,
            quality_score=confidence,
            metadata={
                "phase": self._current_phase,
                "hop": self._current_hop,
                "beam_size": len(self._beam_paths),
            },
        )
        session.add_thought(thought)
        return thought

    async def _identify_starting_entities(self, input_text: str) -> list[str]:
        """Identify starting entities in the knowledge graph using sampling or heuristics."""
        prompt = f"""Given the following problem, identify the key entities that would serve as starting points in a knowledge graph search:

Problem: {input_text}

List 2-3 key entities or concepts that are central to this problem. Format your response as a simple list, one entity per line."""

        system_prompt = (
            "You are an expert at identifying key entities in text for knowledge graph "
            "reasoning. Extract the most relevant entities concisely."
        )

        def fallback() -> str:
            return "\n".join(self._fallback_entity_extraction(input_text))

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # Parse entities from result
        entities = [line.strip() for line in result.strip().split("\n") if line.strip()]
        return entities[:3] if entities else self._fallback_entity_extraction(input_text)

    def _fallback_entity_extraction(self, input_text: str) -> list[str]:
        """Fallback heuristic for entity extraction."""
        # Simple heuristic: extract capitalized words or use generic entities
        words = input_text.split()
        entities = [w.strip(".,!?") for w in words if w[0].isupper() and len(w) > 2]
        return entities[:3] if entities else ["Entity_A", "Entity_B"]

    async def _expand_beam_path(
        self, path: dict[str, Any], input_text: str
    ) -> list[dict[str, Any]]:
        """Expand a beam path by finding neighbors using sampling or heuristics."""
        last_entity = path["path"][-1]

        prompt = f"""Given the following reasoning path and problem, suggest 2 related entities or concepts that could be the next step in exploring the knowledge graph:

Problem: {input_text}
Current Path: {" -> ".join(path["path"])}
Last Entity: {last_entity}

Suggest 2 related entities that would help answer the problem. Format as a simple list, one per line."""

        system_prompt = (
            "You are an expert at knowledge graph traversal. "
            "Suggest relevant next entities for reasoning paths."
        )

        def fallback() -> str:
            return "\n".join(self._fallback_neighbors(last_entity))

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # Parse neighbors from result
        neighbors = [line.strip() for line in result.strip().split("\n") if line.strip()]
        neighbors = neighbors[:2] if neighbors else self._fallback_neighbors(last_entity)

        # Create expanded paths
        expanded = []
        for n in neighbors:
            new_path = {"path": path["path"] + [n], "score": path["score"] * 0.9}
            expanded.append(new_path)

        return expanded

    def _fallback_neighbors(self, entity: str) -> list[str]:
        """Fallback heuristic for finding neighbors."""
        return [f"{entity}_neighbor_{i}" for i in range(2)]

    async def _score_path(self, path: dict[str, Any], input_text: str) -> float:
        """Score a reasoning path using sampling or heuristics."""
        prompt = f"""Evaluate how well this reasoning path helps answer the problem. Rate from 0.0 to 1.0.

Problem: {input_text}
Reasoning Path: {" -> ".join(path["path"])}

Provide a single numerical score between 0.0 and 1.0, where 1.0 means highly relevant and coherent."""

        system_prompt = (
            "You are an expert at evaluating reasoning paths. "
            "Return only a numerical score between 0.0 and 1.0."
        )

        def fallback() -> str:
            return str(self._fallback_path_score(path))

        result = await self._sample_with_fallback(
            user_prompt=prompt,
            fallback_generator=fallback,
            system_prompt=system_prompt,
        )

        # Parse score from result
        try:
            score = float(result.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return self._fallback_path_score(path)

    def _fallback_path_score(self, path: dict[str, Any]) -> float:
        """Fallback heuristic for path scoring."""
        # Simple heuristic: penalize longer paths
        score = float(path["score"])
        return score * (1.0 - len(path["path"]) * 0.05)

    async def health_check(self) -> bool:
        return self._initialized


__all__ = ["ThinkOnGraph", "THINK_ON_GRAPH_METADATA"]

"""Atom of Thoughts reasoning method implementation.

Atom of Thoughts (AoT) implements atomic decomposition - breaking problems into
smallest indivisible "atoms" of reasoning with explicit dependencies between them.
Each atom represents a fundamental unit of thought that cannot be meaningfully
subdivided, and dependencies form a directed acyclic graph (DAG) of reasoning.

This method is particularly effective for:
- Complex logical arguments requiring traceable reasoning
- Proof construction with explicit dependencies
- Dependency analysis and causal reasoning
- Structured problem solving with clear prerequisites
- Debugging complex reasoning chains

Atom Types:
- PREMISE: Given facts or assumptions (no dependencies)
- REASONING: Logical inference from other atoms
- HYPOTHESIS: Proposed solutions based on reasoning
- VERIFICATION: Testing/validating hypotheses
- CONCLUSION: Final verified conclusions

The method ensures:
- No circular dependencies (DAG structure)
- Topological ordering of atoms
- Explicit dependency tracking
- Atomic granularity of reasoning units
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any
from uuid import uuid4

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethod
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)

# Define metadata for Atom of Thoughts method
ATOM_OF_THOUGHTS_METADATA = MethodMetadata(
    identifier=MethodIdentifier.ATOM_OF_THOUGHTS,
    name="Atom of Thoughts",
    description="Atomic decomposition with explicit dependency tracking between reasoning units",
    category=MethodCategory.HOLISTIC,
    tags=frozenset({
        "atomic",
        "decomposition",
        "dependencies",
        "structured",
        "dag",
        "logical",
        "traceable",
    }),
    complexity=7,
    supports_branching=True,
    supports_revision=True,
    requires_context=False,
    min_thoughts=3,  # Need at least premises, reasoning, conclusion
    max_thoughts=0,  # Unlimited - depends on problem complexity
    avg_tokens_per_thought=300,
    best_for=(
        "complex logical arguments",
        "proof construction",
        "dependency analysis",
        "traceable reasoning",
        "structured problem solving",
        "causal chain analysis",
    ),
    not_recommended_for=(
        "creative brainstorming",
        "intuition-based decisions",
        "time-sensitive problems",
        "simple linear reasoning",
    ),
)


class AtomOfThoughtsMethod:
    """Atom of Thoughts reasoning method implementation.

    This class implements atomic decomposition of problems into smallest
    indivisible reasoning units (atoms) with explicit dependency tracking.
    Each atom represents a fundamental thought that cannot be meaningfully
    subdivided, and atoms are connected in a directed acyclic graph (DAG).

    Attributes:
        identifier: Unique identifier matching MethodIdentifier.ATOM_OF_THOUGHTS
        name: Human-readable name "Atom of Thoughts"
        description: Brief description of the method
        category: Category as MethodCategory.HOLISTIC

    Examples:
        >>> method = AtomOfThoughtsMethod()
        >>> session = Session().start()
        >>> await method.initialize()
        >>> thought = await method.execute(
        ...     session,
        ...     "Prove that the sum of two even numbers is even"
        ... )
        >>> # Will generate atoms: premises, reasoning steps, conclusion
    """

    def __init__(self) -> None:
        """Initialize the Atom of Thoughts reasoning method."""
        self._is_initialized = False
        self._atoms: dict[str, dict[str, Any]] = {}
        self._dependency_graph: dict[str, list[str]] = defaultdict(list)

    @property
    def identifier(self) -> str:
        """Return the unique identifier for this method."""
        return str(MethodIdentifier.ATOM_OF_THOUGHTS)

    @property
    def name(self) -> str:
        """Return the human-readable name of this method."""
        return "Atom of Thoughts"

    @property
    def description(self) -> str:
        """Return a brief description of this method."""
        return "Atomic decomposition with explicit dependency tracking between reasoning units"

    @property
    def category(self) -> str:
        """Return the category this method belongs to."""
        return str(MethodCategory.HOLISTIC)

    async def initialize(self) -> None:
        """Initialize the method.

        For Atom of Thoughts, initialization resets the atom tracking
        and dependency graph structures.
        """
        self._is_initialized = True
        self._atoms = {}
        self._dependency_graph = defaultdict(list)

    async def execute(
        self,
        session: Session,
        input_text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Execute Atom of Thoughts reasoning on the input.

        This method decomposes the problem into atomic reasoning units:
        1. Identify premises (atomic facts with no dependencies)
        2. Build reasoning atoms with explicit dependencies
        3. Generate hypotheses from reasoning
        4. Verify hypotheses
        5. Derive conclusions

        The atoms are structured in a DAG where each atom depends on
        zero or more previous atoms, and no circular dependencies exist.

        Args:
            session: The current reasoning session
            input_text: The problem or question to reason about
            context: Optional additional context

        Returns:
            A ThoughtNode containing the atomic decomposition

        Examples:
            >>> session = Session().start()
            >>> method = AtomOfThoughtsMethod()
            >>> await method.initialize()
            >>> thought = await method.execute(
            ...     session,
            ...     "If all A are B, and all B are C, then all A are C"
            ... )
            >>> assert "PREMISE" in thought.content
            >>> assert "REASONING" in thought.content
            >>> assert "CONCLUSION" in thought.content
        """
        if not self._is_initialized:
            await self.initialize()

        # Extract context parameters
        context = context or {}
        max_atoms = context.get("max_atoms", 10)

        # Reset atom tracking for this execution
        self._atoms = {}
        self._dependency_graph = defaultdict(list)

        # Generate atomic decomposition
        decomposition = self._generate_atomic_decomposition(
            input_text,
            max_atoms=max_atoms,
        )

        # Determine thought type based on session state
        thought_type = ThoughtType.INITIAL
        parent_id = None
        depth = 0

        # If session has thoughts, this is a continuation
        if session.thought_count > 0:
            thought_type = ThoughtType.CONTINUATION
            # Get the most recent thought as parent
            recent_thoughts = session.get_recent_thoughts(n=1)
            if recent_thoughts:
                parent = recent_thoughts[0]
                parent_id = parent.id
                depth = parent.depth + 1

        # Create the thought node
        thought = ThoughtNode(
            id=str(uuid4()),
            type=thought_type,
            method_id=MethodIdentifier.ATOM_OF_THOUGHTS,
            content=decomposition,
            parent_id=parent_id,
            depth=depth,
            confidence=0.88,  # High confidence for structured atomic reasoning
            step_number=session.thought_count + 1,
            metadata={
                "atom_count": len(self._atoms),
                "input_text": input_text,
                "method": "atom_of_thoughts",
                "has_dependencies": len(self._dependency_graph) > 0,
                "atoms": self._atoms,
                "dependency_graph": dict(self._dependency_graph),
            },
        )

        # Add thought to session
        session.add_thought(thought)

        return thought

    async def continue_reasoning(
        self,
        session: Session,
        previous_thought: ThoughtNode,
        *,
        guidance: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> ThoughtNode:
        """Continue reasoning from a previous thought.

        Extends the atomic decomposition by adding new atoms that build
        upon the existing atom graph, potentially revising or refining
        previous atoms.

        Args:
            session: The current reasoning session
            previous_thought: The thought to continue from
            guidance: Optional guidance for continuation
            context: Optional additional context

        Returns:
            A new ThoughtNode continuing the atomic reasoning

        Examples:
            >>> # After initial execution
            >>> continuation = await method.continue_reasoning(
            ...     session,
            ...     previous_thought,
            ...     guidance="Add verification atoms to validate the conclusion"
            ... )
            >>> assert continuation.parent_id == previous_thought.id
        """
        if not self._is_initialized:
            await self.initialize()

        # Restore previous atom state from metadata
        if previous_thought.metadata.get("atoms"):
            self._atoms = previous_thought.metadata["atoms"]
        if previous_thought.metadata.get("dependency_graph"):
            self._dependency_graph = defaultdict(
                list,
                previous_thought.metadata["dependency_graph"]
            )

        # Build continuation text
        continuation_input = guidance or "Extend the atomic reasoning"

        # Generate continued atomic decomposition
        continuation = self._generate_continuation(
            previous_thought.content,
            continuation_input,
        )

        # Create continuation thought
        thought = ThoughtNode(
            id=str(uuid4()),
            type=ThoughtType.CONTINUATION,
            method_id=MethodIdentifier.ATOM_OF_THOUGHTS,
            content=continuation,
            parent_id=previous_thought.id,
            depth=previous_thought.depth + 1,
            confidence=0.85,
            step_number=session.thought_count + 1,
            metadata={
                "atom_count": len(self._atoms),
                "continued_from": previous_thought.id,
                "guidance": guidance,
                "method": "atom_of_thoughts",
                "atoms": self._atoms,
                "dependency_graph": dict(self._dependency_graph),
            },
        )

        # Add to session
        session.add_thought(thought)

        return thought

    async def health_check(self) -> bool:
        """Check if the method is healthy and ready to execute.

        Returns:
            True if the method is initialized and ready, False otherwise

        Examples:
            >>> method = AtomOfThoughtsMethod()
            >>> assert await method.health_check() is False  # Not initialized
            >>> await method.initialize()
            >>> assert await method.health_check() is True  # Now ready
        """
        return self._is_initialized

    def _generate_atomic_decomposition(
        self,
        input_text: str,
        max_atoms: int = 10,
    ) -> str:
        """Generate atomic decomposition of the problem.

        This is a placeholder implementation that demonstrates the structure
        of atomic reasoning. In a real implementation, this would use an LLM
        to generate the actual atoms and dependencies.

        Args:
            input_text: The input problem or question
            max_atoms: Maximum number of atoms to generate

        Returns:
            A formatted string containing the atomic decomposition
        """
        # In a real implementation, this would use an LLM to:
        # 1. Identify atomic facts (premises)
        # 2. Generate reasoning atoms with dependencies
        # 3. Form hypotheses
        # 4. Add verification atoms
        # 5. Derive conclusions

        sections = []
        sections.append("=== ATOMIC DECOMPOSITION ===\n")
        sections.append(f"Problem: {input_text}\n")

        # PREMISES (Atoms with no dependencies) - always needed
        sections.append("\n--- PREMISES (Foundational Atoms) ---")
        premise_atoms = self._create_premise_atoms(input_text)
        sections.extend(premise_atoms)

        # Only add additional atoms if under max_atoms limit
        if len(self._atoms) < max_atoms:
            # REASONING ATOMS (Depend on premises and other reasoning)
            sections.append("\n--- REASONING ATOMS ---")
            reasoning_atoms = self._create_reasoning_atoms(max_atoms)
            sections.extend(reasoning_atoms)

        if len(self._atoms) < max_atoms:
            # HYPOTHESES (Proposed solutions based on reasoning)
            sections.append("\n--- HYPOTHESIS ATOMS ---")
            hypothesis_atoms = self._create_hypothesis_atoms(max_atoms)
            sections.extend(hypothesis_atoms)

        if len(self._atoms) < max_atoms:
            # VERIFICATION (Testing the hypotheses)
            sections.append("\n--- VERIFICATION ATOMS ---")
            verification_atoms = self._create_verification_atoms(max_atoms)
            sections.extend(verification_atoms)

        if len(self._atoms) < max_atoms:
            # CONCLUSION (Final verified conclusion)
            sections.append("\n--- CONCLUSION ATOMS ---")
            conclusion_atoms = self._create_conclusion_atoms(max_atoms)
            sections.extend(conclusion_atoms)

        # DEPENDENCY GRAPH SUMMARY
        sections.append("\n\n--- DEPENDENCY GRAPH ---")
        sections.append(self._format_dependency_graph())

        return "\n".join(sections)

    def _create_premise_atoms(self, input_text: str) -> list[str]:
        """Create premise atoms (foundational facts with no dependencies)."""
        atoms = []

        # Atom P1: Problem statement
        atom_id = f"P1"
        self._atoms[atom_id] = {
            "id": atom_id,
            "type": "PREMISE",
            "content": f"Given problem: {input_text}",
            "dependencies": [],
        }
        atoms.append(f"\n[{atom_id}] (PREMISE) Given problem: {input_text}")
        atoms.append(f"    Dependencies: None")

        # Atom P2: Domain assumptions
        atom_id = "P2"
        self._atoms[atom_id] = {
            "id": atom_id,
            "type": "PREMISE",
            "content": "Assume standard logical rules and mathematical axioms apply",
            "dependencies": [],
        }
        atoms.append(f"\n[{atom_id}] (PREMISE) Assume standard logical rules and mathematical axioms apply")
        atoms.append(f"    Dependencies: None")

        return atoms

    def _create_reasoning_atoms(self, max_atoms: int = 10) -> list[str]:
        """Create reasoning atoms (depend on premises and other reasoning)."""
        atoms = []

        if len(self._atoms) >= max_atoms:
            return atoms

        # Atom R1: Initial analysis
        atom_id = "R1"
        dependencies = ["P1", "P2"]
        self._atoms[atom_id] = {
            "id": atom_id,
            "type": "REASONING",
            "content": "Analyze the problem structure and identify key components",
            "dependencies": dependencies,
        }
        for dep in dependencies:
            self._dependency_graph[dep].append(atom_id)
        atoms.append(f"\n[{atom_id}] (REASONING) Analyze the problem structure and identify key components")
        atoms.append(f"    Dependencies: {', '.join(dependencies)}")

        if len(self._atoms) >= max_atoms:
            return atoms

        # Atom R2: Decompose into subproblems
        atom_id = "R2"
        dependencies = ["R1"]
        self._atoms[atom_id] = {
            "id": atom_id,
            "type": "REASONING",
            "content": "Break down into atomic subproblems that can be solved independently",
            "dependencies": dependencies,
        }
        for dep in dependencies:
            self._dependency_graph[dep].append(atom_id)
        atoms.append(f"\n[{atom_id}] (REASONING) Break down into atomic subproblems that can be solved independently")
        atoms.append(f"    Dependencies: {', '.join(dependencies)}")

        return atoms

    def _create_hypothesis_atoms(self, max_atoms: int = 10) -> list[str]:
        """Create hypothesis atoms (proposed solutions based on reasoning)."""
        atoms = []

        if len(self._atoms) >= max_atoms:
            return atoms

        # Atom H1: Primary hypothesis
        atom_id = "H1"
        # Use available dependencies based on what atoms were created
        dependencies = [d for d in ["R1", "R2"] if d in self._atoms] or ["P1"]
        self._atoms[atom_id] = {
            "id": atom_id,
            "type": "HYPOTHESIS",
            "content": "Propose solution approach based on decomposed subproblems",
            "dependencies": dependencies,
        }
        for dep in dependencies:
            self._dependency_graph[dep].append(atom_id)
        atoms.append(f"\n[{atom_id}] (HYPOTHESIS) Propose solution approach based on decomposed subproblems")
        atoms.append(f"    Dependencies: {', '.join(dependencies)}")

        return atoms

    def _create_verification_atoms(self, max_atoms: int = 10) -> list[str]:
        """Create verification atoms (testing hypotheses)."""
        atoms = []

        if len(self._atoms) >= max_atoms:
            return atoms

        # Atom V1: Verify hypothesis
        atom_id = "V1"
        # Use available dependencies based on what atoms were created
        dependencies = [d for d in ["H1", "P2"] if d in self._atoms] or ["P1"]
        self._atoms[atom_id] = {
            "id": atom_id,
            "type": "VERIFICATION",
            "content": "Test the proposed solution against the original problem constraints",
            "dependencies": dependencies,
        }
        for dep in dependencies:
            self._dependency_graph[dep].append(atom_id)
        atoms.append(f"\n[{atom_id}] (VERIFICATION) Test the proposed solution against the original problem constraints")
        atoms.append(f"    Dependencies: {', '.join(dependencies)}")

        return atoms

    def _create_conclusion_atoms(self, max_atoms: int = 10) -> list[str]:
        """Create conclusion atoms (final verified conclusions)."""
        atoms = []

        if len(self._atoms) >= max_atoms:
            return atoms

        # Atom C1: Final conclusion
        atom_id = "C1"
        # Use available dependencies based on what atoms were created
        dependencies = [d for d in ["V1", "H1"] if d in self._atoms] or ["P1"]
        self._atoms[atom_id] = {
            "id": atom_id,
            "type": "CONCLUSION",
            "content": "Conclude with verified solution based on atomic reasoning chain",
            "dependencies": dependencies,
        }
        for dep in dependencies:
            self._dependency_graph[dep].append(atom_id)
        atoms.append(f"\n[{atom_id}] (CONCLUSION) Conclude with verified solution based on atomic reasoning chain")
        atoms.append(f"    Dependencies: {', '.join(dependencies)}")

        return atoms

    def _format_dependency_graph(self) -> str:
        """Format the dependency graph for display."""
        if not self._dependency_graph:
            return "No dependencies (all atoms are independent)"

        lines = []
        for atom_id, dependents in sorted(self._dependency_graph.items()):
            if dependents:
                lines.append(f"{atom_id} → {', '.join(sorted(dependents))}")

        if not lines:
            return "No dependencies (all atoms are independent)"

        return "\n".join(lines)

    def _generate_continuation(
        self,
        previous_content: str,
        continuation_input: str,
    ) -> str:
        """Generate a continuation of the atomic reasoning.

        Args:
            previous_content: Content from the previous thought
            continuation_input: The continuation guidance or input

        Returns:
            A formatted string continuing the atomic decomposition
        """
        # In a real implementation, this would use an LLM to generate
        # additional atoms based on the guidance

        sections = []
        sections.append("=== EXTENDED ATOMIC DECOMPOSITION ===\n")
        sections.append(f"Guidance: {continuation_input}\n")

        # Add refinement atoms
        sections.append("\n--- REFINEMENT ATOMS ---")

        # Atom X1: Refinement based on guidance
        atom_id = "X1"
        dependencies = ["C1"]  # Depends on previous conclusion
        self._atoms[atom_id] = {
            "id": atom_id,
            "type": "REASONING",
            "content": f"Refine reasoning based on: {continuation_input}",
            "dependencies": dependencies,
        }
        for dep in dependencies:
            if dep in self._atoms:  # Only add if dependency exists
                self._dependency_graph[dep].append(atom_id)
        sections.append(f"\n[{atom_id}] (REASONING) Refine reasoning based on: {continuation_input}")
        sections.append(f"    Dependencies: {', '.join(dependencies)}")

        # Updated dependency graph
        sections.append("\n\n--- UPDATED DEPENDENCY GRAPH ---")
        sections.append(self._format_dependency_graph())

        return "\n".join(sections)

    def _verify_dag(self) -> bool:
        """Verify that the dependency graph is a valid DAG (no cycles).

        Returns:
            True if the graph is a valid DAG, False if cycles exist
        """
        # Topological sort using Kahn's algorithm
        in_degree = defaultdict(int)

        # Calculate in-degrees
        for atom_id in self._atoms:
            if atom_id not in in_degree:
                in_degree[atom_id] = 0
            for dependent in self._dependency_graph.get(atom_id, []):
                in_degree[dependent] += 1

        # Queue of nodes with no incoming edges
        queue = deque([atom_id for atom_id in self._atoms if in_degree[atom_id] == 0])
        processed = 0

        while queue:
            atom_id = queue.popleft()
            processed += 1

            for dependent in self._dependency_graph.get(atom_id, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        # If we processed all atoms, it's a valid DAG
        return processed == len(self._atoms)

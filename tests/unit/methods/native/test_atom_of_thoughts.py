"""Unit tests for AtomOfThoughts reasoning method.

This module provides comprehensive tests for the AtomOfThoughtsMethod implementation,
covering initialization, execution, atom types, dependency tracking, DAG structure,
topological ordering, conclusion synthesis, and edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

from reasoning_mcp.methods.native.atom_of_thoughts import (
    ATOM_OF_THOUGHTS_METADATA,
    AtomOfThoughtsMethod,
)
from reasoning_mcp.models import Session, ThoughtNode
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier, ThoughtType

# Fixtures


@pytest.fixture
def aot_method() -> AtomOfThoughtsMethod:
    """Create an AtomOfThoughtsMethod instance for testing.

    Returns:
        A fresh AtomOfThoughtsMethod instance
    """
    return AtomOfThoughtsMethod()


@pytest.fixture
def initialized_method() -> AtomOfThoughtsMethod:
    """Create an AtomOfThoughtsMethod instance (initialization handled in tests).

    Returns:
        An AtomOfThoughtsMethod instance ready for initialization
    """
    return AtomOfThoughtsMethod()


@pytest.fixture
def session() -> Session:
    """Create a fresh session for testing.

    Returns:
        A new Session instance in ACTIVE status
    """
    return Session().start()


@pytest.fixture
def sample_problem() -> str:
    """Provide a sample logical problem for testing.

    Returns:
        A sample problem string
    """
    return "If all A are B, and all B are C, then prove that all A are C"


@pytest.fixture
def math_problem() -> str:
    """Provide a mathematical problem for testing.

    Returns:
        A mathematical problem string
    """
    return "Prove that the sum of two even numbers is always even"


@pytest.fixture
def complex_problem() -> str:
    """Provide a complex problem requiring multiple reasoning steps.

    Returns:
        A complex problem string
    """
    return (
        "Given a directed graph with weighted edges, design an algorithm to find "
        "the shortest path between two nodes, prove its correctness, and analyze "
        "its time complexity"
    )


# Test Metadata


class TestMetadata:
    """Tests for AtomOfThoughts metadata constant."""

    def test_metadata_identifier(self):
        """Test that metadata has the correct identifier."""
        assert ATOM_OF_THOUGHTS_METADATA.identifier == MethodIdentifier.ATOM_OF_THOUGHTS

    def test_metadata_name(self):
        """Test that metadata has the correct name."""
        assert ATOM_OF_THOUGHTS_METADATA.name == "Atom of Thoughts"

    def test_metadata_description(self):
        """Test that metadata has descriptive text."""
        assert len(ATOM_OF_THOUGHTS_METADATA.description) > 0
        assert "atomic" in ATOM_OF_THOUGHTS_METADATA.description.lower()
        assert "dependency" in ATOM_OF_THOUGHTS_METADATA.description.lower()

    def test_metadata_category(self):
        """Test that metadata has correct category."""
        assert ATOM_OF_THOUGHTS_METADATA.category == MethodCategory.HOLISTIC

    def test_metadata_tags(self):
        """Test that metadata contains expected tags."""
        expected_tags = {"atomic", "decomposition", "dependencies", "dag", "traceable"}
        assert expected_tags.issubset(ATOM_OF_THOUGHTS_METADATA.tags)

    def test_metadata_complexity(self):
        """Test that metadata has appropriate complexity rating."""
        assert ATOM_OF_THOUGHTS_METADATA.complexity == 7

    def test_metadata_supports_branching(self):
        """Test that metadata correctly indicates branching support."""
        assert ATOM_OF_THOUGHTS_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test that metadata correctly indicates revision support."""
        assert ATOM_OF_THOUGHTS_METADATA.supports_revision is True

    def test_metadata_requires_context(self):
        """Test that metadata correctly indicates no context requirement."""
        assert ATOM_OF_THOUGHTS_METADATA.requires_context is False

    def test_metadata_min_thoughts(self):
        """Test that metadata defines minimum thoughts."""
        assert ATOM_OF_THOUGHTS_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self):
        """Test that metadata has unlimited max thoughts."""
        assert ATOM_OF_THOUGHTS_METADATA.max_thoughts == 0

    def test_metadata_best_for(self):
        """Test that metadata lists appropriate use cases."""
        best_for = ATOM_OF_THOUGHTS_METADATA.best_for
        assert "logical" in " ".join(best_for).lower() or "proof" in " ".join(best_for).lower()

    def test_metadata_not_recommended_for(self):
        """Test that metadata lists inappropriate use cases."""
        not_recommended = ATOM_OF_THOUGHTS_METADATA.not_recommended_for
        assert len(not_recommended) > 0


# Test Initialization


class TestInitialization:
    """Tests for AtomOfThoughtsMethod initialization and setup."""

    def test_create_method(self, aot_method: AtomOfThoughtsMethod):
        """Test that AtomOfThoughtsMethod can be instantiated."""
        assert aot_method is not None
        assert isinstance(aot_method, AtomOfThoughtsMethod)

    def test_initial_state(self, aot_method: AtomOfThoughtsMethod):
        """Test that a new method starts in the correct initial state."""
        assert aot_method._is_initialized is False
        assert len(aot_method._atoms) == 0
        assert len(aot_method._dependency_graph) == 0

    async def test_initialize(self, aot_method: AtomOfThoughtsMethod):
        """Test that initialize() sets up the method correctly."""
        await aot_method.initialize()
        assert aot_method._is_initialized is True
        assert len(aot_method._atoms) == 0
        assert len(aot_method._dependency_graph) == 0

    async def test_initialize_resets_state(self):
        """Test that initialize() resets atom structures."""
        method = AtomOfThoughtsMethod()
        await method.initialize()

        # Add some atoms
        method._atoms["test"] = {"id": "test", "type": "PREMISE"}
        method._dependency_graph["test"] = ["dep1"]

        # Re-initialize
        await method.initialize()
        assert len(method._atoms) == 0
        assert len(method._dependency_graph) == 0
        assert method._is_initialized is True

    async def test_health_check_not_initialized(self, aot_method: AtomOfThoughtsMethod):
        """Test that health_check returns False before initialization."""
        result = await aot_method.health_check()
        assert result is False

    async def test_health_check_initialized(self, aot_method: AtomOfThoughtsMethod):
        """Test that health_check returns True after initialization."""
        await aot_method.initialize()
        result = await aot_method.health_check()
        assert result is True


# Test Properties


class TestProperties:
    """Tests for AtomOfThoughtsMethod property accessors."""

    def test_identifier_property(self, aot_method: AtomOfThoughtsMethod):
        """Test that identifier returns the correct method identifier."""
        assert aot_method.identifier == str(MethodIdentifier.ATOM_OF_THOUGHTS)

    def test_name_property(self, aot_method: AtomOfThoughtsMethod):
        """Test that name returns the correct human-readable name."""
        assert aot_method.name == "Atom of Thoughts"

    def test_description_property(self, aot_method: AtomOfThoughtsMethod):
        """Test that description returns the correct method description."""
        description = aot_method.description.lower()
        assert "atomic" in description
        assert "dependency" in description

    def test_category_property(self, aot_method: AtomOfThoughtsMethod):
        """Test that category returns the correct method category."""
        assert aot_method.category == str(MethodCategory.HOLISTIC)


# Test Basic Execution


class TestBasicExecution:
    """Tests for basic execute() functionality."""

    async def test_execute_basic(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test basic execution creates a thought with atomic decomposition."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        assert thought is not None
        assert isinstance(thought, ThoughtNode)
        assert thought.content != ""
        assert thought.method_id == MethodIdentifier.ATOM_OF_THOUGHTS

    async def test_execute_auto_initializes(
        self,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute auto-initializes if not initialized."""
        method = AtomOfThoughtsMethod()
        assert method._is_initialized is False

        thought = await method.execute(session, sample_problem)

        assert thought is not None
        assert method._is_initialized is True

    async def test_execute_creates_initial_thought(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute creates an INITIAL thought type for first execution."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        assert thought.type == ThoughtType.INITIAL
        assert thought.parent_id is None
        assert thought.depth == 0

    async def test_execute_creates_continuation_thought(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute creates CONTINUATION thought when session has thoughts."""
        await aot_method.initialize()

        # First thought
        first_thought = await aot_method.execute(session, sample_problem)
        assert first_thought.type == ThoughtType.INITIAL

        # Second thought should be continuation
        second_thought = await aot_method.execute(session, "Extend the proof")
        assert second_thought.type == ThoughtType.CONTINUATION
        assert second_thought.parent_id == first_thought.id
        assert second_thought.depth == first_thought.depth + 1

    async def test_execute_adds_to_session(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute adds thought to the session."""
        await aot_method.initialize()
        initial_count = session.thought_count

        await aot_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_execute_sets_confidence(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute sets a confidence score."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        assert 0.0 <= thought.confidence <= 1.0
        assert thought.confidence == 0.88  # High confidence for structured reasoning

    async def test_execute_sets_step_number(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that execute sets correct step numbers."""
        await aot_method.initialize()

        first = await aot_method.execute(session, sample_problem)
        assert first.step_number == 1

        second = await aot_method.execute(session, "Continue")
        assert second.step_number == 2


# Test Atom Types


class TestAtomTypes:
    """Tests for different atom types in the decomposition."""

    async def test_premise_atoms_present(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that PREMISE atoms are created."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        assert "PREMISE" in thought.content
        assert "P1" in thought.content or "P2" in thought.content

        # Check metadata
        atoms = thought.metadata.get("atoms", {})
        premise_atoms = [a for a in atoms.values() if a["type"] == "PREMISE"]
        assert len(premise_atoms) > 0

    async def test_reasoning_atoms_present(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that REASONING atoms are created."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        assert "REASONING" in thought.content
        assert "R1" in thought.content or "R2" in thought.content

        # Check metadata
        atoms = thought.metadata.get("atoms", {})
        reasoning_atoms = [a for a in atoms.values() if a["type"] == "REASONING"]
        assert len(reasoning_atoms) > 0

    async def test_hypothesis_atoms_present(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that HYPOTHESIS atoms are created."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        assert "HYPOTHESIS" in thought.content
        assert "H1" in thought.content

        # Check metadata
        atoms = thought.metadata.get("atoms", {})
        hypothesis_atoms = [a for a in atoms.values() if a["type"] == "HYPOTHESIS"]
        assert len(hypothesis_atoms) > 0

    async def test_verification_atoms_present(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that VERIFICATION atoms are created."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        assert "VERIFICATION" in thought.content
        assert "V1" in thought.content

        # Check metadata
        atoms = thought.metadata.get("atoms", {})
        verification_atoms = [a for a in atoms.values() if a["type"] == "VERIFICATION"]
        assert len(verification_atoms) > 0

    async def test_conclusion_atoms_present(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that CONCLUSION atoms are created."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        assert "CONCLUSION" in thought.content
        assert "C1" in thought.content

        # Check metadata
        atoms = thought.metadata.get("atoms", {})
        conclusion_atoms = [a for a in atoms.values() if a["type"] == "CONCLUSION"]
        assert len(conclusion_atoms) > 0

    async def test_all_atom_types_present(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that all atom types are present in a complete decomposition."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        atoms = thought.metadata.get("atoms", {})
        atom_types = {a["type"] for a in atoms.values()}

        expected_types = {"PREMISE", "REASONING", "HYPOTHESIS", "VERIFICATION", "CONCLUSION"}
        assert expected_types.issubset(atom_types)


# Test Configuration


class TestConfiguration:
    """Tests for configuration options."""

    async def test_max_atoms_configuration(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that max_atoms configuration is respected."""
        await aot_method.initialize()

        context: dict[str, Any] = {"max_atoms": 5}
        thought = await aot_method.execute(session, sample_problem, context=context)

        # Atom count should not exceed max_atoms
        atom_count = thought.metadata.get("atom_count", 0)
        assert atom_count <= 5

    async def test_default_max_atoms(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test default max_atoms value."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        # Default is 10
        atom_count = thought.metadata.get("atom_count", 0)
        assert atom_count <= 10
        assert atom_count > 0

    async def test_empty_context(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test execution with empty context dictionary."""
        await aot_method.initialize()

        thought = await aot_method.execute(session, sample_problem, context={})

        assert thought is not None
        assert thought.metadata.get("atom_count", 0) > 0

    async def test_none_context(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test execution with None context (default)."""
        await aot_method.initialize()

        thought = await aot_method.execute(session, sample_problem, context=None)

        assert thought is not None
        assert thought.metadata.get("atom_count", 0) > 0


# Test Continue Reasoning


class TestContinueReasoning:
    """Tests for continue_reasoning() method."""

    async def test_continue_reasoning_basic(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test basic continuation of reasoning."""
        await aot_method.initialize()

        # Create initial thought
        initial = await aot_method.execute(session, sample_problem)

        # Continue reasoning
        continuation = await aot_method.continue_reasoning(
            session,
            initial,
            guidance="Add verification atoms to validate the conclusion",
        )

        assert continuation is not None
        assert isinstance(continuation, ThoughtNode)

    async def test_continue_auto_initializes(
        self,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning auto-initializes if needed."""
        method = AtomOfThoughtsMethod()
        await method.initialize()

        initial = await method.execute(session, sample_problem)

        # Create new uninitialized method
        method2 = AtomOfThoughtsMethod()
        assert method2._is_initialized is False

        continuation = await method2.continue_reasoning(session, initial)
        assert continuation is not None
        assert method2._is_initialized is True

    async def test_continue_creates_continuation_type(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continue_reasoning creates CONTINUATION thought type."""
        await aot_method.initialize()
        initial = await aot_method.execute(session, sample_problem)

        continuation = await aot_method.continue_reasoning(session, initial)

        assert continuation.type == ThoughtType.CONTINUATION

    async def test_continue_sets_parent(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation has correct parent_id."""
        await aot_method.initialize()
        initial = await aot_method.execute(session, sample_problem)

        continuation = await aot_method.continue_reasoning(session, initial)

        assert continuation.parent_id == initial.id

    async def test_continue_increments_depth(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation increments depth."""
        await aot_method.initialize()
        initial = await aot_method.execute(session, sample_problem)

        continuation = await aot_method.continue_reasoning(session, initial)

        assert continuation.depth == initial.depth + 1

    async def test_continue_with_guidance(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test continuation with custom guidance."""
        await aot_method.initialize()
        initial = await aot_method.execute(session, sample_problem)

        guidance = "Refine the logical chain with additional premises"
        continuation = await aot_method.continue_reasoning(
            session,
            initial,
            guidance=guidance,
        )

        assert "guidance" in continuation.metadata
        assert continuation.metadata["guidance"] == guidance
        assert guidance in continuation.content

    async def test_continue_without_guidance(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test continuation without explicit guidance."""
        await aot_method.initialize()
        initial = await aot_method.execute(session, sample_problem)

        continuation = await aot_method.continue_reasoning(session, initial)

        assert continuation.content != ""
        assert "continued_from" in continuation.metadata

    async def test_continue_adds_to_session(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation is added to session."""
        await aot_method.initialize()
        initial = await aot_method.execute(session, sample_problem)
        count_before = session.thought_count

        await aot_method.continue_reasoning(session, initial)

        assert session.thought_count == count_before + 1

    async def test_continue_adds_atoms(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation adds new atoms to the decomposition."""
        await aot_method.initialize()
        initial = await aot_method.execute(session, sample_problem)
        initial_atom_count = initial.metadata.get("atom_count", 0)

        continuation = await aot_method.continue_reasoning(
            session,
            initial,
            guidance="Add more reasoning steps",
        )

        # Should have more atoms (or at least the same if state is restored)
        continuation_atom_count = continuation.metadata.get("atom_count", 0)
        assert continuation_atom_count >= initial_atom_count

    async def test_continue_preserves_atoms(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation preserves previous atoms."""
        await aot_method.initialize()
        initial = await aot_method.execute(session, sample_problem)

        continuation = await aot_method.continue_reasoning(session, initial)

        # Should have atoms metadata
        assert "atoms" in continuation.metadata
        assert "dependency_graph" in continuation.metadata


# Test Dependency Tracking


class TestDependencyTracking:
    """Tests for explicit dependency tracking between atoms."""

    async def test_dependencies_tracked(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that dependencies are tracked in metadata."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        assert "dependency_graph" in thought.metadata
        dep_graph = thought.metadata["dependency_graph"]
        assert isinstance(dep_graph, dict)

    async def test_premises_have_no_dependencies(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that PREMISE atoms have no dependencies."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        atoms = thought.metadata.get("atoms", {})
        premise_atoms = [a for a in atoms.values() if a["type"] == "PREMISE"]

        for premise in premise_atoms:
            assert len(premise["dependencies"]) == 0

    async def test_reasoning_atoms_have_dependencies(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that REASONING atoms have dependencies."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        atoms = thought.metadata.get("atoms", {})
        reasoning_atoms = [a for a in atoms.values() if a["type"] == "REASONING"]

        # At least some reasoning atoms should have dependencies
        has_dependencies = any(len(r["dependencies"]) > 0 for r in reasoning_atoms)
        assert has_dependencies

    async def test_conclusion_depends_on_verification(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that CONCLUSION atoms depend on VERIFICATION atoms."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        atoms = thought.metadata.get("atoms", {})
        conclusion_atoms = [
            a for a_id, a in atoms.items() if a["type"] == "CONCLUSION" and a_id == "C1"
        ]

        if conclusion_atoms:
            conclusion = conclusion_atoms[0]
            dependencies = conclusion["dependencies"]
            # C1 should depend on V1 and H1
            assert "V1" in dependencies
            assert "H1" in dependencies

    async def test_dependency_graph_structure(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that dependency graph has proper structure."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        dep_graph = thought.metadata.get("dependency_graph", {})

        # Dependency graph maps atom_id -> [dependent_atom_ids]
        assert isinstance(dep_graph, dict)
        for _atom_id, dependents in dep_graph.items():
            assert isinstance(dependents, list)

    async def test_explicit_dependencies_in_content(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that dependencies are shown in content."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        assert "Dependencies:" in thought.content
        assert "DEPENDENCY GRAPH" in thought.content


# Test DAG Structure


class TestDAGStructure:
    """Tests for directed acyclic graph structure."""

    async def test_verify_dag_method_exists(self, aot_method: AtomOfThoughtsMethod):
        """Test that _verify_dag method exists."""
        assert hasattr(aot_method, "_verify_dag")
        assert callable(aot_method._verify_dag)

    async def test_no_circular_dependencies(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that the dependency graph has no circular dependencies."""
        await aot_method.initialize()
        await aot_method.execute(session, sample_problem)

        # Verify DAG property
        is_dag = aot_method._verify_dag()
        assert is_dag is True

    async def test_dag_with_multiple_executions(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test DAG property is maintained across multiple executions."""
        await aot_method.initialize()

        await aot_method.execute(session, sample_problem)
        assert aot_method._verify_dag() is True

        await aot_method.execute(session, "Another problem")
        assert aot_method._verify_dag() is True

    async def test_dag_after_continuation(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test DAG property after continuation."""
        await aot_method.initialize()

        initial = await aot_method.execute(session, sample_problem)
        await aot_method.continue_reasoning(session, initial)

        # Should still be a valid DAG
        is_dag = aot_method._verify_dag()
        assert is_dag is True


# Test Atom Ordering


class TestAtomOrdering:
    """Tests for topological ordering of atoms."""

    async def test_premises_come_first(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that PREMISE atoms appear before dependent atoms."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        content = thought.content

        # Find positions of atom types
        premise_pos = content.find("PREMISES")
        reasoning_pos = content.find("REASONING ATOMS")
        conclusion_pos = content.find("CONCLUSION ATOMS")

        # Premises should come before reasoning and conclusion
        assert premise_pos < reasoning_pos
        assert reasoning_pos < conclusion_pos

    async def test_verification_before_conclusion(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that VERIFICATION atoms appear before CONCLUSION."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        content = thought.content

        verification_pos = content.find("VERIFICATION ATOMS")
        conclusion_pos = content.find("CONCLUSION ATOMS")

        # Verification should come before conclusion
        assert verification_pos < conclusion_pos

    async def test_topological_ordering_in_graph(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that dependency graph represents topological ordering."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        atoms = thought.metadata.get("atoms", {})
        thought.metadata.get("dependency_graph", {})

        # For each atom, its dependencies should not depend on it (acyclic)
        for atom_id, atom in atoms.items():
            for dep_id in atom["dependencies"]:
                # dep_id should not have atom_id in its dependents
                dep_atom = atoms.get(dep_id)
                if dep_atom:
                    assert atom_id not in dep_atom["dependencies"]


# Test Conclusion Synthesis


class TestConclusionSynthesis:
    """Tests for building conclusion from verified atoms."""

    async def test_conclusion_synthesizes_from_verification(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that conclusion is synthesized from verification results."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        atoms = thought.metadata.get("atoms", {})

        # Get conclusion atom
        conclusion_atoms = [a for a in atoms.values() if a["type"] == "CONCLUSION"]
        assert len(conclusion_atoms) > 0

        # Conclusion should reference verification
        conclusion = conclusion_atoms[0]
        assert "V1" in conclusion["dependencies"]

    async def test_conclusion_includes_reasoning_chain(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that conclusion references the reasoning chain."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        content = thought.content

        # Conclusion section should reference atomic reasoning
        conclusion_section = content[content.find("CONCLUSION ATOMS") :]
        assert "atomic reasoning chain" in conclusion_section.lower()

    async def test_conclusion_atom_has_metadata(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that conclusion atom has proper metadata."""
        await aot_method.initialize()
        thought = await aot_method.execute(session, sample_problem)

        atoms = thought.metadata.get("atoms", {})
        conclusion_atoms = [a for a in atoms.values() if a["type"] == "CONCLUSION"]

        assert len(conclusion_atoms) > 0
        conclusion = conclusion_atoms[0]

        assert "id" in conclusion
        assert "type" in conclusion
        assert "content" in conclusion
        assert "dependencies" in conclusion


# Test Edge Cases


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    async def test_empty_problem_string(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
    ):
        """Test execution with empty problem string."""
        await aot_method.initialize()

        thought = await aot_method.execute(session, "")

        assert thought is not None
        assert thought.metadata.get("atom_count", 0) > 0

    async def test_very_short_problem(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
    ):
        """Test execution with very short problem."""
        await aot_method.initialize()

        thought = await aot_method.execute(session, "Prove P")

        assert thought is not None
        assert thought.content != ""

    async def test_very_long_problem(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        complex_problem: str,
    ):
        """Test execution with long, complex problem."""
        await aot_method.initialize()

        thought = await aot_method.execute(session, complex_problem)

        assert thought is not None
        assert complex_problem in thought.content

    async def test_special_characters_in_problem(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
    ):
        """Test execution with special characters."""
        await aot_method.initialize()

        problem = "Prove: ∀x ∈ ℝ, x² ≥ 0 → √(x²) = |x|"
        thought = await aot_method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""

    async def test_single_atom_minimal(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test with minimal atoms (max_atoms=1)."""
        await aot_method.initialize()

        context: dict[str, Any] = {"max_atoms": 1}
        thought = await aot_method.execute(session, sample_problem, context=context)

        assert thought is not None
        # Should still create at least the basic structure

    async def test_many_atoms(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        complex_problem: str,
    ):
        """Test with many atoms (max_atoms=20)."""
        await aot_method.initialize()

        context: dict[str, Any] = {"max_atoms": 20}
        thought = await aot_method.execute(session, complex_problem, context=context)

        atom_count = thought.metadata.get("atom_count", 0)
        assert atom_count <= 20

    async def test_multiple_executions_same_session(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
    ):
        """Test multiple executions in the same session."""
        await aot_method.initialize()

        thoughts = []
        for i in range(3):
            thought = await aot_method.execute(session, f"Prove statement {i}")
            thoughts.append(thought)

        assert len(thoughts) == 3
        assert session.thought_count == 3
        # First should be INITIAL, rest should be CONTINUATION
        assert thoughts[0].type == ThoughtType.INITIAL
        for thought in thoughts[1:]:
            assert thought.type == ThoughtType.CONTINUATION

    async def test_nested_reasoning_depth(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test deeply nested reasoning chain."""
        await aot_method.initialize()

        current = await aot_method.execute(session, sample_problem)

        # Create a chain of continuations
        for i in range(3):
            current = await aot_method.continue_reasoning(
                session,
                current,
                guidance=f"Refine atom chain {i + 1}",
            )

        # Last thought should have depth of 3
        assert current.depth >= 3

    async def test_thought_ids_are_unique(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
    ):
        """Test that generated thoughts have unique IDs."""
        await aot_method.initialize()

        thoughts = []
        for i in range(5):
            thought = await aot_method.execute(session, f"Problem {i}")
            thoughts.append(thought)

        # All IDs should be unique
        ids = [t.id for t in thoughts]
        assert len(ids) == len(set(ids))

    async def test_unicode_in_problem(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
    ):
        """Test execution with Unicode characters."""
        await aot_method.initialize()

        problem = "证明: 如果所有A是B，所有B是C，那么所有A是C (Prove transitivity)"
        thought = await aot_method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""

    async def test_newlines_in_problem(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
    ):
        """Test execution with newlines in problem."""
        await aot_method.initialize()

        problem = """Problem:
Given: All A are B
Given: All B are C
Prove: All A are C"""
        thought = await aot_method.execute(session, problem)

        assert thought is not None
        assert thought.content != ""


# Test Metadata Fields


class TestMetadataFields:
    """Tests for metadata fields in thoughts."""

    async def test_metadata_contains_atom_count(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that metadata tracks atom count."""
        await aot_method.initialize()

        thought = await aot_method.execute(session, sample_problem)

        assert "atom_count" in thought.metadata
        assert isinstance(thought.metadata["atom_count"], int)
        assert thought.metadata["atom_count"] > 0

    async def test_metadata_contains_input_text(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that metadata tracks the input text."""
        await aot_method.initialize()

        thought = await aot_method.execute(session, sample_problem)

        assert "input_text" in thought.metadata
        assert thought.metadata["input_text"] == sample_problem

    async def test_metadata_contains_method(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that metadata identifies the method."""
        await aot_method.initialize()

        thought = await aot_method.execute(session, sample_problem)

        assert "method" in thought.metadata
        assert thought.metadata["method"] == "atom_of_thoughts"

    async def test_metadata_contains_atoms(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that metadata contains atoms dictionary."""
        await aot_method.initialize()

        thought = await aot_method.execute(session, sample_problem)

        assert "atoms" in thought.metadata
        assert isinstance(thought.metadata["atoms"], dict)
        assert len(thought.metadata["atoms"]) > 0

    async def test_metadata_contains_dependency_graph(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that metadata contains dependency graph."""
        await aot_method.initialize()

        thought = await aot_method.execute(session, sample_problem)

        assert "dependency_graph" in thought.metadata
        assert isinstance(thought.metadata["dependency_graph"], dict)

    async def test_metadata_has_dependencies_flag(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that metadata indicates whether dependencies exist."""
        await aot_method.initialize()

        thought = await aot_method.execute(session, sample_problem)

        assert "has_dependencies" in thought.metadata
        assert thought.metadata["has_dependencies"] is True

    async def test_continuation_metadata_references_parent(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that continuation metadata references parent."""
        await aot_method.initialize()

        initial = await aot_method.execute(session, sample_problem)
        continuation = await aot_method.continue_reasoning(session, initial)

        assert "continued_from" in continuation.metadata
        assert continuation.metadata["continued_from"] == initial.id


# Test Session Integration


class TestSessionIntegration:
    """Tests for integration with Session model."""

    async def test_session_thought_count_updates(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that session thought count updates correctly."""
        await aot_method.initialize()

        initial_count = session.thought_count
        await aot_method.execute(session, sample_problem)

        assert session.thought_count == initial_count + 1

    async def test_session_metrics_update(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that session metrics update after execution."""
        await aot_method.initialize()

        await aot_method.execute(session, sample_problem)

        assert session.metrics.total_thoughts > 0
        assert session.metrics.average_confidence > 0.0

    async def test_session_method_tracking(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that session tracks method usage."""
        await aot_method.initialize()

        await aot_method.execute(session, sample_problem)

        method_key = str(MethodIdentifier.ATOM_OF_THOUGHTS)
        assert method_key in session.metrics.methods_used
        assert session.metrics.methods_used[method_key] > 0

    async def test_session_can_retrieve_thoughts_by_method(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that session can filter thoughts by method."""
        await aot_method.initialize()

        await aot_method.execute(session, sample_problem)

        aot_thoughts = session.get_thoughts_by_method(MethodIdentifier.ATOM_OF_THOUGHTS)
        assert len(aot_thoughts) > 0

    async def test_session_graph_structure(
        self,
        aot_method: AtomOfThoughtsMethod,
        session: Session,
        sample_problem: str,
    ):
        """Test that thoughts are properly linked in session graph."""
        await aot_method.initialize()

        initial = await aot_method.execute(session, sample_problem)
        continuation = await aot_method.continue_reasoning(session, initial)

        # Check graph structure
        assert session.graph.node_count >= 2
        assert continuation.id in session.graph.nodes
        assert initial.id in session.graph.nodes

        # Check parent-child relationship
        parent_node = session.graph.get_node(initial.id)
        assert parent_node is not None
        assert continuation.id in parent_node.children_ids

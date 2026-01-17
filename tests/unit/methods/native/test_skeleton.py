"""Comprehensive unit tests for SkeletonOfThought reasoning method.

This module tests the SkeletonOfThought class, which implements a three-phase
reasoning approach:
1. Skeleton Generation: Create high-level outline
2. Parallel Expansion: Expand each skeleton point independently
3. Final Assembly: Synthesize expanded points into coherent answer

Test coverage includes:
- Initialization and health checks
- Basic execution flow
- Two-phase structure (skeleton -> expansion -> assembly)
- Outline generation and extraction
- Configuration options
- Continue reasoning for expansions
- Parallel expansion capabilities
- Point ordering and quality
- Edge cases and error handling
"""

from __future__ import annotations

import pytest

from reasoning_mcp.methods.native.skeleton import (
    SKELETON_OF_THOUGHT_METADATA,
    SkeletonOfThought,
)
from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    ThoughtType,
)
from reasoning_mcp.models.session import Session


class TestSkeletonOfThoughtMetadata:
    """Tests for SKELETON_OF_THOUGHT_METADATA configuration."""

    def test_metadata_identifier(self):
        """Test that metadata has correct identifier."""
        assert SKELETON_OF_THOUGHT_METADATA.identifier == MethodIdentifier.SKELETON_OF_THOUGHT

    def test_metadata_name(self):
        """Test that metadata has correct name."""
        assert SKELETON_OF_THOUGHT_METADATA.name == "Skeleton of Thought"

    def test_metadata_category(self):
        """Test that metadata is in SPECIALIZED category."""
        assert SKELETON_OF_THOUGHT_METADATA.category == MethodCategory.SPECIALIZED

    def test_metadata_complexity(self):
        """Test that metadata has medium complexity (4)."""
        assert SKELETON_OF_THOUGHT_METADATA.complexity == 4

    def test_metadata_supports_branching(self):
        """Test that metadata supports branching for parallel expansion."""
        assert SKELETON_OF_THOUGHT_METADATA.supports_branching is True

    def test_metadata_supports_revision(self):
        """Test that metadata supports revision."""
        assert SKELETON_OF_THOUGHT_METADATA.supports_revision is True

    def test_metadata_requires_context(self):
        """Test that metadata does not require special context."""
        assert SKELETON_OF_THOUGHT_METADATA.requires_context is False

    def test_metadata_min_thoughts(self):
        """Test that metadata has minimum 3 thoughts (skeleton + expansion + synthesis)."""
        assert SKELETON_OF_THOUGHT_METADATA.min_thoughts == 3

    def test_metadata_max_thoughts(self):
        """Test that metadata has no hard limit (0 = unlimited)."""
        assert SKELETON_OF_THOUGHT_METADATA.max_thoughts == 0

    def test_metadata_avg_tokens(self):
        """Test that metadata has medium token count per thought."""
        assert SKELETON_OF_THOUGHT_METADATA.avg_tokens_per_thought == 400

    def test_metadata_tags(self):
        """Test that metadata includes expected tags."""
        expected_tags = {
            "skeleton",
            "outline",
            "parallel",
            "structured",
            "long-form",
            "hierarchical",
        }
        assert expected_tags.issubset(SKELETON_OF_THOUGHT_METADATA.tags)

    def test_metadata_best_for(self):
        """Test that metadata includes best use cases."""
        assert "long-form answers" in SKELETON_OF_THOUGHT_METADATA.best_for
        assert "structured responses" in SKELETON_OF_THOUGHT_METADATA.best_for
        assert "essays and reports" in SKELETON_OF_THOUGHT_METADATA.best_for

    def test_metadata_not_recommended_for(self):
        """Test that metadata includes cases where method is not recommended."""
        assert "simple yes/no questions" in SKELETON_OF_THOUGHT_METADATA.not_recommended_for
        assert "single-step problems" in SKELETON_OF_THOUGHT_METADATA.not_recommended_for


class TestSkeletonOfThoughtInitialization:
    """Tests for SkeletonOfThought initialization and health checks."""

    def test_init_creates_instance(self):
        """Test that __init__ creates a valid instance."""
        method = SkeletonOfThought()
        assert method is not None
        assert isinstance(method, SkeletonOfThought)

    def test_init_sets_not_initialized(self):
        """Test that newly created instance is not initialized."""
        method = SkeletonOfThought()
        assert method._initialized is False

    def test_init_resets_internal_state(self):
        """Test that __init__ sets initial internal state."""
        method = SkeletonOfThought()
        assert method._step_counter == 0
        assert method._skeleton_points == []
        assert method._expanded_points == {}
        assert method._phase == "skeleton"

    @pytest.mark.asyncio
    async def test_initialize_sets_initialized_flag(self):
        """Test that initialize() sets the initialized flag."""
        method = SkeletonOfThought()
        await method.initialize()
        assert method._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_resets_state(self):
        """Test that initialize() resets all internal state."""
        method = SkeletonOfThought()
        # Mess up the internal state
        method._step_counter = 5
        method._skeleton_points = ["1", "2", "3"]
        method._expanded_points = {0: "expansion"}
        method._phase = "assembly"

        # Initialize should reset everything
        await method.initialize()
        assert method._step_counter == 0
        assert method._skeleton_points == []
        assert method._expanded_points == {}
        assert method._phase == "skeleton"

    @pytest.mark.asyncio
    async def test_health_check_before_initialize(self):
        """Test that health_check returns False before initialization."""
        method = SkeletonOfThought()
        health = await method.health_check()
        assert health is False

    @pytest.mark.asyncio
    async def test_health_check_after_initialize(self):
        """Test that health_check returns True after initialization."""
        method = SkeletonOfThought()
        await method.initialize()
        health = await method.health_check()
        assert health is True


class TestSkeletonOfThoughtProperties:
    """Tests for SkeletonOfThought property accessors."""

    def test_identifier_property(self):
        """Test that identifier property returns correct value."""
        method = SkeletonOfThought()
        assert method.identifier == MethodIdentifier.SKELETON_OF_THOUGHT

    def test_name_property(self):
        """Test that name property returns correct value."""
        method = SkeletonOfThought()
        assert method.name == "Skeleton of Thought"

    def test_description_property(self):
        """Test that description property returns metadata description."""
        method = SkeletonOfThought()
        # Description mentions skeleton-first approach and expansion
        assert "skeleton" in method.description.lower()
        assert "expand" in method.description.lower()

    def test_category_property(self):
        """Test that category property returns SPECIALIZED."""
        method = SkeletonOfThought()
        assert method.category == MethodCategory.SPECIALIZED


class TestSkeletonOfThoughtExecution:
    """Tests for SkeletonOfThought execute() method."""

    @pytest.mark.asyncio
    async def test_execute_requires_initialization(self):
        """Test that execute() raises error if not initialized."""
        method = SkeletonOfThought()
        session = Session().start()

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.execute(session=session, input_text="Test question")

    @pytest.mark.asyncio
    async def test_execute_creates_initial_thought(self):
        """Test that execute() creates an INITIAL thought."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session, input_text="Explain machine learning concepts"
        )

        assert thought.type == ThoughtType.INITIAL
        assert thought.method_id == MethodIdentifier.SKELETON_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_execute_generates_skeleton_content(self):
        """Test that execute() generates skeleton/outline content."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(
            session=session, input_text="Explain object-oriented programming"
        )

        assert "Skeleton of Thought" in thought.content
        assert "1." in thought.content  # Should have numbered points
        assert "2." in thought.content

    @pytest.mark.asyncio
    async def test_execute_sets_step_number(self):
        """Test that execute() sets step_number to 1."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session=session, input_text="Test input")

        assert thought.step_number == 1

    @pytest.mark.asyncio
    async def test_execute_sets_depth_zero(self):
        """Test that execute() sets depth to 0 for initial thought."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session=session, input_text="Test input")

        assert thought.depth == 0

    @pytest.mark.asyncio
    async def test_execute_sets_confidence(self):
        """Test that execute() sets a high confidence for structural outline."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session=session, input_text="Test input")

        assert thought.confidence == 0.8

    @pytest.mark.asyncio
    async def test_execute_includes_metadata(self):
        """Test that execute() includes expected metadata."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session=session, input_text="Test input")

        assert "input" in thought.metadata
        assert "phase" in thought.metadata
        assert "skeleton_points" in thought.metadata
        assert "total_points" in thought.metadata
        assert thought.metadata["phase"] == "skeleton"

    @pytest.mark.asyncio
    async def test_execute_with_context(self):
        """Test that execute() accepts and stores context."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        context = {"domain": "computer science", "level": "beginner"}
        thought = await method.execute(session=session, input_text="Test input", context=context)

        assert thought.metadata["context"] == context

    @pytest.mark.asyncio
    async def test_execute_adds_thought_to_session(self):
        """Test that execute() adds thought to session."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        await method.execute(session=session, input_text="Test input")

        assert session.thought_count == 1

    @pytest.mark.asyncio
    async def test_execute_sets_current_method(self):
        """Test that execute() sets current_method on session."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        await method.execute(session=session, input_text="Test input")

        assert session.current_method == MethodIdentifier.SKELETON_OF_THOUGHT

    @pytest.mark.asyncio
    async def test_execute_transitions_to_expansion_phase(self):
        """Test that execute() transitions to expansion phase."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        await method.execute(session=session, input_text="Test input")

        assert method._phase == "expansion"

    @pytest.mark.asyncio
    async def test_execute_extracts_skeleton_points(self):
        """Test that execute() extracts skeleton points."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        thought = await method.execute(session=session, input_text="Test input")

        # Should have extracted 5 main points (based on implementation)
        assert len(method._skeleton_points) == 5
        assert method._skeleton_points == ["1", "2", "3", "4", "5"]
        assert thought.metadata["total_points"] == 5


class TestSkeletonOfThoughtContinueReasoning:
    """Tests for SkeletonOfThought continue_reasoning() method."""

    @pytest.mark.asyncio
    async def test_continue_requires_initialization(self):
        """Test that continue_reasoning() requires initialization."""
        method = SkeletonOfThought()
        session = Session().start()

        # Create a dummy thought
        thought = await self._create_skeleton_thought(session)

        with pytest.raises(RuntimeError, match="must be initialized"):
            await method.continue_reasoning(session=session, previous_thought=thought)

    @pytest.mark.asyncio
    async def test_continue_increments_step_counter(self):
        """Test that continue_reasoning() increments step counter."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")
        assert method._step_counter == 1

        await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )
        assert method._step_counter == 2

    @pytest.mark.asyncio
    async def test_continue_expansion_phase_creates_branch(self):
        """Test that continue_reasoning() in expansion phase creates BRANCH thought."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        expansion = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )

        assert expansion.type == ThoughtType.BRANCH

    @pytest.mark.asyncio
    async def test_continue_expansion_has_parent(self):
        """Test that expansion thought has parent_id set."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        expansion = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )

        assert expansion.parent_id == skeleton.id

    @pytest.mark.asyncio
    async def test_continue_expansion_increases_depth(self):
        """Test that expansion thought has increased depth."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        expansion = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )

        assert expansion.depth == skeleton.depth + 1

    @pytest.mark.asyncio
    async def test_continue_expansion_has_branch_id(self):
        """Test that expansion thought has branch_id set."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        expansion = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )

        assert expansion.branch_id is not None
        assert "point_" in expansion.branch_id

    @pytest.mark.asyncio
    async def test_continue_expansion_metadata(self):
        """Test that expansion thought has correct metadata."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        expansion = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 2"
        )

        assert expansion.metadata["phase"] == "expansion"
        assert "point_number" in expansion.metadata
        assert "expansions_complete" in expansion.metadata
        assert "total_points" in expansion.metadata

    @pytest.mark.asyncio
    async def test_continue_expansion_extracts_point_number_from_guidance(self):
        """Test that continue_reasoning() extracts point number from guidance."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        expansion = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 3"
        )

        assert expansion.metadata["point_number"] == 3

    @pytest.mark.asyncio
    async def test_continue_expansion_without_guidance(self):
        """Test that continue_reasoning() works without guidance."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        expansion = await method.continue_reasoning(session=session, previous_thought=skeleton)

        # Should default to point 1
        assert expansion.metadata["point_number"] == 1

    @pytest.mark.asyncio
    async def test_continue_expansion_tracks_expansions(self):
        """Test that continue_reasoning() tracks expanded points."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")
        assert len(method._expanded_points) == 0

        await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )
        assert len(method._expanded_points) == 1

        await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 2"
        )
        assert len(method._expanded_points) == 2

    @pytest.mark.asyncio
    async def test_continue_transitions_to_assembly_when_complete(self):
        """Test that continue_reasoning() transitions to assembly after all points expanded."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")
        assert method._phase == "expansion"

        # Expand all 5 points
        for i in range(1, 6):
            await method.continue_reasoning(
                session=session, previous_thought=skeleton, guidance=f"Expand point {i}"
            )

        # Should transition to assembly phase
        assert method._phase == "assembly"

    @pytest.mark.asyncio
    async def test_continue_assembly_phase_creates_synthesis(self):
        """Test that continue_reasoning() in assembly phase creates SYNTHESIS thought."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        # Expand all points to trigger assembly phase
        for i in range(1, 6):
            await method.continue_reasoning(
                session=session, previous_thought=skeleton, guidance=f"Expand point {i}"
            )

        # Now continue should create synthesis
        synthesis = await method.continue_reasoning(session=session, previous_thought=skeleton)

        assert synthesis.type == ThoughtType.SYNTHESIS

    @pytest.mark.asyncio
    async def test_continue_assembly_metadata(self):
        """Test that assembly thought has correct metadata."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        # Expand all points
        for i in range(1, 6):
            await method.continue_reasoning(
                session=session, previous_thought=skeleton, guidance=f"Expand point {i}"
            )

        # Create synthesis
        synthesis = await method.continue_reasoning(session=session, previous_thought=skeleton)

        assert synthesis.metadata["phase"] == "assembly"
        assert synthesis.metadata["points_assembled"] == 5

    @pytest.mark.asyncio
    async def test_continue_adds_thoughts_to_session(self):
        """Test that continue_reasoning() adds thoughts to session."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")
        assert session.thought_count == 1

        await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )
        assert session.thought_count == 2

    async def _create_skeleton_thought(self, session: Session):
        """Helper to create a skeleton thought for testing."""
        from reasoning_mcp.models.thought import ThoughtNode

        thought = ThoughtNode(
            type=ThoughtType.INITIAL,
            method_id=MethodIdentifier.SKELETON_OF_THOUGHT,
            content="Test skeleton",
            step_number=1,
            depth=0,
            metadata={"phase": "skeleton"},
        )
        session.add_thought(thought)
        return thought


class TestSkeletonOfThoughtOutlineGeneration:
    """Tests for outline generation and point extraction."""

    def test_generate_skeleton_includes_main_points(self):
        """Test that _generate_skeleton_heuristic creates numbered main points."""
        method = SkeletonOfThought()
        skeleton = method._generate_skeleton_heuristic("Test question", None)

        # Should have 5 main numbered points
        assert "1." in skeleton
        assert "2." in skeleton
        assert "3." in skeleton
        assert "4." in skeleton
        assert "5." in skeleton

    def test_generate_skeleton_includes_input(self):
        """Test that _generate_skeleton_heuristic references the input question."""
        method = SkeletonOfThought()
        input_text = "Explain quantum computing"
        skeleton = method._generate_skeleton_heuristic(input_text, None)

        assert input_text in skeleton

    def test_generate_skeleton_has_structure(self):
        """Test that _generate_skeleton_heuristic creates a structured outline."""
        method = SkeletonOfThought()
        skeleton = method._generate_skeleton_heuristic("Test", None)

        # Should mention creating an outline/skeleton
        assert "Skeleton" in skeleton or "outline" in skeleton.lower()

    def test_extract_skeleton_points_from_numbered_list(self):
        """Test that _extract_skeleton_points extracts numbered points."""
        method = SkeletonOfThought()
        content = """
        Outline:
        1. First point
        2. Second point
        3. Third point
        """

        points = method._extract_skeleton_points(content)
        assert "1" in points
        assert "2" in points
        assert "3" in points

    def test_extract_skeleton_points_handles_no_numbers(self):
        """Test that _extract_skeleton_points returns default if no numbers found."""
        method = SkeletonOfThought()
        content = "Just some text without numbered points"

        points = method._extract_skeleton_points(content)
        # Should return default points
        assert points == ["1", "2", "3", "4", "5"]

    def test_extract_point_number_from_guidance(self):
        """Test that _extract_point_number extracts number from guidance string."""
        method = SkeletonOfThought()

        assert method._extract_point_number("Expand point 1") == 1
        assert method._extract_point_number("Expand point 5") == 5
        assert method._extract_point_number("Point 3 expansion") == 3
        assert method._extract_point_number("Detail for 2") == 2

    def test_extract_point_number_with_no_number(self):
        """Test that _extract_point_number defaults when no number in guidance."""
        method = SkeletonOfThought()
        # No expanded points yet, should default to 1
        assert method._extract_point_number("Expand next point") == 1

    def test_extract_point_number_with_none_guidance(self):
        """Test that _extract_point_number handles None guidance."""
        method = SkeletonOfThought()
        # Should default to next point (1 when no points expanded)
        assert method._extract_point_number(None) == 1

        # After expanding one point
        method._expanded_points[0] = "content"
        assert method._extract_point_number(None) == 2


class TestSkeletonOfThoughtExpansionGeneration:
    """Tests for expansion content generation."""

    def test_generate_expansion_includes_point_number(self):
        """Test that _generate_expansion_heuristic includes the point number."""
        method = SkeletonOfThought()
        expansion = method._generate_expansion_heuristic(
            point_num=2, skeleton_points=["1", "2", "3"], guidance=None, context=None
        )

        assert "Point 2" in expansion or "point 2" in expansion

    def test_generate_expansion_with_guidance(self):
        """Test that _generate_expansion_heuristic includes guidance when provided."""
        method = SkeletonOfThought()
        expansion = method._generate_expansion_heuristic(
            point_num=1, skeleton_points=["1", "2", "3"], guidance="Focus on examples", context=None
        )

        assert "Focus on examples" in expansion

    def test_generate_expansion_without_guidance(self):
        """Test that _generate_expansion_heuristic works without guidance."""
        method = SkeletonOfThought()
        expansion = method._generate_expansion_heuristic(
            point_num=1, skeleton_points=["1", "2", "3"], guidance=None, context=None
        )

        assert "Expansion" in expansion
        assert "Point 1" in expansion or "point 1" in expansion


class TestSkeletonOfThoughtAssembly:
    """Tests for final assembly generation."""

    def test_generate_assembly_includes_point_count(self):
        """Test that _generate_assembly_heuristic mentions number of points assembled."""
        method = SkeletonOfThought()
        expanded = {0: "exp1", 1: "exp2", 2: "exp3"}

        assembly = method._generate_assembly_heuristic(
            expanded_points=expanded, skeleton_points=["1", "2", "3"], context=None
        )

        assert "3" in assembly  # Should mention 3 points

    def test_generate_assembly_has_structure(self):
        """Test that _generate_assembly_heuristic creates structured final answer."""
        method = SkeletonOfThought()
        expanded = {0: "exp1", 1: "exp2"}

        assembly = method._generate_assembly_heuristic(
            expanded_points=expanded, skeleton_points=["1", "2"], context=None
        )

        assert "Final" in assembly or "Assembly" in assembly
        assert "comprehensive" in assembly.lower() or "complete" in assembly.lower()


class TestSkeletonOfThoughtParallelExpansion:
    """Tests for parallel expansion capabilities."""

    @pytest.mark.asyncio
    async def test_can_expand_points_in_any_order(self):
        """Test that points can be expanded in any order (supporting parallelism)."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        # Expand points out of order: 3, 1, 5, 2, 4
        exp3 = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 3"
        )
        exp1 = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )
        exp5 = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 5"
        )

        assert exp3.metadata["point_number"] == 3
        assert exp1.metadata["point_number"] == 1
        assert exp5.metadata["point_number"] == 5

    @pytest.mark.asyncio
    async def test_expansions_are_independent_branches(self):
        """Test that each expansion has unique branch_id."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        exp1 = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )
        exp2 = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 2"
        )

        assert exp1.branch_id != exp2.branch_id
        assert exp1.branch_id == "point_1"
        assert exp2.branch_id == "point_2"

    @pytest.mark.asyncio
    async def test_all_expansions_share_same_parent(self):
        """Test that all expansions reference the skeleton as parent."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        exp1 = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )
        exp2 = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 2"
        )
        exp3 = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 3"
        )

        assert exp1.parent_id == skeleton.id
        assert exp2.parent_id == skeleton.id
        assert exp3.parent_id == skeleton.id


class TestSkeletonOfThoughtEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_point_outline(self):
        """Test handling of outline with single point by overriding after skeleton generation."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Simple question")

        # Manually set to single point AFTER execute to test single-point behavior
        method._skeleton_points = ["1"]
        method._total_points = 1

        # Expand the single point
        await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )

        # After expanding the only point, should transition to assembly
        assert method._phase == "assembly"

    @pytest.mark.asyncio
    async def test_many_points_outline(self):
        """Test handling of outline with many points."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Complex question")

        # Should handle standard 5 points
        assert len(method._skeleton_points) == 5

        # Expand all points
        for i in range(1, 6):
            await method.continue_reasoning(
                session=session, previous_thought=skeleton, guidance=f"Expand point {i}"
            )

        assert len(method._expanded_points) == 5

    @pytest.mark.asyncio
    async def test_duplicate_point_expansion(self):
        """Test expanding the same point multiple times overwrites."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        # Expand point 2 twice
        await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 2"
        )
        await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 2"
        )

        # Point 2 (index 1) should be overwritten
        assert 1 in method._expanded_points
        # Only 1 unique point expanded (point 2)
        assert len(method._expanded_points) == 1

    @pytest.mark.asyncio
    async def test_empty_input_text(self):
        """Test handling of empty input text."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        # Should handle empty input gracefully
        thought = await method.execute(session=session, input_text="")

        assert thought is not None
        assert thought.type == ThoughtType.INITIAL

    @pytest.mark.asyncio
    async def test_very_long_input_text(self):
        """Test handling of very long input text."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        long_input = "Explain " + "a very complex topic " * 100

        thought = await method.execute(session=session, input_text=long_input)

        assert thought is not None
        assert long_input in thought.content

    @pytest.mark.asyncio
    async def test_multiple_execute_calls_reset_state(self):
        """Test that calling execute() multiple times resets state properly."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        # First execution
        thought1 = await method.execute(session=session, input_text="First question")
        await method.continue_reasoning(
            session=session, previous_thought=thought1, guidance="Expand point 1"
        )

        # Second execution should reset
        await method.execute(session=session, input_text="Second question")

        assert method._step_counter == 1  # Reset to 1
        assert len(method._expanded_points) == 0  # Cleared
        assert method._phase == "expansion"  # Reset to expansion


class TestSkeletonOfThoughtQualityMetrics:
    """Tests for quality and confidence metrics."""

    @pytest.mark.asyncio
    async def test_skeleton_has_high_confidence(self):
        """Test that skeleton thought has high confidence (0.8)."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        assert skeleton.confidence == 0.8

    @pytest.mark.asyncio
    async def test_expansion_has_medium_confidence(self):
        """Test that expansion thoughts have medium confidence (0.75)."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")
        expansion = await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )

        assert expansion.confidence == 0.75

    @pytest.mark.asyncio
    async def test_assembly_has_highest_confidence(self):
        """Test that assembly thought has highest confidence (0.85)."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        # Expand all points
        for i in range(1, 6):
            await method.continue_reasoning(
                session=session, previous_thought=skeleton, guidance=f"Expand point {i}"
            )

        # Create assembly
        assembly = await method.continue_reasoning(session=session, previous_thought=skeleton)

        assert assembly.confidence == 0.85


class TestSkeletonOfThoughtIntegration:
    """Integration tests for complete reasoning workflows."""

    @pytest.mark.asyncio
    async def test_complete_workflow_skeleton_to_assembly(self):
        """Test complete workflow from skeleton through assembly."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        # Phase 1: Create skeleton
        skeleton = await method.execute(
            session=session, input_text="Explain the principles of software engineering"
        )
        assert skeleton.type == ThoughtType.INITIAL
        assert method._phase == "expansion"
        assert session.thought_count == 1

        # Phase 2: Expand all points
        expansions = []
        for i in range(1, 6):
            exp = await method.continue_reasoning(
                session=session, previous_thought=skeleton, guidance=f"Expand point {i}"
            )
            expansions.append(exp)
            assert exp.type == ThoughtType.BRANCH

        assert len(expansions) == 5
        assert method._phase == "assembly"
        assert session.thought_count == 6  # 1 skeleton + 5 expansions

        # Phase 3: Final assembly
        assembly = await method.continue_reasoning(session=session, previous_thought=skeleton)
        assert assembly.type == ThoughtType.SYNTHESIS
        assert session.thought_count == 7  # + 1 assembly

    @pytest.mark.asyncio
    async def test_session_metrics_updated_correctly(self):
        """Test that session metrics are updated correctly throughout workflow."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        skeleton = await method.execute(session=session, input_text="Test")

        # Check metrics after skeleton
        assert session.metrics.total_thoughts == 1
        assert session.metrics.max_depth_reached == 0

        # Expand one point
        await method.continue_reasoning(
            session=session, previous_thought=skeleton, guidance="Expand point 1"
        )

        # Check metrics after expansion
        assert session.metrics.total_thoughts == 2
        assert session.metrics.max_depth_reached == 1
        assert session.metrics.branches_created >= 1

    @pytest.mark.asyncio
    async def test_can_query_thoughts_by_method(self):
        """Test that thoughts can be queried by method from session."""
        method = SkeletonOfThought()
        await method.initialize()
        session = Session().start()

        await method.execute(session=session, input_text="Test")

        thoughts = session.get_thoughts_by_method(MethodIdentifier.SKELETON_OF_THOUGHT)

        assert len(thoughts) == 1
        assert thoughts[0].method_id == MethodIdentifier.SKELETON_OF_THOUGHT

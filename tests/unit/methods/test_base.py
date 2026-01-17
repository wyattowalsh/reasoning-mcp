"""Unit tests for ReasoningMethod protocol and MethodMetadata."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from reasoning_mcp.methods.base import MethodMetadata, ReasoningMethod
from reasoning_mcp.models.core import MethodCategory, MethodIdentifier

if TYPE_CHECKING:
    from reasoning_mcp.models import Session, ThoughtNode


class TestMethodMetadata:
    """Tests for MethodMetadata dataclass."""

    def test_create_with_required_fields(self):
        """Test creation with only required fields."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="Chain of Thought",
            description="Step-by-step reasoning",
            category=MethodCategory.CORE,
        )
        assert metadata.identifier == MethodIdentifier.CHAIN_OF_THOUGHT
        assert metadata.name == "Chain of Thought"
        assert metadata.description == "Step-by-step reasoning"
        assert metadata.category == MethodCategory.CORE
        assert metadata.complexity == 5  # default
        assert metadata.supports_branching is False  # default
        assert metadata.supports_revision is False  # default
        assert metadata.requires_context is False  # default
        assert metadata.min_thoughts == 1  # default
        assert metadata.max_thoughts == 0  # default (unlimited)
        assert metadata.avg_tokens_per_thought == 500  # default
        assert metadata.tags == frozenset()  # default empty frozenset
        assert metadata.best_for == ()  # default empty tuple
        assert metadata.not_recommended_for == ()  # default empty tuple

    def test_create_with_all_fields(self):
        """Test creation with all fields."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.TREE_OF_THOUGHTS,
            name="Tree of Thoughts",
            description="Branching exploration",
            category=MethodCategory.CORE,
            tags=frozenset({"branching", "exploration"}),
            complexity=7,
            supports_branching=True,
            supports_revision=True,
            requires_context=False,
            min_thoughts=3,
            max_thoughts=10,
            avg_tokens_per_thought=600,
            best_for=("complex decisions", "creative problems"),
            not_recommended_for=("simple queries",),
        )
        assert metadata.identifier == MethodIdentifier.TREE_OF_THOUGHTS
        assert metadata.name == "Tree of Thoughts"
        assert metadata.description == "Branching exploration"
        assert metadata.category == MethodCategory.CORE
        assert metadata.complexity == 7
        assert metadata.supports_branching is True
        assert metadata.supports_revision is True
        assert metadata.requires_context is False
        assert metadata.min_thoughts == 3
        assert metadata.max_thoughts == 10
        assert metadata.avg_tokens_per_thought == 600
        assert "branching" in metadata.tags
        assert "exploration" in metadata.tags
        assert metadata.best_for == ("complex decisions", "creative problems")
        assert metadata.not_recommended_for == ("simple queries",)

    def test_immutability(self):
        """Test that MethodMetadata is immutable."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="Chain of Thought",
            description="Step-by-step",
            category=MethodCategory.CORE,
        )
        with pytest.raises(AttributeError):
            metadata.name = "New Name"

    def test_immutability_frozen_fields(self):
        """Test that frozen fields like tags are immutable."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="Chain of Thought",
            description="Step-by-step",
            category=MethodCategory.CORE,
        )
        # Cannot reassign frozen attribute
        with pytest.raises(AttributeError):
            metadata.tags = frozenset({"new_tag"})

    def test_complexity_validation_too_low(self):
        """Test that complexity must be at least 1."""
        with pytest.raises(ValueError, match="complexity must be 1-10"):
            MethodMetadata(
                identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="CoT",
                description="Test",
                category=MethodCategory.CORE,
                complexity=0,
            )

    def test_complexity_validation_too_high(self):
        """Test that complexity must be at most 10."""
        with pytest.raises(ValueError, match="complexity must be 1-10"):
            MethodMetadata(
                identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="CoT",
                description="Test",
                category=MethodCategory.CORE,
                complexity=11,
            )

    def test_complexity_validation_negative(self):
        """Test that complexity cannot be negative."""
        with pytest.raises(ValueError, match="complexity must be 1-10"):
            MethodMetadata(
                identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="CoT",
                description="Test",
                category=MethodCategory.CORE,
                complexity=-1,
            )

    def test_complexity_boundary_valid_min(self):
        """Test that complexity=1 is valid."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            complexity=1,
        )
        assert metadata.complexity == 1

    def test_complexity_boundary_valid_max(self):
        """Test that complexity=10 is valid."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            complexity=10,
        )
        assert metadata.complexity == 10

    def test_min_thoughts_validation(self):
        """Test that min_thoughts must be at least 1."""
        with pytest.raises(ValueError, match="min_thoughts must be >= 1"):
            MethodMetadata(
                identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="CoT",
                description="Test",
                category=MethodCategory.CORE,
                min_thoughts=0,
            )

    def test_min_thoughts_validation_negative(self):
        """Test that min_thoughts cannot be negative."""
        with pytest.raises(ValueError, match="min_thoughts must be >= 1"):
            MethodMetadata(
                identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="CoT",
                description="Test",
                category=MethodCategory.CORE,
                min_thoughts=-5,
            )

    def test_max_thoughts_validation_negative(self):
        """Test that max_thoughts cannot be negative."""
        with pytest.raises(ValueError, match="max_thoughts must be >= 0"):
            MethodMetadata(
                identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="CoT",
                description="Test",
                category=MethodCategory.CORE,
                max_thoughts=-1,
            )

    def test_max_thoughts_less_than_min(self):
        """Test that max_thoughts must be >= min_thoughts when non-zero."""
        with pytest.raises(ValueError, match="max_thoughts .* must be >= min_thoughts"):
            MethodMetadata(
                identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="CoT",
                description="Test",
                category=MethodCategory.CORE,
                min_thoughts=5,
                max_thoughts=3,
            )

    def test_max_thoughts_zero_unlimited(self):
        """Test that max_thoughts=0 means unlimited."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            min_thoughts=5,
            max_thoughts=0,
        )
        assert metadata.max_thoughts == 0
        # max_thoughts=0 is special case for unlimited, no error should be raised

    def test_max_thoughts_equals_min_thoughts(self):
        """Test that max_thoughts can equal min_thoughts."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            min_thoughts=5,
            max_thoughts=5,
        )
        assert metadata.min_thoughts == 5
        assert metadata.max_thoughts == 5

    def test_max_thoughts_greater_than_min_thoughts(self):
        """Test that max_thoughts can be greater than min_thoughts."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            min_thoughts=3,
            max_thoughts=10,
        )
        assert metadata.min_thoughts == 3
        assert metadata.max_thoughts == 10

    def test_all_method_identifiers_accepted(self):
        """Test that all MethodIdentifier enum values are accepted."""
        # Test a few different method identifiers
        identifiers_to_test = [
            MethodIdentifier.CHAIN_OF_THOUGHT,
            MethodIdentifier.TREE_OF_THOUGHTS,
            MethodIdentifier.REACT,
            MethodIdentifier.SELF_CONSISTENCY,
            MethodIdentifier.ETHICAL_REASONING,
            MethodIdentifier.MCTS,
        ]
        for identifier in identifiers_to_test:
            metadata = MethodMetadata(
                identifier=identifier,
                name=f"Test {identifier}",
                description="Test method",
                category=MethodCategory.CORE,
            )
            assert metadata.identifier == identifier

    def test_all_method_categories_accepted(self):
        """Test that all MethodCategory enum values are accepted."""
        categories = [
            MethodCategory.CORE,
            MethodCategory.HIGH_VALUE,
            MethodCategory.SPECIALIZED,
            MethodCategory.ADVANCED,
            MethodCategory.HOLISTIC,
        ]
        for category in categories:
            metadata = MethodMetadata(
                identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
                name="Test Method",
                description="Test description",
                category=category,
            )
            assert metadata.category == category

    def test_tags_immutable_frozenset(self):
        """Test that tags is a frozenset and cannot be modified."""
        tags = frozenset({"tag1", "tag2"})
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            tags=tags,
        )
        assert metadata.tags == tags
        # frozenset is immutable
        assert isinstance(metadata.tags, frozenset)

    def test_best_for_tuple(self):
        """Test that best_for is stored as a tuple."""
        best_for = ("problem1", "problem2", "problem3")
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            best_for=best_for,
        )
        assert metadata.best_for == best_for
        assert isinstance(metadata.best_for, tuple)

    def test_not_recommended_for_tuple(self):
        """Test that not_recommended_for is stored as a tuple."""
        not_recommended = ("problem1", "problem2")
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            not_recommended_for=not_recommended,
        )
        assert metadata.not_recommended_for == not_recommended
        assert isinstance(metadata.not_recommended_for, tuple)

    def test_avg_tokens_per_thought_custom_value(self):
        """Test setting custom avg_tokens_per_thought."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.CHAIN_OF_THOUGHT,
            name="CoT",
            description="Test",
            category=MethodCategory.CORE,
            avg_tokens_per_thought=750,
        )
        assert metadata.avg_tokens_per_thought == 750

    def test_requires_context_true(self):
        """Test setting requires_context to True."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.REACT,
            name="ReAct",
            description="Test",
            category=MethodCategory.CORE,
            requires_context=True,
        )
        assert metadata.requires_context is True

    def test_supports_branching_true(self):
        """Test setting supports_branching to True."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.TREE_OF_THOUGHTS,
            name="ToT",
            description="Test",
            category=MethodCategory.CORE,
            supports_branching=True,
        )
        assert metadata.supports_branching is True

    def test_supports_revision_true(self):
        """Test setting supports_revision to True."""
        metadata = MethodMetadata(
            identifier=MethodIdentifier.SELF_REFLECTION,
            name="Self-Reflection",
            description="Test",
            category=MethodCategory.HIGH_VALUE,
            supports_revision=True,
        )
        assert metadata.supports_revision is True


class TestReasoningMethodProtocol:
    """Tests for ReasoningMethod protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that ReasoningMethod is runtime checkable."""
        # The @runtime_checkable decorator adds special attributes
        assert hasattr(ReasoningMethod, "__protocol_attrs__") or hasattr(
            ReasoningMethod, "__subclasshook__"
        )

    def test_valid_implementation(self):
        """Test that a valid implementation satisfies the protocol."""

        class ValidMethod:
            streaming_context = None

            @property
            def identifier(self) -> str:
                return "test"

            @property
            def name(self) -> str:
                return "Test Method"

            @property
            def description(self) -> str:
                return "A test method"

            @property
            def category(self) -> str:
                return "core"

            async def initialize(self) -> None:
                pass

            async def execute(self, session, input_text, *, context=None, execution_context=None):
                pass

            async def continue_reasoning(
                self,
                session,
                previous_thought,
                *,
                guidance=None,
                context=None,
                execution_context=None,
            ):
                pass

            async def health_check(self) -> bool:
                return True

            async def emit_thought(self, content: str, confidence: float | None = None) -> None:
                pass

        method = ValidMethod()
        assert isinstance(method, ReasoningMethod)

    def test_invalid_implementation_missing_identifier(self):
        """Test that missing identifier property fails protocol check."""

        class InvalidMethod:
            @property
            def name(self) -> str:
                return "Test Method"

            @property
            def description(self) -> str:
                return "A test method"

            @property
            def category(self) -> str:
                return "core"

            async def initialize(self) -> None:
                pass

            async def execute(self, session, input_text, *, context=None):
                pass

            async def continue_reasoning(
                self, session, previous_thought, *, guidance=None, context=None
            ):
                pass

            async def health_check(self) -> bool:
                return True

        method = InvalidMethod()
        assert not isinstance(method, ReasoningMethod)

    def test_invalid_implementation_missing_name(self):
        """Test that missing name property fails protocol check."""

        class InvalidMethod:
            @property
            def identifier(self) -> str:
                return "test"

            @property
            def description(self) -> str:
                return "A test method"

            @property
            def category(self) -> str:
                return "core"

            async def initialize(self) -> None:
                pass

            async def execute(self, session, input_text, *, context=None):
                pass

            async def continue_reasoning(
                self, session, previous_thought, *, guidance=None, context=None
            ):
                pass

            async def health_check(self) -> bool:
                return True

        method = InvalidMethod()
        assert not isinstance(method, ReasoningMethod)

    def test_invalid_implementation_missing_description(self):
        """Test that missing description property fails protocol check."""

        class InvalidMethod:
            @property
            def identifier(self) -> str:
                return "test"

            @property
            def name(self) -> str:
                return "Test Method"

            @property
            def category(self) -> str:
                return "core"

            async def initialize(self) -> None:
                pass

            async def execute(self, session, input_text, *, context=None):
                pass

            async def continue_reasoning(
                self, session, previous_thought, *, guidance=None, context=None
            ):
                pass

            async def health_check(self) -> bool:
                return True

        method = InvalidMethod()
        assert not isinstance(method, ReasoningMethod)

    def test_invalid_implementation_missing_category(self):
        """Test that missing category property fails protocol check."""

        class InvalidMethod:
            @property
            def identifier(self) -> str:
                return "test"

            @property
            def name(self) -> str:
                return "Test Method"

            @property
            def description(self) -> str:
                return "A test method"

            async def initialize(self) -> None:
                pass

            async def execute(self, session, input_text, *, context=None):
                pass

            async def continue_reasoning(
                self, session, previous_thought, *, guidance=None, context=None
            ):
                pass

            async def health_check(self) -> bool:
                return True

        method = InvalidMethod()
        assert not isinstance(method, ReasoningMethod)

    def test_invalid_implementation_missing_initialize(self):
        """Test that missing initialize method fails protocol check."""

        class InvalidMethod:
            @property
            def identifier(self) -> str:
                return "test"

            @property
            def name(self) -> str:
                return "Test Method"

            @property
            def description(self) -> str:
                return "A test method"

            @property
            def category(self) -> str:
                return "core"

            async def execute(self, session, input_text, *, context=None):
                pass

            async def continue_reasoning(
                self, session, previous_thought, *, guidance=None, context=None
            ):
                pass

            async def health_check(self) -> bool:
                return True

        method = InvalidMethod()
        assert not isinstance(method, ReasoningMethod)

    def test_invalid_implementation_missing_execute(self):
        """Test that missing execute method fails protocol check."""

        class InvalidMethod:
            @property
            def identifier(self) -> str:
                return "test"

            @property
            def name(self) -> str:
                return "Test Method"

            @property
            def description(self) -> str:
                return "A test method"

            @property
            def category(self) -> str:
                return "core"

            async def initialize(self) -> None:
                pass

            async def continue_reasoning(
                self, session, previous_thought, *, guidance=None, context=None
            ):
                pass

            async def health_check(self) -> bool:
                return True

        method = InvalidMethod()
        assert not isinstance(method, ReasoningMethod)

    def test_invalid_implementation_missing_continue_reasoning(self):
        """Test that missing continue_reasoning method fails protocol check."""

        class InvalidMethod:
            @property
            def identifier(self) -> str:
                return "test"

            @property
            def name(self) -> str:
                return "Test Method"

            @property
            def description(self) -> str:
                return "A test method"

            @property
            def category(self) -> str:
                return "core"

            async def initialize(self) -> None:
                pass

            async def execute(self, session, input_text, *, context=None):
                pass

            async def health_check(self) -> bool:
                return True

        method = InvalidMethod()
        assert not isinstance(method, ReasoningMethod)

    def test_invalid_implementation_missing_health_check(self):
        """Test that missing health_check method fails protocol check."""

        class InvalidMethod:
            @property
            def identifier(self) -> str:
                return "test"

            @property
            def name(self) -> str:
                return "Test Method"

            @property
            def description(self) -> str:
                return "A test method"

            @property
            def category(self) -> str:
                return "core"

            async def initialize(self) -> None:
                pass

            async def execute(self, session, input_text, *, context=None):
                pass

            async def continue_reasoning(
                self, session, previous_thought, *, guidance=None, context=None
            ):
                pass

        method = InvalidMethod()
        assert not isinstance(method, ReasoningMethod)

    def test_protocol_with_regular_method_instead_of_property(self):
        """Test that runtime_checkable doesn't distinguish methods from properties.

        NOTE: Python's @runtime_checkable Protocol cannot distinguish between
        regular methods and properties at runtime - it only checks if the
        attribute exists. This is a known limitation. Type checkers (mypy)
        would catch this at static analysis time, but isinstance() won't.
        """

        class MethodInsteadOfProperty:
            streaming_context = None

            def identifier(self) -> str:  # Should be @property, but callable works
                return "test"

            @property
            def name(self) -> str:
                return "Test Method"

            @property
            def description(self) -> str:
                return "A test method"

            @property
            def category(self) -> str:
                return "core"

            async def initialize(self) -> None:
                pass

            async def execute(self, session, input_text, *, context=None, execution_context=None):
                pass

            async def continue_reasoning(
                self,
                session,
                previous_thought,
                *,
                guidance=None,
                context=None,
                execution_context=None,
            ):
                pass

            async def health_check(self) -> bool:
                return True

            async def emit_thought(self, content: str, confidence: float | None = None) -> None:
                pass

        method = MethodInsteadOfProperty()
        # This passes because runtime_checkable only checks attribute existence
        # Type checkers would catch this, but isinstance() won't
        assert isinstance(method, ReasoningMethod)

    def test_valid_implementation_with_additional_methods(self):
        """Test that implementations can have additional methods beyond the protocol."""

        class ExtendedMethod:
            streaming_context = None

            @property
            def identifier(self) -> str:
                return "test"

            @property
            def name(self) -> str:
                return "Test Method"

            @property
            def description(self) -> str:
                return "A test method"

            @property
            def category(self) -> str:
                return "core"

            async def initialize(self) -> None:
                pass

            async def execute(self, session, input_text, *, context=None, execution_context=None):
                pass

            async def continue_reasoning(
                self,
                session,
                previous_thought,
                *,
                guidance=None,
                context=None,
                execution_context=None,
            ):
                pass

            async def health_check(self) -> bool:
                return True

            async def emit_thought(self, content: str, confidence: float | None = None) -> None:
                pass

            # Additional methods
            def extra_method(self) -> str:
                return "extra"

            @property
            def extra_property(self) -> int:
                return 42

        method = ExtendedMethod()
        assert isinstance(method, ReasoningMethod)
        assert method.extra_method() == "extra"
        assert method.extra_property == 42

    def test_valid_implementation_with_type_annotations(self):
        """Test that proper type annotations work with the protocol."""

        class TypedMethod:
            streaming_context = None

            @property
            def identifier(self) -> str:
                return "test"

            @property
            def name(self) -> str:
                return "Test Method"

            @property
            def description(self) -> str:
                return "A test method"

            @property
            def category(self) -> str:
                return "core"

            async def initialize(self) -> None:
                pass

            async def execute(
                self,
                session: Session,
                input_text: str,
                *,
                context: dict[str, Any] | None = None,
                execution_context=None,
            ) -> ThoughtNode:
                # This would normally return a real ThoughtNode
                raise NotImplementedError

            async def continue_reasoning(
                self,
                session: Session,
                previous_thought: ThoughtNode,
                *,
                guidance: str | None = None,
                context: dict[str, Any] | None = None,
                execution_context=None,
            ) -> ThoughtNode:
                # This would normally return a real ThoughtNode
                raise NotImplementedError

            async def health_check(self) -> bool:
                return True

            async def emit_thought(self, content: str, confidence: float | None = None) -> None:
                pass

        method = TypedMethod()
        assert isinstance(method, ReasoningMethod)

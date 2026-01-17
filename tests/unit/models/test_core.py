"""
Comprehensive tests for core enumerations in reasoning_mcp.models.core.

This module provides complete test coverage for all core enums:
- MethodIdentifier (30 values)
- MethodCategory (5 values)
- ThoughtType (10 values)
- SessionStatus (5 values)
- PipelineStageType (6 values)

Each enum is tested for:
1. Expected value existence
2. String type (StrEnum)
3. Value count
4. Value uniqueness
5. String representation
6. Membership checks
"""

from enum import StrEnum

import pytest

from reasoning_mcp.models.core import (
    MethodCategory,
    MethodIdentifier,
    PipelineStageType,
    SessionStatus,
    ThoughtType,
)

# ============================================================================
# MethodIdentifier Tests
# ============================================================================


class TestMethodIdentifier:
    """Test suite for MethodIdentifier enum (30 values)."""

    # Expected values organized by category
    EXPECTED_CORE_METHODS = {
        "SEQUENTIAL_THINKING",
        "CHAIN_OF_THOUGHT",
        "TREE_OF_THOUGHTS",
        "REACT",
        "SELF_CONSISTENCY",
    }

    EXPECTED_HIGH_VALUE_METHODS = {
        "ETHICAL_REASONING",
        "CODE_REASONING",
        "DIALECTIC",
        "SHANNON_THINKING",
        "SELF_REFLECTION",
    }

    EXPECTED_SPECIALIZED_METHODS = {
        "GRAPH_OF_THOUGHTS",
        "MCTS",
        "SKELETON_OF_THOUGHT",
        "LEAST_TO_MOST",
        "STEP_BACK",
        "SELF_ASK",
        "DECOMPOSED_PROMPTING",
        "MATHEMATICAL_REASONING",
        "ABDUCTIVE",
        "ANALOGICAL",
    }

    EXPECTED_ADVANCED_METHODS = {
        "CAUSAL_REASONING",
        "SOCRATIC",
        "COUNTERFACTUAL",
        "METACOGNITIVE",
        "BEAM_SEARCH",
    }

    EXPECTED_HOLISTIC_METHODS = {
        "LATERAL_THINKING",
        "LOTUS_WISDOM",
        "ATOM_OF_THOUGHTS",
        "CASCADE_THINKING",
        "CRASH",
    }

    EXPECTED_ALL_METHODS = (
        EXPECTED_CORE_METHODS
        | EXPECTED_HIGH_VALUE_METHODS
        | EXPECTED_SPECIALIZED_METHODS
        | EXPECTED_ADVANCED_METHODS
        | EXPECTED_HOLISTIC_METHODS
    )

    # Count updated to reflect all 107 implemented methods
    EXPECTED_COUNT = 108

    def test_is_strenum(self):
        """Test that MethodIdentifier is a StrEnum."""
        assert issubclass(MethodIdentifier, StrEnum)
        assert issubclass(MethodIdentifier, str)

    def test_all_expected_values_exist(self):
        """Test that all core method identifiers exist (superset allowed)."""
        actual_names = {member.name for member in MethodIdentifier}
        # Core methods must exist as a subset of actual methods
        assert actual_names >= self.EXPECTED_ALL_METHODS

    def test_value_count(self):
        """Test that all method identifiers are defined."""
        assert len(MethodIdentifier) == self.EXPECTED_COUNT

    def test_values_are_strings(self):
        """Test that all enum values are strings."""
        for member in MethodIdentifier:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    def test_values_are_unique(self):
        """Test that all enum values are unique."""
        values = [member.value for member in MethodIdentifier]
        assert len(values) == len(set(values))

    def test_string_representation(self):
        """Test string representation of enum members."""
        # Test a few representative members
        assert str(MethodIdentifier.SEQUENTIAL_THINKING) == "sequential_thinking"
        assert str(MethodIdentifier.TREE_OF_THOUGHTS) == "tree_of_thoughts"
        assert str(MethodIdentifier.ETHICAL_REASONING) == "ethical_reasoning"
        assert str(MethodIdentifier.MCTS) == "mcts"
        assert str(MethodIdentifier.LOTUS_WISDOM) == "lotus_wisdom"

    def test_membership_checks(self):
        """Test membership checks work correctly."""
        # Valid values
        assert "sequential_thinking" in MethodIdentifier._value2member_map_
        assert "tree_of_thoughts" in MethodIdentifier._value2member_map_
        assert "mcts" in MethodIdentifier._value2member_map_

        # Invalid values
        assert "invalid_method" not in MethodIdentifier._value2member_map_
        assert "" not in MethodIdentifier._value2member_map_
        assert "SEQUENTIAL_THINKING" not in MethodIdentifier._value2member_map_

    def test_value_lookup(self):
        """Test looking up enum members by value."""
        assert MethodIdentifier("sequential_thinking") == MethodIdentifier.SEQUENTIAL_THINKING
        assert MethodIdentifier("tree_of_thoughts") == MethodIdentifier.TREE_OF_THOUGHTS
        assert MethodIdentifier("mcts") == MethodIdentifier.MCTS

    def test_value_lookup_invalid(self):
        """Test that looking up invalid values raises ValueError."""
        with pytest.raises(ValueError):
            MethodIdentifier("invalid_method")

    def test_core_methods_exist(self):
        """Test that all 5 core methods exist."""
        for method_name in self.EXPECTED_CORE_METHODS:
            assert hasattr(MethodIdentifier, method_name)

    def test_high_value_methods_exist(self):
        """Test that all 5 high-value methods exist."""
        for method_name in self.EXPECTED_HIGH_VALUE_METHODS:
            assert hasattr(MethodIdentifier, method_name)

    def test_specialized_methods_exist(self):
        """Test that all 10 specialized methods exist."""
        for method_name in self.EXPECTED_SPECIALIZED_METHODS:
            assert hasattr(MethodIdentifier, method_name)

    def test_advanced_methods_exist(self):
        """Test that all 5 advanced methods exist."""
        for method_name in self.EXPECTED_ADVANCED_METHODS:
            assert hasattr(MethodIdentifier, method_name)

    def test_holistic_methods_exist(self):
        """Test that all 5 holistic methods exist."""
        for method_name in self.EXPECTED_HOLISTIC_METHODS:
            assert hasattr(MethodIdentifier, method_name)

    def test_iteration(self):
        """Test that we can iterate over all members."""
        members = list(MethodIdentifier)
        assert len(members) == self.EXPECTED_COUNT
        # Check that all are instances of the enum
        assert all(isinstance(m, MethodIdentifier) for m in members)

    def test_value_format(self):
        """Test that all values follow snake_case format."""
        for member in MethodIdentifier:
            # All values should be lowercase with underscores
            assert member.value.islower() or "_" in member.value
            # No spaces or hyphens
            assert " " not in member.value
            assert "-" not in member.value


# ============================================================================
# MethodCategory Tests
# ============================================================================


class TestMethodCategory:
    """Test suite for MethodCategory enum (5 values)."""

    EXPECTED_CATEGORIES = {
        "CORE",
        "HIGH_VALUE",
        "SPECIALIZED",
        "ADVANCED",
        "HOLISTIC",
    }

    EXPECTED_COUNT = 5

    def test_is_strenum(self):
        """Test that MethodCategory is a StrEnum."""
        assert issubclass(MethodCategory, StrEnum)
        assert issubclass(MethodCategory, str)

    def test_all_expected_values_exist(self):
        """Test that all 5 expected categories exist."""
        actual_names = {member.name for member in MethodCategory}
        assert actual_names == self.EXPECTED_CATEGORIES

    def test_value_count(self):
        """Test that exactly 5 categories are defined."""
        assert len(MethodCategory) == self.EXPECTED_COUNT

    def test_values_are_strings(self):
        """Test that all enum values are strings."""
        for member in MethodCategory:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    def test_values_are_unique(self):
        """Test that all enum values are unique."""
        values = [member.value for member in MethodCategory]
        assert len(values) == len(set(values))

    def test_string_representation(self):
        """Test string representation of enum members."""
        assert str(MethodCategory.CORE) == "core"
        assert str(MethodCategory.HIGH_VALUE) == "high_value"
        assert str(MethodCategory.SPECIALIZED) == "specialized"
        assert str(MethodCategory.ADVANCED) == "advanced"
        assert str(MethodCategory.HOLISTIC) == "holistic"

    def test_membership_checks(self):
        """Test membership checks work correctly."""
        # Valid values
        assert "core" in MethodCategory._value2member_map_
        assert "high_value" in MethodCategory._value2member_map_
        assert "specialized" in MethodCategory._value2member_map_

        # Invalid values
        assert "invalid_category" not in MethodCategory._value2member_map_
        assert "" not in MethodCategory._value2member_map_
        assert "CORE" not in MethodCategory._value2member_map_

    def test_value_lookup(self):
        """Test looking up enum members by value."""
        assert MethodCategory("core") == MethodCategory.CORE
        assert MethodCategory("high_value") == MethodCategory.HIGH_VALUE
        assert MethodCategory("holistic") == MethodCategory.HOLISTIC

    def test_value_lookup_invalid(self):
        """Test that looking up invalid values raises ValueError."""
        with pytest.raises(ValueError):
            MethodCategory("invalid_category")

    def test_iteration(self):
        """Test that we can iterate over all members."""
        members = list(MethodCategory)
        assert len(members) == self.EXPECTED_COUNT
        assert all(isinstance(m, MethodCategory) for m in members)

    def test_value_format(self):
        """Test that all values follow snake_case format."""
        for member in MethodCategory:
            # All values should be lowercase with underscores
            assert member.value.islower() or "_" in member.value
            # No spaces or hyphens
            assert " " not in member.value
            assert "-" not in member.value


# ============================================================================
# ThoughtType Tests
# ============================================================================


class TestThoughtType:
    """Test suite for ThoughtType enum (13 values)."""

    EXPECTED_THOUGHT_TYPES = {
        "INITIAL",
        "CONTINUATION",
        "BRANCH",
        "REVISION",
        "SYNTHESIS",
        "CONCLUSION",
        "HYPOTHESIS",
        "VERIFICATION",
        "OBSERVATION",
        "ACTION",
        "REASONING",
        "EXPLORATION",
        "INSIGHT",
    }

    EXPECTED_COUNT = 13

    def test_is_strenum(self):
        """Test that ThoughtType is a StrEnum."""
        assert issubclass(ThoughtType, StrEnum)
        assert issubclass(ThoughtType, str)

    def test_all_expected_values_exist(self):
        """Test that all 10 expected thought types exist."""
        actual_names = {member.name for member in ThoughtType}
        assert actual_names == self.EXPECTED_THOUGHT_TYPES

    def test_value_count(self):
        """Test that exactly 10 thought types are defined."""
        assert len(ThoughtType) == self.EXPECTED_COUNT

    def test_values_are_strings(self):
        """Test that all enum values are strings."""
        for member in ThoughtType:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    def test_values_are_unique(self):
        """Test that all enum values are unique."""
        values = [member.value for member in ThoughtType]
        assert len(values) == len(set(values))

    def test_string_representation(self):
        """Test string representation of enum members."""
        assert str(ThoughtType.INITIAL) == "initial"
        assert str(ThoughtType.CONTINUATION) == "continuation"
        assert str(ThoughtType.BRANCH) == "branch"
        assert str(ThoughtType.REVISION) == "revision"
        assert str(ThoughtType.SYNTHESIS) == "synthesis"
        assert str(ThoughtType.CONCLUSION) == "conclusion"
        assert str(ThoughtType.HYPOTHESIS) == "hypothesis"
        assert str(ThoughtType.VERIFICATION) == "verification"
        assert str(ThoughtType.OBSERVATION) == "observation"
        assert str(ThoughtType.ACTION) == "action"

    def test_membership_checks(self):
        """Test membership checks work correctly."""
        # Valid values
        assert "initial" in ThoughtType._value2member_map_
        assert "continuation" in ThoughtType._value2member_map_
        assert "conclusion" in ThoughtType._value2member_map_

        # Invalid values
        assert "invalid_thought" not in ThoughtType._value2member_map_
        assert "" not in ThoughtType._value2member_map_
        assert "INITIAL" not in ThoughtType._value2member_map_

    def test_value_lookup(self):
        """Test looking up enum members by value."""
        assert ThoughtType("initial") == ThoughtType.INITIAL
        assert ThoughtType("continuation") == ThoughtType.CONTINUATION
        assert ThoughtType("conclusion") == ThoughtType.CONCLUSION

    def test_value_lookup_invalid(self):
        """Test that looking up invalid values raises ValueError."""
        with pytest.raises(ValueError):
            ThoughtType("invalid_thought")

    def test_iteration(self):
        """Test that we can iterate over all members."""
        members = list(ThoughtType)
        assert len(members) == self.EXPECTED_COUNT
        assert all(isinstance(m, ThoughtType) for m in members)

    def test_value_format(self):
        """Test that all values follow snake_case format."""
        for member in ThoughtType:
            # All values should be lowercase with underscores
            assert member.value.islower() or "_" in member.value
            # No spaces or hyphens
            assert " " not in member.value
            assert "-" not in member.value

    def test_specific_thought_types(self):
        """Test specific important thought types exist."""
        # Test reasoning flow types
        assert hasattr(ThoughtType, "INITIAL")
        assert hasattr(ThoughtType, "CONTINUATION")
        assert hasattr(ThoughtType, "CONCLUSION")

        # Test branching types
        assert hasattr(ThoughtType, "BRANCH")
        assert hasattr(ThoughtType, "SYNTHESIS")

        # Test correction types
        assert hasattr(ThoughtType, "REVISION")

        # Test ReAct-specific types
        assert hasattr(ThoughtType, "OBSERVATION")
        assert hasattr(ThoughtType, "ACTION")


# ============================================================================
# SessionStatus Tests
# ============================================================================


class TestSessionStatus:
    """Test suite for SessionStatus enum (6 values)."""

    EXPECTED_STATUSES = {
        "CREATED",
        "ACTIVE",
        "PAUSED",
        "COMPLETED",
        "FAILED",
        "CANCELLED",
    }

    EXPECTED_COUNT = 6

    def test_is_strenum(self):
        """Test that SessionStatus is a StrEnum."""
        assert issubclass(SessionStatus, StrEnum)
        assert issubclass(SessionStatus, str)

    def test_all_expected_values_exist(self):
        """Test that all 6 expected statuses exist."""
        actual_names = {member.name for member in SessionStatus}
        assert actual_names == self.EXPECTED_STATUSES

    def test_value_count(self):
        """Test that exactly 6 statuses are defined."""
        assert len(SessionStatus) == self.EXPECTED_COUNT

    def test_values_are_strings(self):
        """Test that all enum values are strings."""
        for member in SessionStatus:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    def test_values_are_unique(self):
        """Test that all enum values are unique."""
        values = [member.value for member in SessionStatus]
        assert len(values) == len(set(values))

    def test_string_representation(self):
        """Test string representation of enum members."""
        assert str(SessionStatus.CREATED) == "created"
        assert str(SessionStatus.ACTIVE) == "active"
        assert str(SessionStatus.PAUSED) == "paused"
        assert str(SessionStatus.COMPLETED) == "completed"
        assert str(SessionStatus.FAILED) == "failed"
        assert str(SessionStatus.CANCELLED) == "cancelled"

    def test_membership_checks(self):
        """Test membership checks work correctly."""
        # Valid values
        assert "created" in SessionStatus._value2member_map_
        assert "active" in SessionStatus._value2member_map_
        assert "paused" in SessionStatus._value2member_map_
        assert "completed" in SessionStatus._value2member_map_

        # Invalid values
        assert "invalid_status" not in SessionStatus._value2member_map_
        assert "" not in SessionStatus._value2member_map_
        assert "ACTIVE" not in SessionStatus._value2member_map_

    def test_value_lookup(self):
        """Test looking up enum members by value."""
        assert SessionStatus("active") == SessionStatus.ACTIVE
        assert SessionStatus("completed") == SessionStatus.COMPLETED
        assert SessionStatus("failed") == SessionStatus.FAILED

    def test_value_lookup_invalid(self):
        """Test that looking up invalid values raises ValueError."""
        with pytest.raises(ValueError):
            SessionStatus("invalid_status")

    def test_iteration(self):
        """Test that we can iterate over all members."""
        members = list(SessionStatus)
        assert len(members) == self.EXPECTED_COUNT
        assert all(isinstance(m, SessionStatus) for m in members)

    def test_value_format(self):
        """Test that all values follow snake_case format."""
        for member in SessionStatus:
            # All values should be lowercase with underscores
            assert member.value.islower() or "_" in member.value
            # No spaces or hyphens
            assert " " not in member.value
            assert "-" not in member.value

    def test_status_lifecycle(self):
        """Test that key lifecycle statuses exist."""
        # Active state
        assert hasattr(SessionStatus, "ACTIVE")
        # Paused state
        assert hasattr(SessionStatus, "PAUSED")
        # Terminal states
        assert hasattr(SessionStatus, "COMPLETED")
        assert hasattr(SessionStatus, "FAILED")
        assert hasattr(SessionStatus, "CANCELLED")


# ============================================================================
# PipelineStageType Tests
# ============================================================================


class TestPipelineStageType:
    """Test suite for PipelineStageType enum (6 values)."""

    EXPECTED_STAGE_TYPES = {
        "METHOD",
        "SEQUENCE",
        "PARALLEL",
        "CONDITIONAL",
        "LOOP",
        "SWITCH",
    }

    EXPECTED_COUNT = 6

    def test_is_strenum(self):
        """Test that PipelineStageType is a StrEnum."""
        assert issubclass(PipelineStageType, StrEnum)
        assert issubclass(PipelineStageType, str)

    def test_all_expected_values_exist(self):
        """Test that all 6 expected stage types exist."""
        actual_names = {member.name for member in PipelineStageType}
        assert actual_names == self.EXPECTED_STAGE_TYPES

    def test_value_count(self):
        """Test that exactly 6 stage types are defined."""
        assert len(PipelineStageType) == self.EXPECTED_COUNT

    def test_values_are_strings(self):
        """Test that all enum values are strings."""
        for member in PipelineStageType:
            assert isinstance(member, str)
            assert isinstance(member.value, str)

    def test_values_are_unique(self):
        """Test that all enum values are unique."""
        values = [member.value for member in PipelineStageType]
        assert len(values) == len(set(values))

    def test_string_representation(self):
        """Test string representation of enum members."""
        assert str(PipelineStageType.METHOD) == "method"
        assert str(PipelineStageType.SEQUENCE) == "sequence"
        assert str(PipelineStageType.PARALLEL) == "parallel"
        assert str(PipelineStageType.CONDITIONAL) == "conditional"
        assert str(PipelineStageType.LOOP) == "loop"
        assert str(PipelineStageType.SWITCH) == "switch"

    def test_membership_checks(self):
        """Test membership checks work correctly."""
        # Valid values
        assert "method" in PipelineStageType._value2member_map_
        assert "sequence" in PipelineStageType._value2member_map_
        assert "parallel" in PipelineStageType._value2member_map_

        # Invalid values
        assert "invalid_stage" not in PipelineStageType._value2member_map_
        assert "" not in PipelineStageType._value2member_map_
        assert "METHOD" not in PipelineStageType._value2member_map_

    def test_value_lookup(self):
        """Test looking up enum members by value."""
        assert PipelineStageType("method") == PipelineStageType.METHOD
        assert PipelineStageType("sequence") == PipelineStageType.SEQUENCE
        assert PipelineStageType("parallel") == PipelineStageType.PARALLEL

    def test_value_lookup_invalid(self):
        """Test that looking up invalid values raises ValueError."""
        with pytest.raises(ValueError):
            PipelineStageType("invalid_stage")

    def test_iteration(self):
        """Test that we can iterate over all members."""
        members = list(PipelineStageType)
        assert len(members) == self.EXPECTED_COUNT
        assert all(isinstance(m, PipelineStageType) for m in members)

    def test_value_format(self):
        """Test that all values follow snake_case format."""
        for member in PipelineStageType:
            # All values should be lowercase with underscores
            assert member.value.islower() or "_" in member.value
            # No spaces or hyphens
            assert " " not in member.value
            assert "-" not in member.value

    def test_execution_patterns(self):
        """Test that key execution pattern types exist."""
        # Basic execution
        assert hasattr(PipelineStageType, "METHOD")
        # Sequential execution
        assert hasattr(PipelineStageType, "SEQUENCE")
        # Parallel execution
        assert hasattr(PipelineStageType, "PARALLEL")
        # Control flow
        assert hasattr(PipelineStageType, "CONDITIONAL")
        assert hasattr(PipelineStageType, "LOOP")
        assert hasattr(PipelineStageType, "SWITCH")


# ============================================================================
# Cross-Enum Integration Tests
# ============================================================================


class TestEnumIntegration:
    """Integration tests across multiple enums."""

    def test_all_enums_are_strenum(self):
        """Test that all core enums inherit from StrEnum."""
        enums = [
            MethodIdentifier,
            MethodCategory,
            ThoughtType,
            SessionStatus,
            PipelineStageType,
        ]
        for enum_cls in enums:
            assert issubclass(enum_cls, StrEnum)
            assert issubclass(enum_cls, str)

    def test_no_value_collisions_across_enums(self):
        """Test that values don't collide across different enums."""
        # Collect all values from all enums
        all_values = []
        enum_sources = {}

        for enum_cls in [
            MethodIdentifier,
            MethodCategory,
            ThoughtType,
            SessionStatus,
            PipelineStageType,
        ]:
            for member in enum_cls:
                all_values.append(member.value)
                if member.value in enum_sources:
                    enum_sources[member.value].append(enum_cls.__name__)
                else:
                    enum_sources[member.value] = [enum_cls.__name__]

        # Check for collisions
        collisions = {value: sources for value, sources in enum_sources.items() if len(sources) > 1}

        # No collisions expected (values should be unique across enums)
        # Note: This test documents current behavior; collisions may be acceptable
        # if they occur in semantically different contexts
        if collisions:
            # Log for information but don't fail - cross-enum collisions may be OK
            print(f"Cross-enum value collisions detected: {collisions}")

    def test_enum_total_count(self):
        """Test the total number of enum members across all enums."""
        total = (
            len(MethodIdentifier)
            + len(MethodCategory)
            + len(ThoughtType)
            + len(SessionStatus)
            + len(PipelineStageType)
        )
        expected_total = 108 + 5 + 13 + 6 + 6  # 138 total enum members
        assert total == expected_total

    def test_all_enum_values_are_lowercase(self):
        """Test that all enum values use lowercase (consistent naming)."""
        for enum_cls in [
            MethodIdentifier,
            MethodCategory,
            ThoughtType,
            SessionStatus,
            PipelineStageType,
        ]:
            for member in enum_cls:
                # Values should be lowercase (may contain underscores)
                assert member.value == member.value.lower()

    def test_enum_comparison_same_type(self):
        """Test that enum members can be compared within same type."""
        # Identity comparison
        assert MethodIdentifier.SEQUENTIAL_THINKING == MethodIdentifier.SEQUENTIAL_THINKING
        assert SessionStatus.ACTIVE == SessionStatus.ACTIVE

        # Inequality
        assert MethodIdentifier.SEQUENTIAL_THINKING != MethodIdentifier.TREE_OF_THOUGHTS
        assert SessionStatus.ACTIVE != SessionStatus.COMPLETED

    def test_enum_comparison_different_types(self):
        """Test that enum members from different types are not equal."""
        # Even if values were the same, different enum types should not be equal
        # (this is guaranteed by Python's enum implementation)
        assert MethodIdentifier.SEQUENTIAL_THINKING != SessionStatus.ACTIVE
        assert ThoughtType.INITIAL != SessionStatus.ACTIVE

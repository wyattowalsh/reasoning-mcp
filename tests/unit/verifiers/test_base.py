"""Unit tests for base verifier module.

Tests the VerifierBase protocol and VerifierMetadata dataclass.
"""

from __future__ import annotations

import pytest

from reasoning_mcp.models.core import VerifierIdentifier
from reasoning_mcp.verifiers.base import VerifierBase, VerifierMetadata


class TestVerifierMetadataCreation:
    """Tests for VerifierMetadata creation."""

    def test_create_minimal_metadata(self) -> None:
        """Test creating metadata with minimal required fields."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test Verifier",
            description="A test verifier",
        )
        assert metadata.identifier == VerifierIdentifier.GEN_PRM
        assert metadata.name == "Test Verifier"
        assert metadata.description == "A test verifier"

    def test_create_full_metadata(self) -> None:
        """Test creating metadata with all fields."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Full Verifier",
            description="A fully configured verifier",
            tags=frozenset({"test", "full"}),
            complexity=7,
            supports_step_level=True,
            supports_outcome_level=True,
            supports_cot_verification=True,
            best_for=("testing", "validation"),
            not_recommended_for=("production",),
        )
        assert metadata.complexity == 7
        assert "test" in metadata.tags
        assert metadata.supports_cot_verification is True
        assert "testing" in metadata.best_for


class TestVerifierMetadataDefaults:
    """Tests for VerifierMetadata default values."""

    def test_default_tags_empty_frozenset(self) -> None:
        """Test default tags is empty frozenset."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
        )
        assert metadata.tags == frozenset()

    def test_default_complexity_five(self) -> None:
        """Test default complexity is 5."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
        )
        assert metadata.complexity == 5

    def test_default_supports_step_level_true(self) -> None:
        """Test default supports_step_level is True."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
        )
        assert metadata.supports_step_level is True

    def test_default_supports_outcome_level_true(self) -> None:
        """Test default supports_outcome_level is True."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
        )
        assert metadata.supports_outcome_level is True

    def test_default_supports_cot_verification_false(self) -> None:
        """Test default supports_cot_verification is False."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
        )
        assert metadata.supports_cot_verification is False

    def test_default_best_for_empty(self) -> None:
        """Test default best_for is empty tuple."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
        )
        assert metadata.best_for == ()

    def test_default_not_recommended_for_empty(self) -> None:
        """Test default not_recommended_for is empty tuple."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
        )
        assert metadata.not_recommended_for == ()


class TestVerifierMetadataValidation:
    """Tests for VerifierMetadata validation."""

    def test_complexity_too_low_raises_error(self) -> None:
        """Test complexity below 1 raises ValueError."""
        with pytest.raises(ValueError, match="complexity must be 1-10"):
            VerifierMetadata(
                identifier=VerifierIdentifier.GEN_PRM,
                name="Test",
                description="Test",
                complexity=0,
            )

    def test_complexity_too_high_raises_error(self) -> None:
        """Test complexity above 10 raises ValueError."""
        with pytest.raises(ValueError, match="complexity must be 1-10"):
            VerifierMetadata(
                identifier=VerifierIdentifier.GEN_PRM,
                name="Test",
                description="Test",
                complexity=11,
            )

    def test_complexity_boundary_low_valid(self) -> None:
        """Test complexity at lower boundary (1) is valid."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
            complexity=1,
        )
        assert metadata.complexity == 1

    def test_complexity_boundary_high_valid(self) -> None:
        """Test complexity at upper boundary (10) is valid."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
            complexity=10,
        )
        assert metadata.complexity == 10

    def test_neither_step_nor_outcome_raises_error(self) -> None:
        """Test verifier must support at least one verification type."""
        with pytest.raises(ValueError, match="must support at least one"):
            VerifierMetadata(
                identifier=VerifierIdentifier.GEN_PRM,
                name="Test",
                description="Test",
                supports_step_level=False,
                supports_outcome_level=False,
            )

    def test_step_only_valid(self) -> None:
        """Test verifier supporting only step-level is valid."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
            supports_step_level=True,
            supports_outcome_level=False,
        )
        assert metadata.supports_step_level is True
        assert metadata.supports_outcome_level is False

    def test_outcome_only_valid(self) -> None:
        """Test verifier supporting only outcome-level is valid."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
            supports_step_level=False,
            supports_outcome_level=True,
        )
        assert metadata.supports_step_level is False
        assert metadata.supports_outcome_level is True


class TestVerifierMetadataImmutability:
    """Tests for VerifierMetadata immutability."""

    def test_metadata_is_frozen(self) -> None:
        """Test metadata is frozen (immutable)."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
        )
        with pytest.raises(AttributeError):
            metadata.name = "Changed"  # type: ignore[misc]

    def test_metadata_hashable(self) -> None:
        """Test metadata is hashable."""
        metadata = VerifierMetadata(
            identifier=VerifierIdentifier.GEN_PRM,
            name="Test",
            description="Test",
        )
        # Should not raise - frozen dataclass is hashable
        hash(metadata)


class TestVerifierBaseProtocol:
    """Tests for VerifierBase protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """Test VerifierBase is runtime checkable."""

        class MockVerifier:
            """Mock verifier for testing."""

            @property
            def identifier(self) -> str:
                return "mock"

            @property
            def name(self) -> str:
                return "Mock"

            @property
            def description(self) -> str:
                return "Mock verifier"

            async def initialize(self) -> None:
                pass

            async def verify(self, solution: str, *, context: dict | None = None):
                return (0.5, "Mock verification")

            async def score_steps(self, steps: list[str], *, context: dict | None = None):
                return [0.5] * len(steps)

            async def health_check(self) -> bool:
                return True

        mock = MockVerifier()
        assert isinstance(mock, VerifierBase)

    def test_incomplete_implementation_not_protocol(self) -> None:
        """Test incomplete implementation is not a VerifierBase."""

        class IncompleteVerifier:
            """Incomplete verifier missing methods."""

            @property
            def identifier(self) -> str:
                return "incomplete"

        incomplete = IncompleteVerifier()
        # Should not be considered a VerifierBase
        assert not isinstance(incomplete, VerifierBase)

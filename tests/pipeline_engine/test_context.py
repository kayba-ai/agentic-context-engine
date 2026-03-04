"""Unit tests for pipeline.StepContext.

Tests cover the generic base class (sample + metadata) and the subclassing
pattern that consuming applications use to add domain fields.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pytest

from pipeline import StepContext

# ---------------------------------------------------------------------------
# Test-local subclass — validates the subclassing pattern
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainContext(StepContext):
    """Minimal subclass used only in these tests."""

    output: Any = None
    score: float = 0.0


# ---------------------------------------------------------------------------
# Base class defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStepContextDefaults:
    def test_sample_defaults_to_none(self):
        ctx = StepContext()
        assert ctx.sample is None

    def test_metadata_defaults_to_empty_mappingproxy(self):
        ctx = StepContext(sample="s")
        assert ctx.metadata == MappingProxyType({})
        assert isinstance(ctx.metadata, MappingProxyType)

    def test_only_two_fields_on_base_class(self):
        field_names = {f.name for f in dataclasses.fields(StepContext)}
        assert field_names == {"sample", "metadata"}


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStepContextImmutability:
    def test_setting_sample_raises(self):
        ctx = StepContext(sample="s")
        with pytest.raises(
            (dataclasses.FrozenInstanceError, AttributeError, TypeError)
        ):
            ctx.sample = "other"  # type: ignore[misc]

    def test_setting_metadata_raises(self):
        ctx = StepContext(sample="s")
        with pytest.raises(
            (dataclasses.FrozenInstanceError, AttributeError, TypeError)
        ):
            ctx.metadata = MappingProxyType({"x": 1})  # type: ignore[misc]

    def test_metadata_mappingproxy_is_not_mutable(self):
        ctx = StepContext(sample="s", metadata={"k": "v"})
        with pytest.raises(TypeError):
            ctx.metadata["k"] = "overwrite"  # type: ignore[index]


# ---------------------------------------------------------------------------
# Coercion
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStepContextCoercion:
    def test_plain_dict_metadata_coerced_to_mappingproxy(self):
        ctx = StepContext(sample="s", metadata={"x": 1})
        assert isinstance(ctx.metadata, MappingProxyType)
        assert ctx.metadata["x"] == 1

    def test_existing_mappingproxy_not_double_wrapped(self):
        mp = MappingProxyType({"x": 1})
        ctx = StepContext(sample="s", metadata=mp)
        assert ctx.metadata is mp


# ---------------------------------------------------------------------------
# replace()
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStepContextReplace:
    def test_replace_returns_new_object(self):
        ctx = StepContext(sample="s")
        ctx2 = ctx.replace(sample="t")
        assert ctx2 is not ctx

    def test_replace_does_not_mutate_original(self):
        ctx = StepContext(sample="s")
        ctx.replace(sample="t")
        assert ctx.sample == "s"

    def test_replace_updates_target_field(self):
        ctx = StepContext(sample="s")
        ctx2 = ctx.replace(sample="t")
        assert ctx2.sample == "t"

    def test_replace_preserves_other_fields(self):
        ctx = StepContext(sample="s", metadata={"k": 1})
        ctx2 = ctx.replace(sample="t")
        assert ctx2.metadata["k"] == 1

    def test_replace_metadata_immutable_pattern(self):
        ctx = StepContext(sample="s", metadata={"x": 1})
        ctx2 = ctx.replace(metadata=MappingProxyType({**ctx.metadata, "y": 2}))
        assert ctx2.metadata["x"] == 1
        assert ctx2.metadata["y"] == 2
        assert "y" not in ctx.metadata  # original unchanged


# ---------------------------------------------------------------------------
# Equality
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStepContextEquality:
    def test_equal_contexts(self):
        ctx1 = StepContext(sample="s")
        ctx2 = StepContext(sample="s")
        assert ctx1 == ctx2

    def test_different_sample_not_equal(self):
        assert StepContext(sample="a") != StepContext(sample="b")

    def test_context_not_hashable(self):
        """StepContext is frozen but NOT hashable: MappingProxyType wraps a dict,
        which is unhashable, so Python cannot derive a hash for the dataclass."""
        ctx = StepContext(sample="s")
        with pytest.raises(TypeError):
            hash(ctx)


# ---------------------------------------------------------------------------
# Subclassing pattern
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestStepContextSubclassing:
    def test_subclass_has_base_fields(self):
        ctx = DomainContext(sample="s")
        assert ctx.sample == "s"
        assert ctx.metadata == MappingProxyType({})

    def test_subclass_has_domain_fields(self):
        ctx = DomainContext(sample="s", output="answer", score=0.95)
        assert ctx.output == "answer"
        assert ctx.score == 0.95

    def test_subclass_defaults(self):
        ctx = DomainContext(sample="s")
        assert ctx.output is None
        assert ctx.score == 0.0

    def test_subclass_is_frozen(self):
        ctx = DomainContext(sample="s", output="x")
        with pytest.raises(
            (dataclasses.FrozenInstanceError, AttributeError, TypeError)
        ):
            ctx.output = "y"  # type: ignore[misc]

    def test_subclass_replace_returns_same_type(self):
        ctx = DomainContext(sample="s")
        ctx2 = ctx.replace(output="answer")
        assert isinstance(ctx2, DomainContext)
        assert ctx2.output == "answer"

    def test_subclass_replace_preserves_base_fields(self):
        ctx = DomainContext(sample="s", metadata={"k": 1})
        ctx2 = ctx.replace(output="x")
        assert ctx2.sample == "s"
        assert ctx2.metadata["k"] == 1

    def test_subclass_replace_preserves_domain_fields(self):
        ctx = DomainContext(sample="s", output="a", score=0.9)
        ctx2 = ctx.replace(sample="t")
        assert ctx2.output == "a"
        assert ctx2.score == 0.9

    def test_subclass_metadata_coercion(self):
        ctx = DomainContext(sample="s", metadata={"x": 1})
        assert isinstance(ctx.metadata, MappingProxyType)

    def test_subclass_isinstance_of_step_context(self):
        ctx = DomainContext(sample="s")
        assert isinstance(ctx, StepContext)

    def test_subclass_equality(self):
        a = DomainContext(sample="s", output="x")
        b = DomainContext(sample="s", output="x")
        assert a == b

    def test_subclass_inequality_on_domain_field(self):
        a = DomainContext(sample="s", output="x")
        b = DomainContext(sample="s", output="y")
        assert a != b


# ---------------------------------------------------------------------------
# Multi-level subclassing (SubSubContext)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExtendedContext(DomainContext):
    """Two-level subclass: StepContext → DomainContext → ExtendedContext."""

    label: str = ""


@pytest.mark.unit
class TestMultiLevelSubclassing:
    def test_has_all_ancestor_fields(self):
        ctx = ExtendedContext(sample="s", output="x", score=0.5, label="L")
        assert ctx.sample == "s"
        assert ctx.metadata == MappingProxyType({})
        assert ctx.output == "x"
        assert ctx.score == 0.5
        assert ctx.label == "L"

    def test_defaults_from_all_levels(self):
        ctx = ExtendedContext(sample="s")
        assert ctx.output is None  # from DomainContext
        assert ctx.score == 0.0  # from DomainContext
        assert ctx.label == ""  # from ExtendedContext

    def test_replace_returns_correct_type(self):
        ctx = ExtendedContext(sample="s")
        ctx2 = ctx.replace(label="new")
        assert isinstance(ctx2, ExtendedContext)
        assert ctx2.label == "new"

    def test_replace_preserves_all_levels(self):
        ctx = ExtendedContext(sample="s", output="x", score=0.9, label="L")
        ctx2 = ctx.replace(sample="t")
        assert ctx2.output == "x"
        assert ctx2.score == 0.9
        assert ctx2.label == "L"

    def test_isinstance_chain(self):
        ctx = ExtendedContext(sample="s")
        assert isinstance(ctx, StepContext)
        assert isinstance(ctx, DomainContext)
        assert isinstance(ctx, ExtendedContext)

    def test_is_frozen(self):
        ctx = ExtendedContext(sample="s", label="L")
        with pytest.raises(
            (dataclasses.FrozenInstanceError, AttributeError, TypeError)
        ):
            ctx.label = "new"  # type: ignore[misc]

    def test_metadata_coercion(self):
        ctx = ExtendedContext(sample="s", metadata={"k": 1})
        assert isinstance(ctx.metadata, MappingProxyType)

    def test_field_count(self):
        field_names = {f.name for f in dataclasses.fields(ExtendedContext)}
        assert field_names == {"sample", "metadata", "output", "score", "label"}


# ---------------------------------------------------------------------------
# Multi-field replace
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestMultiFieldReplace:
    def test_replace_multiple_fields_at_once(self):
        ctx = DomainContext(sample="s", output="a", score=0.1)
        ctx2 = ctx.replace(sample="t", output="b", score=0.9)
        assert ctx2.sample == "t"
        assert ctx2.output == "b"
        assert ctx2.score == 0.9

    def test_replace_base_and_domain_fields_together(self):
        ctx = DomainContext(sample="s", metadata={"k": 1}, output="a")
        ctx2 = ctx.replace(
            sample="t",
            metadata=MappingProxyType({"k": 2}),
            output="b",
        )
        assert ctx2.sample == "t"
        assert ctx2.metadata["k"] == 2
        assert ctx2.output == "b"

    def test_replace_all_fields_on_multi_level_subclass(self):
        ctx = ExtendedContext(sample="s", output="a", score=0.1, label="L")
        ctx2 = ctx.replace(sample="t", output="b", score=0.9, label="M")
        assert isinstance(ctx2, ExtendedContext)
        assert ctx2.sample == "t"
        assert ctx2.output == "b"
        assert ctx2.score == 0.9
        assert ctx2.label == "M"

    def test_original_unchanged_after_multi_field_replace(self):
        ctx = DomainContext(sample="s", output="a", score=0.1)
        ctx.replace(sample="t", output="b", score=0.9)
        assert ctx.sample == "s"
        assert ctx.output == "a"
        assert ctx.score == 0.1


# ---------------------------------------------------------------------------
# Cross-type equality
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OtherContext(StepContext):
    """A different subclass with the same field name as DomainContext."""

    output: Any = None


@pytest.mark.unit
class TestCrossTypeEquality:
    def test_different_subclass_types_not_equal(self):
        a = DomainContext(sample="s", output="x")
        b = OtherContext(sample="s", output="x")
        assert a != b

    def test_base_not_equal_to_subclass(self):
        base = StepContext(sample="s")
        sub = DomainContext(sample="s")
        assert base != sub

    def test_subclass_not_equal_to_sub_subclass(self):
        parent = DomainContext(sample="s")
        child = ExtendedContext(sample="s")
        assert parent != child

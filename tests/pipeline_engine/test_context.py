"""Unit tests for pipeline.StepContext."""

from __future__ import annotations

import dataclasses
from types import MappingProxyType

import pytest

from pipeline import StepContext


@pytest.mark.unit
class TestStepContextDefaults:
    def test_all_optional_fields_default_to_none(self):
        ctx = StepContext(sample="s")
        assert ctx.agent_output is None
        assert ctx.environment_result is None
        assert ctx.reflection is None
        assert ctx.skill_manager_output is None
        assert ctx.skillbook is None
        assert ctx.environment is None

    def test_counter_defaults(self):
        ctx = StepContext(sample="s")
        assert ctx.epoch == 1
        assert ctx.total_epochs == 1
        assert ctx.step_index == 0
        assert ctx.total_steps == 0

    def test_recent_reflections_defaults_to_empty_tuple(self):
        ctx = StepContext(sample="s")
        assert ctx.recent_reflections == ()
        assert isinstance(ctx.recent_reflections, tuple)

    def test_metadata_defaults_to_empty_mappingproxy(self):
        ctx = StepContext(sample="s")
        assert ctx.metadata == MappingProxyType({})
        assert isinstance(ctx.metadata, MappingProxyType)


@pytest.mark.unit
class TestStepContextImmutability:
    def test_setting_named_field_raises(self):
        ctx = StepContext(sample="s")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            ctx.agent_output = "oops"  # type: ignore[misc]

    def test_setting_metadata_raises(self):
        ctx = StepContext(sample="s")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            ctx.metadata = MappingProxyType({"x": 1})  # type: ignore[misc]

    def test_setting_sample_raises(self):
        ctx = StepContext(sample="s")
        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError, TypeError)):
            ctx.sample = "other"  # type: ignore[misc]

    def test_metadata_mappingproxy_is_not_mutable(self):
        ctx = StepContext(sample="s", metadata={"k": "v"})
        with pytest.raises(TypeError):
            ctx.metadata["k"] = "overwrite"  # type: ignore[index]


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

    def test_list_recent_reflections_coerced_to_tuple(self):
        ctx = StepContext(sample="s", recent_reflections=["r1", "r2"])
        assert isinstance(ctx.recent_reflections, tuple)
        assert ctx.recent_reflections == ("r1", "r2")

    def test_tuple_recent_reflections_unchanged(self):
        t = ("r1", "r2")
        ctx = StepContext(sample="s", recent_reflections=t)
        assert ctx.recent_reflections is t


@pytest.mark.unit
class TestStepContextReplace:
    def test_replace_returns_new_object(self):
        ctx = StepContext(sample="s")
        ctx2 = ctx.replace(agent_output="out")
        assert ctx2 is not ctx

    def test_replace_does_not_mutate_original(self):
        ctx = StepContext(sample="s")
        ctx.replace(agent_output="out")
        assert ctx.agent_output is None

    def test_replace_updates_target_field(self):
        ctx = StepContext(sample="s")
        ctx2 = ctx.replace(agent_output="out")
        assert ctx2.agent_output == "out"

    def test_replace_preserves_other_fields(self):
        ctx = StepContext(sample="s", epoch=3)
        ctx2 = ctx.replace(agent_output="x")
        assert ctx2.sample == "s"
        assert ctx2.epoch == 3

    def test_replace_multiple_fields_at_once(self):
        ctx = StepContext(sample="s")
        ctx2 = ctx.replace(agent_output="a", reflection="r", epoch=2)
        assert ctx2.agent_output == "a"
        assert ctx2.reflection == "r"
        assert ctx2.epoch == 2

    def test_replace_metadata_immutable_pattern(self):
        ctx = StepContext(sample="s", metadata={"x": 1})
        ctx2 = ctx.replace(metadata=MappingProxyType({**ctx.metadata, "y": 2}))
        assert ctx2.metadata["x"] == 1
        assert ctx2.metadata["y"] == 2
        assert "y" not in ctx.metadata  # original unchanged

    def test_replace_recent_reflections(self):
        ctx = StepContext(sample="s", recent_reflections=("r1",))
        ctx2 = ctx.replace(recent_reflections=(*ctx.recent_reflections, "r2"))
        assert ctx2.recent_reflections == ("r1", "r2")
        assert ctx.recent_reflections == ("r1",)  # original unchanged


@pytest.mark.unit
class TestStepContextEquality:
    def test_equal_contexts(self):
        ctx1 = StepContext(sample="s", epoch=2)
        ctx2 = StepContext(sample="s", epoch=2)
        assert ctx1 == ctx2

    def test_different_sample_not_equal(self):
        assert StepContext(sample="a") != StepContext(sample="b")

    def test_different_named_field_not_equal(self):
        assert (
            StepContext(sample="s", agent_output="x")
            != StepContext(sample="s", agent_output="y")
        )

    def test_context_not_hashable(self):
        """StepContext is frozen but NOT hashable: MappingProxyType wraps a dict,
        which is unhashable, so Python cannot derive a hash for the dataclass."""
        ctx = StepContext(sample="s")
        with pytest.raises(TypeError):
            hash(ctx)

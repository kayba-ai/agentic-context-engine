"""Regression tests: temperature is only sent when explicitly set."""

import pytest

from ace.providers.config import ModelConfig
from ace.providers.pydantic_ai import build_model_settings, settings_from_config


@pytest.mark.unit
class TestTemperatureDefault:
    def test_model_config_default_is_none(self):
        assert ModelConfig(model="m").temperature is None

    def test_to_dict_omits_unset_temperature(self):
        assert "temperature" not in ModelConfig(model="m").to_dict()

    def test_to_dict_keeps_explicit_zero(self):
        assert ModelConfig(model="m", temperature=0.0).to_dict()["temperature"] == 0.0

    def test_settings_from_config_omits_unset_temperature(self):
        settings = settings_from_config(ModelConfig(model="m", max_tokens=99))
        assert "temperature" not in settings
        assert settings["max_tokens"] == 99

    def test_settings_from_config_keeps_explicit_temperature(self):
        settings = settings_from_config(ModelConfig(model="m", temperature=0.3))
        assert settings["temperature"] == 0.3

    def test_build_model_settings_roundtrip(self):
        assert "temperature" not in build_model_settings(max_tokens=1)
        assert build_model_settings(max_tokens=1, temperature=0.0)["temperature"] == 0.0

    def test_from_model_omits_temperature_by_default(self):
        from ace.runners.litellm import ACELiteLLM

        ace = ACELiteLLM.from_model("test-model")
        assert "temperature" not in ace.agent._agent.model_settings

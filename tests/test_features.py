"""Tests for ace/features.py — optional dependency detection."""

import unittest
from unittest.mock import patch

import pytest

from ace import features
from ace.features import (
    _check_import,
    _FEATURE_CACHE,
    has_opik,
    has_litellm,
    has_langchain,
    has_transformers,
    has_torch,
    has_browser_use,
    has_playwright,
    has_instructor,
    has_numpy,
    has_sentence_transformers,
    get_available_features,
    print_feature_status,
)


@pytest.mark.unit
class TestCheckImport(unittest.TestCase):
    """Tests for _check_import() caching and import behaviour."""

    def setUp(self):
        self._saved = _FEATURE_CACHE.copy()
        _FEATURE_CACHE.clear()

    def tearDown(self):
        _FEATURE_CACHE.clear()
        _FEATURE_CACHE.update(self._saved)

    def test_returns_true_for_stdlib_module(self):
        result = _check_import("json")
        self.assertTrue(result)

    def test_returns_false_for_nonexistent_module(self):
        result = _check_import("__nonexistent_module_xyz__")
        self.assertFalse(result)

    def test_caches_positive_result(self):
        _check_import("json")
        self.assertIn("json", _FEATURE_CACHE)
        self.assertTrue(_FEATURE_CACHE["json"])

    def test_caches_negative_result(self):
        _check_import("__nonexistent_module_xyz__")
        self.assertIn("__nonexistent_module_xyz__", _FEATURE_CACHE)
        self.assertFalse(_FEATURE_CACHE["__nonexistent_module_xyz__"])

    def test_cache_hit_skips_import(self):
        _FEATURE_CACHE["fake_module"] = True
        # Even though the module doesn't exist, cache says True
        self.assertTrue(_check_import("fake_module"))

    def test_cache_hit_returns_false_from_cache(self):
        _FEATURE_CACHE["json"] = False  # override with False
        self.assertFalse(_check_import("json"))

    def test_package_parameter_accepted(self):
        result = _check_import("os", package="path")
        self.assertTrue(result)


@pytest.mark.unit
class TestHasFunctions(unittest.TestCase):
    """Tests for individual has_*() convenience functions."""

    def setUp(self):
        self._saved = _FEATURE_CACHE.copy()
        _FEATURE_CACHE.clear()

    def tearDown(self):
        _FEATURE_CACHE.clear()
        _FEATURE_CACHE.update(self._saved)

    def test_has_opik_calls_check_import_with_opik(self):
        with patch.object(features, "_check_import", return_value=True) as m:
            self.assertTrue(has_opik())
            m.assert_called_once_with("opik")

    def test_has_litellm_checks_litellm(self):
        with patch.object(features, "_check_import", return_value=False) as m:
            self.assertFalse(has_litellm())
            m.assert_called_once_with("litellm")

    def test_has_langchain_checks_langchain_core(self):
        with patch.object(features, "_check_import", return_value=True) as m:
            self.assertTrue(has_langchain())
            m.assert_called_once_with("langchain_core")

    def test_has_transformers(self):
        with patch.object(features, "_check_import", return_value=False) as m:
            self.assertFalse(has_transformers())
            m.assert_called_once_with("transformers")

    def test_has_torch(self):
        with patch.object(features, "_check_import", return_value=True) as m:
            self.assertTrue(has_torch())
            m.assert_called_once_with("torch")

    def test_has_browser_use(self):
        with patch.object(features, "_check_import", return_value=False) as m:
            self.assertFalse(has_browser_use())
            m.assert_called_once_with("browser_use")

    def test_has_playwright(self):
        with patch.object(features, "_check_import", return_value=True) as m:
            self.assertTrue(has_playwright())
            m.assert_called_once_with("playwright")

    def test_has_instructor(self):
        with patch.object(features, "_check_import", return_value=True) as m:
            self.assertTrue(has_instructor())
            m.assert_called_once_with("instructor")

    def test_has_numpy(self):
        with patch.object(features, "_check_import", return_value=False) as m:
            self.assertFalse(has_numpy())
            m.assert_called_once_with("numpy")

    def test_has_sentence_transformers(self):
        with patch.object(features, "_check_import", return_value=True) as m:
            self.assertTrue(has_sentence_transformers())
            m.assert_called_once_with("sentence_transformers")


@pytest.mark.unit
class TestGetAvailableFeatures(unittest.TestCase):
    """Tests for get_available_features() and print_feature_status()."""

    def test_returns_dict_with_all_feature_keys(self):
        result = get_available_features()
        expected_keys = {
            "opik",
            "litellm",
            "langchain",
            "transformers",
            "torch",
            "browser_use",
            "playwright",
            "instructor",
            "numpy",
            "sentence_transformers",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_all_values_are_booleans(self):
        result = get_available_features()
        for key, value in result.items():
            self.assertIsInstance(value, bool, f"{key} is not bool")

    def test_print_feature_status_runs_without_error(
        self,
    ):
        # Just verify it doesn't raise
        print_feature_status()


if __name__ == "__main__":
    unittest.main()

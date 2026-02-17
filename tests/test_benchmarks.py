"""
Tests for the ACE benchmarking system (TAU-bench only).
"""

import unittest
from unittest.mock import patch

import pytest

from benchmarks.base import DataLoader


@pytest.mark.unit
class TestDataLoader(unittest.TestCase):
    """Test DataLoader abstract base class."""

    def test_cannot_instantiate(self):
        """DataLoader is abstract and cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            DataLoader()

    def test_subclass_must_implement(self):
        """Subclass must implement load() and supports_source()."""

        class IncompleteLoader(DataLoader):
            pass

        with self.assertRaises(TypeError):
            IncompleteLoader()

    def test_concrete_subclass(self):
        """A concrete subclass with both methods works."""

        class TestLoader(DataLoader):
            def load(self, **kwargs):
                yield {"id": 1}

            def supports_source(self, source):
                return source == "test"

        loader = TestLoader()
        self.assertTrue(loader.supports_source("test"))
        self.assertFalse(loader.supports_source("other"))
        self.assertEqual(list(loader.load()), [{"id": 1}])


@pytest.mark.unit
class TestTau2LoaderImport(unittest.TestCase):
    """Test that Tau2Loader can be imported (does not require tau2 at import time)."""

    def test_import(self):
        from benchmarks.loaders.tau2 import Tau2Loader

        loader = Tau2Loader()
        self.assertTrue(loader.supports_source("tau2"))
        self.assertFalse(loader.supports_source("huggingface"))

    def test_domains(self):
        from benchmarks.loaders.tau2 import Tau2Loader

        loader = Tau2Loader()
        domains = loader.get_domains()
        self.assertIn("airline", domains)
        self.assertIn("retail", domains)

    def test_load_without_tau2_raises(self):
        """Loading without tau2 installed raises ImportError."""
        from benchmarks.loaders.tau2 import Tau2Loader

        loader = Tau2Loader()
        with patch.dict("sys.modules", {"tau2": None, "tau2.registry": None}):
            with self.assertRaises(ImportError):
                list(loader.load(domain="airline"))


if __name__ == "__main__":
    unittest.main()

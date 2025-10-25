"""
Benchmark task manager for configuration-driven evaluation.

This module implements the BenchmarkTaskManager which handles YAML config loading,
task discovery, and benchmark instantiation following lm-evaluation-harness patterns.
"""

from __future__ import annotations

import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Type

# Try to import tomllib (Python 3.11+) or tomli for TOML support
try:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        import tomli as tomllib
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

from .base import BenchmarkConfig, BenchmarkEnvironment, DataLoader
from .loaders.huggingface import HuggingFaceLoader


class BenchmarkTaskManager:
    """
    Manages benchmark tasks with automatic discovery and configuration loading.

    Features:
    - Automatic YAML config discovery in tasks/ directory
    - Pluggable data loader system
    - Lazy loading of benchmark instances
    - Environment variable configuration

    Usage:
        >>> manager = BenchmarkTaskManager()
        >>> benchmark = manager.get_benchmark("finer")
        >>> config = manager.get_config("finer")
        >>> available = manager.list_benchmarks()
    """

    def __init__(self, tasks_dir: Optional[Path] = None):
        """
        Initialize the benchmark task manager.

        Args:
            tasks_dir: Directory containing benchmark task configs.
                      Defaults to benchmarks/tasks/ relative to this module.
        """
        if tasks_dir is None:
            tasks_dir = Path(__file__).parent / "tasks"

        self.tasks_dir = Path(tasks_dir)
        self._configs: Dict[str, BenchmarkConfig] = {}
        self._benchmarks: Dict[str, BenchmarkEnvironment] = {}

        # Register available data loaders
        self._loaders: Dict[str, DataLoader] = {
            "huggingface": HuggingFaceLoader(),
        }

        # Try to import and register AppWorld loader if available
        try:
            from .loaders.appworld import AppWorldLoader
            self._loaders["appworld"] = AppWorldLoader()
        except ImportError:
            pass

        # Discover all available task configs
        self._discover_configs()

    def _discover_configs(self) -> None:
        """Scan tasks directory for YAML and TOML configuration files."""
        if not self.tasks_dir.exists():
            self.tasks_dir.mkdir(parents=True, exist_ok=True)
            return

        # Load YAML configs
        for yaml_file in self.tasks_dir.rglob("*.yaml"):
            try:
                config_dict = yaml.safe_load(yaml_file.read_text())
                config = BenchmarkConfig.from_dict(config_dict)
                self._configs[config.task] = config
            except Exception as e:
                print(f"Warning: Failed to load YAML config from {yaml_file}: {e}")

        # Load TOML configs if available
        if TOML_AVAILABLE:
            for toml_file in self.tasks_dir.rglob("*.toml"):
                try:
                    config_dict = tomllib.loads(toml_file.read_text())
                    config = BenchmarkConfig.from_dict(config_dict)
                    # TOML configs take precedence over YAML if same task name
                    self._configs[config.task] = config
                except Exception as e:
                    print(f"Warning: Failed to load TOML config from {toml_file}: {e}")

    def list_benchmarks(self) -> List[str]:
        """Return list of available benchmark task names."""
        return list(self._configs.keys())

    def get_config(self, task_name: str) -> BenchmarkConfig:
        """Get configuration for a specific benchmark task."""
        if task_name not in self._configs:
            raise ValueError(f"Unknown benchmark task: {task_name}")
        return self._configs[task_name]

    def get_benchmark(self, task_name: str) -> BenchmarkEnvironment:
        """
        Get benchmark environment instance for a task.

        Uses lazy loading - benchmark is instantiated only when first requested.
        """
        if task_name not in self._benchmarks:
            config = self.get_config(task_name)

            # Determine benchmark environment class based on task
            env_class = self._get_environment_class(config)
            self._benchmarks[task_name] = env_class(config)

        return self._benchmarks[task_name]

    def get_data_loader(self, source: str) -> DataLoader:
        """Get data loader for the specified source."""
        if source not in self._loaders:
            raise ValueError(f"Unknown data source: {source}")
        return self._loaders[source]

    def load_benchmark_data(self, task_name: str):
        """Load data for a specific benchmark task."""
        config = self.get_config(task_name)
        data_config = config.data.copy()  # Make a copy to avoid modifying original
        source = data_config["source"]

        # Extract limit and remove from loader args
        limit = data_config.pop("limit", None)

        # Add benchmark name for processor selection
        data_config["benchmark_name"] = task_name

        loader = self.get_data_loader(source)
        data_iter = loader.load(**data_config)

        # Apply limit if specified in config
        if limit:
            import itertools
            data_iter = itertools.islice(data_iter, limit)

        return data_iter

    def _get_environment_class(self, config: BenchmarkConfig) -> Type[BenchmarkEnvironment]:
        """
        Determine the appropriate environment class for a benchmark.

        This can be extended to support custom environment classes
        based on task configuration.
        """
        # Import here to avoid circular imports
        try:
            from .environments import (
                FiNEREnvironment,
                XBRLMathEnvironment,
                AppWorldEnvironment,
                GenericBenchmarkEnvironment
            )
        except ImportError:
            # Fallback to base class if environments module not available
            return BenchmarkEnvironment

        task_name = config.task.lower()

        if "finer" in task_name:
            return FiNEREnvironment
        elif "xbrl" in task_name or "math" in task_name:
            return XBRLMathEnvironment
        elif "appworld" in task_name:
            return AppWorldEnvironment
        else:
            return GenericBenchmarkEnvironment

    def register_loader(self, source: str, loader: DataLoader) -> None:
        """Register a custom data loader for a source."""
        self._loaders[source] = loader

    def reload_configs(self) -> None:
        """Reload all configuration files from tasks directory."""
        self._configs.clear()
        self._benchmarks.clear()
        self._discover_configs()

    def validate_config(self, task_name: str) -> List[str]:
        """
        Validate a benchmark configuration and return any issues found.

        Returns:
            List of validation error messages, empty if valid.
        """
        errors = []

        try:
            config = self.get_config(task_name)
        except ValueError as e:
            return [str(e)]

        # Validate data source
        source = config.data.get("source")
        if not source:
            errors.append("Missing 'source' in data configuration")
        elif source not in self._loaders:
            errors.append(f"Unknown data source: {source}")

        # Validate required fields
        required_fields = ["task", "version", "data", "preprocessing", "metrics"]
        for field in required_fields:
            if not hasattr(config, field) or getattr(config, field) is None:
                errors.append(f"Missing required field: {field}")

        # Validate metrics configuration
        if config.metrics:
            for metric in config.metrics:
                if "name" not in metric:
                    errors.append("Metric missing 'name' field")

        return errors
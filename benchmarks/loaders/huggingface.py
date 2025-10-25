"""
HuggingFace datasets loader with streaming support.

This module provides efficient data loading from HuggingFace Hub using streaming
to avoid downloading large datasets while supporting caching for repeated access.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Any

from ..base import DataLoader, get_cache_dir
from ..processors import get_processor


class HuggingFaceLoader(DataLoader):
    """
    Data loader for HuggingFace datasets with streaming support.

    Features:
    - Streaming mode for zero local storage during iteration
    - Automatic caching in user-configurable directory
    - Support for dataset subsets and custom splits
    - Memory-efficient loading for large datasets

    Example:
        >>> loader = HuggingFaceLoader()
        >>> data = loader.load(
        ...     dataset_path="gtfintechlab/finer-ord",
        ...     split="test",
        ...     streaming=True
        ... )
        >>> for sample in data:
        ...     print(sample["gold_token"], sample["gold_label"])
    """

    def __init__(self, default_cache_dir: Optional[str] = None):
        """
        Initialize HuggingFace loader.

        Args:
            default_cache_dir: Default cache directory.
                              Falls back to environment variables or HF default.
        """
        self.default_cache_dir = default_cache_dir

    def supports_source(self, source: str) -> bool:
        """Check if this loader supports the given data source."""
        return source == "huggingface"

    def load(self, **kwargs) -> Iterator[Dict[str, Any]]:
        """
        Load HuggingFace dataset with streaming support and dataset-specific processing.

        Args:
            dataset_path: HuggingFace dataset identifier (e.g., "gtfintechlab/finer-ord")
            split: Dataset split to load (default: "test")
            streaming: Whether to use streaming mode (default: True)
            cache_dir: Override default cache directory
            subset: Dataset subset/config name
            columns: List of columns to load (for efficiency)
            benchmark_name: Name of benchmark for processor selection
            **kwargs: Additional arguments passed to load_dataset

        Yields:
            Dictionary containing sample data from the dataset (potentially processed)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "datasets library is required for HuggingFace loader. "
                "Install with: pip install datasets"
            )

        # Extract parameters
        dataset_path = kwargs.get("dataset_path")
        if not dataset_path:
            raise ValueError("dataset_path is required for HuggingFace loader")

        split = kwargs.get("split", "test")
        streaming = kwargs.get("streaming", True)
        subset = kwargs.get("subset")
        columns = kwargs.get("columns")

        # Determine cache directory
        cache_dir = kwargs.get("cache_dir")
        if not cache_dir:
            cache_dir = self.default_cache_dir
        if not cache_dir:
            cache_dir = self._get_cache_dir()

        # Prepare load_dataset arguments
        load_args = {
            "path": dataset_path,
            "split": split,
            "streaming": streaming,
            "cache_dir": cache_dir,
        }

        # Add optional parameters
        if subset:
            load_args["name"] = subset
        if columns and streaming:
            # Column filtering works with streaming datasets
            load_args["columns"] = columns

        # Add any additional kwargs, filtering out our custom ones
        custom_keys = {
            "dataset_path", "split", "streaming", "cache_dir",
            "subset", "columns", "source", "benchmark_name"  # Add benchmark_name to filter
        }
        for key, value in kwargs.items():
            if key not in custom_keys:
                load_args[key] = value

        # Load dataset
        dataset = load_dataset(**load_args)

        # Check if we need dataset-specific processing
        benchmark_name = kwargs.get('benchmark_name')
        processor = get_processor(benchmark_name) if benchmark_name else None

        # Handle FiNER special case - needs token grouping
        if benchmark_name == 'finer_ord' and processor:
            # For FiNER, we need to collect all tokens and process them
            token_stream = dataset if streaming else iter(dataset)

            # Use processor to convert tokens to sentences
            for processed_sample in processor.process_token_stream(token_stream):
                # Convert Sample back to dict for compatibility
                yield {
                    'question': processed_sample.question,
                    'ground_truth': processed_sample.ground_truth,
                    'context': getattr(processed_sample, 'context', '')
                }
        else:
            # Standard processing for other datasets
            sample_iter = dataset if streaming else iter(dataset)
            for sample in sample_iter:
                yield sample

    @lru_cache(maxsize=1)
    def _get_cache_dir(self) -> str:
        """Get cache directory with fallback hierarchy."""
        # 1. Check for benchmark-specific cache
        cache_dir = os.getenv("BENCHMARK_CACHE_DIR")
        if cache_dir:
            return str(get_cache_dir("huggingface"))

        # 2. Check for HuggingFace datasets cache
        cache_dir = os.getenv("HF_DATASETS_CACHE")
        if cache_dir:
            return cache_dir

        # 3. Fall back to default HuggingFace location
        return os.path.expanduser("~/.cache/huggingface/datasets")

    def get_dataset_info(self, dataset_path: str, subset: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metadata about a HuggingFace dataset without downloading.

        Args:
            dataset_path: HuggingFace dataset identifier
            subset: Optional dataset subset/config name

        Returns:
            Dictionary with dataset information including features, splits, etc.
        """
        try:
            from datasets import load_dataset_builder
        except ImportError:
            raise ImportError(
                "datasets library is required for HuggingFace loader. "
                "Install with: pip install datasets"
            )

        builder_args = {"path": dataset_path}
        if subset:
            builder_args["name"] = subset

        builder = load_dataset_builder(**builder_args)

        return {
            "description": builder.info.description,
            "features": dict(builder.info.features),
            "splits": list(builder.info.splits.keys()) if builder.info.splits else [],
            "dataset_size": builder.info.dataset_size,
            "download_size": builder.info.download_size,
            "citation": builder.info.citation,
            "license": builder.info.license,
        }

    def validate_dataset(self, dataset_path: str, subset: Optional[str] = None) -> bool:
        """
        Validate that a HuggingFace dataset exists and is accessible.

        Args:
            dataset_path: HuggingFace dataset identifier
            subset: Optional dataset subset/config name

        Returns:
            True if dataset is valid and accessible, False otherwise
        """
        try:
            self.get_dataset_info(dataset_path, subset)
            return True
        except Exception:
            return False

    def list_dataset_configs(self, dataset_path: str) -> List[str]:
        """
        List available configurations/subsets for a dataset.

        Args:
            dataset_path: HuggingFace dataset identifier

        Returns:
            List of available configuration names
        """
        try:
            from datasets import get_dataset_config_names
            return get_dataset_config_names(dataset_path)
        except ImportError:
            raise ImportError(
                "datasets library is required for HuggingFace loader. "
                "Install with: pip install datasets"
            )
        except Exception:
            return []
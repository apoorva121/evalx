"""Tests for evalx_sdk.introspection module."""

import inspect

import pytest

from evalx_sdk.introspection import (
    extract_dataset_metadata,
    extract_experiment_metadata,
    extract_function_metadata,
)


def sample_function(x, y):
    """Sample function for testing."""
    return x + y


async def sample_async_function(x):
    """Sample async function for testing."""
    return x * 2


class TestExtractFunctionMetadata:
    """Test extract_function_metadata function."""

    def test_regular_function(self):
        """Test extracting metadata from regular function."""
        metadata = extract_function_metadata(sample_function)

        assert metadata["name"] == "sample_function"
        assert "test_introspection" in metadata["module"]
        assert metadata["is_lambda"] is False
        assert metadata["is_async"] is False
        assert metadata["signature"] == "(x, y)"
        assert metadata["docstring"] == "Sample function for testing."
        assert metadata["file_path"] is not None
        assert metadata["line_number"] is not None

    def test_async_function(self):
        """Test extracting metadata from async function."""
        metadata = extract_function_metadata(sample_async_function)

        assert metadata["name"] == "sample_async_function"
        assert metadata["is_async"] is True
        assert metadata["is_lambda"] is False

    def test_lambda_function(self):
        """Test extracting metadata from lambda function."""
        lambda_func = lambda x: x * 2
        metadata = extract_function_metadata(lambda_func)

        assert metadata["name"] == "<lambda>"
        assert metadata["is_lambda"] is True
        assert metadata["is_async"] is False

    def test_builtin_function(self):
        """Test extracting metadata from builtin function."""
        metadata = extract_function_metadata(len)

        assert metadata["name"] == "len"
        assert metadata["file_path"] is None  # Builtins don't have file paths
        assert metadata["line_number"] is None

    def test_function_without_docstring(self):
        """Test function without docstring."""

        def no_doc():
            pass

        metadata = extract_function_metadata(no_doc)

        assert metadata["docstring"] is None


class TestExtractDatasetMetadata:
    """Test extract_dataset_metadata function."""

    def test_list_of_dicts(self):
        """Test extracting metadata from list of dicts."""
        data = [
            {"input": "test1", "expected": "result1"},
            {"input": "test2", "expected": "result2"},
        ]

        metadata = extract_dataset_metadata(data)

        assert metadata["type"] == "list"
        assert metadata["size"] == 2
        assert metadata["sample_keys"] == ["input", "expected"]

    def test_empty_list(self):
        """Test extracting metadata from empty list."""
        data = []

        metadata = extract_dataset_metadata(data)

        assert metadata["type"] == "list"
        assert metadata["size"] == 0
        assert metadata["sample_keys"] is None

    def test_list_of_strings(self):
        """Test extracting metadata from list of strings."""
        data = ["item1", "item2", "item3"]

        metadata = extract_dataset_metadata(data)

        assert metadata["type"] == "list"
        assert metadata["size"] == 3
        assert metadata["sample_keys"] is None

    def test_dict(self):
        """Test extracting metadata from dict."""
        data = {"key1": "value1", "key2": "value2"}

        metadata = extract_dataset_metadata(data)

        assert metadata["type"] == "dict"
        assert metadata["size"] == 2
        assert set(metadata["sample_keys"]) == {"key1", "key2"}

    def test_large_dict(self):
        """Test extracting metadata from large dict (only first 10 keys)."""
        data = {f"key{i}": f"value{i}" for i in range(20)}

        metadata = extract_dataset_metadata(data)

        assert metadata["type"] == "dict"
        assert metadata["size"] == 20
        assert len(metadata["sample_keys"]) == 10

    def test_non_sized_object(self):
        """Test extracting metadata from object without len()."""

        class CustomObject:
            pass

        data = CustomObject()

        metadata = extract_dataset_metadata(data)

        assert metadata["type"] == "CustomObject"
        assert metadata["size"] is None


class TestExtractExperimentMetadata:
    """Test extract_experiment_metadata function."""

    def test_all_parameters(self):
        """Test extracting metadata with all parameters."""
        data = [{"input": "test"}]
        task = sample_function
        evaluators = [lambda x: x > 0]
        run_evaluators = [lambda results: sum(results)]

        metadata = extract_experiment_metadata(
            name="Test Experiment",
            data=data,
            task=task,
            evaluators=evaluators,
            run_evaluators=run_evaluators,
        )

        assert metadata["experiment_name"] == "Test Experiment"
        assert "task" in metadata
        assert metadata["task"]["name"] == "sample_function"
        assert "dataset" in metadata
        assert metadata["dataset"]["size"] == 1
        assert "evaluators" in metadata
        assert len(metadata["evaluators"]) == 1
        assert "run_evaluators" in metadata
        assert len(metadata["run_evaluators"]) == 1

    def test_only_name(self):
        """Test extracting metadata with only name."""
        metadata = extract_experiment_metadata(name="Test")

        assert metadata["experiment_name"] == "Test"
        assert "task" not in metadata
        assert "dataset" not in metadata

    def test_no_parameters(self):
        """Test extracting metadata with no parameters."""
        metadata = extract_experiment_metadata()

        assert metadata == {"additional_params": {}}

    def test_additional_kwargs(self):
        """Test extracting additional kwargs."""
        metadata = extract_experiment_metadata(
            name="Test", max_concurrency=5, custom_param="value"
        )

        assert metadata["experiment_name"] == "Test"
        assert metadata["additional_params"]["max_concurrency"] == 5
        assert metadata["additional_params"]["custom_param"] == "value"

    def test_non_callable_evaluators(self):
        """Test with non-callable evaluators."""

        class CustomEvaluator:
            pass

        evaluator = CustomEvaluator()
        metadata = extract_experiment_metadata(evaluators=[evaluator])

        assert len(metadata["evaluators"]) == 1
        assert metadata["evaluators"][0]["type"] == "CustomEvaluator"

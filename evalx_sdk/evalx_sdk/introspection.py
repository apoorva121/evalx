"""Introspection utilities for extracting metadata from functions and data."""

import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def extract_function_metadata(func: Callable) -> Dict[str, Any]:
    """
    Extract metadata from a function.

    Args:
        func: The function to introspect

    Returns:
        Dictionary containing function metadata:
        - name: Function name
        - module: Module name
        - file_path: Absolute path to file containing the function
        - line_number: Line number where function is defined
        - signature: Function signature as string
        - docstring: Function docstring (first line)
        - is_async: Whether function is async
        - is_lambda: Whether function is a lambda
    """
    metadata = {
        "name": func.__name__,
        "module": func.__module__ if hasattr(func, "__module__") else None,
        "is_lambda": func.__name__ == "<lambda>",
        "is_async": inspect.iscoroutinefunction(func),
    }

    # Extract file path
    try:
        file_path = inspect.getfile(func)
        metadata["file_path"] = str(Path(file_path).resolve())
    except (TypeError, OSError):
        metadata["file_path"] = None

    # Extract line number
    try:
        _, line_number = inspect.getsourcelines(func)
        metadata["line_number"] = line_number
    except (TypeError, OSError):
        metadata["line_number"] = None

    # Extract signature
    try:
        sig = inspect.signature(func)
        metadata["signature"] = str(sig)
    except (ValueError, TypeError):
        metadata["signature"] = None

    # Extract docstring (first line only)
    if func.__doc__:
        docstring = func.__doc__.strip().split("\n")[0]
        metadata["docstring"] = docstring
    else:
        metadata["docstring"] = None

    return metadata


def extract_dataset_metadata(data: Any) -> Dict[str, Any]:
    """
    Extract metadata from dataset.

    Args:
        data: The dataset (list, dict, or other iterable)

    Returns:
        Dictionary containing dataset metadata:
        - type: Type of data structure
        - size: Number of items (if applicable)
        - sample_keys: Sample keys for dict-like structures
    """
    metadata = {
        "type": type(data).__name__,
    }

    # Extract size
    try:
        metadata["size"] = len(data)
    except TypeError:
        metadata["size"] = None

    # Extract sample keys for dict-like structures
    if isinstance(data, list) and len(data) > 0:
        first_item = data[0]
        if isinstance(first_item, dict):
            metadata["sample_keys"] = list(first_item.keys())
        else:
            metadata["sample_keys"] = None
    elif isinstance(data, dict):
        metadata["sample_keys"] = list(data.keys())[:10]  # First 10 keys
    else:
        metadata["sample_keys"] = None

    return metadata


def extract_experiment_metadata(
    name: Optional[str] = None,
    data: Optional[Any] = None,
    task: Optional[Callable] = None,
    evaluators: Optional[List[Callable]] = None,
    run_evaluators: Optional[List[Callable]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Extract comprehensive metadata from experiment parameters.

    This function can be called before run_experiment to gather metadata.

    Args:
        name: Experiment name
        data: Dataset for experiment
        task: Task function
        evaluators: List of item-level evaluator functions
        run_evaluators: List of aggregate run evaluator functions
        **kwargs: Additional experiment parameters

    Returns:
        Dictionary containing all extracted metadata
    """
    metadata = {}

    # Basic info
    if name:
        metadata["experiment_name"] = name

    # Extract task metadata
    if task and callable(task):
        metadata["task"] = extract_function_metadata(task)

    # Extract dataset metadata
    if data is not None:
        metadata["dataset"] = extract_dataset_metadata(data)

    # Extract evaluator metadata if provided
    if evaluators:
        if isinstance(evaluators, list):
            metadata["evaluators"] = [
                extract_function_metadata(evaluator)
                if callable(evaluator)
                else {"type": type(evaluator).__name__}
                for evaluator in evaluators
            ]

    # Extract run_evaluator metadata if provided
    if run_evaluators:
        if isinstance(run_evaluators, list):
            metadata["run_evaluators"] = [
                extract_function_metadata(evaluator)
                if callable(evaluator)
                else {"type": type(evaluator).__name__}
                for evaluator in run_evaluators
            ]

    # Add any additional kwargs
    metadata["additional_params"] = {
        k: v
        for k, v in kwargs.items()
        if k not in ["evaluators", "run_evaluators"]
    }

    return metadata

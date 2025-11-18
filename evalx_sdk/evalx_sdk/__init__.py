"""EvalX SDK - Context management for evaluation tracking."""

from evalx_sdk.context import EvalContext, get_eval_context, set_eval_context, init_context
from evalx_sdk.decorators import track_run
from evalx_sdk.run import EvalRun
from evalx_sdk.client import EvalXRunClient
from evalx_sdk.introspection import (
    extract_function_metadata,
    extract_dataset_metadata,
    extract_experiment_metadata,
)

__version__ = "0.1.0"

__all__ = [
    "EvalContext",
    "get_eval_context",
    "set_eval_context",
    "init_context",
    "track_run",
    "EvalRun",
    "EvalXRunClient",
    "extract_function_metadata",
    "extract_dataset_metadata",
    "extract_experiment_metadata",
]

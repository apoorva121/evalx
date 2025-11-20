"""Context manager for tracking evaluation runs."""

import inspect
import os
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Literal, Optional

from evalx_sdk.client import EvalXRunClient
from evalx_sdk.context import get_eval_context
from evalx_sdk.introspection import extract_experiment_metadata


class EvalRun:
    """
    Context manager to track evaluation runs with EvalX service.

    Usage:
        >>> from evalx_sdk import EvalRun
        >>>
        >>> # Environment, mode, service URL, and token are auto-detected from env vars
        >>> with EvalRun():
        >>>     result = langfuse.run_experiment(...)

    The context manager automatically:
    1. Loads service_url and bearer_token from environment variables
    2. Detects env from EVALX_ENV (default: "local")
    3. Detects mode from EVALX_MODE or CI environment (default: "local", or "ci" if CI=true)
    4. Starts a run in EvalX service on entry
    5. Finishes the run with success status on normal exit
    6. Finishes the run with fail status on exception
    """

    def __init__(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize run tracker.

        All configuration is loaded from environment variables.
        Metadata is automatically extracted from run_experiment call parameters.

        Args:
            metadata: Optional custom metadata to attach to the run

        Environment Variables:
            EVALX_SERVICE_URL: Service URL (default: http://localhost:8000)
            EVALX_BEARER_TOKEN: Bearer token for authentication
            EVALX_ENV: Environment (local, e2e, prod). Default: "local"
            EVALX_MODE: Mode (local, ci, prod). Auto-detects CI environment if not set
            CI: Detected to auto-set mode to "ci" if set to "true"

        Example:
            >>> # Simple usage - metadata extracted automatically
            >>> with EvalRun():
            >>>     result = langfuse.run_experiment(
            >>>         name="My Test",
            >>>         data=my_data,
            >>>         task=my_task,
            >>>     )
        """
        # Load service configuration from environment
        self.service_url = os.getenv("EVALX_SERVICE_URL", "http://localhost:8000")
        self.bearer_token = os.getenv("EVALX_BEARER_TOKEN")

        # Auto-detect env and mode from environment
        self.env = os.getenv("EVALX_ENV", "local")

        # Auto-detect mode: check if running in CI
        default_mode = "ci" if os.getenv("CI", "").lower() == "true" else "local"
        self.mode = os.getenv("EVALX_MODE", default_mode)

        # Start with provided metadata
        self.metadata = metadata or {}
        self._original_run_experiment = None
        self._original_dataset_run_experiment = None
        self._original_trace_dataset_run_evaluation = None

        self.run_id: Optional[str] = None
        self.client: Optional[EvalXRunClient] = None

    def _create_wrapper(self, original_func):
        """Create a wrapper that extracts metadata from run_experiment parameters."""

        @wraps(original_func)
        def wrapper(*args, **kwargs):
            # Extract parameters from kwargs for metadata
            name = kwargs.get("name")
            data = kwargs.get("data")
            task = kwargs.get("task")
            evaluators = kwargs.get("evaluators")
            run_evaluators = kwargs.get("run_evaluators")

            # Extract metadata from these parameters
            if name or data or task or evaluators or run_evaluators:
                extracted = extract_experiment_metadata(
                    name=name,
                    data=data,
                    task=task,
                    evaluators=evaluators,
                    run_evaluators=run_evaluators,
                )
                # Update metadata (only once, for the first call)
                if not self.metadata.get("experiment_captured"):
                    self.metadata.update(extracted)
                    self.metadata["experiment_captured"] = True

            # Call original function
            return original_func(*args, **kwargs)

        return wrapper

    def __enter__(self):
        """Start the run when entering the context."""
        # Get eval context
        context = get_eval_context()
        if not context:
            print(
                "⚠️  Warning: No EvalX context found. Run tracking disabled. "
                "Run 'evalx init' first."
            )
            return self

        # Monkey-patch langfuse.run_experiment and TraceDataset.run_experiment to intercept calls
        try:
            import langfuse

            # Patch langfuse client's run_experiment
            if hasattr(langfuse, "Langfuse"):
                self._original_run_experiment = langfuse.Langfuse.run_experiment
                langfuse.Langfuse.run_experiment = self._create_wrapper(
                    self._original_run_experiment
                )

            # Patch dataset's run_experiment
            if hasattr(langfuse, "LangfuseDataset"):
                self._original_dataset_run_experiment = (
                    langfuse.LangfuseDataset.run_experiment
                )
                langfuse.LangfuseDataset.run_experiment = self._create_wrapper(
                    self._original_dataset_run_experiment
                )
        except ImportError:
            pass  # Langfuse not installed, skip patching

        # Patch TraceDataset's run_evaluation
        try:
            from evalx_sdk.trace_dataset import TraceDataset

            self._original_trace_dataset_run_evaluation = TraceDataset.run_evaluation
            TraceDataset.run_evaluation = self._create_wrapper(
                self._original_trace_dataset_run_evaluation
            )
        except ImportError:
            pass  # TraceDataset not available

        # Initialize client
        self.client = EvalXRunClient(
            base_url=self.service_url, bearer_token=self.bearer_token
        )

        # Start run
        try:
            run_start = self.client.start_run(
                eval_id=context.eval_id,
                asset_id=context.asset_id,
                env=self.env,
                mode=self.mode,
                metadata={
                    **self.metadata,
                    "started_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            self.run_id = run_start.run_id
            print(
                f"🚀 Run started: run_id={run_start.run_id}, eval_id={context.eval_id}"
            )
        except Exception as e:
            print(f"⚠️  Warning: Failed to start run: {e}")
            print("   Continuing without run tracking...")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish the run when exiting the context."""
        # Restore original run_experiment methods
        try:
            import langfuse

            if self._original_run_experiment:
                langfuse.Langfuse.run_experiment = self._original_run_experiment

            if self._original_dataset_run_experiment:
                langfuse.LangfuseDataset.run_experiment = (
                    self._original_dataset_run_experiment
                )
        except ImportError:
            pass

        # Restore TraceDataset's run_evaluation
        try:
            from evalx_sdk.trace_dataset import TraceDataset

            if self._original_trace_dataset_run_evaluation:
                TraceDataset.run_evaluation = self._original_trace_dataset_run_evaluation
        except ImportError:
            pass

        if not self.client or not self.run_id:
            return False  # Don't suppress exceptions

        # Remove internal tracking flag
        self.metadata.pop("experiment_captured", None)

        # Determine status based on whether exception occurred
        status = "fail" if exc_type else "success"

        # Finish run
        try:
            run_finish = self.client.finish_run(
                run_id=self.run_id,
                status=status,
                metadata={
                    **self.metadata,
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                    "error": str(exc_val) if exc_val else None,
                },
            )
            print(
                f"✅ Run finished: run_id={run_finish.run_id}, status={run_finish.status}"
            )
        except Exception as e:
            print(f"⚠️  Warning: Failed to finish run: {e}")

        return False  # Don't suppress exceptions



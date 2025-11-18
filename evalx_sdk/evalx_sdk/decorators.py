"""Decorators for tracking evaluation runs."""

import functools
import os
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional

from evalx_sdk.client import EvalXRunClient
from evalx_sdk.context import get_eval_context


def track_run(
    env: Literal["local", "e2e", "prod"] = "local",
    mode: Literal["local", "ci", "prod"] = "local",
    service_url: Optional[str] = None,
    bearer_token: Optional[str] = None,
):
    """
    Decorator to track evaluation runs with EvalX service.

    Automatically starts a run before the decorated function executes,
    and finishes it with success/fail status after completion.

    Args:
        env: Environment (local, e2e, prod). Default: "local"
        mode: Mode (local, ci, prod). Default: "local"
        service_url: Optional EvalX service URL (defaults to env var or localhost)
        bearer_token: Optional bearer token (defaults to env var)

    Usage:
        >>> from evalx_sdk import track_run
        >>>
        >>> @track_run(env="local", mode="local")
        >>> def run_experiment():
        >>>     result = langfuse.run_experiment(...)
        >>>     return result

    Environment Variables:
        EVALX_SERVICE_URL: Service URL (default: http://localhost:8000)
        EVALX_BEARER_TOKEN: Bearer token for authentication

    Returns:
        Decorated function that tracks runs
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get eval context
            context = get_eval_context()
            if not context:
                print(
                    "⚠️  Warning: No EvalX context found. Run tracking disabled. "
                    "Run 'evalx init' first."
                )
                return func(*args, **kwargs)

            # Get service configuration
            url = service_url or os.getenv("EVALX_SERVICE_URL", "http://localhost:8000")
            token = bearer_token or os.getenv("EVALX_BEARER_TOKEN")

            # Initialize client
            client = EvalXRunClient(base_url=url, bearer_token=token)

            # Start run
            try:
                run_start = client.start_run(
                    eval_id=context.eval_id,
                    asset_id=context.asset_id,
                    env=env,
                    mode=mode,
                    metadata={
                        "function": func.__name__,
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(
                    f"🚀 Run started: run_id={run_start.run_id}, eval_id={context.eval_id}"
                )
                run_id = run_start.run_id
            except Exception as e:
                print(f"⚠️  Warning: Failed to start run: {e}")
                print("   Continuing without run tracking...")
                return func(*args, **kwargs)

            # Execute function and track result
            error = None
            result = None
            try:
                result = func(*args, **kwargs)
                status = "success"
            except Exception as e:
                error = e
                status = "fail"

            # Finish run
            try:
                run_finish = client.finish_run(
                    run_id=run_id,
                    status=status,
                    metadata={
                        "function": func.__name__,
                        "ended_at": datetime.now(timezone.utc).isoformat(),
                        "error": str(error) if error else None,
                    },
                )
                print(
                    f"✅ Run finished: run_id={run_finish.run_id}, status={run_finish.status}"
                )
            except Exception as e:
                print(f"⚠️  Warning: Failed to finish run: {e}")

            # Re-raise error if function failed
            if error:
                raise error

            return result

        return wrapper

    return decorator

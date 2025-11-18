"""Context management for EvalX evaluations."""

import json
import os
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class EvalContext(BaseModel):
    """Evaluation context containing eval_id and metadata."""

    eval_id: str = Field(..., description="UUID of the evaluation")
    asset_id: int = Field(..., description="Asset ID associated with the evaluation")
    name: str = Field(..., description="Name of the evaluation")
    description: Optional[str] = Field(None, description="Description of the evaluation")

    class Config:
        frozen = True


# Global context variable
_eval_context: Optional[EvalContext] = None


def set_eval_context(context: EvalContext) -> None:
    """
    Set the global evaluation context.

    Args:
        context: EvalContext instance to set as global context
    """
    global _eval_context
    _eval_context = context


def get_eval_context() -> Optional[EvalContext]:
    """
    Get the current evaluation context.

    Returns:
        Current EvalContext if set, None otherwise
    """
    return _eval_context


def load_eval_config(config_path: Optional[Path] = None) -> Optional[EvalContext]:
    """
    Load evaluation context from a config file.

    Args:
        config_path: Path to the config file. If None, looks for .evalx/config.json
                    in current directory or parent directories.

    Returns:
        EvalContext if config file exists and is valid, None otherwise
    """
    if config_path is None:
        # Search for .evalx/config.json starting from current directory
        current_dir = Path.cwd()
        while current_dir != current_dir.parent:
            config_file = current_dir / ".evalx" / "config.json"
            if config_file.exists():
                config_path = config_file
                break
            current_dir = current_dir.parent
        else:
            # Check in current directory as last resort
            config_file = Path.cwd() / ".evalx" / "config.json"
            if config_file.exists():
                config_path = config_file

    if config_path is None or not Path(config_path).exists():
        return None

    try:
        with open(config_path, "r") as f:
            data = json.load(f)
        return EvalContext(**data)
    except (json.JSONDecodeError, ValueError, IOError):
        return None


def save_eval_config(context: EvalContext, config_path: Optional[Path] = None) -> Path:
    """
    Save evaluation context to a config file.

    Args:
        context: EvalContext to save
        config_path: Path to save the config. If None, saves to .evalx/config.json

    Returns:
        Path where the config was saved
    """
    if config_path is None:
        config_dir = Path.cwd() / ".evalx"
        config_path = config_dir / "config.json"
    else:
        config_path = Path(config_path)
        config_dir = config_path.parent

    # Create directory if it doesn't exist
    config_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(config_path, "w") as f:
        json.dump(context.model_dump(), f, indent=2)

    return config_path


def init_context(verbose: bool = True) -> Optional[EvalContext]:
    """
    Initialize EvalX context by loading config and setting global context.

    This is a convenience function that:
    1. Loads the eval config from .evalx/config.json
    2. Sets it as the global context
    3. Optionally prints status messages

    Args:
        verbose: If True, prints status messages

    Returns:
        EvalContext if loaded successfully, None otherwise

    Example:
        >>> from evalx_sdk import init_context
        >>> context = init_context()
        ✅ EvalX context loaded: eval_id=..., asset_id=...
    """
    context = load_eval_config()

    if context:
        set_eval_context(context)
        if verbose:
            print(
                f"✅ EvalX context loaded: eval_id={context.eval_id}, asset_id={context.asset_id}"
            )
    else:
        if verbose:
            print("ℹ️  No EvalX config found. Run 'evalx init' to initialize.")

    return context

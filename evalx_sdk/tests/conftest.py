"""Pytest configuration and shared fixtures."""

import json
import os
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def eval_config_data() -> dict:
    """Sample evaluation configuration data."""
    return {
        "eval_id": "550e8400-e29b-41d4-a716-446655440000",
        "asset_id": 123,
        "name": "Test Evaluation",
        "description": "Test description",
    }


@pytest.fixture
def eval_config_file(temp_dir: Path, eval_config_data: dict) -> Path:
    """Create a temporary eval config file."""
    config_dir = temp_dir / ".evalx"
    config_dir.mkdir()
    config_path = config_dir / "config.json"

    with open(config_path, "w") as f:
        json.dump(eval_config_data, f)

    return config_path


@pytest.fixture
def mock_env_vars() -> Generator[dict, None, None]:
    """Mock environment variables for testing."""
    original_env = os.environ.copy()

    # Set test environment variables
    test_env = {
        "EVALX_SERVICE_URL": "http://test.example.com",
        "EVALX_BEARER_TOKEN": "test_token",
        "EVALX_ENV": "test",
        "EVALX_MODE": "test",
    }

    for key, value in test_env.items():
        os.environ[key] = value

    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def reset_global_context():
    """Reset global context before each test."""
    from evalx_sdk.context import _eval_context

    # Store original value
    original = _eval_context

    yield

    # Reset after test
    import evalx_sdk.context

    evalx_sdk.context._eval_context = None

"""Pytest configuration and shared fixtures for evalx_cli tests."""

import os
import sys
from pathlib import Path
from typing import Dict

import pytest

# Add evalx_sdk to Python path for testing
evalx_root = Path(__file__).parent.parent.parent
evalx_sdk_path = evalx_root / "evalx_sdk"
if str(evalx_sdk_path) not in sys.path:
    sys.path.insert(0, str(evalx_sdk_path))


@pytest.fixture
def eval_response_data() -> Dict:
    """Sample evaluation response data."""
    return {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "asset_id": 123,
        "name": "Test Evaluation",
        "description": "Test description",
        "created_at": "2024-01-01T00:00:00Z",
        "updated_at": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def clean_env(monkeypatch):
    """Clean environment variables before each test."""
    monkeypatch.delenv("EVALX_SERVICE_URL", raising=False)
    monkeypatch.delenv("EVALX_BEARER_TOKEN", raising=False)
    monkeypatch.delenv("EVALX_ENV", raising=False)
    monkeypatch.delenv("EVALX_MODE", raising=False)
    monkeypatch.delenv("CI", raising=False)


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for tests."""
    return tmp_path


@pytest.fixture
def mock_config_dir(tmp_path):
    """Create a temporary .evalx config directory."""
    config_dir = tmp_path / ".evalx"
    config_dir.mkdir()
    return config_dir

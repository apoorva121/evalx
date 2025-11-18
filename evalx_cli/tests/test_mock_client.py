"""Tests for evalx_cli.mock_client module."""

import uuid
from datetime import datetime, timezone

import pytest

from evalx_cli.client import EvalResponse
from evalx_cli.mock_client import MockEvalXClient


class TestMockEvalXClient:
    """Test MockEvalXClient for testing without service."""

    def test_init_with_defaults(self):
        """Test mock client initialization."""
        client = MockEvalXClient(base_url="http://test.com")

        assert client.base_url == "http://test.com"
        assert client.bearer_token is None

    def test_init_with_bearer_token(self):
        """Test mock client initialization with token."""
        client = MockEvalXClient(base_url="http://test.com", bearer_token="token123")

        assert client.base_url == "http://test.com"
        assert client.bearer_token == "token123"

    def test_create_or_get_eval_generates_uuid(self):
        """Test that mock client generates valid UUID."""
        client = MockEvalXClient(base_url="http://test.com")

        response = client.create_or_get_eval(asset_id=123, name="Test Eval")

        # Check that id is a valid UUID
        try:
            uuid.UUID(response.id)
        except ValueError:
            pytest.fail("Generated ID is not a valid UUID")

    def test_create_or_get_eval_with_required_fields(self):
        """Test mock eval creation with required fields."""
        client = MockEvalXClient(base_url="http://test.com")

        response = client.create_or_get_eval(asset_id=123, name="Test Eval")

        assert isinstance(response, EvalResponse)
        assert response.asset_id == 123
        assert response.name == "Test Eval"
        assert response.description is None
        assert response.created_at is not None
        assert response.updated_at is not None

    def test_create_or_get_eval_with_description(self):
        """Test mock eval creation with description."""
        client = MockEvalXClient(base_url="http://test.com")

        response = client.create_or_get_eval(
            asset_id=456, name="Test Eval", description="Test description"
        )

        assert response.asset_id == 456
        assert response.name == "Test Eval"
        assert response.description == "Test description"

    def test_create_or_get_eval_timestamps_format(self):
        """Test that timestamps are in ISO format."""
        client = MockEvalXClient(base_url="http://test.com")

        response = client.create_or_get_eval(asset_id=123, name="Test Eval")

        # Verify timestamps can be parsed
        try:
            datetime.fromisoformat(response.created_at.replace("Z", "+00:00"))
            datetime.fromisoformat(response.updated_at.replace("Z", "+00:00"))
        except ValueError:
            pytest.fail("Timestamps are not in valid ISO format")

    def test_create_or_get_eval_generates_unique_ids(self):
        """Test that multiple calls generate unique IDs."""
        client = MockEvalXClient(base_url="http://test.com")

        response1 = client.create_or_get_eval(asset_id=123, name="Test Eval 1")
        response2 = client.create_or_get_eval(asset_id=456, name="Test Eval 2")

        assert response1.id != response2.id

    def test_health_check_always_returns_true(self):
        """Test that mock health check always returns True."""
        client = MockEvalXClient(base_url="http://nonexistent.invalid")

        result = client.health_check()

        assert result is True

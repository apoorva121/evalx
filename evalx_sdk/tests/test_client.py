"""Tests for evalx_sdk.client module."""

from datetime import datetime, timezone

import pytest
import responses

from evalx_sdk.client import EvalXRunClient, RunFinishResponse, RunStartResponse


class TestEvalXRunClient:
    """Test EvalXRunClient class."""

    def test_init_with_defaults(self):
        """Test client initialization with defaults."""
        client = EvalXRunClient()

        assert client.base_url == "http://localhost:8000"
        assert client.bearer_token is None

    def test_init_with_custom_values(self):
        """Test client initialization with custom values."""
        client = EvalXRunClient(
            base_url="https://api.example.com", bearer_token="test_token"
        )

        assert client.base_url == "https://api.example.com"
        assert client.bearer_token == "test_token"

    @responses.activate
    def test_start_run_success(self):
        """Test successful run start."""
        client = EvalXRunClient(
            base_url="http://test.com", bearer_token="test_token"
        )

        # Mock API response
        responses.post(
            "http://test.com/v1/runs",
            json={
                "run_id": "run_123",
                "eval_id": "eval_456",
                "asset_id": 789,
                "env": "local",
                "mode": "local",
                "status": "running",
                "started_at": "2024-01-01T00:00:00Z",
            },
            status=200,
        )

        response = client.start_run(
            eval_id="eval_456",
            asset_id=789,
            env="local",
            mode="local",
            metadata={"test": "value"},
        )

        assert isinstance(response, RunStartResponse)
        assert response.run_id == "run_123"
        assert response.status == "running"
        assert response.started_at is not None

        # Check request
        assert len(responses.calls) == 1
        assert responses.calls[0].request.headers["Authorization"] == "Bearer test_token"

    @responses.activate
    def test_start_run_without_metadata(self):
        """Test starting run without metadata."""
        client = EvalXRunClient(base_url="http://test.com")

        responses.post(
            "http://test.com/v1/runs",
            json={
                "run_id": "run_123",
                "eval_id": "eval_456",
                "asset_id": 789,
                "env": "local",
                "mode": "local",
                "status": "running",
                "started_at": "2024-01-01T00:00:00Z",
            },
            status=200,
        )

        response = client.start_run(
            eval_id="eval_456", asset_id=789, env="local", mode="local"
        )

        assert response.run_id == "run_123"

    @responses.activate
    def test_start_run_http_error(self):
        """Test start_run with HTTP error."""
        client = EvalXRunClient(base_url="http://test.com")

        responses.post(
            "http://test.com/v1/runs",
            json={"error": "Bad request"},
            status=400,
        )

        with pytest.raises(Exception) as exc_info:
            client.start_run(
                eval_id="eval_456", asset_id=789, env="local", mode="local"
            )

        assert "400" in str(exc_info.value)

    @responses.activate
    def test_finish_run_success(self):
        """Test successful run finish."""
        client = EvalXRunClient(
            base_url="http://test.com", bearer_token="test_token"
        )

        responses.put(
            "http://test.com/v1/runs/run_123",
            json={
                "run_id": "run_123",
                "eval_id": "eval_456",
                "asset_id": 789,
                "env": "local",
                "mode": "local",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:01:00Z",
                "metadata": {},
            },
            status=200,
        )

        response = client.finish_run(
            run_id="run_123",
            status="success",
            metadata={"result": "completed"},
        )

        assert isinstance(response, RunFinishResponse)
        assert response.run_id == "run_123"
        assert response.status == "success"
        assert response.ended_at is not None

    @responses.activate
    def test_finish_run_with_datetime(self):
        """Test finishing run with custom ended_at datetime."""
        client = EvalXRunClient(base_url="http://test.com")

        responses.put(
            "http://test.com/v1/runs/run_123",
            json={
                "run_id": "run_123",
                "eval_id": "eval_456",
                "asset_id": 789,
                "env": "local",
                "mode": "local",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:02:00Z",
                "metadata": {},
            },
            status=200,
        )

        ended_at = datetime(2024, 1, 1, 0, 2, 0, tzinfo=timezone.utc)
        response = client.finish_run(
            run_id="run_123", status="success", ended_at=ended_at
        )

        assert response.status == "success"
        assert response.ended_at is not None

    @responses.activate
    def test_finish_run_fail_status(self):
        """Test finishing run with fail status."""
        client = EvalXRunClient(base_url="http://test.com")

        responses.put(
            "http://test.com/v1/runs/run_123",
            json={
                "run_id": "run_123",
                "eval_id": "eval_456",
                "asset_id": 789,
                "env": "local",
                "mode": "local",
                "status": "fail",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:01:00Z",
                "metadata": {},
            },
            status=200,
        )

        response = client.finish_run(run_id="run_123", status="fail")

        assert response.status == "fail"

    @responses.activate
    def test_finish_run_http_error(self):
        """Test finish_run with HTTP error."""
        client = EvalXRunClient(base_url="http://test.com")

        responses.put(
            "http://test.com/v1/runs/run_123",
            json={"error": "Not found"},
            status=404,
        )

        with pytest.raises(Exception) as exc_info:
            client.finish_run(run_id="run_123", status="success")

        assert "404" in str(exc_info.value)

    @responses.activate
    def test_client_without_bearer_token(self):
        """Test client without bearer token (no auth header)."""
        client = EvalXRunClient(base_url="http://test.com", bearer_token=None)

        responses.post(
            "http://test.com/v1/runs",
            json={
                "run_id": "run_123",
                "eval_id": "eval_456",
                "asset_id": 789,
                "env": "local",
                "mode": "local",
                "status": "running",
                "started_at": "2024-01-01T00:00:00Z",
            },
            status=200,
        )

        client.start_run(eval_id="eval_456", asset_id=789, env="local", mode="local")

        # Check that Authorization header is not present
        assert "Authorization" not in responses.calls[0].request.headers

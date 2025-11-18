"""Tests for evalx_sdk.run module."""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest
import responses

from evalx_sdk.context import EvalContext, set_eval_context
from evalx_sdk.run import EvalRun


class TestEvalRun:
    """Test EvalRun context manager."""

    def test_init_with_defaults(self, monkeypatch):
        """Test initialization with default values."""
        monkeypatch.setenv("EVALX_SERVICE_URL", "http://test.com")
        monkeypatch.setenv("EVALX_BEARER_TOKEN", "token123")
        monkeypatch.setenv("EVALX_ENV", "prod")
        monkeypatch.setenv("EVALX_MODE", "ci")

        run = EvalRun()

        assert run.service_url == "http://test.com"
        assert run.bearer_token == "token123"
        assert run.env == "prod"
        assert run.mode == "ci"

    def test_init_with_ci_auto_detection(self, monkeypatch):
        """Test that CI environment is auto-detected."""
        monkeypatch.setenv("CI", "true")
        monkeypatch.delenv("EVALX_MODE", raising=False)

        run = EvalRun()

        assert run.mode == "ci"

    def test_init_without_ci(self, monkeypatch):
        """Test default mode when not in CI."""
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("EVALX_MODE", raising=False)

        run = EvalRun()

        assert run.mode == "local"

    def test_init_with_custom_metadata(self):
        """Test initialization with custom metadata."""
        run = EvalRun(metadata={"custom": "value"})

        assert run.metadata == {"custom": "value"}

    def test_enter_without_context(self, capsys):
        """Test entering context manager without eval context."""
        # Clear any existing context
        import evalx_sdk.context
        evalx_sdk.context._eval_context = None

        run = EvalRun()
        result = run.__enter__()

        assert result == run
        captured = capsys.readouterr()
        assert "No EvalX context found" in captured.out

    @responses.activate
    def test_enter_with_context_success(self, eval_config_data, capsys):
        """Test successful run start with context."""
        # Set eval context
        context = EvalContext(**eval_config_data)
        set_eval_context(context)

        # Mock API response
        responses.post(
            "http://localhost:8000/v1/runs",
            json={
                "run_id": "run_123",
                "eval_id": eval_config_data["eval_id"],
                "asset_id": eval_config_data["asset_id"],
                "env": "local",
                "mode": "local",
                "status": "running",
                "started_at": "2024-01-01T00:00:00Z",
            },
            status=200,
        )

        run = EvalRun()
        run.__enter__()

        assert run.run_id == "run_123"
        assert run.client is not None

        captured = capsys.readouterr()
        assert "Run started" in captured.out

    @responses.activate
    def test_enter_with_api_error(self, eval_config_data, capsys):
        """Test run start with API error."""
        context = EvalContext(**eval_config_data)
        set_eval_context(context)

        # Mock API error
        responses.post(
            "http://localhost:8000/v1/runs",
            json={"error": "Server error"},
            status=500,
        )

        run = EvalRun()
        run.__enter__()

        assert run.run_id is None

        captured = capsys.readouterr()
        assert "Failed to start run" in captured.out

    @responses.activate
    def test_exit_success(self, eval_config_data):
        """Test successful run finish on exit."""
        context = EvalContext(**eval_config_data)
        set_eval_context(context)

        # Mock start run
        responses.post(
            "http://localhost:8000/v1/runs",
            json={
                "run_id": "run_123",
                "eval_id": eval_config_data["eval_id"],
                "asset_id": eval_config_data["asset_id"],
                "env": "local",
                "mode": "local",
                "status": "running",
                "started_at": "2024-01-01T00:00:00Z",
            },
            status=200,
        )

        # Mock finish run
        responses.put(
            "http://localhost:8000/v1/runs/run_123",
            json={
                "run_id": "run_123",
                "eval_id": eval_config_data["eval_id"],
                "asset_id": eval_config_data["asset_id"],
                "env": "local",
                "mode": "local",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:01:00Z",
            },
            status=200,
        )

        run = EvalRun()
        run.__enter__()
        result = run.__exit__(None, None, None)

        assert result is False  # Don't suppress exceptions

    @responses.activate
    def test_exit_with_exception(self, eval_config_data):
        """Test run finish with exception (fail status)."""
        context = EvalContext(**eval_config_data)
        set_eval_context(context)

        # Mock start run
        responses.post(
            "http://localhost:8000/v1/runs",
            json={
                "run_id": "run_123",
                "eval_id": eval_config_data["eval_id"],
                "asset_id": eval_config_data["asset_id"],
                "env": "local",
                "mode": "local",
                "status": "running",
                "started_at": "2024-01-01T00:00:00Z",
            },
            status=200,
        )

        # Mock finish run
        responses.put(
            "http://localhost:8000/v1/runs/run_123",
            json={
                "run_id": "run_123",
                "eval_id": eval_config_data["eval_id"],
                "asset_id": eval_config_data["asset_id"],
                "env": "local",
                "mode": "local",
                "status": "fail",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:01:00Z",
            },
            status=200,
        )

        run = EvalRun()
        run.__enter__()

        # Simulate exception
        exc = ValueError("Test error")
        result = run.__exit__(ValueError, exc, None)

        assert result is False

    def test_exit_without_client(self):
        """Test exit when client was not initialized."""
        run = EvalRun()
        result = run.__exit__(None, None, None)

        assert result is False

    @responses.activate
    def test_context_manager_with_statement(self, eval_config_data):
        """Test using EvalRun as context manager with 'with' statement."""
        context = EvalContext(**eval_config_data)
        set_eval_context(context)

        # Mock start run
        responses.post(
            "http://localhost:8000/v1/runs",
            json={
                "run_id": "run_123",
                "eval_id": eval_config_data["eval_id"],
                "asset_id": eval_config_data["asset_id"],
                "env": "local",
                "mode": "local",
                "status": "running",
                "started_at": "2024-01-01T00:00:00Z",
            },
            status=200,
        )

        # Mock finish run
        responses.put(
            "http://localhost:8000/v1/runs/run_123",
            json={
                "run_id": "run_123",
                "eval_id": eval_config_data["eval_id"],
                "asset_id": eval_config_data["asset_id"],
                "env": "local",
                "mode": "local",
                "status": "success",
                "started_at": "2024-01-01T00:00:00Z",
                "ended_at": "2024-01-01T00:01:00Z",
            },
            status=200,
        )

        executed = False
        with EvalRun() as run:
            executed = True
            assert run.run_id == "run_123"

        assert executed

    def test_monkey_patch_langfuse(self, eval_config_data):
        """Test that run_experiment is monkey-patched."""
        context = EvalContext(**eval_config_data)
        set_eval_context(context)

        # Create mock langfuse module
        import sys
        from unittest.mock import MagicMock

        mock_langfuse = MagicMock()
        mock_client_class = MagicMock()
        mock_original_method = MagicMock()
        mock_client_class.run_experiment = mock_original_method
        mock_langfuse.Langfuse = mock_client_class

        mock_dataset_class = MagicMock()
        mock_dataset_method = MagicMock()
        mock_dataset_class.run_experiment = mock_dataset_method
        mock_langfuse.LangfuseDataset = mock_dataset_class

        # Inject mock into sys.modules
        sys.modules["langfuse"] = mock_langfuse

        try:
            run = EvalRun()
            run.client = Mock()
            run.client.start_run = Mock(
                return_value=Mock(run_id="run_123", eval_id="eval_123")
            )

            run.__enter__()

            # Verify methods were stored and wrapped
            assert run._original_run_experiment == mock_original_method
            assert run._original_dataset_run_experiment == mock_dataset_method
            assert mock_client_class.run_experiment != mock_original_method
            assert mock_dataset_class.run_experiment != mock_dataset_method

            run.__exit__(None, None, None)

            # Verify methods were restored
            assert mock_client_class.run_experiment == mock_original_method
            assert mock_dataset_class.run_experiment == mock_dataset_method
        finally:
            # Clean up
            if "langfuse" in sys.modules:
                del sys.modules["langfuse"]

    def test_create_wrapper_extracts_metadata(self):
        """Test that wrapper extracts metadata from run_experiment call."""
        run = EvalRun()

        def mock_run_experiment(**kwargs):
            return "result"

        def sample_task(item):
            return item

        wrapper = run._create_wrapper(mock_run_experiment)

        # Call wrapper with experiment parameters
        result = wrapper(
            name="Test Exp",
            data=[{"input": "test"}],
            task=sample_task,
            evaluators=[lambda x: x > 0],
            run_evaluators=[lambda r: sum(r)],
        )

        assert result == "result"
        assert run.metadata["experiment_name"] == "Test Exp"
        assert "task" in run.metadata
        assert "dataset" in run.metadata
        assert "evaluators" in run.metadata
        assert "run_evaluators" in run.metadata

    def test_create_wrapper_only_captures_once(self):
        """Test that metadata is only captured on first call."""
        run = EvalRun()

        def mock_run_experiment(**kwargs):
            return "result"

        wrapper = run._create_wrapper(mock_run_experiment)

        # First call
        wrapper(name="First", data=[1, 2, 3])
        assert run.metadata["experiment_name"] == "First"

        # Second call - should not override
        wrapper(name="Second", data=[4, 5, 6])
        assert run.metadata["experiment_name"] == "First"  # Not changed

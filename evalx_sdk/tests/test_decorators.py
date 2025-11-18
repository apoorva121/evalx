"""Tests for evalx_sdk.decorators module."""

import pytest
import responses

from evalx_sdk.context import EvalContext, set_eval_context
from evalx_sdk.decorators import track_run


class TestTrackRunDecorator:
    """Test track_run decorator."""

    @responses.activate
    def test_track_run_basic(self, eval_config_data):
        """Test basic usage of track_run decorator."""
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
                "metadata": {},
            },
            status=200,
        )

        @track_run()
        def my_function():
            return "success"

        result = my_function()

        assert result == "success"

    @responses.activate
    def test_track_run_with_args(self, eval_config_data):
        """Test track_run with function arguments."""
        context = EvalContext(**eval_config_data)
        set_eval_context(context)

        # Mock API calls
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
                "metadata": {},
            },
            status=200,
        )

        @track_run()
        def add_numbers(a, b):
            return a + b

        result = add_numbers(5, 3)

        assert result == 8

    @responses.activate
    def test_track_run_with_exception(self, eval_config_data):
        """Test track_run handles exceptions."""
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

        # Mock finish run with fail status
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
                "metadata": {},
            },
            status=200,
        )

        @track_run()
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            failing_function()

    def test_track_run_without_context(self, capsys):
        """Test track_run without eval context."""
        # Clear context
        import evalx_sdk.context

        evalx_sdk.context._eval_context = None

        @track_run()
        def my_function():
            return "success"

        result = my_function()

        assert result == "success"
        captured = capsys.readouterr()
        assert "No EvalX context found" in captured.out

    @responses.activate
    def test_track_run_preserves_function_name(self, eval_config_data):
        """Test track_run preserves function name and docstring."""
        context = EvalContext(**eval_config_data)
        set_eval_context(context)

        # Mock API calls
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
                "metadata": {},
            },
            status=200,
        )

        @track_run()
        def my_documented_function():
            """This is a documented function."""
            return "success"

        assert my_documented_function.__name__ == "my_documented_function"
        assert my_documented_function.__doc__ == "This is a documented function."
        result = my_documented_function()
        assert result == "success"

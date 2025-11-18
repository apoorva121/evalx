"""Tests for evalx_cli.cli module."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import responses
from click.testing import CliRunner

from evalx_cli.cli import init, main, setup_env_file


class TestMain:
    """Test main CLI group."""

    def test_main_command_exists(self):
        """Test that main command is accessible."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "EvalX CLI" in result.output

    def test_main_version_option(self):
        """Test version option."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "0.1.0" in result.output


class TestInitCommand:
    """Test init command."""

    def test_init_help(self):
        """Test init command help."""
        runner = CliRunner()
        result = runner.invoke(init, ["--help"])

        assert result.exit_code == 0
        assert "Initialize a new evaluation" in result.output
        assert "--asset-id" in result.output
        assert "--name" in result.output

    @responses.activate
    def test_init_success_with_test_mode(self, eval_response_data, tmp_path):
        """Test successful init with test mode."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                init,
                [
                    "--asset-id",
                    "123",
                    "--name",
                    "Test Eval",
                    "--description",
                    "Test description",
                    "--test-mode",
                ],
            )

            assert result.exit_code == 0
            assert "Running in test mode" in result.output
            assert "Evaluation initialized successfully" in result.output
            assert "Eval ID:" in result.output
            assert "Asset ID: 123" in result.output
            assert "Name: Test Eval" in result.output
            assert "Description: Test description" in result.output

            # Verify config file was created
            config_path = Path.cwd() / ".evalx" / "config.json"
            assert config_path.exists()

    @responses.activate
    def test_init_success_without_description(self, eval_response_data, tmp_path):
        """Test successful init without description."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                init,
                [
                    "--asset-id",
                    "123",
                    "--name",
                    "Test Eval",
                    "--test-mode",
                ],
            )

            assert result.exit_code == 0
            assert "Evaluation initialized successfully" in result.output
            # Description should not appear in output
            assert result.output.count("Description:") == 0

    @responses.activate
    def test_init_with_service_url_and_token(self, eval_response_data, tmp_path, monkeypatch):
        """Test init with custom service URL and token."""
        runner = CliRunner()

        # Set default environment to localhost (what Settings will load)
        monkeypatch.setenv("EVALX_SERVICE_URL", "http://custom.com")
        monkeypatch.setenv("EVALX_BEARER_TOKEN", "custom_token")

        # Mock health check and create_or_get_eval
        responses.get(
            "http://custom.com/health",
            json={"status": "healthy"},
            status=200,
        )

        responses.post(
            "http://custom.com/v1/evals",
            json=eval_response_data,
            status=200,
        )

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                init,
                [
                    "--asset-id",
                    "123",
                    "--name",
                    "Test Eval",
                    "--service-url",
                    "http://custom.com",
                    "--bearer-token",
                    "custom_token",
                    "--setup-env",  # Auto-setup .env without prompting
                ],
            )

            assert result.exit_code == 0
            assert "Connecting to EvalX service at http://custom.com" in result.output
            assert "Evaluation initialized successfully" in result.output

    @responses.activate
    def test_init_health_check_failure(self, tmp_path, monkeypatch):
        """Test init with failed health check."""
        runner = CliRunner()

        # Mock health check failure
        responses.get(
            "http://localhost:8000/health",
            json={"status": "unhealthy"},
            status=500,
        )

        # Set environment to avoid relying on .env file
        monkeypatch.setenv("EVALX_SERVICE_URL", "http://localhost:8000")

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                init,
                [
                    "--asset-id",
                    "123",
                    "--name",
                    "Test Eval",
                ],
            )

            assert result.exit_code == 1
            assert "Cannot connect to EvalX service" in result.output

    @responses.activate
    def test_init_api_error(self, tmp_path):
        """Test init with API error."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Mock error by not using test mode and having no service
            result = runner.invoke(
                init,
                [
                    "--asset-id",
                    "123",
                    "--name",
                    "Test Eval",
                ],
                catch_exceptions=False,  # Let exceptions bubble up
            )

            # Health check will fail since no service is running
            assert result.exit_code == 1
            assert "Cannot connect to EvalX service" in result.output

    def test_init_missing_required_options(self):
        """Test init without required options."""
        runner = CliRunner()

        # Missing --asset-id
        result = runner.invoke(init, ["--name", "Test Eval"])
        assert result.exit_code != 0
        assert "Missing option '--asset-id'" in result.output

        # Missing --name
        result = runner.invoke(init, ["--asset-id", "123"])
        assert result.exit_code != 0
        assert "Missing option '--name'" in result.output

    @responses.activate
    def test_init_with_env_and_mode_options(self, eval_response_data, tmp_path):
        """Test init with env and mode options."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                init,
                [
                    "--asset-id",
                    "123",
                    "--name",
                    "Test Eval",
                    "--env",
                    "prod",
                    "--mode",
                    "ci",
                    "--test-mode",
                ],
            )

            assert result.exit_code == 0
            assert "Evaluation initialized successfully" in result.output

    @responses.activate
    def test_init_with_setup_env_flag(self, eval_response_data, tmp_path):
        """Test init with --setup-env flag."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(
                init,
                [
                    "--asset-id",
                    "123",
                    "--name",
                    "Test Eval",
                    "--setup-env",
                    "--test-mode",
                ],
            )

            assert result.exit_code == 0
            assert "Evaluation initialized successfully" in result.output

            # .env file should be created (but not in test mode)
            env_path = Path.cwd() / ".env"
            # In test mode, .env is not created
            assert not env_path.exists()

    @responses.activate
    def test_init_declines_env_setup(self, eval_response_data, tmp_path):
        """Test init when user declines .env setup."""
        runner = CliRunner()

        with runner.isolated_filesystem(temp_dir=tmp_path):
            # Simulate user declining .env setup by providing 'n' as input
            result = runner.invoke(
                init,
                [
                    "--asset-id",
                    "123",
                    "--name",
                    "Test Eval",
                    "--test-mode",
                ],
                input="n\n",  # Decline .env setup
            )

            assert result.exit_code == 0


class TestSetupEnvFile:
    """Test setup_env_file function."""

    def test_setup_env_file_creates_new_file(self, tmp_path):
        """Test creating new .env file."""
        env_path = tmp_path / ".env"

        setup_env_file(
            env_path=env_path,
            service_url="http://test.com",
            bearer_token="test_token",
            env_name="local",
            mode="local",
            interactive=False,
        )

        assert env_path.exists()
        content = env_path.read_text()
        assert "EVALX_SERVICE_URL=http://test.com" in content
        assert "EVALX_BEARER_TOKEN=test_token" in content
        assert "EVALX_ENV=local" in content
        assert "EVALX_MODE=local" in content

    def test_setup_env_file_updates_existing_file(self, tmp_path):
        """Test updating existing .env file."""
        env_path = tmp_path / ".env"

        # Create existing .env with other content
        env_path.write_text("OTHER_VAR=value\nANOTHER_VAR=123\n")

        setup_env_file(
            env_path=env_path,
            service_url="http://updated.com",
            bearer_token="new_token",
            env_name="prod",
            mode="ci",
            interactive=False,
        )

        content = env_path.read_text()
        # Original content should be preserved
        assert "OTHER_VAR=value" in content
        assert "ANOTHER_VAR=123" in content
        # New config should be added
        assert "EVALX_SERVICE_URL=http://updated.com" in content
        assert "EVALX_BEARER_TOKEN=new_token" in content
        assert "EVALX_ENV=prod" in content
        assert "EVALX_MODE=ci" in content

    def test_setup_env_file_replaces_old_evalx_config(self, tmp_path):
        """Test replacing old EvalX configuration."""
        env_path = tmp_path / ".env"

        # Create existing .env with old EvalX config
        old_content = """OTHER_VAR=value

# EvalX Service Configuration (for run tracking)
EVALX_SERVICE_URL=http://old.com
EVALX_BEARER_TOKEN=old_token
EVALX_ENV=local
EVALX_MODE=local

ANOTHER_VAR=123
"""
        env_path.write_text(old_content)

        setup_env_file(
            env_path=env_path,
            service_url="http://new.com",
            bearer_token="new_token",
            env_name="prod",
            mode="ci",
            interactive=False,
        )

        content = env_path.read_text()
        # Old config should be replaced
        assert "http://old.com" not in content
        assert "old_token" not in content
        # New config should be present
        assert "EVALX_SERVICE_URL=http://new.com" in content
        assert "EVALX_BEARER_TOKEN=new_token" in content
        assert "EVALX_ENV=prod" in content
        assert "EVALX_MODE=ci" in content
        # Other vars should be preserved
        assert "OTHER_VAR=value" in content
        assert "ANOTHER_VAR=123" in content

    def test_setup_env_file_with_none_token(self, tmp_path):
        """Test setup with None bearer token."""
        env_path = tmp_path / ".env"

        setup_env_file(
            env_path=env_path,
            service_url="http://test.com",
            bearer_token=None,
            env_name="local",
            mode="local",
            interactive=False,
        )

        content = env_path.read_text()
        # Should use default token when None provided
        assert "EVALX_BEARER_TOKEN=test123" in content

    def test_setup_env_file_preserves_existing_token_when_none(self, tmp_path):
        """Test that existing token is preserved when None is provided."""
        env_path = tmp_path / ".env"

        # Create existing .env with token
        env_path.write_text("EVALX_BEARER_TOKEN=existing_token\n")

        setup_env_file(
            env_path=env_path,
            service_url="http://test.com",
            bearer_token=None,
            env_name="local",
            mode="local",
            interactive=False,
        )

        content = env_path.read_text()
        # Should preserve existing token from .env
        assert "EVALX_BEARER_TOKEN=existing_token" in content

    def test_setup_env_file_interactive_mode(self, tmp_path, monkeypatch):
        """Test interactive mode with user prompts."""
        env_path = tmp_path / ".env"

        # Mock click.prompt to return specific values
        with patch("evalx_cli.cli.click.prompt") as mock_prompt:
            mock_prompt.side_effect = [
                "http://interactive.com",  # service_url
                "interactive_token",  # bearer_token
                "e2e",  # env_name
                "ci",  # mode
            ]

            setup_env_file(
                env_path=env_path,
                service_url="http://default.com",
                bearer_token="default_token",
                env_name="local",
                mode="local",
                interactive=True,
            )

            # Verify prompts were called
            assert mock_prompt.call_count == 4

            # Verify file contains prompted values
            content = env_path.read_text()
            assert "EVALX_SERVICE_URL=http://interactive.com" in content
            assert "EVALX_BEARER_TOKEN=interactive_token" in content
            assert "EVALX_ENV=e2e" in content
            assert "EVALX_MODE=ci" in content

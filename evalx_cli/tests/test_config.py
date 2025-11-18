"""Tests for evalx_cli.config module."""

import os

import pytest

from evalx_cli.config import Settings, get_settings


class TestSettings:
    """Test Settings model."""

    def test_default_settings(self, clean_env):
        """Test default settings values."""
        settings = Settings()

        assert settings.evalx_service_url == "http://localhost:8000"
        assert settings.evalx_bearer_token is None

    def test_settings_from_env(self, monkeypatch):
        """Test loading settings from environment variables."""
        monkeypatch.setenv("EVALX_SERVICE_URL", "http://test.com")
        monkeypatch.setenv("EVALX_BEARER_TOKEN", "test_token_123")

        settings = Settings()

        assert settings.evalx_service_url == "http://test.com"
        assert settings.evalx_bearer_token == "test_token_123"

    def test_settings_case_insensitive(self, monkeypatch):
        """Test that settings are case-insensitive."""
        monkeypatch.setenv("evalx_service_url", "http://lowercase.com")
        monkeypatch.setenv("evalx_bearer_token", "lowercase_token")

        settings = Settings()

        assert settings.evalx_service_url == "http://lowercase.com"
        assert settings.evalx_bearer_token == "lowercase_token"

    def test_settings_ignores_extra_env_vars(self, monkeypatch):
        """Test that extra environment variables are ignored."""
        monkeypatch.setenv("EVALX_SERVICE_URL", "http://test.com")
        monkeypatch.setenv("RANDOM_EXTRA_VAR", "should_be_ignored")

        # Should not raise validation error
        settings = Settings()

        assert settings.evalx_service_url == "http://test.com"
        assert not hasattr(settings, "random_extra_var")


class TestGetSettings:
    """Test get_settings function."""

    def test_get_settings_returns_settings_instance(self, clean_env):
        """Test that get_settings returns a Settings instance."""
        settings = get_settings()

        assert isinstance(settings, Settings)
        assert settings.evalx_service_url == "http://localhost:8000"

    def test_get_settings_with_env_vars(self, monkeypatch):
        """Test get_settings loads from environment."""
        monkeypatch.setenv("EVALX_SERVICE_URL", "http://custom.com")
        monkeypatch.setenv("EVALX_BEARER_TOKEN", "custom_token")

        settings = get_settings()

        assert settings.evalx_service_url == "http://custom.com"
        assert settings.evalx_bearer_token == "custom_token"

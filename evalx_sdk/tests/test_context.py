"""Tests for evalx_sdk.context module."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from evalx_sdk.context import (
    EvalContext,
    get_eval_context,
    init_context,
    load_eval_config,
    save_eval_config,
    set_eval_context,
)


class TestEvalContext:
    """Test EvalContext model."""

    def test_create_valid_context(self, eval_config_data):
        """Test creating a valid EvalContext."""
        context = EvalContext(**eval_config_data)

        assert context.eval_id == eval_config_data["eval_id"]
        assert context.asset_id == eval_config_data["asset_id"]
        assert context.name == eval_config_data["name"]
        assert context.description == eval_config_data["description"]

    def test_create_context_without_description(self):
        """Test creating context without optional description."""
        context = EvalContext(
            eval_id="550e8400-e29b-41d4-a716-446655440000",
            asset_id=123,
            name="Test",
        )

        assert context.description is None

    def test_context_is_immutable(self, eval_config_data):
        """Test that EvalContext is immutable (frozen)."""
        context = EvalContext(**eval_config_data)

        with pytest.raises(ValidationError):
            context.eval_id = "new_id"

    def test_create_context_missing_required_fields(self):
        """Test that missing required fields raise validation error."""
        with pytest.raises(ValidationError):
            EvalContext(eval_id="123", asset_id=456)  # Missing name


class TestSetGetEvalContext:
    """Test global context management."""

    def test_set_and_get_context(self, eval_config_data):
        """Test setting and getting global context."""
        context = EvalContext(**eval_config_data)
        set_eval_context(context)

        retrieved = get_eval_context()
        assert retrieved == context
        assert retrieved.eval_id == context.eval_id

    def test_get_context_when_none_set(self):
        """Test getting context when none is set."""
        result = get_eval_context()
        assert result is None


class TestLoadEvalConfig:
    """Test load_eval_config function."""

    def test_load_from_explicit_path(self, eval_config_file, eval_config_data):
        """Test loading config from explicit path."""
        context = load_eval_config(eval_config_file)

        assert context is not None
        assert context.eval_id == eval_config_data["eval_id"]
        assert context.asset_id == eval_config_data["asset_id"]

    def test_load_from_current_directory(self, temp_dir, eval_config_data, monkeypatch):
        """Test loading config from current directory."""
        # Change to temp directory
        monkeypatch.chdir(temp_dir)

        # Create config in current directory
        config_dir = temp_dir / ".evalx"
        config_dir.mkdir()
        config_path = config_dir / "config.json"

        with open(config_path, "w") as f:
            json.dump(eval_config_data, f)

        context = load_eval_config()

        assert context is not None
        assert context.eval_id == eval_config_data["eval_id"]

    def test_load_from_parent_directory(self, temp_dir, eval_config_data, monkeypatch):
        """Test loading config from parent directory."""
        # Create config in temp directory
        config_dir = temp_dir / ".evalx"
        config_dir.mkdir()
        config_path = config_dir / "config.json"

        with open(config_path, "w") as f:
            json.dump(eval_config_data, f)

        # Change to subdirectory
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        monkeypatch.chdir(subdir)

        context = load_eval_config()

        assert context is not None
        assert context.eval_id == eval_config_data["eval_id"]

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file returns None."""
        result = load_eval_config(Path("/nonexistent/path/config.json"))
        assert result is None

    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON returns None."""
        config_dir = temp_dir / ".evalx"
        config_dir.mkdir()
        config_path = config_dir / "config.json"

        with open(config_path, "w") as f:
            f.write("invalid json{")

        result = load_eval_config(config_path)
        assert result is None

    def test_load_invalid_data(self, temp_dir):
        """Test loading data with invalid schema returns None."""
        config_dir = temp_dir / ".evalx"
        config_dir.mkdir()
        config_path = config_dir / "config.json"

        with open(config_path, "w") as f:
            json.dump({"invalid": "data"}, f)

        result = load_eval_config(config_path)
        assert result is None

    def test_load_no_config_anywhere(self, temp_dir, monkeypatch):
        """Test loading when no config exists anywhere."""
        monkeypatch.chdir(temp_dir)
        result = load_eval_config()
        assert result is None


class TestSaveEvalConfig:
    """Test save_eval_config function."""

    def test_save_to_explicit_path(self, temp_dir, eval_config_data):
        """Test saving config to explicit path."""
        context = EvalContext(**eval_config_data)
        config_path = temp_dir / "custom" / "config.json"

        saved_path = save_eval_config(context, config_path)

        assert saved_path == config_path
        assert config_path.exists()

        with open(config_path) as f:
            data = json.load(f)

        assert data["eval_id"] == eval_config_data["eval_id"]

    def test_save_to_default_path(self, temp_dir, eval_config_data, monkeypatch):
        """Test saving config to default .evalx/config.json."""
        monkeypatch.chdir(temp_dir)
        context = EvalContext(**eval_config_data)

        saved_path = save_eval_config(context)

        expected_path = temp_dir / ".evalx" / "config.json"
        assert saved_path == expected_path
        assert expected_path.exists()

    def test_save_creates_directory(self, temp_dir, eval_config_data):
        """Test that save creates directory if it doesn't exist."""
        context = EvalContext(**eval_config_data)
        config_path = temp_dir / "new" / "dir" / "config.json"

        save_eval_config(context, config_path)

        assert config_path.parent.exists()
        assert config_path.exists()

    def test_save_without_description(self, temp_dir):
        """Test saving context without optional description."""
        context = EvalContext(
            eval_id="550e8400-e29b-41d4-a716-446655440000",
            asset_id=123,
            name="Test",
        )
        config_path = temp_dir / "config.json"

        save_eval_config(context, config_path)

        with open(config_path) as f:
            data = json.load(f)

        assert data["description"] is None


class TestInitContext:
    """Test init_context function."""

    def test_init_with_valid_config(
        self, temp_dir, eval_config_data, monkeypatch, capsys
    ):
        """Test initializing context with valid config."""
        monkeypatch.chdir(temp_dir)

        # Create config file
        config_dir = temp_dir / ".evalx"
        config_dir.mkdir()
        config_path = config_dir / "config.json"

        with open(config_path, "w") as f:
            json.dump(eval_config_data, f)

        context = init_context(verbose=True)

        assert context is not None
        assert context.eval_id == eval_config_data["eval_id"]

        # Check that context was set globally
        assert get_eval_context() == context

        # Check verbose output
        captured = capsys.readouterr()
        assert "✅ EvalX context loaded" in captured.out
        assert eval_config_data["eval_id"] in captured.out

    def test_init_without_config(self, temp_dir, monkeypatch, capsys):
        """Test initializing when no config exists."""
        monkeypatch.chdir(temp_dir)

        context = init_context(verbose=True)

        assert context is None

        # Check verbose output
        captured = capsys.readouterr()
        assert "No EvalX config found" in captured.out

    def test_init_verbose_false(self, temp_dir, eval_config_data, monkeypatch, capsys):
        """Test initializing with verbose=False."""
        monkeypatch.chdir(temp_dir)

        # Create config file
        config_dir = temp_dir / ".evalx"
        config_dir.mkdir()
        config_path = config_dir / "config.json"

        with open(config_path, "w") as f:
            json.dump(eval_config_data, f)

        context = init_context(verbose=False)

        assert context is not None

        # Check no output
        captured = capsys.readouterr()
        assert captured.out == ""

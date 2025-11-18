# EvalX SDK

Python SDK for EvalX - Context management for evaluation tracking.

## Overview

The EvalX SDK provides a lightweight library for managing evaluation context in your Python applications. It handles loading, saving, and accessing evaluation configurations that are created by the EvalX CLI.

## Features

- Load evaluation context from `.evalx/config.json`
- Save evaluation context to local config files
- Global context management for easy access across your application
- Automatic config file discovery in parent directories
- Type-safe with Pydantic models
- **Automatic run tracking** with `EvalRun` context manager
- **Environment detection** for local, CI, and production modes

## Installation

### Development Mode (Current)

For local development without packaging:

```bash
# From evalX root directory
cp .env.sample .env
source .env  # Adds evalx_sdk to PYTHONPATH

# Install dependencies
cd evalx_sdk
poetry install
```

### Production Mode (Future)

When ready for production, install as a proper package:

```bash
cd evalx_sdk
poetry build
pip install dist/evalx_sdk-0.1.0.tar.gz
```

## Usage

### Quick Start (Recommended)

The simplest way to use EvalX SDK is with automatic initialization and run tracking:

```python
from evalx_sdk import init_context, EvalRun
from langfuse import get_client

# Initialize EvalX context (loads .evalx/config.json if exists)
init_context()

# Track your experiment run automatically
with EvalRun():
    # Your evaluation code here
    result = get_client().run_experiment(
        name="My Experiment",
        data=my_data,
        task=my_task,
    )
```

This automatically:
- Loads eval context from `.evalx/config.json`
- Detects environment (local, e2e, prod) from `EVALX_ENV`
- Detects mode (local, ci, prod) from `EVALX_MODE` or CI environment
- Starts a run in EvalX service
- Finishes the run with success/fail status
- Handles errors gracefully

### Loading Evaluation Config

The SDK automatically searches for `.evalx/config.json` starting from the current directory and walking up to parent directories:

```python
from evalx_sdk import load_eval_config, set_eval_context, get_eval_context

# Load config from .evalx/config.json (searches current and parent directories)
context = load_eval_config()

if context:
    # Set as global context
    set_eval_context(context)

    # Access context anywhere in your application
    current_context = get_eval_context()
    print(f"Eval ID: {current_context.eval_id}")
    print(f"Asset ID: {current_context.asset_id}")
    print(f"Name: {current_context.name}")
```

### Simple Initialization

Use the one-liner `init_context()` to load config and set global context:

```python
from evalx_sdk import init_context

# Load config and set global context in one line
init_context()  # Prints status and returns context
```

### Run Tracking with EvalRun

The `EvalRun` context manager automatically tracks evaluation runs:

```python
from evalx_sdk import init_context, EvalRun

# Initialize context first
init_context()

# Track a run
with EvalRun():
    # Your evaluation code here
    # Run is automatically started and finished
    pass
```

**Environment Configuration:**

The `EvalRun` context manager auto-detects environment and mode:

- `env`: From `EVALX_ENV` environment variable (default: "local")
  - Values: `local`, `e2e`, `prod`
- `mode`: From `EVALX_MODE` or auto-detected from CI environment (default: "local")
  - Values: `local`, `ci`, `prod`
  - Auto-detects: If `CI=true`, defaults to "ci"

**Optional Parameters:**

```python
with EvalRun(
    service_url="http://localhost:8000",  # Override service URL
    bearer_token="your-token",            # Override bearer token
    metadata={"key": "value"}             # Add custom metadata
):
    # Your code here
    pass
```

**Environment Variables:**

```bash
# Required for run tracking
EVALX_SERVICE_URL=http://localhost:8000  # EvalX service URL
EVALX_BEARER_TOKEN=your-token            # Authentication token

# Optional - Auto-detected
EVALX_ENV=local                          # Environment (local, e2e, prod)
EVALX_MODE=local                         # Mode (local, ci, prod)
CI=true                                  # Auto-sets mode to "ci" if present
```

**Status on Exit:**

- Normal exit: Run marked as `success`
- Exception exit: Run marked as `fail` with error details in metadata

### Manual Context Management

You can also create and manage context manually:

```python
from evalx_sdk import EvalContext, save_eval_config

# Create context
context = EvalContext(
    eval_id="550e8400-e29b-41d4-a716-446655440000",
    asset_id=123,
    name="My Evaluation",
    description="Testing my model"
)

# Save to custom location
config_path = save_eval_config(context, Path("/path/to/config.json"))

# Or save to default location (.evalx/config.json)
config_path = save_eval_config(context)
```

### EvalContext Model

```python
class EvalContext:
    eval_id: str              # UUID of the evaluation
    asset_id: int             # Asset ID associated with the evaluation
    name: str                 # Name of the evaluation
    description: Optional[str] # Description of the evaluation
```

## Configuration File Format

The SDK uses a JSON configuration file stored in `.evalx/config.json`:

```json
{
  "eval_id": "550e8400-e29b-41d4-a716-446655440000",
  "asset_id": 123,
  "name": "My Evaluation",
  "description": "Testing my model"
}
```

## API Reference

### Functions

#### `load_eval_config(config_path: Optional[Path] = None) -> Optional[EvalContext]`

Load evaluation context from a config file.

- **Args:**
  - `config_path`: Path to the config file. If None, searches for `.evalx/config.json` in current and parent directories.
- **Returns:** EvalContext if found and valid, None otherwise

#### `save_eval_config(context: EvalContext, config_path: Optional[Path] = None) -> Path`

Save evaluation context to a config file.

- **Args:**
  - `context`: EvalContext to save
  - `config_path`: Path to save the config. If None, saves to `.evalx/config.json`
- **Returns:** Path where the config was saved

#### `set_eval_context(context: EvalContext) -> None`

Set the global evaluation context.

- **Args:**
  - `context`: EvalContext instance to set as global context

#### `get_eval_context() -> Optional[EvalContext]`

Get the current evaluation context.

- **Returns:** Current EvalContext if set, None otherwise

#### `init_context(verbose: bool = True) -> Optional[EvalContext]`

Initialize EvalX context by loading config and setting global context.

- **Args:**
  - `verbose`: Whether to print status messages (default: True)
- **Returns:** EvalContext if found and loaded, None otherwise

### Classes

#### `EvalRun`

Context manager for automatic run tracking.

**Constructor:**
```python
EvalRun(
    service_url: Optional[str] = None,
    bearer_token: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
)
```

- **Args:**
  - `service_url`: EvalX service URL (defaults to `EVALX_SERVICE_URL` env var or localhost)
  - `bearer_token`: Bearer token for authentication (defaults to `EVALX_BEARER_TOKEN` env var)
  - `metadata`: Custom metadata to attach to the run

**Usage:**
```python
with EvalRun():
    # Your evaluation code here
    pass
```

#### `EvalXRunClient`

Low-level API client for run endpoints.

**Methods:**
- `start_run(eval_id, asset_id, env, mode, metadata)`: Start a new run
- `finish_run(run_id, status, ended_at, metadata)`: Finish an existing run

## Requirements

- Python 3.11.9+
- pydantic >= 2.0.0

## Development

```bash
# Install dependencies
cd evalx_sdk
pip install pydantic requests pytest pytest-cov pytest-mock responses

# Run tests
pytest

# Run tests with coverage
pytest --cov=evalx_sdk --cov-report=term-missing

# Run tests with HTML coverage report
pytest --cov=evalx_sdk --cov-report=html
# Open htmlcov/index.html in browser to view

# Run specific test file
pytest tests/test_context.py

# Run specific test
pytest tests/test_context.py::TestEvalContext::test_create_valid_context
```

### Test Coverage

The test suite achieves **97% code coverage** with 65 comprehensive tests covering:
- Context management (loading, saving, initialization)
- API client (start/finish runs with various scenarios)
- Run tracking (EvalRun context manager, monkey-patching)
- Introspection (function/dataset/experiment metadata extraction)
- Decorators (track_run decorator)
- Error handling and edge cases

## Integration with EvalX CLI

This SDK is designed to work seamlessly with the EvalX CLI. When you run `evalx init`, it creates a `.evalx/config.json` file that this SDK can load:

```bash
# Initialize evaluation with CLI
evalx init --asset-id 123 --name "My Eval"

# Load in your Python code
from evalx_sdk import load_eval_config
context = load_eval_config()  # Automatically finds .evalx/config.json
```

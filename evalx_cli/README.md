# EvalX CLI

Command-line interface for EvalX - Initialize and manage evaluations.

## Overview

The EvalX CLI provides a simple command-line tool to initialize evaluations and manage their lifecycle. It communicates with the EvalX service to create or retrieve evaluations and stores the configuration locally for use by your evaluation scripts.

## Features

- Initialize evaluations with the EvalX service
- Store eval configuration locally in `.evalx/config.json`
- Health check for EvalX service connectivity
- Test mode for offline development
- Integration with EvalX SDK for context management

## Installation

### Development Mode (Current)

For local development without packaging:

```bash
# From evalX root directory
cp .env.sample .env
source .env  # Sets EVALX_DEV_MODE=true and adds to PYTHONPATH

# Install dependencies
cd evalx_cli
poetry install

# Run CLI
cd ..
./evalx init --asset-id 123 --name "My Eval"
```

**⚠️ IMPORTANT:** The `./evalx` script requires `EVALX_DEV_MODE=true` in `.env` to run. This safety check prevents accidental use of development paths in production.

### Production Mode (Future)

When ready for production, install as a proper package:

```bash
cd evalx_cli
poetry build
pip install dist/evalx_cli-0.1.0.tar.gz

# Use installed CLI (no ./ prefix)
evalx init --asset-id 123 --name "My Eval"
```

## Configuration

The CLI uses environment variables for configuration.

### Development Configuration

Copy `.env.sample` to `.env` in the evalX root directory:

```bash
# ============================================
# DEVELOPMENT ONLY - DO NOT USE IN PRODUCTION
# ============================================

# Development Mode Flag - Required for ./evalx script to run
EVALX_DEV_MODE=true

# Python Path - Adds evalx_sdk and evalx_cli to import path
PYTHONPATH=${PYTHONPATH}:${PWD}/evalx_sdk:${PWD}/evalx_cli

# EvalX Service Configuration
EVALX_SERVICE_URL=http://localhost:8000
EVALX_BEARER_TOKEN=your-secret-token-here
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `EVALX_SERVICE_URL` | No | `http://localhost:8000` | URL of the EvalX service |
| `EVALX_BEARER_TOKEN` | No | None | Bearer token for authentication |

## Usage

### Initialize an Evaluation

Create or retrieve an evaluation and save its configuration locally:

```bash
evalx init --asset-id 123 --name "My Evaluation" --description "Testing my model"
```

**Options:**
- `--asset-id` (required): Unique asset identifier for the evaluation
- `--name` (required): Name of the evaluation
- `--description` (optional): Description of the evaluation

**Output:**
```
Connecting to EvalX service at http://localhost:8000...
Initializing evaluation for asset_id=123...
✅ Evaluation initialized successfully!
   Eval ID: 550e8400-e29b-41d4-a716-446655440000
   Asset ID: 123
   Name: My Evaluation
   Description: Testing my model
   Config saved to: /path/to/project/.evalx/config.json
```

### Configuration File

The `evalx init` command creates a `.evalx/config.json` file in your current directory:

```json
{
  "eval_id": "550e8400-e29b-41d4-a716-446655440000",
  "asset_id": 123,
  "name": "My Evaluation",
  "description": "Testing my model"
}
```

This file is automatically discovered by the EvalX SDK when you use it in your evaluation scripts.

## Test Mode

For offline development or testing without the EvalX service, you can use test mode (hidden flag):

```bash
evalx init --asset-id 123 --name "Test Eval" --test-mode
```

This generates a mock eval_id without connecting to the service.

## Integration with EvalX SDK

After initializing an evaluation, use the EvalX SDK to load the context in your Python code:

```python
from evalx_sdk import load_eval_config, set_eval_context

# Load the config created by 'evalx init'
context = load_eval_config()
if context:
    set_eval_context(context)
    print(f"Using eval_id: {context.eval_id}")
```

## Requirements

- Python 3.11.9+
- click >= 8.1.0
- requests >= 2.31.0
- pydantic >= 2.0.0
- python-dotenv >= 1.0.0

## Development

```bash
# Install dependencies
cd evalx_cli
pip install click pydantic pydantic-settings requests python-dotenv pytest pytest-cov pytest-mock responses

# Run tests
pytest

# Run tests with coverage
pytest --cov=evalx_cli --cov-report=term-missing

# Run tests with HTML coverage report
pytest --cov=evalx_cli --cov-report=html
# Open htmlcov/index.html in browser to view

# Run specific test file
pytest tests/test_cli.py

# Run specific test
pytest tests/test_cli.py::TestInitCommand::test_init_success_with_test_mode
```

### Test Coverage

The test suite achieves **97% code coverage** with 46 comprehensive tests covering:
- CLI commands (init, help, version)
- API client (create_or_get_eval, health_check)
- Configuration management (settings, environment variables)
- Mock client for testing
- .env file setup and management
- Error handling and edge cases

## Troubleshooting

### Cannot connect to EvalX service

**Error:**
```
❌ Error: Cannot connect to EvalX service at http://localhost:8000
   Make sure the service is running or check EVALX_SERVICE_URL
```

**Solutions:**
1. Check if the EvalX service is running: `curl http://localhost:8000/health`
2. Verify `EVALX_SERVICE_URL` in your `.env` file
3. Ensure the service is accessible from your network

### Authentication errors

If you receive `401 Unauthorized` errors, ensure your `EVALX_BEARER_TOKEN` matches the token configured in the EvalX service.

### Config file not found

The `.evalx/config.json` file is created in your current working directory when you run `evalx init`. Make sure to:
1. Run `evalx init` before using the SDK
2. Run your evaluation scripts from the same directory or a subdirectory
3. The SDK will automatically search parent directories for `.evalx/config.json`

## Command Reference

### `evalx init`

Initialize a new evaluation and save config locally.

**Synopsis:**
```bash
evalx init --asset-id ID --name NAME [--description DESC]
```

**Options:**
- `--asset-id INTEGER` - Unique asset identifier (required)
- `--name TEXT` - Name of the evaluation (required)
- `--description TEXT` - Optional description
- `--help` - Show help message

**Examples:**
```bash
# Basic initialization
evalx init --asset-id 123 --name "Geography Quiz"

# With description
evalx init --asset-id 456 --name "Math Test" --description "Testing basic arithmetic"
```

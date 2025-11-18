# EvalX Client

Langfuse-based evaluation client for running experiments with EvalX tracking.

## Overview

The eval-client runs LLM evaluation experiments using Langfuse and integrates with EvalX SDK to automatically load evaluation context.

## Setup

### Prerequisites

1. **Install Python 3.11.9:**
   ```bash
   pyenv install 3.11.9
   pyenv local 3.11.9
   ```

2. **Install dependencies:**
   ```bash
   cd eval-client
   poetry env use 3.11.9
   poetry install
   ```

### Development Mode

```bash
# 1. Setup environment
cd eval-client
cp .env.sample .env

# 2. Edit .env with your API keys
# Required:
# - LANGFUSE_SECRET_KEY=sk-lf-...
# - LANGFUSE_PUBLIC_KEY=pk-lf-...
# - OPENAI_API_KEY=sk-proj-...

# 3. Initialize evaluation (creates .evalx/config.json)
./evalx init --asset-id 123 --name "Geography Quiz"
# OR for test mode (no service needed):
./evalx init --asset-id 123 --name "Geography Quiz" --test-mode

# 4. Run evaluation
source .env  # Load PYTHONPATH
poetry run python eval.py
```

## Configuration

The `.env.sample` file shows all required configuration. Copy and edit it:

```bash
cp .env.sample .env
```

**Required environment variables:**
```bash
# Langfuse Configuration
LANGFUSE_SECRET_KEY=sk-lf-...  # Get from Langfuse dashboard
LANGFUSE_PUBLIC_KEY=pk-lf-...   # Get from Langfuse dashboard
LANGFUSE_BASE_URL=https://cloud.langfuse.com

# OpenAI Configuration
OPENAI_API_KEY=sk-proj-...  # Get from OpenAI dashboard

# Python Path (auto-configured in .env.sample)
export PYTHONPATH="${PYTHONPATH}:${PWD}/../evalx_sdk"
```

**⚠️ IMPORTANT:**
- Never commit `.env` files to git - they contain secrets
- Always use `source .env` before running eval.py
- The `export` statement in PYTHONPATH is required for bash to export the variable

## Usage

### 1. Initialize Evaluation

First, initialize an evaluation using the EvalX CLI:

```bash
cd ..
./evalx init --asset-id 123 --name "Geography Quiz" --description "Testing basic functionality"
```

This creates `.evalx/config.json` with the eval_id.

### 2. Run Evaluation

The eval.py script automatically loads the eval context:

```bash
cd eval-client
poetry run python eval.py
```

Output:
```
✅ EvalX context loaded: eval_id=550e8400-e29b-41d4-a716-446655440000, asset_id=123
✅ Langfuse configured
...
```

## How It Works

The `eval.py` script uses EvalX SDK to automatically load evaluation context:

```python
from evalx_sdk import init_context

# Load .evalx/config.json if exists
init_context()
```

This single line:
1. Searches for `.evalx/config.json` in current and parent directories
2. Loads the eval_id and other context
3. Sets it as global context for later use
4. Prints status message

## Production Setup (Future)

For production, install evalx_sdk as a proper package:

```bash
# Install evalx_sdk
pip install /path/to/evalx_sdk

# No PYTHONPATH needed in .env
```

Then your code works the same way without environment modifications.

## Requirements

- Python 3.11.9+
- langfuse >= 3.10.0
- openai >= 2.8.1
- python-dotenv >= 1.2.1
- evalx_sdk (via PYTHONPATH in dev mode)

## Notes

- **Model Issue:** Line 17 uses `model="gpt-4.1"` which doesn't exist. Change to `gpt-4`, `gpt-4-turbo`, or `gpt-4o`
- **EvalX Config:** If `.evalx/config.json` is not found, the script will print a message but continue running
- **Development Only:** The PYTHONPATH approach is for fast development iteration. Use proper package installation in production.

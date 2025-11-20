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

## Debug Utilities

The eval-client includes several diagnostic utilities to help you explore and debug your Langfuse traces and evaluations:

### 1. `quick_trace_check.py` - Quick Trace Existence Check

Quickly verify that traces exist in your Langfuse project without fetching all details.

```bash
poetry run python quick_trace_check.py
```

**Output:**
```
✅ Langfuse configured
Found 4 traces in last 1 hour
First few traces:
  - ID: 15782e0d..., Name: experiment-item-run, Time: 2025-11-20 04:26:49
```

**Use Cases:**
- Verify traces are being created before running evaluations
- Quick sanity check before running time-consuming operations
- Debugging trace creation issues

### 2. `explore_traces.py` - Comprehensive Trace Explorer

Analyze your Langfuse traces to understand their structure and characteristics.

```bash
poetry run python explore_traces.py
```

**Features:**
- Statistics on trace characteristics (inputs, outputs, metadata)
- Sample traces with full details
- Analysis of tags, user IDs, sessions, releases
- Recommendations for filters based on your data

**Output:**
```
📊 Statistics from 50 traces:
   - Traces with input: 20/50 (40.0%)
   - Traces with output: 50/50 (100.0%)
   - Unique user IDs: 0
🏷️  Tags: production (15), staging (10)
```

**Use Cases:**
- Understand what data exists in your Langfuse project
- Choose appropriate filters for TraceToDatasetBuilder
- Identify traces suitable for evaluation

### 3. `inspect_dataset_item.py` - Dataset Item Structure Inspector

Inspect the structure of dataset items built from traces.

```bash
poetry run python inspect_dataset_item.py
```

**Output:**
```json
{
  "trace_id": "15782e0d...",
  "trace_name": "experiment-item-run",
  "input": "What is the capital of Germany?",
  "output": "The capital of Germany is **Berlin**.",
  "trace_core": {...}
}
```

**Use Cases:**
- Understand what data is available in evaluators
- Debug evaluator functions
- Verify trace data is being parsed correctly

### 4. `test_dataset_conversion.py` - Dataset Conversion Tester

Test the `to_langfuse_dataset()` method to verify traces are correctly converted to Langfuse datasets.

```bash
poetry run python test_dataset_conversion.py
```

**Features:**
- Creates a Langfuse dataset from traces
- Verifies dataset client methods
- Checks dataset items have correct attributes

**Output:**
```
✅ Created Langfuse dataset: trace_dataset_Test Dataset_20251120
  - Total items: 2
  - First item has 'input' attribute: True
  - First item has 'expected_output' attribute: True
✅ All tests passed!
```

**Use Cases:**
- Verify dataset conversion works before running experiments
- Debug dataset creation issues
- Test Langfuse dataset integration

### 5. `trace.py` - Production Trace Evaluation

Complete example of evaluating production traces with item-level and run-level evaluators.

```bash
poetry run python trace.py
```

**Features:**
- Batch processing of traces (memory efficient)
- Item-level evaluators (per-trace evaluation)
- Run-level evaluators (aggregate statistics)
- Trace-specific evaluator signature with `trace_item` parameter
- Prints trace_id for each evaluation

**Output:**
```
📦 Processing batch 1 (2 traces)...
  [accuracy_evaluator] Evaluating trace_id=15782e0d...
  [trace_quality_evaluator] Evaluating trace_id=15782e0d...
✅ Batch 1 complete

============================================================
Evaluation Complete: Production Trace Evaluation
============================================================
Total traces: 4
Processed: 4
Scores by evaluator:
  accuracy: Avg: 0.000, Count: 4
  trace_quality: Avg: 0.000, Count: 4
```

**Use Cases:**
- Evaluate production traces for quality/accuracy
- Monitor trace characteristics over time
- Compute aggregate statistics across traces

### 6. `trace_simple.py` - Simplified Trace Evaluation

Minimal example for quick testing and learning.

### 7. `trace_debug.py` - Trace Debugging Suite

Comprehensive debugging with multiple test scenarios for different time windows and filters.

## Example Workflow

1. **Check traces exist:**
   ```bash
   poetry run python quick_trace_check.py
   ```

2. **Explore trace characteristics:**
   ```bash
   poetry run python explore_traces.py
   ```

3. **Inspect dataset item structure:**
   ```bash
   poetry run python inspect_dataset_item.py
   ```

4. **Run evaluation on production traces:**
   ```bash
   poetry run python trace.py
   ```

## Notes

- **Model Issue:** Line 17 uses `model="gpt-4.1"` which doesn't exist. Change to `gpt-4`, `gpt-4-turbo`, or `gpt-4o`
- **EvalX Config:** If `.evalx/config.json` is not found, the script will print a message but continue running
- **Development Only:** The PYTHONPATH approach is for fast development iteration. Use proper package installation in production.
- **Rate Limits:** Langfuse may rate limit API calls. The debug utilities use small page sizes and limits to avoid triggering rate limits.

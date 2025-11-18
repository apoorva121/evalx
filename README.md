# EvalX

Evaluation tracking and management system with FastAPI service, Python SDK, and CLI.

## Project Structure

```
evalX/
├── evalx_sdk/           # Python SDK for context management
├── evalx_cli/           # CLI tool for initialization
├── eval-client/         # Langfuse experiment runner
├── eval-service/        # FastAPI service
├── evalx                # CLI entrypoint (development mode)
├── .env.sample          # Environment template
└── .gitignore           # Git ignore rules
```

## Quick Start (Development Mode)

### Prerequisites

1. **Install Python 3.11.9 with pyenv:**
   ```bash
   pyenv install 3.11.9
   ```

2. **Install Poetry:**
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

### Step 1: Setup Python Environment

```bash
cd evalX

# Set Python version for all projects
pyenv local 3.11.9

# Setup evalx_sdk
cd evalx_sdk
poetry env use 3.11.9
poetry install
cd ..

# Setup evalx_cli
cd evalx_cli
poetry env use 3.11.9
poetry install
cd ..

# Setup eval-client
cd eval-client
poetry env use 3.11.9
poetry install
cd ..
```

### Step 2: Configure Environment

**In evalX root:**
```bash
cd evalX
cp .env.sample .env
# The default EVALX_BEARER_TOKEN=test123 matches eval-service's default
# Edit if you changed the service token
```

**In eval-client:**
```bash
cd eval-client
cp .env.sample .env
# Edit .env and add your API keys:
# - LANGFUSE_SECRET_KEY
# - LANGFUSE_PUBLIC_KEY
# - OPENAI_API_KEY
```

### Step 3: Start EvalX Service (Optional)

**If you want to use the real service (not test mode):**

```bash
cd eval-service
poetry install
docker compose up -d  # Start PostgreSQL
poetry run alembic upgrade head
poetry run uvicorn app.main:app --reload
```

Service will be available at: http://localhost:8000

### Step 4: Initialize Evaluation

**From eval-client directory:**
```bash
cd eval-client

# With real service (if running):
./evalx init --asset-id 123 --name "Geography Quiz" --description "Testing"

# OR with test mode (no service needed):
./evalx init --asset-id 123 --name "Geography Quiz" --test-mode
```

This creates `.evalx/config.json` with your eval_id.

### Step 5: Run Evaluation

**From eval-client directory:**
```bash
cd eval-client
source .env  # Load PYTHONPATH
poetry run python eval.py
```

**Output:**
```
✅ EvalX context loaded: eval_id=..., asset_id=123
✅ Langfuse configured
[Experiment runs...]
```

## Development vs Production

### ⚠️ Current Setup: Development Mode

**Characteristics:**
- `EVALX_DEV_MODE=true` required in `.env`
- Uses `PYTHONPATH` to add modules to import path
- Fast iteration - no build/install needed
- Run CLI via `./evalx` script
- **NOT production-ready**

**Security Guards:**
- `./evalx` script checks for `EVALX_DEV_MODE=true` or exits
- All `.env` files in `.gitignore`
- Clear "DEVELOPMENT ONLY" warnings in config files

### 🚀 Future: Production Mode

When ready for production deployment:

1. **Build packages:**
   ```bash
   cd evalx_sdk && poetry build
   cd evalx_cli && poetry build
   ```

2. **Install packages:**
   ```bash
   pip install ./evalx_sdk/dist/evalx_sdk-0.1.0.tar.gz
   pip install ./evalx_cli/dist/evalx_cli-0.1.0.tar.gz
   ```

3. **Use without dev mode:**
   ```bash
   # No EVALX_DEV_MODE needed
   # No PYTHONPATH needed
   evalx init --asset-id 123 --name "My Eval"  # Works directly
   ```

## Environment Variables

### Development (.env in evalX root)

```bash
# REQUIRED for development
EVALX_DEV_MODE=true
PYTHONPATH=${PYTHONPATH}:${PWD}/evalx_sdk:${PWD}/evalx_cli

# Service configuration
EVALX_SERVICE_URL=http://localhost:8000
EVALX_BEARER_TOKEN=your-token-here
```

### Production (Never use these in production!)

```bash
# Only these needed in production
EVALX_SERVICE_URL=https://evalx.example.com
EVALX_BEARER_TOKEN=production-token-here

# DO NOT SET:
# EVALX_DEV_MODE=true  ❌
# PYTHONPATH=...       ❌
```

## Components

### evalx_sdk
- **Purpose:** Context management for eval_id
- **Key Function:** `init_context()` - One line to load eval config
- **Location:** `evalx_sdk/`
- [📖 SDK Documentation](./evalx_sdk/README.md)

### evalx_cli
- **Purpose:** Initialize evaluations via API
- **Main Command:** `evalx init`
- **Location:** `evalx_cli/`
- [📖 CLI Documentation](./evalx_cli/README.md)

### eval-client
- **Purpose:** Run Langfuse experiments with EvalX tracking
- **Integration:** Uses `evalx_sdk` to load context
- **Location:** `eval-client/`
- [📖 Client Documentation](./eval-client/README.md)

### eval-service
- **Purpose:** FastAPI service for managing evals and runs
- **API:** RESTful endpoints for CRUD operations
- **Location:** `eval-service/`
- [📖 Service Documentation](./eval-service/README.md)

## Workflow Example

```bash
# Terminal 1: Start service
cd eval-service
docker compose up -d
poetry run uvicorn app.main:app --reload

# Terminal 2: Initialize and run evaluation
cd evalX
source .env

# Initialize
./evalx init --asset-id 123 --name "Test Eval"
# Output: ✅ Evaluation initialized successfully!
#         Eval ID: 550e8400-e29b-41d4-a716-446655440000
#         Config saved to: .evalx/config.json

# Run evaluation
cd eval-client
source .env
poetry run python eval.py
# Output: ✅ EvalX context loaded: eval_id=550e8400-...
#         ✅ Langfuse configured
#         [Experiment runs...]
```

## Test Mode (CLI Only)

For testing CLI without the service:

```bash
./evalx init --asset-id 123 --name "Test" --test-mode
```

This uses a mock client and generates a fake eval_id without connecting to the service. Hidden flag for testing only.

## Security & Best Practices

### ✅ DO

- Use `.env.sample` as templates
- Set different secrets per environment
- Use proper package installation in production
- Keep `.env` files in `.gitignore`

### ❌ DON'T

- Commit `.env` files to git
- Use `EVALX_DEV_MODE=true` in production
- Use `PYTHONPATH` modifications in production
- Share secrets via email/chat

## Requirements

- Python 3.11.9+
- Poetry for dependency management
- Docker & Docker Compose (for PostgreSQL)
- API keys: Langfuse, OpenAI (for eval-client)

## Troubleshooting

### "Building a package is not possible in non-package mode"

**Cause:** Trying to install packages with path dependencies when `package-mode = false`

**Solution:** Use `PYTHONPATH` in development mode:
```bash
source .env  # Loads PYTHONPATH
```

### "./evalx: EVALX_DEV_MODE not set"

**Cause:** Running `./evalx` without development mode flag

**Solution:**
```bash
# Make sure .env exists in evalX root
cd ~/Code/ged/evalX
cp .env.sample .env  # If not already done
```

### "401 Unauthorized" when using evalx init

**Cause:** Bearer token mismatch between CLI and service

**Solution:**
```bash
# Check token in eval-service
cat eval-service/.env | grep BEARER_SECRET

# Update evalX root .env to match
# In evalX/.env:
EVALX_BEARER_TOKEN=test123  # Or whatever the service uses
```

### "ModuleNotFoundError: No module named 'evalx_sdk'"

**Cause:** PYTHONPATH not set when running eval.py

**Solution:**
```bash
cd eval-client
source .env  # This sets PYTHONPATH
poetry run python eval.py
```

### "Cannot connect to EvalX service"

**Cause:** Service not running

**Solution:**
```bash
cd eval-service
docker compose up -d
poetry run uvicorn app.main:app --reload
```

### "No EvalX config found"

**Cause:** Haven't run `evalx init` yet

**Solution:**
```bash
./evalx init --asset-id 123 --name "My Eval"
```

## Documentation

- [EvalX SDK Documentation](./evalx_sdk/README.md)
- [EvalX CLI Documentation](./evalx_cli/README.md)
- [Eval Client Documentation](./eval-client/README.md)
- [Eval Service Documentation](./eval-service/README.md)

## Contributing

When contributing:
1. Never commit `.env` files or secrets
2. Test in development mode first
3. Update relevant README files
4. Follow existing code patterns
5. Use Python 3.11.9 and Poetry

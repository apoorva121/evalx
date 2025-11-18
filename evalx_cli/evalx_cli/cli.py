"""CLI commands for EvalX."""

import os
import sys
from pathlib import Path
from typing import Optional

import click

from evalx_cli.client import EvalXClient
from evalx_cli.config import get_settings


@click.group()
@click.version_option(version="0.1.0")
def main():
    """EvalX CLI - Initialize and manage evaluations."""
    pass


@main.command()
@click.option(
    "--asset-id",
    type=int,
    required=True,
    help="Unique asset identifier for the evaluation",
)
@click.option(
    "--name",
    type=str,
    required=True,
    help="Name of the evaluation",
)
@click.option(
    "--description",
    type=str,
    default=None,
    help="Optional description of the evaluation",
)
@click.option(
    "--service-url",
    type=str,
    default=None,
    help="EvalX service URL (default: from env or http://localhost:8000)",
)
@click.option(
    "--bearer-token",
    type=str,
    default=None,
    help="Bearer token for authentication (default: from env)",
)
@click.option(
    "--env",
    "env_name",
    type=click.Choice(["local", "e2e", "prod"], case_sensitive=False),
    default="local",
    help="Environment name (default: local)",
)
@click.option(
    "--mode",
    type=click.Choice(["local", "ci", "prod"], case_sensitive=False),
    default="local",
    help="Execution mode (default: local)",
)
@click.option(
    "--setup-env",
    is_flag=True,
    default=False,
    help="Automatically setup .env file without prompting",
)
@click.option(
    "--test-mode",
    is_flag=True,
    hidden=True,
    help="Run in test mode without connecting to service (for testing only)",
)
def init(
    asset_id: int,
    name: str,
    description: Optional[str],
    service_url: Optional[str],
    bearer_token: Optional[str],
    env_name: str,
    mode: str,
    setup_env: bool,
    test_mode: bool,
):
    """
    Initialize a new evaluation and save config locally.

    Creates or retrieves an evaluation from EvalX service and saves
    the eval_id to .evalx/config.json for use in subsequent runs.
    """
    settings = get_settings()

    # Use mock client only if explicitly requested via hidden flag
    if test_mode:
        from evalx_cli.mock_client import MockEvalXClient

        client = MockEvalXClient(settings.evalx_service_url, settings.evalx_bearer_token)
        click.echo("⚠️  Running in test mode (mock client)")
    else:
        client = EvalXClient(settings.evalx_service_url, settings.evalx_bearer_token)

        # Health check
        click.echo(f"Connecting to EvalX service at {settings.evalx_service_url}...")
        if not client.health_check():
            click.echo(
                f"❌ Error: Cannot connect to EvalX service at {settings.evalx_service_url}",
                err=True,
            )
            click.echo(
                "   Make sure the service is running or check EVALX_SERVICE_URL", err=True
            )
            sys.exit(1)

    # Create or get eval
    try:
        click.echo(f"Initializing evaluation for asset_id={asset_id}...")
        eval_response = client.create_or_get_eval(
            asset_id=asset_id, name=name, description=description
        )

        # Save config locally using SDK
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "evalx_sdk"))
        from evalx_sdk.context import EvalContext, save_eval_config

        context = EvalContext(
            eval_id=eval_response.id,
            asset_id=eval_response.asset_id,
            name=eval_response.name,
            description=eval_response.description,
        )

        # Save config in the directory where evalx was called from
        original_cwd = Path.cwd()
        config_path = save_eval_config(context)

        click.echo(f"✅ Evaluation initialized successfully!")
        click.echo(f"   Eval ID: {eval_response.id}")
        click.echo(f"   Asset ID: {eval_response.asset_id}")
        click.echo(f"   Name: {eval_response.name}")
        if eval_response.description:
            click.echo(f"   Description: {eval_response.description}")
        click.echo(f"   Config saved to: {config_path}")
        click.echo()

        # Setup .env file if requested
        env_path = Path.cwd() / ".env"
        if not test_mode:
            # Use provided values or defaults
            final_service_url = service_url or settings.evalx_service_url
            final_bearer_token = bearer_token or settings.evalx_bearer_token

            if setup_env or click.confirm(
                "📝 Would you like to setup/update .env file for run tracking?",
                default=True,
            ):
                setup_env_file(
                    env_path=env_path,
                    service_url=final_service_url,
                    bearer_token=final_bearer_token,
                    env_name=env_name,
                    mode=mode,
                    interactive=not setup_env,
                )

    except Exception as e:
        click.echo(f"❌ Error: {str(e)}", err=True)
        sys.exit(1)


def setup_env_file(
    env_path: Path,
    service_url: str,
    bearer_token: Optional[str],
    env_name: str,
    mode: str,
    interactive: bool = True,
):
    """Setup or update .env file with EvalX configuration."""
    click.echo()
    click.echo("Setting up EvalX service configuration...")
    click.echo()

    # Read existing .env if it exists
    existing_env = {}
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    existing_env[key.strip()] = value.strip()

    # Get values interactively or use defaults
    if interactive:
        service_url = click.prompt(
            "EvalX Service URL", default=service_url, show_default=True
        )

        default_token = bearer_token or existing_env.get("EVALX_BEARER_TOKEN", "test123")
        bearer_token = click.prompt(
            "Bearer Token", default=default_token, show_default=True
        )

        env_name = click.prompt(
            "Environment (local/e2e/prod)", default=env_name, show_default=True
        )

        mode = click.prompt(
            "Mode (local/ci/prod)", default=mode, show_default=True
        )
    else:
        # Use provided values or fallback to existing
        if not bearer_token:
            bearer_token = existing_env.get("EVALX_BEARER_TOKEN", "test123")

    # Update or create .env file
    evalx_config = f"""
# EvalX Service Configuration (for run tracking)
EVALX_SERVICE_URL={service_url}
EVALX_BEARER_TOKEN={bearer_token}
EVALX_ENV={env_name}
EVALX_MODE={mode}
"""

    if env_path.exists():
        # Update existing file
        with open(env_path, "r") as f:
            content = f.read()

        # Remove existing EVALX_ configuration
        lines = content.split("\n")
        new_lines = []
        skip_section = False
        for line in lines:
            if line.strip().startswith("# EvalX Service Configuration"):
                skip_section = True
                continue
            if skip_section and line.strip().startswith("EVALX_"):
                continue
            if skip_section and (line.strip() == "" or not line.strip().startswith("EVALX_")):
                skip_section = False
            if not skip_section:
                new_lines.append(line)

        # Add new configuration at the end
        content = "\n".join(new_lines).rstrip() + evalx_config

        with open(env_path, "w") as f:
            f.write(content)
        click.echo(f"✅ Updated {env_path}")
    else:
        # Create new file
        with open(env_path, "w") as f:
            f.write(evalx_config.lstrip())
        click.echo(f"✅ Created {env_path}")

    click.echo()
    click.echo("📝 Next steps:")
    click.echo("   1. Source the .env file: source .env")
    click.echo("   2. Run your evaluation script with run tracking enabled")


if __name__ == "__main__":
    main()

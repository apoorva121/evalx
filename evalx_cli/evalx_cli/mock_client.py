"""Mock client for testing without EvalX service."""

import uuid
from datetime import datetime, timezone
from typing import Optional

from evalx_cli.client import EvalResponse


class MockEvalXClient:
    """Mock client for testing without EvalX service."""

    def __init__(self, base_url: str, bearer_token: Optional[str] = None):
        """
        Initialize mock EvalX client.

        Args:
            base_url: Base URL (ignored in mock mode)
            bearer_token: Bearer token (ignored in mock mode)
        """
        self.base_url = base_url
        self.bearer_token = bearer_token

    def create_or_get_eval(
        self, asset_id: int, name: str, description: Optional[str] = None
    ) -> EvalResponse:
        """
        Mock create evaluation - generates fake response.

        Args:
            asset_id: Unique asset identifier
            name: Name of the evaluation
            description: Optional description

        Returns:
            EvalResponse with mocked eval details
        """
        now = datetime.now(timezone.utc).isoformat()
        return EvalResponse(
            id=str(uuid.uuid4()),
            asset_id=asset_id,
            name=name,
            description=description,
            created_at=now,
            updated_at=now,
        )

    def health_check(self) -> bool:
        """Mock health check - always returns True."""
        return True

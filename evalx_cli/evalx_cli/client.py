"""API client for EvalX service."""

from typing import Dict, Optional

import requests
from pydantic import BaseModel


class EvalCreateRequest(BaseModel):
    """Request schema for creating an evaluation."""

    asset_id: int
    name: str
    description: Optional[str] = None


class EvalResponse(BaseModel):
    """Response schema for evaluation."""

    id: str
    asset_id: int
    name: str
    description: Optional[str]
    created_at: str
    updated_at: str


class EvalXClient:
    """Client for interacting with EvalX service API."""

    def __init__(self, base_url: str, bearer_token: Optional[str] = None):
        """
        Initialize EvalX API client.

        Args:
            base_url: Base URL of the EvalX service
            bearer_token: Optional bearer token for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token

    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def create_or_get_eval(
        self, asset_id: int, name: str, description: Optional[str] = None
    ) -> EvalResponse:
        """
        Create a new evaluation or get existing one by asset_id.

        Args:
            asset_id: Unique asset identifier
            name: Name of the evaluation
            description: Optional description

        Returns:
            EvalResponse with eval details

        Raises:
            requests.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v1/evals"
        payload = EvalCreateRequest(
            asset_id=asset_id, name=name, description=description
        )

        response = requests.post(
            url, json=payload.model_dump(), headers=self._get_headers(), timeout=10
        )
        response.raise_for_status()

        return EvalResponse(**response.json())

    def health_check(self) -> bool:
        """
        Check if the EvalX service is healthy.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            url = f"{self.base_url}/health"
            response = requests.get(url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

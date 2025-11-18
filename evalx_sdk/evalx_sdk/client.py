"""API client for EvalX service - Run tracking."""

from datetime import datetime
from typing import Any, Dict, Literal, Optional

import requests
from pydantic import BaseModel


Env = Literal["local", "e2e", "prod"]
Mode = Literal["local", "ci", "prod"]
Status = Literal["running", "success", "fail"]


class RunStartRequest(BaseModel):
    """Request schema for starting a run."""

    eval_id: str
    asset_id: int
    env: Env
    mode: Mode
    metadata: Dict[str, Any] = {}


class RunStartResponse(BaseModel):
    """Response schema for run start."""

    run_id: str
    status: Status
    started_at: datetime


class RunFinishRequest(BaseModel):
    """Request schema for finishing a run."""

    status: Status
    ended_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


class RunFinishResponse(BaseModel):
    """Response schema for run finish."""

    run_id: str
    eval_id: str
    asset_id: int
    env: Env
    mode: Mode
    status: Status
    started_at: datetime
    ended_at: Optional[datetime]
    metadata: Dict[str, Any]


class EvalXRunClient:
    """Client for interacting with EvalX run tracking API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        bearer_token: Optional[str] = None,
    ):
        """
        Initialize EvalX run tracking client.

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

    def start_run(
        self,
        eval_id: str,
        asset_id: int,
        env: Env = "local",
        mode: Mode = "local",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RunStartResponse:
        """
        Start a new evaluation run.

        Args:
            eval_id: UUID of the evaluation
            asset_id: Asset ID associated with the evaluation
            env: Environment (local, e2e, prod)
            mode: Mode (local, ci, prod)
            metadata: Optional metadata for the run

        Returns:
            RunStartResponse with run_id, status, and started_at

        Raises:
            requests.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v1/runs"
        payload = RunStartRequest(
            eval_id=eval_id,
            asset_id=asset_id,
            env=env,
            mode=mode,
            metadata=metadata or {},
        )

        response = requests.post(
            url, json=payload.model_dump(), headers=self._get_headers(), timeout=10
        )
        response.raise_for_status()

        return RunStartResponse(**response.json())

    def finish_run(
        self,
        run_id: str,
        status: Literal["success", "fail"],
        ended_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RunFinishResponse:
        """
        Finish an evaluation run.

        Args:
            run_id: UUID of the run to finish
            status: Final status (success or fail)
            ended_at: Optional end timestamp (defaults to current time)
            metadata: Optional metadata to merge with existing run metadata

        Returns:
            RunFinishResponse with complete run details

        Raises:
            requests.HTTPError: If the API request fails
        """
        url = f"{self.base_url}/v1/runs/{run_id}"
        payload = RunFinishRequest(
            status=status, ended_at=ended_at, metadata=metadata
        )

        response = requests.put(
            url,
            json=payload.model_dump(mode="json", exclude_none=True),
            headers=self._get_headers(),
            timeout=10,
        )
        response.raise_for_status()

        return RunFinishResponse(**response.json())

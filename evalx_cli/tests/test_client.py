"""Tests for evalx_cli.client module."""

import pytest
import responses

from evalx_cli.client import EvalCreateRequest, EvalResponse, EvalXClient


class TestEvalCreateRequest:
    """Test EvalCreateRequest model."""

    def test_create_request_with_required_fields(self):
        """Test creating request with required fields only."""
        request = EvalCreateRequest(asset_id=123, name="Test Eval")

        assert request.asset_id == 123
        assert request.name == "Test Eval"
        assert request.description is None

    def test_create_request_with_description(self):
        """Test creating request with description."""
        request = EvalCreateRequest(
            asset_id=456, name="Test Eval", description="Test description"
        )

        assert request.asset_id == 456
        assert request.name == "Test Eval"
        assert request.description == "Test description"


class TestEvalResponse:
    """Test EvalResponse model."""

    def test_eval_response_parsing(self, eval_response_data):
        """Test parsing EvalResponse from JSON data."""
        response = EvalResponse(**eval_response_data)

        assert response.id == eval_response_data["id"]
        assert response.asset_id == eval_response_data["asset_id"]
        assert response.name == eval_response_data["name"]
        assert response.description == eval_response_data["description"]
        assert response.created_at == eval_response_data["created_at"]
        assert response.updated_at == eval_response_data["updated_at"]


class TestEvalXClient:
    """Test EvalXClient API interactions."""

    def test_init_with_defaults(self):
        """Test client initialization with defaults."""
        client = EvalXClient(base_url="http://test.com")

        assert client.base_url == "http://test.com"
        assert client.bearer_token is None

    def test_init_with_bearer_token(self):
        """Test client initialization with bearer token."""
        client = EvalXClient(base_url="http://test.com", bearer_token="token123")

        assert client.base_url == "http://test.com"
        assert client.bearer_token == "token123"

    def test_init_strips_trailing_slash(self):
        """Test that trailing slash is removed from base_url."""
        client = EvalXClient(base_url="http://test.com/")

        assert client.base_url == "http://test.com"

    def test_get_headers_without_token(self):
        """Test headers without bearer token."""
        client = EvalXClient(base_url="http://test.com")
        headers = client._get_headers()

        assert headers == {"Content-Type": "application/json"}

    def test_get_headers_with_token(self):
        """Test headers with bearer token."""
        client = EvalXClient(base_url="http://test.com", bearer_token="token123")
        headers = client._get_headers()

        assert headers == {
            "Content-Type": "application/json",
            "Authorization": "Bearer token123",
        }

    @responses.activate
    def test_create_or_get_eval_success(self, eval_response_data):
        """Test successful eval creation."""
        client = EvalXClient(base_url="http://test.com", bearer_token="token123")

        # Mock API response
        responses.post(
            "http://test.com/v1/evals",
            json=eval_response_data,
            status=200,
        )

        response = client.create_or_get_eval(
            asset_id=123, name="Test Eval", description="Test description"
        )

        assert response.id == eval_response_data["id"]
        assert response.asset_id == eval_response_data["asset_id"]
        assert response.name == eval_response_data["name"]
        assert response.description == eval_response_data["description"]

        # Verify request
        assert len(responses.calls) == 1
        request = responses.calls[0].request
        assert request.headers["Authorization"] == "Bearer token123"
        assert request.headers["Content-Type"] == "application/json"

    @responses.activate
    def test_create_or_get_eval_without_description(self, eval_response_data):
        """Test eval creation without description."""
        client = EvalXClient(base_url="http://test.com")

        # Mock API response
        eval_data = eval_response_data.copy()
        eval_data["description"] = None
        responses.post(
            "http://test.com/v1/evals",
            json=eval_data,
            status=200,
        )

        response = client.create_or_get_eval(asset_id=123, name="Test Eval")

        assert response.id == eval_data["id"]
        assert response.description is None

    @responses.activate
    def test_create_or_get_eval_api_error(self):
        """Test eval creation with API error."""
        client = EvalXClient(base_url="http://test.com")

        # Mock API error
        responses.post(
            "http://test.com/v1/evals",
            json={"error": "Server error"},
            status=500,
        )

        with pytest.raises(Exception):  # requests.HTTPError
            client.create_or_get_eval(asset_id=123, name="Test Eval")

    @responses.activate
    def test_health_check_success(self):
        """Test successful health check."""
        client = EvalXClient(base_url="http://test.com")

        # Mock health endpoint
        responses.get(
            "http://test.com/health",
            json={"status": "healthy"},
            status=200,
        )

        result = client.health_check()

        assert result is True

    @responses.activate
    def test_health_check_failure(self):
        """Test health check with service down."""
        client = EvalXClient(base_url="http://test.com")

        # Mock health endpoint failure
        responses.get(
            "http://test.com/health",
            json={"status": "unhealthy"},
            status=500,
        )

        result = client.health_check()

        assert result is False

    def test_health_check_connection_error(self):
        """Test health check with connection error."""
        client = EvalXClient(base_url="http://nonexistent.invalid")

        result = client.health_check()

        assert result is False

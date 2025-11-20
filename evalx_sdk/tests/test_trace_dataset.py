"""Comprehensive tests for trace_dataset module with full endpoint mocking."""

import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch

import pytest
from pydantic import ValidationError
from langfuse.experiment import Evaluation

from evalx_sdk.trace_dataset import (
    ObservationData,
    TraceCore,
    TraceData,
    TraceDataset,
    TraceIO,
    TraceMetadata,
    TraceToDatasetBuilder,
)


# ============================================================================
# MOCK HELPERS
# ============================================================================


class MockTrace:
    """Mock trace object that mimics Langfuse trace structure."""

    def __init__(self, trace_id, **kwargs):
        self.id = str(trace_id)  # Ensure ID is string
        self.name = kwargs.get("name", f"trace-{trace_id}")
        self.user_id = kwargs.get("user_id", f"user-{trace_id}")
        self.session_id = kwargs.get("session_id", f"session-{trace_id}")
        self.timestamp = kwargs.get("timestamp", datetime.now())
        self.release = kwargs.get("release", "v1.0")
        self.version = kwargs.get("version", "1.0.0")
        self.tags = kwargs.get("tags", [])
        self.input = kwargs.get("input", {"question": f"Q{trace_id}"})
        self.output = kwargs.get("output", {"answer": f"A{trace_id}"})
        self.metadata = kwargs.get("metadata", {})


class MockObservation:
    """Mock observation object that mimics Langfuse observation structure."""

    def __init__(self, obs_id, **kwargs):
        self.id = obs_id
        self.type = kwargs.get("type", "GENERATION")
        self.name = kwargs.get("name", f"obs-{obs_id}")
        self.input = kwargs.get("input", f"input-{obs_id}")
        self.output = kwargs.get("output", f"output-{obs_id}")
        self.metadata = kwargs.get("metadata", {})
        self.start_time = kwargs.get("start_time", datetime.now())
        self.end_time = kwargs.get("end_time", datetime.now())


class MockLangfuseClient:
    """Comprehensive mock of Langfuse client with configurable responses."""

    def __init__(self, num_traces=10, traces_per_page=5):
        self.num_traces = num_traces
        self.traces_per_page = traces_per_page
        self.traces = [MockTrace(i) for i in range(num_traces)]
        self.observations_per_trace = 2

        # Track API calls
        self.trace_list_calls = []
        self.observation_calls = []
        self.dataset_create_calls = []
        self.dataset_item_calls = []
        self.score_calls = []
        self.flush_calls = []

        # Mock API structure
        self.api = Mock()
        self.api.trace = Mock()
        self.api.trace.list = Mock(side_effect=self._mock_trace_list)
        self.api.observations = Mock()
        self.api.observations.get_many = Mock(side_effect=self._mock_get_observations)

        # Mock dataset operations
        self.create_dataset = Mock(side_effect=self._mock_create_dataset)
        self.create_dataset_item = Mock(side_effect=self._mock_create_dataset_item)
        self.get_dataset = Mock(side_effect=self._mock_get_dataset)
        self.run_experiment = Mock(side_effect=self._mock_run_experiment)

        # Mock evaluation operations (new for batch processing)
        self.create_score = Mock(side_effect=self._mock_create_score)
        self.flush = Mock(side_effect=self._mock_flush)

    def _mock_trace_list(self, **kwargs):
        """Mock trace.list endpoint with pagination."""
        self.trace_list_calls.append(kwargs)

        page = kwargs.get("page", 1)
        limit = kwargs.get("limit", self.traces_per_page)

        start_idx = (page - 1) * limit
        end_idx = start_idx + limit

        page_traces = self.traces[start_idx:end_idx]

        response = Mock()
        response.data = page_traces
        return response

    def _mock_get_observations(self, **kwargs):
        """Mock observations.get_many endpoint."""
        self.observation_calls.append(kwargs)

        trace_id = kwargs["trace_id"]
        obs_type = kwargs.get("type")
        obs_limit = kwargs.get("limit", 100)

        # Generate observations for this trace
        observations = []
        for i in range(self.observations_per_trace):
            obs = MockObservation(
                f"{trace_id}-obs-{i}",
                type=obs_type or "GENERATION",
                name=f"step-{i}"
            )
            observations.append(obs)

        # Respect limit
        observations = observations[:obs_limit]

        response = Mock()
        response.data = observations
        return response

    def _mock_create_dataset(self, **kwargs):
        """Mock dataset creation."""
        self.dataset_create_calls.append(kwargs)
        # Returns nothing - just creates the dataset
        return None

    def _mock_create_dataset_item(self, **kwargs):
        """Mock dataset item creation."""
        self.dataset_item_calls.append(kwargs)
        return Mock()  # Return a mock dataset item

    def _mock_get_dataset(self, **kwargs):
        """Mock get_dataset to return a DatasetClient."""
        mock_dataset = Mock()
        mock_dataset.run_experiment = Mock(return_value=Mock(format=lambda: "Result"))
        return mock_dataset

    def _mock_run_experiment(self, **kwargs):
        """Mock run_experiment directly on client."""
        # Return a mock result with format method
        mock_result = Mock()
        mock_result.format = lambda: "Experiment Result"
        return mock_result

    def _mock_create_item(self, **kwargs):
        """Mock dataset item creation (deprecated method)."""
        self.dataset_item_calls.append(kwargs)

    def _mock_create_score(self, **kwargs):
        """Mock score creation for evaluations."""
        self.score_calls.append(kwargs)
        return None

    def _mock_flush(self):
        """Mock flush operation."""
        self.flush_calls.append(True)
        return None


@pytest.fixture
def mock_langfuse_factory():
    """Factory fixture for creating mock Langfuse clients."""
    def _create_mock(num_traces=10, traces_per_page=5):
        return MockLangfuseClient(num_traces=num_traces, traces_per_page=traces_per_page)
    return _create_mock


# ============================================================================
# UNIT TESTS - Pydantic Models
# ============================================================================


class TestPydanticModels:
    """Test Pydantic model validation and construction."""

    def test_trace_core_all_fields(self):
        """Test TraceCore with all fields."""
        now = datetime.now()
        core = TraceCore(
            id="trace-123",
            name="test-trace",
            user_id="user-456",
            session_id="session-789",
            timestamp=now,
            release="v1.0",
            version="1.0.0",
            tags=["production", "test"],
        )

        assert core.id == "trace-123"
        assert core.name == "test-trace"
        assert core.user_id == "user-456"
        assert core.session_id == "session-789"
        assert core.timestamp == now
        assert core.release == "v1.0"
        assert core.version == "1.0.0"
        assert len(core.tags) == 2
        assert "production" in core.tags

    def test_trace_core_minimal(self):
        """Test TraceCore with minimal fields."""
        core = TraceCore()
        assert core.id is None
        assert core.name is None
        assert core.tags is None

    def test_trace_io_complex_data(self):
        """Test TraceIO with complex nested data."""
        io = TraceIO(
            input={"question": "What is 2+2?", "context": ["math", "basic"]},
            output={"answer": "4", "confidence": 0.99}
        )

        assert io.input["question"] == "What is 2+2?"
        assert len(io.input["context"]) == 2
        assert io.output["confidence"] == 0.99

    def test_trace_metadata_nested(self):
        """Test TraceMetadata with nested structures."""
        metadata = TraceMetadata(
            metadata={
                "env": "prod",
                "version": "1.0",
                "config": {"model": "gpt-4", "temp": 0.7}
            }
        )

        assert metadata.metadata["env"] == "prod"
        assert metadata.metadata["config"]["model"] == "gpt-4"

    def test_trace_data_complete(self):
        """Test complete TraceData with all components."""
        core = TraceCore(id="trace-123", name="test", tags=["prod"])
        io = TraceIO(input="test input", output="test output")
        metadata = TraceMetadata(metadata={"key": "value"})

        trace_data = TraceData(core=core, io=io, metadata=metadata)

        assert trace_data.core.id == "trace-123"
        assert trace_data.core.tags == ["prod"]
        assert trace_data.io.input == "test input"
        assert trace_data.metadata.metadata["key"] == "value"

    def test_observation_data_complete(self):
        """Test ObservationData with all fields."""
        start = datetime.now()
        end = start + timedelta(seconds=2)

        obs = ObservationData(
            id="obs-123",
            type="GENERATION",
            name="completion",
            input="prompt text",
            output="response text",
            metadata={"model": "gpt-4", "tokens": 150},
            start_time=start,
            end_time=end,
        )

        assert obs.id == "obs-123"
        assert obs.type == "GENERATION"
        assert obs.name == "completion"
        assert obs.metadata["tokens"] == 150
        assert (obs.end_time - obs.start_time).seconds == 2


# ============================================================================
# UNIT TESTS - TraceToDatasetBuilder
# ============================================================================


class TestTraceToDatasetBuilder:
    """Test TraceToDatasetBuilder configuration and validation."""

    def test_builder_defaults(self, mock_langfuse_factory):
        """Test builder initialization with default values."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        assert builder.langfuse == client
        assert builder.page_size == 50
        assert builder._fields == ["core", "io"]
        assert builder._fetch_observations is False
        assert builder._max_traces is None
        assert builder._obs_type is None
        assert builder._trace_filter is None
        assert builder._trace_processor is None
        assert builder._sample_rate == 0.01  # Default 1% sampling
        assert builder._batch_size == 10  # Default batch size

    def test_builder_custom_params(self, mock_langfuse_factory):
        """Test builder with custom initialization parameters."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(
            client,
            default_time_range_days=30,
            page_size=100
        )

        assert builder.page_size == 100
        time_diff = builder._end_time - builder._start_time
        assert 29 <= time_diff.days <= 31  # Allow some tolerance

    def test_with_fields_valid_subset(self, mock_langfuse_factory):
        """Test with_fields accepts valid field combinations."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        # Test all valid combinations
        builder.with_fields(["core"])
        assert set(builder._fields) == {"core"}

        builder.with_fields(["core", "io"])
        assert set(builder._fields) == {"core", "io"}

        builder.with_fields(["core", "io", "metadata"])
        assert set(builder._fields) == {"core", "io", "metadata"}

        builder.with_fields(["metadata"])
        assert set(builder._fields) == {"metadata"}

    def test_with_fields_forbidden_scores(self, mock_langfuse_factory):
        """Test with_fields rejects 'scores' field."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        with pytest.raises(ValueError) as exc_info:
            builder.with_fields(["core", "scores"])

        assert "Forbidden fields detected" in str(exc_info.value)
        assert "scores" in str(exc_info.value)

    def test_with_fields_forbidden_observations(self, mock_langfuse_factory):
        """Test with_fields rejects 'observations' field."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        with pytest.raises(ValueError) as exc_info:
            builder.with_fields(["observations"])

        assert "Forbidden fields detected" in str(exc_info.value)

    def test_with_fields_forbidden_metrics(self, mock_langfuse_factory):
        """Test with_fields rejects 'metrics' field."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        with pytest.raises(ValueError) as exc_info:
            builder.with_fields(["metrics", "core"])

        assert "Forbidden fields detected" in str(exc_info.value)

    def test_with_fields_invalid(self, mock_langfuse_factory):
        """Test with_fields rejects completely invalid fields."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        with pytest.raises(ValueError) as exc_info:
            builder.with_fields(["invalid_field", "another_bad_field"])

        assert "Invalid fields" in str(exc_info.value)

    def test_with_time_range_days_parameter(self, mock_langfuse_factory):
        """Test with_time_range using days parameter."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        result = builder.with_time_range(days=7)

        assert result is builder  # Test chaining
        time_diff = builder._end_time - builder._start_time
        assert 6 <= time_diff.days <= 7

    def test_with_time_range_explicit_dates(self, mock_langfuse_factory):
        """Test with_time_range with explicit start and end times."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        start = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 31, 23, 59, 59)

        result = builder.with_time_range(start_time=start, end_time=end)

        assert result is builder
        assert builder._start_time == start
        assert builder._end_time == end

    def test_with_time_range_start_only(self, mock_langfuse_factory):
        """Test with_time_range with only start time."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        original_end = builder._end_time
        start = datetime(2024, 1, 1)

        builder.with_time_range(start_time=start)

        assert builder._start_time == start
        # End time should remain unchanged
        assert builder._end_time == original_end

    def test_with_time_range_hours(self, mock_langfuse_factory):
        """Test with_time_range with hours parameter."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        from datetime import datetime, timedelta

        before = datetime.now()
        result = builder.with_time_range(hours=1)
        after = datetime.now()

        assert result is builder
        # Start time should be approximately 1 hour ago
        expected_start = datetime.now() - timedelta(hours=1)
        # Allow 1 second tolerance for test execution time
        assert abs((builder._start_time - expected_start).total_seconds()) < 1
        # End time should be approximately now
        assert abs((builder._end_time - datetime.now()).total_seconds()) < 1

    def test_with_trace_filter_lambda(self, mock_langfuse_factory):
        """Test with_trace_filter accepts lambda function."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        filter_fn = lambda trace: trace.core.user_id == "user-123"
        result = builder.with_trace_filter(filter_fn)

        assert result is builder
        assert builder._trace_filter == filter_fn

    def test_with_trace_processor_custom(self, mock_langfuse_factory):
        """Test with_trace_processor with custom transformation."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        def processor(trace):
            return {
                "id": trace.core.id,
                "custom_field": "value"
            }

        result = builder.with_trace_processor(processor)

        assert result is builder
        assert builder._trace_processor == processor

    def test_with_observations_all_params(self, mock_langfuse_factory):
        """Test with_observations with all parameters."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        obs_filter = lambda obs: obs.type == "GENERATION"

        result = builder.with_observations(
            fetch=True,
            obs_type="GENERATION",
            obs_limit=25,
            obs_filter=obs_filter
        )

        assert result is builder
        assert builder._fetch_observations is True
        assert builder._obs_type == "GENERATION"
        assert builder._obs_limit == 25
        assert builder._obs_filter == obs_filter

    def test_with_observations_fetch_false(self, mock_langfuse_factory):
        """Test with_observations can disable observation fetching."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        # Enable first
        builder.with_observations(fetch=True)
        assert builder._fetch_observations is True

        # Then disable
        builder.with_observations(fetch=False)
        assert builder._fetch_observations is False

    def test_with_limits_both(self, mock_langfuse_factory):
        """Test with_limits sets both trace and observation limits."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        result = builder.with_limits(
            max_traces=100,
            max_observations_per_trace=5
        )

        assert result is builder
        assert builder._max_traces == 100
        assert builder._max_observations_per_trace == 5

    def test_with_limits_traces_only(self, mock_langfuse_factory):
        """Test with_limits with only trace limit."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        builder.with_limits(max_traces=50)

        assert builder._max_traces == 50
        assert builder._max_observations_per_trace is None

    def test_with_trace_filters_multiple(self, mock_langfuse_factory):
        """Test with_trace_filters with multiple API parameters."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        result = builder.with_trace_filters(
            user_id="user-123",
            session_id="session-456",
            tags=["production", "v2"],
            name="my-trace"
        )

        assert result is builder
        assert builder._trace_list_filters["user_id"] == "user-123"
        assert builder._trace_list_filters["session_id"] == "session-456"
        assert builder._trace_list_filters["tags"] == ["production", "v2"]
        assert builder._trace_list_filters["name"] == "my-trace"

    def test_with_trace_filters_cumulative(self, mock_langfuse_factory):
        """Test with_trace_filters accumulates filters."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        builder.with_trace_filters(user_id="user-123")
        builder.with_trace_filters(tags=["prod"])

        assert builder._trace_list_filters["user_id"] == "user-123"
        assert builder._trace_list_filters["tags"] == ["prod"]

    def test_with_batch_size_custom(self, mock_langfuse_factory):
        """Test with_batch_size sets custom batch size."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        result = builder.with_batch_size(15)

        assert result is builder
        assert builder._batch_size == 15

    def test_with_sample_rate_custom(self, mock_langfuse_factory):
        """Test with_sample_rate sets custom sampling rate."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        result = builder.with_sample_rate(0.1)

        assert result is builder
        assert builder._sample_rate == 0.1

    def test_with_sample_rate_clamps_values(self, mock_langfuse_factory):
        """Test with_sample_rate clamps values to [0, 1] range."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        # Test value > 1.0 gets clamped to 1.0
        builder.with_sample_rate(1.5)
        assert builder._sample_rate == 1.0

        # Test value < 0.0 gets clamped to 0.0
        builder.with_sample_rate(-0.1)
        assert builder._sample_rate == 0.0

    def test_build_creates_dataset(self, mock_langfuse_factory):
        """Test build returns TraceDataset instance."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)

        dataset = builder.build()

        assert isinstance(dataset, TraceDataset)
        assert dataset.langfuse == client
        assert dataset.page_size == 50

    def test_build_with_max_traces_override(self, mock_langfuse_factory):
        """Test build accepts max_traces override."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)
        builder.with_limits(max_traces=100)

        dataset = builder.build(max_traces=50)

        assert dataset.max_traces == 50  # Override should win

    def test_build_with_batch_size_override(self, mock_langfuse_factory):
        """Test build accepts batch_size override."""
        client = mock_langfuse_factory()
        builder = TraceToDatasetBuilder(client)
        builder.with_batch_size(20)

        dataset = builder.build(batch_size=10)

        assert dataset.batch_size == 10  # Override should win

    def test_complete_chaining(self, mock_langfuse_factory):
        """Test complete method chaining in realistic scenario."""
        client = mock_langfuse_factory()

        dataset = (TraceToDatasetBuilder(client, page_size=25)
            .with_fields(["core", "io", "metadata"])
            .with_time_range(days=14)
            .with_trace_filters(user_id="user-123", tags=["prod"])
            .with_trace_filter(lambda t: t.core.session_id is not None)
            .with_observations(
                fetch=True,
                obs_type="GENERATION",
                obs_limit=10,
                obs_filter=lambda o: o.name.startswith("step")
            )
            .with_limits(max_traces=500, max_observations_per_trace=5)
            .with_batch_size(20)
            .build()
        )

        assert isinstance(dataset, TraceDataset)
        assert dataset.page_size == 25
        assert dataset.max_traces == 500
        assert dataset.batch_size == 20
        assert dataset.fetch_observations is True


# ============================================================================
# UNIT TESTS - TraceDataset Core Functionality
# ============================================================================


class TestTraceDatasetCore:
    """Test core TraceDataset methods."""

    def test_parse_trace_all_fields(self, mock_langfuse_factory):
        """Test _parse_trace with all fields."""
        client = mock_langfuse_factory()
        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={},
            fields=["core", "io", "metadata"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        mock_trace = MockTrace(
            "trace-123",
            name="test-trace",
            user_id="user-456",
            tags=["prod", "v2"],
            input={"q": "What?"},
            output={"a": "Answer"},
            metadata={"env": "production"}
        )

        trace_data = dataset._parse_trace(mock_trace)

        assert isinstance(trace_data, TraceData)
        assert trace_data.core.id == "trace-123"
        assert trace_data.core.name == "test-trace"
        assert trace_data.core.user_id == "user-456"
        assert "prod" in trace_data.core.tags
        assert trace_data.io.input == {"q": "What?"}
        assert trace_data.io.output == {"a": "Answer"}
        assert trace_data.metadata.metadata["env"] == "production"

    def test_parse_trace_core_only(self, mock_langfuse_factory):
        """Test _parse_trace with only core fields."""
        client = mock_langfuse_factory()
        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={},
            fields=["core"],  # Only core
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        mock_trace = MockTrace("trace-123")
        trace_data = dataset._parse_trace(mock_trace)

        assert trace_data.core is not None
        assert trace_data.io is None
        assert trace_data.metadata is None

    def test_build_dataset_item_default_structure(self, mock_langfuse_factory):
        """Test _build_dataset_item creates default structure."""
        client = mock_langfuse_factory()
        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={},
            fields=["core", "io", "metadata"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        trace = TraceData(
            core=TraceCore(id="trace-1", name="test", tags=["prod"]),
            io=TraceIO(input={"q": "?"}, output={"a": "!"}),
            metadata=TraceMetadata(metadata={"key": "value"})
        )

        item = dataset._build_dataset_item(trace)

        assert item["trace_id"] == "trace-1"
        assert item["trace_name"] == "test"
        assert item["input"] == {"q": "?"}
        assert item["output"] == {"a": "!"}
        assert item["metadata"] == {"key": "value"}
        assert "trace_core" in item
        assert item["trace_core"]["tags"] == ["prod"]

    def test_build_dataset_item_with_observations(self, mock_langfuse_factory):
        """Test _build_dataset_item includes observations."""
        client = mock_langfuse_factory()
        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        trace = TraceData(
            core=TraceCore(id="trace-1"),
            io=TraceIO(input="in", output="out")
        )

        observations = [
            ObservationData(id="obs-1", name="step-1", type="GENERATION"),
            ObservationData(id="obs-2", name="step-2", type="SPAN"),
        ]

        item = dataset._build_dataset_item(trace, observations)

        assert "observations" in item
        assert len(item["observations"]) == 2
        assert item["observations"][0]["id"] == "obs-1"
        assert item["observations"][1]["name"] == "step-2"

    def test_build_dataset_item_custom_processor(self, mock_langfuse_factory):
        """Test _build_dataset_item uses custom processor."""
        client = mock_langfuse_factory()

        def custom_processor(trace):
            return {
                "trace_identifier": trace.core.id,
                "question": trace.io.input,
                "answer": trace.io.output,
                "custom": "field"
            }

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=custom_processor,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        trace = TraceData(
            core=TraceCore(id="trace-1"),
            io=TraceIO(input="question", output="answer")
        )

        item = dataset._build_dataset_item(trace)

        assert item["trace_identifier"] == "trace-1"
        assert item["question"] == "question"
        assert item["answer"] == "answer"
        assert item["custom"] == "field"
        # Default fields should not be present
        assert "trace_id" not in item

    def test_fetch_observations_basic(self, mock_langfuse_factory):
        """Test _fetch_observations retrieves observations."""
        client = mock_langfuse_factory()
        client.observations_per_trace = 3

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=True,
            obs_type="GENERATION",
            obs_limit=10,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        observations = dataset._fetch_observations("trace-1")

        assert len(observations) == 3
        assert all(isinstance(obs, ObservationData) for obs in observations)
        assert observations[0].id == "trace-1-obs-0"

        # Verify API call
        assert len(client.observation_calls) == 1
        assert client.observation_calls[0]["trace_id"] == "trace-1"
        assert client.observation_calls[0]["type"] == "GENERATION"
        assert client.observation_calls[0]["limit"] == 10

    def test_fetch_observations_with_filter(self, mock_langfuse_factory):
        """Test _fetch_observations applies filter."""
        client = mock_langfuse_factory()
        client.observations_per_trace = 5

        obs_filter = lambda obs: obs.name == "step-1"

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=True,
            obs_type=None,
            obs_limit=None,
            obs_filter=obs_filter,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        observations = dataset._fetch_observations("trace-1")

        # Should only get observations matching filter
        assert len(observations) == 1
        assert observations[0].name == "step-1"

    def test_fetch_observations_respects_max_per_trace(self, mock_langfuse_factory):
        """Test _fetch_observations respects max_observations_per_trace."""
        client = mock_langfuse_factory()
        client.observations_per_trace = 10

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=True,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=3,
            batch_size=10,
            sample_rate=1.0,
        )

        observations = dataset._fetch_observations("trace-1")

        assert len(observations) == 3  # Limited by max_observations_per_trace


# ============================================================================
# INTEGRATION TESTS - TraceDataset Iteration
# ============================================================================


class TestTraceDatasetIteration:
    """Test TraceDataset iteration and pagination."""

    def test_iter_single_page(self, mock_langfuse_factory):
        """Test iteration over single page of traces."""
        client = mock_langfuse_factory(num_traces=5, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        items = list(dataset)

        assert len(items) == 5
        assert all("trace_id" in item for item in items)
        assert items[0]["trace_id"] == "0"
        assert items[4]["trace_id"] == "4"

        # Should only call API once (one page)
        assert len(client.trace_list_calls) == 1

    def test_iter_multiple_pages(self, mock_langfuse_factory):
        """Test iteration with pagination across multiple pages."""
        client = mock_langfuse_factory(num_traces=25, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        items = list(dataset)

        assert len(items) == 25
        # Should call API 3 times (3 pages: 10, 10, 5)
        assert len(client.trace_list_calls) == 3

    def test_iter_with_max_traces(self, mock_langfuse_factory):
        """Test iteration stops at max_traces limit."""
        client = mock_langfuse_factory(num_traces=100, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=25,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        items = list(dataset)

        assert len(items) == 25  # Limited by max_traces
        # Should stop after getting 25 traces (3 pages)
        assert len(client.trace_list_calls) == 3

    def test_iter_with_trace_filter(self, mock_langfuse_factory):
        """Test iteration applies trace filter."""
        client = mock_langfuse_factory(num_traces=20, traces_per_page=10)

        # Modify traces to have different tags
        for i, trace in enumerate(client.traces):
            trace.tags = ["production"] if i % 2 == 0 else ["test"]

        trace_filter = lambda trace: "production" in trace.core.tags

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=trace_filter,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        items = list(dataset)

        # Should only get even-numbered traces (10 total)
        assert len(items) == 10
        assert all("production" in item["trace_core"]["tags"] for item in items)

    def test_iter_with_observations(self, mock_langfuse_factory):
        """Test iteration fetches and includes observations."""
        client = mock_langfuse_factory(num_traces=5, traces_per_page=10)
        client.observations_per_trace = 3

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=True,
            obs_type="GENERATION",
            obs_limit=10,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        items = list(dataset)

        assert len(items) == 5
        assert all("observations" in item for item in items)
        assert all(len(item["observations"]) == 3 for item in items)

        # Should fetch observations for each trace
        assert len(client.observation_calls) == 5

    def test_iter_empty_result(self, mock_langfuse_factory):
        """Test iteration with no traces."""
        client = mock_langfuse_factory(num_traces=0, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        items = list(dataset)

        assert len(items) == 0
        assert len(client.trace_list_calls) == 1

    def test_iter_batches_even_split(self, mock_langfuse_factory):
        """Test iter_batches with even division."""
        client = mock_langfuse_factory(num_traces=10, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=5,
            sample_rate=1.0,
        )

        batches = list(dataset.iter_batches())

        assert len(batches) == 2
        assert len(batches[0]) == 5
        assert len(batches[1]) == 5

    def test_iter_batches_uneven_split(self, mock_langfuse_factory):
        """Test iter_batches with uneven division."""
        client = mock_langfuse_factory(num_traces=10, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=3,
            sample_rate=1.0,
        )

        batches = list(dataset.iter_batches())

        # 10 items / batch_size=3 = [3, 3, 3, 1]
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_iter_batches_single_batch(self, mock_langfuse_factory):
        """Test iter_batches when all items fit in one batch."""
        client = mock_langfuse_factory(num_traces=5, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        batches = list(dataset.iter_batches())

        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_iter_batches_with_observations(self, mock_langfuse_factory):
        """Test iter_batches includes complete observations per trace."""
        client = mock_langfuse_factory(num_traces=6, traces_per_page=10)
        client.observations_per_trace = 2

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=True,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=2,
            sample_rate=1.0,
        )

        batches = list(dataset.iter_batches())

        # Each batch should have complete traces with their observations
        assert len(batches) == 3
        for batch in batches:
            for item in batch:
                assert "observations" in item
                assert len(item["observations"]) == 2

    def test_iter_with_sampling(self, mock_langfuse_factory):
        """Test iteration with sampling reduces trace count deterministically."""
        client = mock_langfuse_factory(num_traces=100, traces_per_page=20)

        # Use 10% sampling
        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 20, "fields": "core,io"},
            fields=["core", "io"],
            page_size=20,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=0.1,
        )

        items = list(dataset)

        # Should get approximately 10% of traces (around 10, but may vary due to hashing)
        # We'll be lenient and check it's less than 100 and more than 0
        assert len(items) < 100
        assert len(items) > 0
        # With 100 traces and 10% sampling, expect around 8-12 traces
        assert 5 <= len(items) <= 15

    def test_iter_with_zero_sampling(self, mock_langfuse_factory):
        """Test iteration with 0% sampling returns no items."""
        client = mock_langfuse_factory(num_traces=10, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=0.0,
        )

        items = list(dataset)

        # 0% sampling should return no items
        assert len(items) == 0


# ============================================================================
# INTEGRATION TESTS - Memory Management
# ============================================================================


class TestMemoryManagement:
    """Test memory tracking and warnings."""

    @patch.dict(os.environ, {"TRACE_MEMORY_THRESHOLD_MB": "50"})
    def test_memory_warning_triggered(self, mock_langfuse_factory):
        """Test memory warning is triggered when threshold exceeded."""
        client = mock_langfuse_factory(num_traces=10, traces_per_page=5)

        # Mock psutil to simulate memory growth
        with patch("evalx_sdk.trace_dataset.psutil.Process") as mock_process:
            # Create mock memory info objects
            call_count = [0]

            def memory_side_effect(*args, **kwargs):
                call_count[0] += 1
                mock_mem_info = Mock()
                # First call: 10MB, subsequent calls: 600MB (over threshold)
                if call_count[0] <= 1:
                    mock_mem_info.rss = 10 * 1024 * 1024
                else:
                    mock_mem_info.rss = 600 * 1024 * 1024
                return mock_mem_info

            mock_instance = Mock()
            mock_instance.memory_info.side_effect = memory_side_effect
            mock_process.return_value = mock_instance

            dataset = TraceDataset(
                langfuse_client=client,
                trace_params={"limit": 5, "fields": "core,io"},
                fields=["core", "io"],
                page_size=5,
                trace_filter=None,
                trace_processor=None,
                fetch_observations=False,
                obs_type=None,
                obs_limit=None,
                obs_filter=None,
                max_traces=None,
                max_observations_per_trace=None,
                batch_size=5,
                sample_rate=1.0,
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # Trigger iteration which checks memory
                items = list(dataset)

                # Should have triggered a warning
                warning_messages = [str(warning.message) for warning in w]
                memory_warnings = [msg for msg in warning_messages if "High memory usage" in msg]
                assert len(memory_warnings) > 0

    def test_memory_warning_not_triggered(self, mock_langfuse_factory):
        """Test no warning when memory stays below threshold."""
        client = mock_langfuse_factory(num_traces=5, traces_per_page=10)

        with patch("evalx_sdk.trace_dataset.psutil.Process") as mock_process:
            mock_instance = mock_process.return_value
            # Keep memory low (20MB)
            mock_instance.memory_info.return_value.rss = 20 * 1024 * 1024

            dataset = TraceDataset(
                langfuse_client=client,
                trace_params={"limit": 10, "fields": "core,io"},
                fields=["core", "io"],
                page_size=10,
                trace_filter=None,
                trace_processor=None,
                fetch_observations=False,
                obs_type=None,
                obs_limit=None,
                obs_filter=None,
                max_traces=None,
                max_observations_per_trace=None,
                batch_size=10,
                sample_rate=1.0,
            )

            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                items = list(dataset)

                # Should not have warnings about memory
                memory_warnings = [warning for warning in w if "memory" in str(warning.message).lower()]
                assert len(memory_warnings) == 0

    def test_memory_warning_only_once(self, mock_langfuse_factory):
        """Test memory warning is only issued once per dataset iteration."""
        client = mock_langfuse_factory(num_traces=20, traces_per_page=5)

        with patch("evalx_sdk.trace_dataset.psutil.Process") as mock_process:
            # Create mock that grows memory over time
            call_count = [0]

            def memory_side_effect(*args, **kwargs):
                call_count[0] += 1
                mock_mem_info = Mock()
                # First call: 10MB, subsequent calls: 1000MB (990MB growth > 10MB threshold)
                if call_count[0] == 1:
                    mock_mem_info.rss = 10 * 1024 * 1024
                else:
                    mock_mem_info.rss = 1000 * 1024 * 1024
                return mock_mem_info

            mock_instance = Mock()
            mock_instance.memory_info.side_effect = memory_side_effect
            mock_process.return_value = mock_instance

            # Patch the MEMORY_THRESHOLD_MB constant directly
            with patch("evalx_sdk.trace_dataset.MEMORY_THRESHOLD_MB", 10):
                dataset = TraceDataset(
                    langfuse_client=client,
                    trace_params={"limit": 5, "fields": "core,io"},
                    fields=["core", "io"],
                    page_size=5,
                    trace_filter=None,
                    trace_processor=None,
                    fetch_observations=False,
                    obs_type=None,
                    obs_limit=None,
                    obs_filter=None,
                    max_traces=None,
                    max_observations_per_trace=None,
                    batch_size=5,
                    sample_rate=1.0,
                )

                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    items = list(dataset)

                    # Should only warn once despite multiple page fetches
                    memory_warnings = [warning for warning in w if "High memory" in str(warning.message)]
                    assert len(memory_warnings) == 1


# ============================================================================
# INTEGRATION TESTS - Dataset Conversion and Evaluation
# ============================================================================


class TestDatasetConversionAndEvaluation:
    """Test to_langfuse_dataset and run_evaluation methods."""

    def test_to_langfuse_dataset_basic(self, mock_langfuse_factory, capsys):
        """Test to_langfuse_dataset creates and populates dataset."""
        client = mock_langfuse_factory(num_traces=5, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io,metadata"},
            fields=["core", "io", "metadata"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        langfuse_dataset = dataset.to_langfuse_dataset("MyDataset")

        # Verify dataset was created
        assert len(client.dataset_create_calls) == 1
        dataset_name = client.dataset_create_calls[0]["name"]
        assert "trace_dataset_MyDataset_" in dataset_name

        # Verify items were added
        assert len(client.dataset_item_calls) == 5

        # Verify output message
        captured = capsys.readouterr()
        assert "Created Langfuse dataset" in captured.out
        assert "5 items" in captured.out

    def test_to_langfuse_dataset_with_observations(self, mock_langfuse_factory):
        """Test to_langfuse_dataset includes observations in metadata."""
        client = mock_langfuse_factory(num_traces=3, traces_per_page=10)
        client.observations_per_trace = 2

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=True,
            obs_type="GENERATION",
            obs_limit=10,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        langfuse_dataset = dataset.to_langfuse_dataset("WithObs")

        # Verify observations were included in metadata
        assert len(client.dataset_item_calls) == 3
        for call in client.dataset_item_calls:
            assert "observations" in call["metadata"]
            assert len(call["metadata"]["observations"]) == 2

    def test_run_evaluation_creates_and_runs(self, mock_langfuse_factory):
        """Test run_evaluation processes traces in batches and creates scores."""
        client = mock_langfuse_factory(num_traces=5, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        def evaluator(*, output, trace_item, **kwargs):
            return Evaluation(name="test_evaluator", value=1.0, comment="test")

        result = dataset.run_evaluation(
            name="TestEval",
            evaluators=[evaluator],
            description="Test",
            metadata={"key": "value"}
        )

        # Verify NO dataset was created (new behavior)
        assert len(client.dataset_create_calls) == 0

        # Verify NO dataset items were created (new behavior)
        assert len(client.dataset_item_calls) == 0

        # Verify scores were created (new behavior - should create scores using langfuse.score())
        assert len(client.score_calls) == 5  # One score per trace

        # Verify flush was called (new behavior)
        assert len(client.flush_calls) > 0

        # Verify result is a statistics dict (new behavior)
        assert isinstance(result, dict)
        assert "evaluation_name" in result
        assert result["evaluation_name"] == "TestEval"
        assert "traces_processed" in result
        assert result["traces_processed"] == 5

    def test_run_evaluation_processes_all_traces(self, mock_langfuse_factory):
        """Test run_evaluation processes all traces successfully."""
        client = mock_langfuse_factory(num_traces=1, traces_per_page=10)

        # Modify trace to have specific output
        client.traces[0].output = {"answer": "42"}

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        result = dataset.run_evaluation(name="Test", evaluators=[])

        # Check that result has correct statistics
        assert result["total_traces"] == 1
        assert result["traces_processed"] == 1
        assert result["traces_skipped"] == 0

    def test_run_evaluation_returns_statistics(self, mock_langfuse_factory):
        """Test run_evaluation returns comprehensive statistics."""
        client = mock_langfuse_factory(num_traces=1, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        result = dataset.run_evaluation(name="Test")

        # Check that statistics dict is returned with correct keys
        assert isinstance(result, dict)
        assert "evaluation_name" in result
        assert "total_traces" in result
        assert "traces_processed" in result
        assert "traces_skipped" in result
        assert "total_scores" in result
        assert "start_time" in result
        assert "trace_ids" in result

    def test_run_evaluation_handles_timeout_gracefully(self, mock_langfuse_factory, capsys):
        """Test run_evaluation handles timeout errors gracefully."""
        client = mock_langfuse_factory(num_traces=5, traces_per_page=10)

        # Create a dataset that will raise a timeout error during iteration
        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        # Mock the trace.list API to raise timeout immediately
        def mock_list_with_timeout(**kwargs):
            raise Exception("httpcore.ReadTimeout: The read operation timed out")

        client.api.trace.list = Mock(side_effect=mock_list_with_timeout)

        # Should raise RuntimeError with helpful message
        with pytest.raises(RuntimeError) as exc_info:
            dataset.run_evaluation(name="Test", evaluators=[])

        assert "Failed to fetch traces from Langfuse due to network timeout" in str(exc_info.value)

        # Check output messages
        captured = capsys.readouterr()
        assert "Network timeout while fetching traces" in captured.out
        assert "Cannot run evaluation with no traces" in captured.out

    def test_run_evaluation_handles_partial_timeout(self, mock_langfuse_factory, capsys):
        """Test run_evaluation handles partial data collection on timeout."""
        client = mock_langfuse_factory(num_traces=15, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=5,  # Use smaller page size to trigger multiple pages
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=3,  # Use batch size smaller than page size to ensure first batch completes
            sample_rate=1.0,
        )

        # Store the original trace.list method
        original_list = client.api.trace.list
        call_count = [0]  # Use list to allow mutation in closure

        # Mock trace.list to return full page (5 traces) first, then timeout on second page
        def mock_list_partial_timeout(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: return full page (5 traces) to trigger next page
                response = Mock()
                response.data = client.traces[:5]
                return response
            else:
                # Second call (next page): raise timeout
                raise Exception("httpcore.ReadTimeout: The read operation timed out")

        client.api.trace.list = Mock(side_effect=mock_list_partial_timeout)

        # Should proceed with partial data
        result = dataset.run_evaluation(name="Test", evaluators=[])

        # Check output messages
        captured = capsys.readouterr()
        assert "Network timeout while fetching traces" in captured.out
        assert "Collected 3 traces before timeout" in captured.out
        assert "Proceeding with 3 traces collected so far" in captured.out

        # Verify result shows partial processing (should process 3 traces before timeout)
        assert result["traces_processed"] == 3


# ============================================================================
# END-TO-END INTEGRATION TESTS
# ============================================================================


class TestEndToEndWorkflows:
    """Comprehensive end-to-end workflow tests."""

    def test_complete_workflow_no_observations(self, mock_langfuse_factory):
        """Test complete workflow from builder to evaluation without observations."""
        client = mock_langfuse_factory(num_traces=10, traces_per_page=5)

        # Build dataset
        builder = TraceToDatasetBuilder(client, page_size=5)
        dataset = (builder
            .with_fields(["core", "io"])
            .with_time_range(days=7)
            .with_limits(max_traces=8)
            .with_batch_size(4)
            .with_sample_rate(1.0)  # 100% sampling for testing
            .build()
        )

        # Iterate
        items = list(dataset)
        assert len(items) == 8

        # Convert to Langfuse dataset
        langfuse_dataset = dataset.to_langfuse_dataset("TestDataset")
        assert len(client.dataset_item_calls) == 8

    def test_complete_workflow_with_observations(self, mock_langfuse_factory):
        """Test complete workflow with observations."""
        client = mock_langfuse_factory(num_traces=6, traces_per_page=10)
        client.observations_per_trace = 3

        builder = TraceToDatasetBuilder(client)
        dataset = (builder
            .with_fields(["core", "io", "metadata"])
            .with_observations(fetch=True, obs_type="GENERATION")
            .with_limits(max_traces=5)
            .with_sample_rate(1.0)  # 100% sampling for testing
            .build()
        )

        items = list(dataset)
        assert len(items) == 5
        assert all(len(item["observations"]) == 3 for item in items)

    def test_complete_workflow_with_filters(self, mock_langfuse_factory):
        """Test complete workflow with trace and observation filters."""
        client = mock_langfuse_factory(num_traces=20, traces_per_page=10)
        client.observations_per_trace = 4

        # Modify traces
        for i, trace in enumerate(client.traces):
            trace.user_id = f"user-{i % 3}"
            trace.tags = ["prod"] if i % 2 == 0 else ["test"]

        builder = TraceToDatasetBuilder(client)
        dataset = (builder
            .with_fields(["core", "io"])
            .with_trace_filters(tags=["prod"])
            .with_trace_filter(lambda t: t.core.user_id == "user-0")
            .with_observations(
                fetch=True,
                obs_filter=lambda obs: obs.name in ["step-0", "step-1"]
            )
            .build()
        )

        items = list(dataset)

        # Verify filters worked
        assert all(item["trace_core"]["user_id"] == "user-0" for item in items)
        assert all("observations" in item for item in items)
        assert all(len(item["observations"]) == 2 for item in items)  # Only step-0 and step-1

    def test_complete_workflow_batch_processing(self, mock_langfuse_factory):
        """Test complete workflow with batch processing."""
        client = mock_langfuse_factory(num_traces=15, traces_per_page=10)

        builder = TraceToDatasetBuilder(client)
        dataset = (builder
            .with_fields(["core", "io"])
            .with_batch_size(4)
            .with_limits(max_traces=11)
            .with_sample_rate(1.0)  # 100% sampling for testing
            .build()
        )

        batches = list(dataset.iter_batches())

        # 11 items / batch_size=4 = [4, 4, 3]
        assert len(batches) == 3
        assert len(batches[0]) == 4
        assert len(batches[1]) == 4
        assert len(batches[2]) == 3

    def test_complete_workflow_run_evaluation(self, mock_langfuse_factory):
        """Test complete end-to-end evaluation workflow."""
        client = mock_langfuse_factory(num_traces=10, traces_per_page=10)

        def accuracy_evaluator(*, output, trace_item, **kwargs):
            return Evaluation(name="accuracy", value=1.0, comment="test")

        builder = TraceToDatasetBuilder(client)
        dataset = (builder
            .with_fields(["core", "io"])
            .with_limits(max_traces=7)
            .with_sample_rate(1.0)  # 100% sampling for testing
            .build()
        )

        result = dataset.run_evaluation(
            name="AccuracyTest",
            evaluators=[accuracy_evaluator],
            metadata={"version": "1.0"}
        )

        # Verify NO dataset was created (new behavior)
        assert len(client.dataset_create_calls) == 0
        assert len(client.dataset_item_calls) == 0

        # Verify scores were created (new behavior)
        assert len(client.score_calls) == 7  # One per trace

        # Verify flush was called
        assert len(client.flush_calls) > 0

        # Verify result is statistics dict
        assert isinstance(result, dict)
        assert result["traces_processed"] == 7

    def test_custom_processor_workflow(self, mock_langfuse_factory):
        """Test workflow with custom trace processor."""
        client = mock_langfuse_factory(num_traces=5, traces_per_page=10)

        def custom_processor(trace):
            return {
                "id": trace.core.id,
                "q": trace.io.input.get("question"),
                "a": trace.io.output.get("answer"),
            }

        builder = TraceToDatasetBuilder(client)
        dataset = (builder
            .with_fields(["core", "io"])
            .with_trace_processor(custom_processor)
            .build()
        )

        items = list(dataset)

        assert all("id" in item for item in items)
        assert all("q" in item for item in items)
        assert all("a" in item for item in items)
        # Default fields should not be present
        assert all("trace_id" not in item for item in items)

    def test_pagination_edge_case_exact_page_boundary(self, mock_langfuse_factory):
        """Test pagination when num_traces exactly matches page size."""
        client = mock_langfuse_factory(num_traces=10, traces_per_page=10)

        dataset = TraceDataset(
            langfuse_client=client,
            trace_params={"limit": 10, "fields": "core,io"},
            fields=["core", "io"],
            page_size=10,
            trace_filter=None,
            trace_processor=None,
            fetch_observations=False,
            obs_type=None,
            obs_limit=None,
            obs_filter=None,
            max_traces=None,
            max_observations_per_trace=None,
            batch_size=10,
            sample_rate=1.0,
        )

        items = list(dataset)

        assert len(items) == 10
        # Should make 2 API calls: one full page, one empty page
        assert len(client.trace_list_calls) == 2

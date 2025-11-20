"""Trace-to-Dataset builder for creating Langfuse datasets from traces."""

import os
import warnings
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Union

import psutil
from pydantic import BaseModel

# Import Langfuse experiment classes for compatibility
from langfuse.experiment import Evaluation, ExperimentResult, ExperimentItemResult

# Memory threshold in MB (configurable via environment variable)
MEMORY_THRESHOLD_MB = int(os.getenv("TRACE_MEMORY_THRESHOLD_MB", "500"))


# Pydantic models for structured trace data
class TraceCore(BaseModel):
    """Core trace fields"""

    id: Optional[str] = None
    name: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: Optional[datetime] = None
    release: Optional[str] = None
    version: Optional[str] = None
    tags: Optional[List[str]] = None


class TraceIO(BaseModel):
    """Trace input/output fields"""

    input: Optional[Any] = None
    output: Optional[Any] = None


class TraceMetadata(BaseModel):
    """Trace metadata fields"""

    metadata: Optional[Dict[str, Any]] = None


class TraceData(BaseModel):
    """Complete trace data structure"""

    core: TraceCore
    io: Optional[TraceIO] = None
    metadata: Optional[TraceMetadata] = None


class ObservationData(BaseModel):
    """Observation data structure"""

    id: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Any] = None
    output: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class TraceDataset:
    """
    Iterator-based dataset that streams traces and observations in batches.

    Provides memory-efficient data access and integrates with Langfuse experiments.
    """

    def __init__(
        self,
        langfuse_client,
        trace_params: Dict[str, Any],
        fields: List[str],
        page_size: int,
        trace_filter: Optional[Callable[[TraceData], bool]],
        trace_processor: Optional[Callable[[TraceData], Dict[str, Any]]],
        fetch_observations: bool,
        obs_type: Optional[str],
        obs_limit: Optional[int],
        obs_filter: Optional[Callable[[ObservationData], bool]],
        max_traces: Optional[int],
        max_observations_per_trace: Optional[int],
        batch_size: int = 10,
        sample_rate: float = 1.0,
    ):
        self.langfuse = langfuse_client
        self.trace_params = trace_params
        self.fields = fields
        self.page_size = page_size
        self.trace_filter = trace_filter
        self.trace_processor = trace_processor
        self.fetch_observations = fetch_observations
        self.obs_type = obs_type
        self.obs_limit = obs_limit
        self.obs_filter = obs_filter
        self.max_traces = max_traces
        self.max_observations_per_trace = max_observations_per_trace
        self.batch_size = batch_size
        # Clamp sample_rate between 0 and 1
        self.sample_rate = max(0.0, min(1.0, sample_rate))

        # Memory tracking
        self._initial_memory: Optional[float] = None
        self._warned_memory: bool = False

    def _check_memory_usage(self) -> None:
        """Check memory usage and warn if threshold exceeded."""
        if self._initial_memory is None:
            process = psutil.Process()
            self._initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            return

        if not self._warned_memory:
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = current_memory - self._initial_memory

            if memory_used > MEMORY_THRESHOLD_MB:
                warnings.warn(
                    f"⚠️  High memory usage detected: {memory_used:.1f} MB "
                    f"(threshold: {MEMORY_THRESHOLD_MB} MB). "
                    f"Consider reducing page_size or batch_size.",
                    ResourceWarning,
                )
                self._warned_memory = True

    def _parse_trace(self, trace_raw) -> TraceData:
        """Parse raw trace object into TraceData structure."""
        core = TraceCore(
            id=getattr(trace_raw, "id", None),
            name=getattr(trace_raw, "name", None),
            user_id=getattr(trace_raw, "user_id", None),
            session_id=getattr(trace_raw, "session_id", None),
            timestamp=getattr(trace_raw, "timestamp", None),
            release=getattr(trace_raw, "release", None),
            version=getattr(trace_raw, "version", None),
            tags=getattr(trace_raw, "tags", None),
        )

        io = None
        if "io" in self.fields:
            io = TraceIO(
                input=getattr(trace_raw, "input", None),
                output=getattr(trace_raw, "output", None),
            )

        metadata = None
        if "metadata" in self.fields:
            metadata = TraceMetadata(metadata=getattr(trace_raw, "metadata", None))

        return TraceData(core=core, io=io, metadata=metadata)

    def _fetch_observations(self, trace_id: str) -> List[ObservationData]:
        """Fetch observations for a trace."""
        observations = []

        # Build observation query parameters
        obs_params = {"trace_id": trace_id}
        if self.obs_type:
            obs_params["type"] = self.obs_type
        if self.obs_limit:
            obs_params["limit"] = self.obs_limit

        # Fetch observations
        obs_response = self.langfuse.api.observations.get_many(**obs_params)

        for obs_raw in getattr(obs_response, "data", []):
            obs_data = ObservationData(
                id=getattr(obs_raw, "id", None),
                type=getattr(obs_raw, "type", None),
                name=getattr(obs_raw, "name", None),
                input=getattr(obs_raw, "input", None),
                output=getattr(obs_raw, "output", None),
                metadata=getattr(obs_raw, "metadata", None),
                start_time=getattr(obs_raw, "start_time", None),
                end_time=getattr(obs_raw, "end_time", None),
            )

            # Apply observation filter
            if self.obs_filter is None or self.obs_filter(obs_data):
                observations.append(obs_data)

                # Check per-trace observation limit
                if (
                    self.max_observations_per_trace
                    and len(observations) >= self.max_observations_per_trace
                ):
                    break

        return observations

    def _build_dataset_item(
        self, trace: TraceData, observations: Optional[List[ObservationData]] = None
    ) -> Dict[str, Any]:
        """Build a single dataset item from trace and observations."""
        # Use custom processor if provided
        if self.trace_processor:
            return self.trace_processor(trace)

        # Default structure
        item = {
            "trace_id": trace.core.id,
            "trace_name": trace.core.name,
        }

        # Add trace IO if requested
        if trace.io:
            item["input"] = trace.io.input
            item["output"] = trace.io.output

        # Add metadata if requested
        if trace.metadata:
            item["metadata"] = trace.metadata.metadata

        # Add core fields
        if "core" in self.fields:
            item["trace_core"] = trace.core.model_dump(exclude_none=True)

        # Add observations if fetched
        if observations:
            item["observations"] = [
                {
                    "id": obs.id,
                    "type": obs.type,
                    "name": obs.name,
                    "input": obs.input,
                    "output": obs.output,
                    "metadata": obs.metadata,
                }
                for obs in observations
            ]

        return item

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over dataset items one by one.

        Yields:
            Dataset items with trace data and observations
        """
        import hashlib

        page = 1
        traces_processed = 0

        # Initialize memory tracking
        self._initial_memory = None
        self._warned_memory = False

        while True:
            # Check memory before fetching next page
            self._check_memory_usage()

            # Fetch page of traces
            params = {**self.trace_params, "page": page}
            traces_response = self.langfuse.api.trace.list(**params)

            trace_list = getattr(traces_response, "data", [])
            if not trace_list:
                break

            # Process each trace
            for trace_raw in trace_list:
                # Parse trace (cheap operation)
                trace = self._parse_trace(trace_raw)

                # Apply sampling BEFORE expensive operations (like fetching observations)
                # Use deterministic sampling based on trace_id (same as Langfuse SDK)
                if self.sample_rate < 1.0:
                    trace_id = trace.core.id
                    # Hash trace_id and normalize to [0, 1)
                    hash_value = int(hashlib.sha256(trace_id.encode()).hexdigest(), 16)
                    normalized = (hash_value % 10000) / 10000.0
                    # Skip if outside sample rate
                    if normalized >= self.sample_rate:
                        continue

                # Apply trace filter
                if self.trace_filter and not self.trace_filter(trace):
                    continue

                # Fetch observations if configured (expensive operation)
                observations = None
                if self.fetch_observations:
                    observations = self._fetch_observations(trace.core.id)

                # Build and yield dataset item
                item = self._build_dataset_item(trace, observations)
                yield item

                traces_processed += 1

                # Check global trace limit
                if self.max_traces and traces_processed >= self.max_traces:
                    return

            # Check if we've fetched all available traces
            if len(trace_list) < self.page_size:
                break

            page += 1

    def iter_batches(self) -> Iterator[List[Dict[str, Any]]]:
        """
        Iterate over dataset in batches.

        Each batch contains complete traces with all their observations.
        The last trace in each batch will have all its observations included.

        Yields:
            Batches of dataset items
        """
        batch = []

        for item in self:
            batch.append(item)

            # Yield batch when it reaches batch_size
            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        # Yield remaining items
        if batch:
            yield batch

    def to_langfuse_dataset(self, name: str) -> Any:
        """
        Convert trace dataset to a Langfuse dataset.

        This creates a Langfuse dataset from the trace data without running
        any experiments. Useful for saving filtered traces as a dataset.

        Args:
            name: Dataset name

        Returns:
            Langfuse dataset object

        Example:
            >>> langfuse_dataset = dataset.to_langfuse_dataset("Production Traces")
            >>> # Use the dataset later
            >>> result = langfuse_dataset.run_experiment(...)
        """
        # Create a Langfuse dataset
        dataset_name = f"trace_dataset_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create dataset in Langfuse (this creates the dataset entity)
        self.langfuse.create_dataset(name=dataset_name)

        # Add items to dataset in batches to avoid memory issues
        item_count = 0
        for batch in self.iter_batches():
            for item in batch:
                self.langfuse.create_dataset_item(
                    dataset_name=dataset_name,
                    input=item.get("input"),
                    expected_output=item.get("output"),
                    metadata={
                        "trace_id": item.get("trace_id"),
                        "trace_name": item.get("trace_name"),
                        "trace_core": item.get("trace_core"),
                        "observations": item.get("observations"),
                        **(item.get("metadata") or {}),
                    },
                    source_trace_id=item.get("trace_id"),
                )
                item_count += 1

        print(f"Created Langfuse dataset '{dataset_name}' with {item_count} items")

        # Get the dataset client for run_experiment
        langfuse_dataset = self.langfuse.get_dataset(name=dataset_name)
        return langfuse_dataset

    def run_evaluation(
        self,
        name: str,
        evaluators: Optional[List[Callable]] = None,
        run_evaluators: Optional[List[Callable]] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        run_evaluators_mode: str = "memory",  # "memory" or "fetch"
    ) -> Dict[str, Any]:
        """
        Run evaluators on the trace dataset in batches.

        This processes traces in batches (respecting batch_size), runs evaluators
        on each item, and creates evaluation scores directly in Langfuse. This
        avoids loading all traces into memory at once.

        Args:
            name: Evaluation run name
            evaluators: List of evaluator functions for item-level evaluation
            run_evaluators: List of evaluator functions for run-level evaluation
            description: Optional evaluation description
            metadata: Optional metadata dict for the evaluation
            run_evaluators_mode: How to provide data to run_evaluators:
                - "fetch": Fetch scores from Langfuse after all items processed (memory-efficient)
                - "memory": Keep all item results in memory (faster but higher memory)

        Returns:
            Dict with evaluation statistics

        Example:
            >>> def accuracy_evaluator(*, output, expected_output, **kwargs):
            ...     score = 1.0 if output == expected_output else 0.0
            ...     return {"score": score, "name": "accuracy"}
            >>>
            >>> result = dataset.run_evaluation(
            ...     name="Production Trace Evaluation",
            ...     evaluators=[accuracy_evaluator],
            ...     metadata={"version": "v1.0"}
            ... )
        """
        from datetime import datetime

        evaluators = evaluators or []
        run_evaluators = run_evaluators or []

        print(f"📊 Starting evaluation '{name}' in batches of {self.batch_size}...")

        # Track statistics
        stats = {
            "evaluation_name": name,
            "total_traces": 0,
            "traces_processed": 0,
            "traces_skipped": 0,
            "total_scores": 0,
            "scores_by_evaluator": {},
            "start_time": datetime.now().isoformat(),
            "trace_ids": [],  # Track all processed trace IDs
        }

        # Store all item results for run-level evaluators (if mode is "memory")
        all_results = [] if run_evaluators_mode == "memory" else None

        batch_num = 0
        try:
            for batch in self.iter_batches():
                batch_num += 1
                print(f"📦 Processing batch {batch_num} ({len(batch)} traces)...")

                for item in batch:
                    stats["total_traces"] += 1

                    # Skip items without input
                    if not item.get("input"):
                        stats["traces_skipped"] += 1
                        print(f"⚠️  Skipping trace {item.get('trace_id')} - no input data")
                        continue

                    trace_id = item.get("trace_id")
                    stats["trace_ids"].append(trace_id)
                    output = item.get("output")

                    # Run item-level evaluators
                    item_scores = []
                    for evaluator in evaluators:
                        try:
                            # Prepare metadata with trace_core and observations for evaluators
                            eval_metadata = item.get("metadata", {}) or {}
                            if isinstance(eval_metadata, dict):
                                # Add trace_core and observations to metadata for evaluator access
                                eval_metadata = {
                                    **eval_metadata,
                                    "trace_core": item.get("trace_core"),
                                    "observations": item.get("observations"),
                                }

                            # Call evaluator with trace-specific signature
                            # For trace-based datasets, we pass the full item as trace_item
                            # This allows evaluators to access all trace data (trace_id, trace_core, etc.)
                            eval_result = evaluator(
                                input=item.get("input"),
                                output=output,
                                trace_item=item,  # Pass full dataset item for trace-based evaluation
                                metadata=eval_metadata,
                            )

                            # Handle both Evaluation objects and dicts (for backward compatibility)
                            if eval_result:
                                # Normalize to list of Evaluation objects
                                evaluations = []
                                if isinstance(eval_result, list):
                                    evaluations = eval_result
                                else:
                                    evaluations = [eval_result]

                                for evaluation in evaluations:
                                    # Extract values from Evaluation object or dict
                                    if isinstance(evaluation, Evaluation):
                                        eval_name = evaluation.name
                                        eval_value = evaluation.value
                                        eval_comment = evaluation.comment
                                        eval_metadata = evaluation.metadata
                                        eval_data_type = evaluation.data_type
                                        eval_config_id = evaluation.config_id
                                    else:
                                        # Backward compatibility with dict format
                                        eval_name = evaluation.get("name", "unnamed_evaluator")
                                        eval_value = evaluation.get("score") or evaluation.get("value")
                                        eval_comment = evaluation.get("comment")
                                        eval_metadata = evaluation.get("metadata")
                                        eval_data_type = evaluation.get("data_type")
                                        eval_config_id = evaluation.get("config_id")

                                    # Create evaluation score in Langfuse (skip if value is None)
                                    if eval_value is not None:
                                        self.langfuse.create_score(
                                            trace_id=trace_id,
                                            name=eval_name,
                                            value=eval_value,
                                            comment=eval_comment,
                                            metadata=eval_metadata,
                                            data_type=eval_data_type,
                                            config_id=eval_config_id,
                                        )

                                    item_scores.append(evaluation)
                                    stats["total_scores"] += 1

                                    # Track per-evaluator stats
                                    if eval_name not in stats["scores_by_evaluator"]:
                                        stats["scores_by_evaluator"][eval_name] = {
                                            "count": 0,
                                            "sum": 0,
                                            "min": None,
                                            "max": None,
                                        }

                                    eval_stats = stats["scores_by_evaluator"][eval_name]
                                    eval_stats["count"] += 1
                                    score_val = eval_value if isinstance(eval_value, (int, float)) else 0
                                eval_stats["sum"] += score_val

                                if eval_stats["min"] is None or score_val < eval_stats["min"]:
                                    eval_stats["min"] = score_val
                                if eval_stats["max"] is None or score_val > eval_stats["max"]:
                                    eval_stats["max"] = score_val

                        except Exception as e:
                            print(f"⚠️  Evaluator '{evaluator.__name__}' failed for trace {trace_id}: {e}")

                    # Store result for run-level evaluators if in memory mode
                    if run_evaluators_mode == "memory" and all_results is not None:
                        # Create ExperimentItemResult for Langfuse compatibility
                        item_result = ExperimentItemResult(
                            item=item,
                            output=output,
                            evaluations=item_scores,  # Already list of Evaluation objects
                            trace_id=trace_id,
                            dataset_run_id=None,
                        )
                        all_results.append(item_result)

                    stats["traces_processed"] += 1

                print(f"✅ Batch {batch_num} complete")

                # Flush scores to Langfuse after each batch
                self.langfuse.flush()

        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                print(f"⚠️  Network timeout while fetching traces from Langfuse.")
                print(f"   This usually means the Langfuse service is slow or unavailable.")
                print(f"   Collected {stats['traces_processed']} traces before timeout.")
                if stats["traces_processed"] == 0:
                    print(f"   Cannot run evaluation with no traces. Please try again later.")
                    raise RuntimeError(
                        "Failed to fetch traces from Langfuse due to network timeout. "
                        "Please check your network connection and try again."
                    ) from e
                print(f"   Proceeding with {stats['traces_processed']} traces collected so far...")
            else:
                print(f"❌ Error while fetching traces: {error_msg}")
                raise

        # Run run-level evaluators
        if run_evaluators and stats["traces_processed"] > 0:
            print(f"🔍 Running {len(run_evaluators)} run-level evaluators...")

            # Fetch data for run evaluators based on mode
            if run_evaluators_mode == "fetch":
                print(f"   Fetching scores from Langfuse for {len(stats['trace_ids'])} traces...")
                # Fetch scores from Langfuse for all processed traces
                all_results = []
                for trace_id in stats["trace_ids"]:
                    try:
                        # Fetch trace with scores
                        trace = self.langfuse.api.trace.get(trace_id)
                        # Get scores for this trace
                        scores_response = self.langfuse.api.score.list(trace_id=trace_id)
                        scores = scores_response.data if hasattr(scores_response, 'data') else []

                        all_results.append({
                            "trace_id": trace_id,
                            "scores": [{"name": s.name, "value": s.value} for s in scores],
                        })
                    except Exception as e:
                        print(f"⚠️  Failed to fetch scores for trace {trace_id}: {e}")

            # Run each run-level evaluator
            for run_evaluator in run_evaluators:
                try:
                    run_result = run_evaluator(item_results=all_results)
                    if run_result:
                        print(f"   {run_evaluator.__name__}: {run_result}")
                except Exception as e:
                    print(f"⚠️  Run evaluator '{run_evaluator.__name__}' failed: {e}")

        # Calculate averages
        for eval_name, eval_stats in stats["scores_by_evaluator"].items():
            if eval_stats["count"] > 0:
                eval_stats["avg"] = eval_stats["sum"] / eval_stats["count"]

        stats["end_time"] = datetime.now().isoformat()

        # Print summary
        print(f"\n{'='*60}")
        print(f"Evaluation Complete: {name}")
        print(f"{'='*60}")
        print(f"Total traces: {stats['total_traces']}")
        print(f"Processed: {stats['traces_processed']}")
        print(f"Skipped (no input): {stats['traces_skipped']}")
        print(f"Total scores: {stats['total_scores']}")

        if stats["scores_by_evaluator"]:
            print(f"\nScores by evaluator:")
            for eval_name, eval_stats in stats["scores_by_evaluator"].items():
                print(f"  {eval_name}:")
                print(f"    Count: {eval_stats['count']}")
                print(f"    Avg: {eval_stats.get('avg', 0):.3f}")
                print(f"    Min: {eval_stats['min']}")
                print(f"    Max: {eval_stats['max']}")

        print(f"{'='*60}\n")

        # Final flush to ensure all scores are sent
        self.langfuse.flush()

        # Create ExperimentResult for compatibility with Langfuse
        return ExperimentResult(
            name=name,
            run_name=name,  # Use same name for run_name
            description=description,
            item_results=all_results if all_results is not None else [],
            run_evaluations=[],  # Run evaluators return arbitrary results, not Evaluation objects yet
            dataset_run_id=None,
            dataset_run_url=None,
        )


class TraceToDatasetBuilder:
    """
    Builder for creating datasets from Langfuse traces with filtering and pagination.

    Modeled after Langfuse's experiment SDK for elegant, chainable API.

    Note:
        By default, sampling is set to 1% (0.01) to manage evaluation costs.
        Use .with_sample_rate(1.0) for 100% sampling if you want to evaluate all traces.

    Example:
        >>> from evalx_sdk import TraceToDatasetBuilder
        >>> from langfuse import get_client
        >>>
        >>> langfuse = get_client()
        >>> builder = TraceToDatasetBuilder(langfuse)
        >>> dataset = (builder
        ...     .with_time_range(days=7)
        ...     .with_fields(["core", "io", "metadata"])
        ...     .with_sample_rate(0.05)  # Sample 5% of traces (default is 1%)
        ...     .with_trace_filter(lambda trace: trace.core.tags and "production" in trace.core.tags)
        ...     .with_observations(
        ...         fetch=True,
        ...         obs_type="GENERATION",
        ...         obs_limit=10,
        ...         obs_filter=lambda obs: obs.name == "completion"
        ...     )
        ...     .build(max_traces=1000))
        >>>
        >>> # Run evaluation on sampled traces (no task needed, avoids trace loops)
        >>> result = dataset.run_evaluation(
        ...     name="Production Trace Evaluation",
        ...     evaluators=[accuracy_evaluator, quality_evaluator]
        ... )
        >>>
        >>> # Or convert to Langfuse dataset for later use
        >>> langfuse_dataset = dataset.to_langfuse_dataset("Production Traces")
    """

    # Mandatory field constraints
    ALLOWED_FIELDS = {"core", "io", "metadata"}
    FORBIDDEN_FIELDS = {"scores", "observations", "metrics"}

    def __init__(
        self,
        langfuse_client,
        default_time_range_days: int = 1,
        page_size: int = 50,
    ):
        """
        Initialize the builder.

        Args:
            langfuse_client: Initialized Langfuse client
            default_time_range_days: Default time range in days (default: 1)
            page_size: Number of traces to fetch per API call (default: 50)
        """
        self.langfuse = langfuse_client
        self.page_size = page_size

        # Filters and configuration
        self._fields: List[str] = ["core", "io"]  # Default fields
        self._trace_filter: Optional[Callable[[TraceData], bool]] = None
        self._trace_processor: Optional[Callable[[TraceData], Dict[str, Any]]] = None

        # Observation configuration
        self._fetch_observations: bool = False
        self._obs_type: Optional[str] = None
        self._obs_limit: Optional[int] = None
        self._obs_filter: Optional[Callable[[ObservationData], bool]] = None

        # Time range (mandatory with default)
        self._start_time: Optional[datetime] = datetime.now() - timedelta(
            days=default_time_range_days
        )
        self._end_time: Optional[datetime] = datetime.now()

        # Limits
        self._max_traces: Optional[int] = None
        self._max_observations_per_trace: Optional[int] = None

        # Sampling (default 1% for cost management)
        self._sample_rate: float = 0.01

        # Batch size
        self._batch_size: int = 10

        # Additional trace list filters (user_id, session_id, etc.)
        self._trace_list_filters: Dict[str, Any] = {}

    def with_fields(
        self, fields: List[Literal["core", "io", "metadata"]]
    ) -> "TraceToDatasetBuilder":
        """
        Set which trace fields to fetch (mandatory).

        Args:
            fields: List of field types. Must be subset of ["core", "io", "metadata"]
                   Cannot include ["scores", "observations", "metrics"]

        Returns:
            Self for chaining

        Raises:
            ValueError: If forbidden fields are provided
        """
        fields_set = set(fields)

        # Check for forbidden fields
        forbidden = fields_set & self.FORBIDDEN_FIELDS
        if forbidden:
            raise ValueError(
                f"Forbidden fields detected: {forbidden}. "
                f"Only {self.ALLOWED_FIELDS} are allowed."
            )

        # Check for invalid fields
        invalid = fields_set - self.ALLOWED_FIELDS
        if invalid:
            raise ValueError(
                f"Invalid fields: {invalid}. " f"Allowed fields: {self.ALLOWED_FIELDS}"
            )

        self._fields = list(fields_set)
        return self

    def with_time_range(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        days: Optional[int] = None,
        hours: Optional[int] = None,
    ) -> "TraceToDatasetBuilder":
        """
        Set time range filter for traces.

        Args:
            start_time: Start datetime (default: now - days/hours)
            end_time: End datetime (default: now)
            days: Number of days back from now (convenience parameter)
            hours: Number of hours back from now (convenience parameter)

        Returns:
            Self for chaining

        Example:
            >>> # Last 24 hours
            >>> builder.with_time_range(hours=24)
            >>>
            >>> # Last 1 hour
            >>> builder.with_time_range(hours=1)
            >>>
            >>> # Last 7 days
            >>> builder.with_time_range(days=7)
        """
        if days is not None:
            self._start_time = datetime.now() - timedelta(days=days)
            self._end_time = datetime.now()
        elif hours is not None:
            self._start_time = datetime.now() - timedelta(hours=hours)
            self._end_time = datetime.now()
        else:
            if start_time:
                self._start_time = start_time
            if end_time:
                self._end_time = end_time

        return self

    def with_trace_filter(
        self, filter_fn: Callable[[TraceData], bool]
    ) -> "TraceToDatasetBuilder":
        """
        Add a custom filter function for traces.

        The filter receives a TraceData object with core, io, and metadata fields
        based on what was requested in with_fields().

        Args:
            filter_fn: Function that takes TraceData and returns True to include

        Returns:
            Self for chaining

        Example:
            >>> builder.with_trace_filter(
            ...     lambda trace: trace.core.tags and "production" in trace.core.tags
            ... )
        """
        self._trace_filter = filter_fn
        return self

    def with_trace_processor(
        self, process_fn: Callable[[TraceData], Dict[str, Any]]
    ) -> "TraceToDatasetBuilder":
        """
        Add a custom processor to transform trace data.

        Args:
            process_fn: Function that takes TraceData and returns a dict

        Returns:
            Self for chaining
        """
        self._trace_processor = process_fn
        return self

    def with_observations(
        self,
        fetch: bool = True,
        obs_type: Optional[Literal["GENERATION", "SPAN", "EVENT"]] = None,
        obs_limit: Optional[int] = 100,
        obs_filter: Optional[Callable[[ObservationData], bool]] = None,
    ) -> "TraceToDatasetBuilder":
        """
        Configure observation fetching for each trace.

        Args:
            fetch: Whether to fetch observations (default: True)
            obs_type: Filter by observation type (GENERATION, SPAN, EVENT)
            obs_limit: Max observations per trace (default: 100)
            obs_filter: Custom filter function for observations

        Returns:
            Self for chaining

        Example:
            >>> builder.with_observations(
            ...     fetch=True,
            ...     obs_type="GENERATION",
            ...     obs_limit=10,
            ...     obs_filter=lambda obs: obs.name == "completion"
            ... )
        """
        self._fetch_observations = fetch
        self._obs_type = obs_type
        self._obs_limit = obs_limit
        self._obs_filter = obs_filter
        return self

    def with_limits(
        self,
        max_traces: Optional[int] = None,
        max_observations_per_trace: Optional[int] = None,
    ) -> "TraceToDatasetBuilder":
        """
        Set global limits on trace and observation counts.

        Args:
            max_traces: Maximum number of traces to process
            max_observations_per_trace: Maximum observations per trace

        Returns:
            Self for chaining
        """
        self._max_traces = max_traces
        self._max_observations_per_trace = max_observations_per_trace
        return self

    def with_trace_filters(self, **kwargs) -> "TraceToDatasetBuilder":
        """
        Add additional filters for trace.list() API call.

        Args:
            **kwargs: user_id, session_id, name, tags, etc.

        Returns:
            Self for chaining

        Example:
            >>> builder.with_trace_filters(user_id="user_123", tags=["production"])
        """
        self._trace_list_filters.update(kwargs)
        return self

    def with_batch_size(self, batch_size: int) -> "TraceToDatasetBuilder":
        """
        Set the batch size for dataset iteration.

        Args:
            batch_size: Number of items per batch (default: 10)

        Returns:
            Self for chaining
        """
        self._batch_size = batch_size
        return self

    def with_sample_rate(self, sample_rate: float) -> "TraceToDatasetBuilder":
        """
        Set the sampling rate for trace evaluation (default: 0.01 = 1%).

        Uses deterministic sampling based on trace_id hash (same as Langfuse SDK).
        Sampling happens before fetching observations to save API calls.

        Args:
            sample_rate: Percentage of traces to sample (0.0 to 1.0)
                        0.01 = 1%, 0.1 = 10%, 1.0 = 100%

        Returns:
            Self for chaining

        Example:
            >>> builder.with_sample_rate(0.05)  # Sample 5% of traces
        """
        self._sample_rate = max(0.0, min(1.0, sample_rate))
        return self

    def build(
        self, max_traces: Optional[int] = None, batch_size: Optional[int] = None
    ) -> TraceDataset:
        """
        Build an iterator-based dataset that streams traces and observations.

        The returned TraceDataset is memory-efficient and can be used directly
        or with run_experiment().

        Args:
            max_traces: Override max_traces limit for this build
            batch_size: Override batch_size for iteration (default: 10)

        Returns:
            TraceDataset iterator with run_experiment() support

        Example:
            >>> dataset = builder.build(max_traces=1000, batch_size=20)
            >>>
            >>> # Iterate through dataset
            >>> for item in dataset:
            ...     process(item)
            >>>
            >>> # Or iterate in batches
            >>> for batch in dataset.iter_batches():
            ...     process_batch(batch)
            >>>
            >>> # Or run experiment
            >>> result = dataset.run_experiment(
            ...     name="Test",
            ...     task=my_task
            ... )
        """
        if max_traces:
            self._max_traces = max_traces

        if batch_size:
            self._batch_size = batch_size

        # Build trace list query parameters
        trace_params = {
            "limit": self.page_size,
            "fields": ",".join(self._fields),
            **self._trace_list_filters,
        }

        # Add time range filters
        if self._start_time:
            trace_params["from_timestamp"] = self._start_time
        if self._end_time:
            trace_params["to_timestamp"] = self._end_time

        # Create and return TraceDataset
        return TraceDataset(
            langfuse_client=self.langfuse,
            trace_params=trace_params,
            fields=self._fields,
            page_size=self.page_size,
            trace_filter=self._trace_filter,
            trace_processor=self._trace_processor,
            fetch_observations=self._fetch_observations,
            obs_type=self._obs_type,
            obs_limit=self._obs_limit,
            obs_filter=self._obs_filter,
            max_traces=self._max_traces,
            max_observations_per_trace=self._max_observations_per_trace,
            batch_size=self._batch_size,
            sample_rate=self._sample_rate,
        )

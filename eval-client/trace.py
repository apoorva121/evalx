from langfuse import get_client
from langfuse.experiment import Evaluation
from dotenv import load_dotenv
from evalx_sdk import init_context, EvalRun, TraceToDatasetBuilder

# Load environment variables
load_dotenv(".env", override=True)

# Initialize EvalX context (loads .evalx/config.json if exists)
init_context()

langfuse = get_client()
print("✅ Langfuse configured")


# Define your evaluator functions (trace evaluation signature)
def accuracy_evaluator(*, input, output, trace_item, metadata, **kwargs):
    """
    Evaluator that checks if output matches expected output.

    Uses trace evaluation signature:
    - input: The input to the original trace
    - output: The output from the original trace
    - trace_item: The complete dataset item (contains trace_id, trace_core, etc.)
    - metadata: Metadata from the trace (contains trace_core, observations, etc.)
    """
    # For trace-based evaluation, you can access the full trace item
    trace_id = trace_item.get("trace_id")
    expected_output = trace_item.get("expected_output")  # Will be None for production traces

    print(f"  [accuracy_evaluator] Evaluating trace_id={trace_id}")

    if not expected_output:
        return Evaluation(
            name="accuracy",
            value=None,
            comment=f"No expected output for trace {trace_id}"
        )

    # Simple exact match
    score = 1.0 if output == expected_output else 0.0

    return Evaluation(
        name="accuracy",
        value=score,
        comment=f"Exact match: {output == expected_output} (trace: {trace_id})"
    )


def trace_quality_evaluator(*, input, output, trace_item, metadata, **kwargs):
    """
    Evaluator that checks trace metadata for quality signals.

    Uses trace evaluation signature:
    - input: The input to the original trace
    - output: The output from the original trace
    - trace_item: The complete dataset item (contains trace_id, trace_core, etc.)
    - metadata: Metadata from the trace (contains trace_core, observations, etc.)
    """
    trace_id = trace_item.get("trace_id")

    print(f"  [trace_quality_evaluator] Evaluating trace_id={trace_id}")

    if not metadata:
        return Evaluation(
            name="trace_quality",
            value=0.0,
            comment=f"No metadata available (trace: {trace_id})"
        )

    # Extract trace-specific data from metadata
    observations = metadata.get("observations") or []
    trace_core = metadata.get("trace_core") or {}

    # Example: Check if trace has observations and tags
    has_observations = len(observations) > 0
    has_tags = len(trace_core.get("tags", [])) > 0

    quality_score = 0.0
    if has_observations:
        quality_score += 0.5
    if has_tags:
        quality_score += 0.5

    return Evaluation(
        name="trace_quality",
        value=quality_score,
        comment=f"Observations: {len(observations)}, Tags: {has_tags} (trace: {trace_id})"
    )


# Define run-level evaluator (operates on all traces)
def aggregate_stats_evaluator(*, item_results, **kwargs):
    """
    Run-level evaluator that computes aggregate statistics across all traces.

    Uses Langfuse-compatible signature:
    - item_results: List of ExperimentItemResult objects containing all processed traces
    """
    if not item_results:
        return Evaluation(
            name="aggregate_stats",
            value=0,
            comment="No traces to evaluate"
        )

    # Collect all accuracy scores
    accuracy_scores = []
    quality_scores = []

    for result in item_results:
        for evaluation in result.evaluations:
            if isinstance(evaluation, Evaluation):
                if evaluation.name == "accuracy" and evaluation.value is not None:
                    accuracy_scores.append(evaluation.value)
                elif evaluation.name == "trace_quality" and evaluation.value is not None:
                    quality_scores.append(evaluation.value)

    # Calculate aggregate metrics
    avg_accuracy = sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

    # Overall score (weighted average)
    overall_score = (avg_accuracy * 0.7 + avg_quality * 0.3)

    return [
        Evaluation(
            name="avg_accuracy",
            value=avg_accuracy,
            comment=f"Average accuracy across {len(accuracy_scores)} traces"
        ),
        Evaluation(
            name="avg_quality",
            value=avg_quality,
            comment=f"Average quality across {len(quality_scores)} traces"
        ),
        Evaluation(
            name="overall_score",
            value=overall_score,
            comment=f"Overall score (70% accuracy, 30% quality) across {len(item_results)} traces"
        ),
    ]


# Build dataset from traces using the chainable API
#
# TIP: Run explore_traces.py first to see what data exists in your Langfuse project
#      and choose appropriate filters based on your actual trace characteristics.
#
# This example shows common filter patterns - adjust based on your needs:

builder = TraceToDatasetBuilder(
    langfuse,
    default_time_range_days=1,
    page_size=10  # REDUCED from 50 to 10 to avoid timeouts
)

dataset = (
    builder
    .with_fields(["core", "io"])  # REDUCED: Removed metadata to speed up fetch
    .with_time_range(hours=1)  # Get traces from last 6 hours
    # .with_time_range(hours=1)  # Or use hours for shorter windows

    # Example filter patterns - uncomment and adjust as needed:

    # Filter by tags (if your traces have tags):
    # .with_trace_filters(tags=["production"])

    # Filter by release version:
    # .with_trace_filters(release="v1.0.0")

    # Filter by user ID pattern (custom function):
    # .with_trace_filter(lambda trace: trace.core.user_id.startswith("user_"))

    # Filter by session (only traces with sessions):
    # .with_trace_filter(lambda trace: trace.core.session_id is not None)

    # Fetch observations for detailed analysis:
    # .with_observations(
    #     fetch=True,
    #     obs_type="GENERATION",  # Only GENERATION observations
    #     obs_limit=10,           # Max 10 observations per trace
    #     obs_filter=lambda obs: obs.name == "completion"  # Filter by observation name
    # )

    .with_limits(max_traces=4)  # Limit to 10 traces for this demo
    .with_batch_size(2)          # Process in batches of 5
    .with_sample_rate(1.0)       # Use 100% of traces (default is 1% for cost management)
    .build()
)

print("✅ Built TraceDataset iterator from Langfuse traces")

# Track the evaluation run with EvalX
# EvalRun automatically extracts metadata from dataset.run_evaluation() call
with EvalRun():
    # Run evaluation on the trace-based dataset
    # This runs evaluators on existing trace outputs (no task needed, avoids trace loops)
    result = dataset.run_evaluation(
        name="Production Trace Evaluation",
        description="Evaluating production traces for quality and accuracy",
        evaluators=[accuracy_evaluator, trace_quality_evaluator],
        run_evaluators=[aggregate_stats_evaluator],  # Run-level evaluators for aggregate stats
        metadata={"evaluation_type": "production_traces", "version": "v1.0"},
    )

    # Result is an ExperimentResult object (Langfuse-compatible)
    # You can use the .format() method for a human-readable summary:
    # print(result.format())
    # Or access individual item results:
    # for item_result in result.item_results:
    #     print(f"Trace: {item_result.trace_id}, Evaluations: {item_result.evaluations}")

    # Ensure all events are flushed before context exits
    langfuse.flush()

# Alternative usage examples:

# 1. Iterate through dataset items one by one (memory efficient)
# for item in dataset:
#     print(f"Trace ID: {item['trace_id']}")
#     print(f"Input: {item.get('input')}")
#     print(f"Output: {item.get('output')}")
#     print(f"Observations: {len(item.get('observations', []))}")

# 2. Iterate in batches (each batch has complete traces with observations)
# for batch in dataset.iter_batches():
#     print(f"Processing batch of {len(batch)} traces")
#     # Process batch...

# 3. Convert to Langfuse dataset for later use (without running evaluation)
# langfuse_dataset = dataset.to_langfuse_dataset("Production Traces Snapshot")
# # Use it later with your own task
# # result = langfuse_dataset.run_experiment(name="Test", task=my_custom_task)

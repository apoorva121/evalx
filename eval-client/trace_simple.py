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


# Define your evaluator functions (using trace evaluation signature)
def accuracy_evaluator(*, input, output, trace_item, metadata, **kwargs):
    """
    Evaluator that checks if output matches expected output.

    Uses trace evaluation signature:
    - input: The input to the original trace
    - output: The output from the original trace
    - trace_item: The complete dataset item (contains trace_id, trace_core, etc.)
    - metadata: Metadata from the trace
    """
    trace_id = trace_item.get("trace_id")
    expected_output = trace_item.get("expected_output")

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

    if not metadata:
        return Evaluation(
            name="trace_quality",
            value=0.0,
            comment=f"No metadata available (trace: {trace_id})"
        )

    observations = metadata.get("observations", [])
    trace_core = metadata.get("trace_core", {})

    # Example: Check if trace has observations and metadata
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


# Build dataset from traces using the chainable API
# Simplified - no filters, just get last 7 days of traces
builder = TraceToDatasetBuilder(langfuse, default_time_range_days=7, page_size=50)

dataset = (
    builder
    .with_fields(["core", "io"])  # Get core trace data and IO
    .with_time_range(days=7)  # Filter last 7 days
    .with_observations(fetch=True)  # Fetch observations for quality checks
    .with_limits(max_traces=10)  # Global limit on traces
    .with_batch_size(5)  # Batch size for iteration
    .with_sample_rate(1.0)  # Use 100% of traces
    .build()
)

print("✅ Built TraceDataset iterator from Langfuse traces")

# First, let's just iterate to see if we get any traces
print("\n🔍 Checking for traces...")
trace_count = 0
for item in dataset:
    trace_count += 1
    print(f"  - Trace {trace_count}: {item.get('trace_id')} - {item.get('input')} -> {item.get('output')}")
    if trace_count >= 3:  # Just show first 3
        break

print(f"\n✅ Found {trace_count} traces")

if trace_count == 0:
    print("⚠️  No traces found. Skipping evaluation.")
    print("   Try creating some traces first by running eval.py or another script.")
else:
    # Track the evaluation run with EvalX
    # EvalRun automatically extracts metadata from dataset.run_evaluation() call
    with EvalRun():
        # Run evaluation on the trace-based dataset
        # This runs evaluators on existing trace outputs (no task needed, avoids trace loops)
        result = dataset.run_evaluation(
            name="Production Trace Evaluation",
            description="Evaluating production traces for quality and accuracy",
            evaluators=[accuracy_evaluator, trace_quality_evaluator],
            metadata={"evaluation_type": "production_traces", "version": "v1.0"},
        )

        # Use format method to display results within context
        print(result.format())

        # Ensure all events are flushed before context exits
        langfuse.flush()

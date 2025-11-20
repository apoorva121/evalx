"""Inspect the structure of dataset items built from traces."""
from langfuse import get_client
from dotenv import load_dotenv
from evalx_sdk import init_context, TraceToDatasetBuilder
import json

# Load environment variables
load_dotenv(".env", override=True)

# Initialize EvalX context
init_context()

langfuse = get_client()
print("✅ Langfuse configured")

# Build dataset from traces
builder = TraceToDatasetBuilder(
    langfuse,
    default_time_range_days=1,
    page_size=10
)

dataset = (
    builder
    .with_fields(["core", "io"])
    .with_time_range(hours=1)
    .with_limits(max_traces=1)  # Only 1 trace to inspect
    .with_batch_size(1)
    .with_sample_rate(1.0)
    .build()
)

print("✅ Built TraceDataset from Langfuse traces\n")

# Get the first item
items = list(dataset)
if items:
    item = items[0]
    print("=" * 80)
    print("DATASET ITEM STRUCTURE")
    print("=" * 80)
    print(json.dumps(item, indent=2, default=str))
    print("=" * 80)
    print("\nTop-level keys:")
    for key in item.keys():
        print(f"  - {key}: {type(item[key]).__name__}")
else:
    print("❌ No items found")

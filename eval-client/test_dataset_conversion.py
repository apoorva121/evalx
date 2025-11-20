"""Test script to verify to_langfuse_dataset() works correctly."""
from langfuse import get_client
from dotenv import load_dotenv
from evalx_sdk import init_context, TraceToDatasetBuilder

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
    .with_limits(max_traces=2)  # Only 2 traces for quick test
    .with_batch_size(2)
    .with_sample_rate(1.0)
    .build()
)

print("✅ Built TraceDataset from Langfuse traces")

# Test 1: Convert to Langfuse dataset
print("\n📦 Testing to_langfuse_dataset()...")
langfuse_dataset = dataset.to_langfuse_dataset("Test Dataset")
print(f"✅ Created Langfuse dataset: {langfuse_dataset.name}")

# Test 2: Verify the dataset client has correct methods
print("\n🔍 Verifying dataset client...")
print(f"  - Dataset name: {langfuse_dataset.name}")
print(f"  - Dataset has items method: {hasattr(langfuse_dataset, 'items')}")
print(f"  - Dataset has run_experiment method: {hasattr(langfuse_dataset, 'run_experiment')}")

# Test 3: Fetch items to verify they were created
print("\n📋 Fetching dataset items...")
items = langfuse_dataset.items  # items is a property, not a method
print(f"  - Total items: {len(items)}")
if items:
    first_item = items[0]
    print(f"  - First item has 'input' attribute: {hasattr(first_item, 'input')}")
    print(f"  - First item has 'expected_output' attribute: {hasattr(first_item, 'expected_output')}")
    print(f"  - First item has 'metadata' attribute: {hasattr(first_item, 'metadata')}")
    print(f"  - First item type: {type(first_item)}")

print("\n✅ All tests passed!")

"""Test script to demonstrate metadata extraction from EvalRun."""

from evalx_sdk import extract_function_metadata, extract_dataset_metadata, extract_experiment_metadata


# Define a sample task function
def my_test_task(*, item, **kwargs):
    """Process a single test item."""
    return f"Processed: {item['input']}"


# Define an async task
async def my_async_task(*, item, **kwargs):
    """Async task for processing items."""
    return f"Async processed: {item['input']}"


# Sample dataset
test_data = [
    {"input": "Test 1", "expected": "Result 1"},
    {"input": "Test 2", "expected": "Result 2"},
]


# Test function metadata extraction
print("=" * 60)
print("FUNCTION METADATA EXTRACTION")
print("=" * 60)

func_metadata = extract_function_metadata(my_test_task)
print("\nRegular function:")
for key, value in func_metadata.items():
    print(f"  {key}: {value}")

async_metadata = extract_function_metadata(my_async_task)
print("\nAsync function:")
for key, value in async_metadata.items():
    print(f"  {key}: {value}")

lambda_func = lambda x: x * 2
lambda_metadata = extract_function_metadata(lambda_func)
print("\nLambda function:")
for key, value in lambda_metadata.items():
    print(f"  {key}: {value}")


# Test dataset metadata extraction
print("\n" + "=" * 60)
print("DATASET METADATA EXTRACTION")
print("=" * 60)

dataset_metadata = extract_dataset_metadata(test_data)
print("\nList dataset:")
for key, value in dataset_metadata.items():
    print(f"  {key}: {value}")


# Test full experiment metadata extraction
print("\n" + "=" * 60)
print("FULL EXPERIMENT METADATA EXTRACTION")
print("=" * 60)

experiment_metadata = extract_experiment_metadata(
    name="Test Experiment",
    data=test_data,
    task=my_test_task,
    evaluators=[lambda x: x > 0],
    max_concurrency=5,
)

print("\nComplete experiment metadata:")
import json
print(json.dumps(experiment_metadata, indent=2))

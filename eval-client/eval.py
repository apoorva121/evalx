from langfuse import get_client
from langfuse.openai import OpenAI
import os
# Import Langfuse for tracing
from dotenv import load_dotenv
from evalx_sdk import init_context, EvalRun

# Load environment variables
load_dotenv('.env', override=True)

# Initialize EvalX context (loads .evalx/config.json if exists)
init_context()

langfuse = get_client()
print("✅ Langfuse configured")

# Define your task function
def my_task(*, item, **kwargs):
    question = item["input"]
    response = OpenAI().chat.completions.create(
        model="gpt-4.1", messages=[{"role": "user", "content": question}]
    )

    return response.choices[0].message.content


# Run experiment on local data
local_data = [
    {"input": "What is the capital of France?", "expected_output": "Paris"},
    {"input": "What is the capital of Germany?", "expected_output": "Berlin"},
]

# Track the experiment run with EvalX
# EvalRun automatically extracts metadata from local variables (my_task, local_data)
with EvalRun():
    result = langfuse.run_experiment(
        name="Geography Quiz",
        description="Testing basic functionality",
        data=local_data,
        task=my_task,
    )

    # Use format method to display results within context
    print(result.format(include_item_results=True))

    # Ensure all events are flushed before context exits
    langfuse.flush()
"""Quick check to see if traces exist in last 1 hour"""
from langfuse import get_client
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv(".env", override=True)

langfuse = get_client()
print("✅ Langfuse configured\n")

# Check last 1 hour
from_time = datetime.now() - timedelta(hours=1)

try:
    response = langfuse.api.trace.list(
        limit=5,
        page=1,
        from_timestamp=from_time,
        fields="id,name,timestamp"
    )

    trace_count = len(response.data) if hasattr(response, 'data') else 0
    print(f"Found {trace_count} traces in last 1 hour")

    if trace_count > 0:
        print("\nFirst few traces:")
        for trace in response.data[:3]:
            print(f"  - ID: {trace.id}, Name: {trace.name}, Time: {trace.timestamp}")
    else:
        print("⚠️  No traces found in last 1 hour")
        print("   Try running with a longer time window (e.g., days=1 or days=7)")

except Exception as e:
    print(f"❌ Error: {e}")

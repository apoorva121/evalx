"""
Debug script to test Langfuse trace fetching with different parameters.
This helps identify if the timeout is due to:
1. Too many traces in the time window
2. Page size too large
3. Network/Langfuse server issues
"""
from langfuse import get_client
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv(".env", override=True)

langfuse = get_client()
print("✅ Langfuse configured\n")

# Test 1: Fetch a small page of traces with minimal fields
print("=" * 60)
print("Test 1: Fetch 5 traces with minimal fields")
print("=" * 60)
try:
    start = time.time()
    response = langfuse.api.trace.list(
        limit=5,
        page=1,
        fields="id,name,timestamp"  # Minimal fields
    )
    elapsed = time.time() - start
    trace_count = len(response.data) if hasattr(response, 'data') else 0
    print(f"✅ Success: Fetched {trace_count} traces in {elapsed:.2f}s")
except Exception as e:
    elapsed = time.time() - start
    print(f"❌ Failed after {elapsed:.2f}s: {e}")

# Test 2: Fetch larger page
print("\n" + "=" * 60)
print("Test 2: Fetch 50 traces with minimal fields")
print("=" * 60)
try:
    start = time.time()
    response = langfuse.api.trace.list(
        limit=50,
        page=1,
        fields="id,name,timestamp"
    )
    elapsed = time.time() - start
    trace_count = len(response.data) if hasattr(response, 'data') else 0
    print(f"✅ Success: Fetched {trace_count} traces in {elapsed:.2f}s")
except Exception as e:
    elapsed = time.time() - start
    print(f"❌ Failed after {elapsed:.2f}s: {e}")

# Test 3: Fetch with all fields (like your script)
print("\n" + "=" * 60)
print("Test 3: Fetch 5 traces with ALL fields (core,io,metadata)")
print("=" * 60)
try:
    start = time.time()
    response = langfuse.api.trace.list(
        limit=5,
        page=1,
        fields="id,name,timestamp,userId,sessionId,release,version,tags,input,output,metadata"
    )
    elapsed = time.time() - start
    trace_count = len(response.data) if hasattr(response, 'data') else 0
    print(f"✅ Success: Fetched {trace_count} traces in {elapsed:.2f}s")

    # Show data size
    if trace_count > 0:
        import sys
        data_size = sys.getsizeof(str(response.data))
        print(f"   Data size: {data_size:,} bytes ({data_size/1024:.1f} KB)")
except Exception as e:
    elapsed = time.time() - start
    print(f"❌ Failed after {elapsed:.2f}s: {e}")

# Test 4: Fetch 50 traces with all fields (your exact scenario)
print("\n" + "=" * 60)
print("Test 4: Fetch 50 traces with ALL fields (your exact scenario)")
print("=" * 60)
try:
    start = time.time()
    response = langfuse.api.trace.list(
        limit=50,
        page=1,
        fields="id,name,timestamp,userId,sessionId,release,version,tags,input,output,metadata"
    )
    elapsed = time.time() - start
    trace_count = len(response.data) if hasattr(response, 'data') else 0
    print(f"✅ Success: Fetched {trace_count} traces in {elapsed:.2f}s")

    if trace_count > 0:
        import sys
        data_size = sys.getsizeof(str(response.data))
        print(f"   Data size: {data_size:,} bytes ({data_size/1024:.1f} KB)")
except Exception as e:
    elapsed = time.time() - start
    print(f"❌ Failed after {elapsed:.2f}s: {e}")

# Test 5: Check total trace count in time window (7 days)
print("\n" + "=" * 60)
print("Test 5: Count total traces in last 7 days")
print("=" * 60)
from datetime import datetime, timedelta
try:
    start = time.time()
    from_time = datetime.now() - timedelta(days=7)

    # Fetch first page to check
    response = langfuse.api.trace.list(
        limit=1,
        page=1,
        from_timestamp=from_time,
        fields="id"
    )
    elapsed = time.time() - start

    # Note: Langfuse doesn't return total count in API, so we just check if any exist
    if hasattr(response, 'data') and response.data:
        print(f"✅ Found traces in last 7 days (fetched in {elapsed:.2f}s)")
        print(f"   Note: Langfuse API doesn't provide total count")
    else:
        print(f"⚠️  No traces found in last 7 days")
except Exception as e:
    elapsed = time.time() - start
    print(f"❌ Failed after {elapsed:.2f}s: {e}")

# Test 6: Fetch traces from last 1 hour
print("\n" + "=" * 60)
print("Test 6: Fetch 10 traces from last 1 hour")
print("=" * 60)
try:
    start = time.time()
    from_time = datetime.now() - timedelta(hours=1)

    response = langfuse.api.trace.list(
        limit=10,
        page=1,
        from_timestamp=from_time,
        fields="id,name,timestamp,input,output"
    )
    elapsed = time.time() - start
    trace_count = len(response.data) if hasattr(response, 'data') else 0
    print(f"✅ Success: Fetched {trace_count} traces in {elapsed:.2f}s")

    if trace_count > 0:
        import sys
        data_size = sys.getsizeof(str(response.data))
        print(f"   Data size: {data_size:,} bytes ({data_size/1024:.1f} KB)")
    else:
        print(f"   ⚠️  No traces found in last 1 hour")
except Exception as e:
    elapsed = time.time() - start
    print(f"❌ Failed after {elapsed:.2f}s: {e}")

# Test 7: Fetch traces from last 30 minutes
print("\n" + "=" * 60)
print("Test 7: Fetch 10 traces from last 30 minutes")
print("=" * 60)
try:
    start = time.time()
    from_time = datetime.now() - timedelta(minutes=30)

    response = langfuse.api.trace.list(
        limit=10,
        page=1,
        from_timestamp=from_time,
        fields="id,name,timestamp,input,output"
    )
    elapsed = time.time() - start
    trace_count = len(response.data) if hasattr(response, 'data') else 0
    print(f"✅ Success: Fetched {trace_count} traces in {elapsed:.2f}s")

    if trace_count > 0:
        import sys
        data_size = sys.getsizeof(str(response.data))
        print(f"   Data size: {data_size:,} bytes ({data_size/1024:.1f} KB)")
    else:
        print(f"   ⚠️  No traces found in last 30 minutes")
except Exception as e:
    elapsed = time.time() - start
    print(f"❌ Failed after {elapsed:.2f}s: {e}")

# Test 8: Fetch traces from last 10 minutes
print("\n" + "=" * 60)
print("Test 8: Fetch 10 traces from last 10 minutes")
print("=" * 60)
try:
    start = time.time()
    from_time = datetime.now() - timedelta(minutes=10)

    response = langfuse.api.trace.list(
        limit=10,
        page=1,
        from_timestamp=from_time,
        fields="id,name,timestamp,input,output"
    )
    elapsed = time.time() - start
    trace_count = len(response.data) if hasattr(response, 'data') else 0
    print(f"✅ Success: Fetched {trace_count} traces in {elapsed:.2f}s")

    if trace_count > 0:
        import sys
        data_size = sys.getsizeof(str(response.data))
        print(f"   Data size: {data_size:,} bytes ({data_size/1024:.1f} KB)")
    else:
        print(f"   ⚠️  No traces found in last 10 minutes")
except Exception as e:
    elapsed = time.time() - start
    print(f"❌ Failed after {elapsed:.2f}s: {e}")

# Test 9: Fetch traces from specific 1-hour window (compare with broader window)
print("\n" + "=" * 60)
print("Test 9: Fetch 10 traces from specific 1-hour window (2-3 hours ago)")
print("=" * 60)
try:
    start = time.time()
    from_time = datetime.now() - timedelta(hours=3)
    to_time = datetime.now() - timedelta(hours=2)

    response = langfuse.api.trace.list(
        limit=10,
        page=1,
        from_timestamp=from_time,
        to_timestamp=to_time,
        fields="id,name,timestamp,input,output"
    )
    elapsed = time.time() - start
    trace_count = len(response.data) if hasattr(response, 'data') else 0
    print(f"✅ Success: Fetched {trace_count} traces in {elapsed:.2f}s")

    if trace_count > 0:
        import sys
        data_size = sys.getsizeof(str(response.data))
        print(f"   Data size: {data_size:,} bytes ({data_size/1024:.1f} KB)")
    else:
        print(f"   ⚠️  No traces found in that time window")
except Exception as e:
    elapsed = time.time() - start
    print(f"❌ Failed after {elapsed:.2f}s: {e}")

print("\n" + "=" * 60)
print("Diagnostics Complete")
print("=" * 60)
print("\nRecommendations:")
print("1. If Test 1 fails: Network or Langfuse server issue")
print("2. If Test 1 passes but Test 2/3 fails: Try smaller page_size")
print("3. If Test 3 passes but Test 4 fails: Reduce page_size or fields")
print("4. If Tests 1-4 pass but 5 fails: Time window too large, try smaller windows")
print("5. If Tests 6-8 succeed: Use hour/minute-based time windows instead of days")
print("6. If all pass: The issue might be intermittent or network-related")

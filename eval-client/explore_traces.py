"""
Explore traces in Langfuse to understand what data is available.
This helps us choose appropriate filters for the trace evaluation example.
"""
from langfuse import get_client
from dotenv import load_dotenv
from collections import Counter

# Load environment variables
load_dotenv(".env", override=True)

langfuse = get_client()
print("✅ Langfuse configured\n")

# Fetch some traces to explore
print("🔍 Fetching traces to explore...")
traces_response = langfuse.api.trace.list(limit=50, page=1)

if not traces_response.data:
    print("❌ No traces found in your Langfuse project.")
    print("   Try running eval.py first to create some traces.")
    exit(0)

traces = traces_response.data
print(f"✅ Found {len(traces)} traces\n")

# Analyze trace characteristics
print("=" * 80)
print("TRACE ANALYSIS")
print("=" * 80)

# Collect statistics
tags_counter = Counter()
user_ids = set()
session_ids = set()
releases = set()
has_input = 0
has_output = 0
has_metadata = 0

for trace in traces:
    # Tags
    if hasattr(trace, 'tags') and trace.tags:
        for tag in trace.tags:
            tags_counter[tag] += 1

    # User IDs
    if hasattr(trace, 'user_id') and trace.user_id:
        user_ids.add(trace.user_id)

    # Session IDs
    if hasattr(trace, 'session_id') and trace.session_id:
        session_ids.add(trace.session_id)

    # Releases
    if hasattr(trace, 'release') and trace.release:
        releases.add(trace.release)

    # Input/Output
    if hasattr(trace, 'input') and trace.input:
        has_input += 1
    if hasattr(trace, 'output') and trace.output:
        has_output += 1
    if hasattr(trace, 'metadata') and trace.metadata:
        has_metadata += 1

# Print statistics
print(f"\n📊 Statistics from {len(traces)} traces:")
print(f"   - Traces with input: {has_input}/{len(traces)} ({100*has_input/len(traces):.1f}%)")
print(f"   - Traces with output: {has_output}/{len(traces)} ({100*has_output/len(traces):.1f}%)")
print(f"   - Traces with metadata: {has_metadata}/{len(traces)} ({100*has_metadata/len(traces):.1f}%)")
print(f"   - Unique user IDs: {len(user_ids)}")
print(f"   - Unique session IDs: {len(session_ids)}")
print(f"   - Unique releases: {len(releases)}")

if tags_counter:
    print(f"\n🏷️  Tags found:")
    for tag, count in tags_counter.most_common(10):
        print(f"   - '{tag}': {count} traces")
else:
    print(f"\n🏷️  No tags found on traces")

if user_ids:
    print(f"\n👤 User IDs (sample):")
    for user_id in list(user_ids)[:5]:
        print(f"   - {user_id}")
    if len(user_ids) > 5:
        print(f"   ... and {len(user_ids) - 5} more")

if releases:
    print(f"\n📦 Releases found:")
    for release in list(releases)[:5]:
        print(f"   - {release}")

# Show sample traces
print(f"\n" + "=" * 80)
print("SAMPLE TRACES")
print("=" * 80)

for i, trace in enumerate(traces[:3], 1):
    print(f"\n--- Trace {i} ---")
    print(f"ID: {trace.id}")
    print(f"Name: {getattr(trace, 'name', 'N/A')}")
    print(f"User ID: {getattr(trace, 'user_id', 'N/A')}")
    print(f"Session ID: {getattr(trace, 'session_id', 'N/A')}")
    print(f"Tags: {getattr(trace, 'tags', [])}")
    print(f"Release: {getattr(trace, 'release', 'N/A')}")
    print(f"Input: {str(getattr(trace, 'input', 'N/A'))[:100]}...")
    print(f"Output: {str(getattr(trace, 'output', 'N/A'))[:100]}...")

    # Check for observations
    try:
        obs_response = langfuse.api.observations.get_many(trace_id=trace.id)
        obs_count = len(obs_response.data) if obs_response.data else 0
        print(f"Observations: {obs_count}")
        if obs_count > 0:
            print(f"  Types: {[obs.type for obs in obs_response.data[:3]]}")
            print(f"  Names: {[obs.name for obs in obs_response.data[:3]]}")
    except:
        print(f"Observations: Unable to fetch")

# Provide recommendations
print(f"\n" + "=" * 80)
print("RECOMMENDATIONS FOR FILTERS")
print("=" * 80)

recommendations = []

if tags_counter:
    most_common_tag = tags_counter.most_common(1)[0][0]
    recommendations.append(f"✅ Filter by tag: .with_trace_filters(tags=[\"{most_common_tag}\"])")

if user_ids and len(user_ids) > 1:
    sample_user = list(user_ids)[0]
    if sample_user.startswith('user'):
        recommendations.append(f"✅ Filter by user_id pattern: .with_trace_filter(lambda trace: trace.core.user_id.startswith('user'))")

if releases and len(releases) > 1:
    sample_release = list(releases)[0]
    recommendations.append(f"✅ Filter by release: .with_trace_filters(release=\"{sample_release}\")")

if has_output >= len(traces) * 0.8:
    recommendations.append(f"✅ Most traces have outputs - good for evaluation!")

if recommendations:
    print("\n📝 Suggested filters based on your data:")
    for rec in recommendations:
        print(f"   {rec}")
else:
    print("\n💡 Your traces don't have many distinguishing features.")
    print("   Consider using simple filters like:")
    print("   - .with_limits(max_traces=10)  # Limit number of traces")
    print("   - .with_time_range(days=1)     # Recent traces only")

print("\n" + "=" * 80)
print(f"✨ Exploration complete!")
print("=" * 80)

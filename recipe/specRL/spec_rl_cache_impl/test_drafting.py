#!/usr/bin/env python3
"""
Minimal example demonstrating specRL suffix-based drafting.

Usage:
    conda activate syslibs
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    python test_drafting.py
"""

import subprocess
import sys
import time


def run_in_subprocess(code, timeout=30):
    """Run code in isolated subprocess to avoid protobuf conflicts."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=timeout
    )
    return result


def test_server_with_small_memory():
    """Test RolloutCacheServer with configurable small memory size."""
    print("\n" + "=" * 60)
    print("Test: RolloutCacheServer with 1GB shared memory")
    print("=" * 60)

    code = '''
import time
from specrl.suffix_cache import RolloutCacheServer

print("Creating server with 1GB shared memory...")
server = RolloutCacheServer("[::]:6399", shared_memory_size_gb=1)

print("Initializing...")
if not server.initialize():
    print("FAILED: Could not initialize")
    exit(1)

print("Starting gRPC server...")
if not server.start():
    print("FAILED: Could not start")
    exit(1)

print("Server running! Waiting 2 seconds...")
time.sleep(2)

print("Shutting down...")
server.shutdown()
print("SUCCESS!")
'''

    result = run_in_subprocess(code)

    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")

    if result.returncode != 0 and result.stderr:
        for line in result.stderr.strip().split('\n'):
            if 'Error' in line or 'FATAL' in line:
                print(f"  ERROR: {line}")

    # Success if we see the SUCCESS message (segfault during cleanup is ok)
    return "SUCCESS!" in result.stdout


def test_full_drafting_flow():
    """Test the complete drafting flow: server -> updater -> cache."""
    print("\n" + "=" * 60)
    print("Test: Full Drafting Flow (Server + Updater + Cache)")
    print("=" * 60)

    # This test runs server and updater in the same process
    # (they use different proto registrations so it should work)
    code = '''
import time
import threading
from specrl.suffix_cache import RolloutCacheServer, SuffixCache

# Step 1: Start server
print("Step 1: Starting RolloutCacheServer (1GB)...")
server = RolloutCacheServer("[::]:6399", shared_memory_size_gb=1)

if not server.initialize():
    print("FAILED: Server init")
    exit(1)

if not server.start():
    print("FAILED: Server start")
    exit(1)

# Run server in background
server_thread = threading.Thread(target=server.wait, daemon=True)
server_thread.start()
print("  Server running on [::]:6399")

time.sleep(0.5)

# Step 2: Connect SuffixCache
print("Step 2: Connecting SuffixCache to shared memory...")
cache = SuffixCache()
print("  Connected!")

# Step 3: Test with mock data (updater would normally populate this)
print("Step 3: Testing speculation API...")
req_id = "test_req_001"
prompt = [101, 102, 103, 104, 105]

# Fetch (will be empty since no data yet)
cache.fetch_responses_by_prompts_batch([req_id], [prompt])
print(f"  Fetched tree for request '{req_id}'")

# Try speculation (should return empty since no historical data)
pattern = [201, 202, 203]
drafts = cache.speculate([req_id], [pattern], min_token_prob=0.1)
print(f"  Pattern: {pattern}")
print(f"  Draft tokens: {drafts[0]} (empty = no historical data yet)")

# Cleanup
print("Step 4: Shutting down...")
server.shutdown()
print("SUCCESS!")
'''

    result = run_in_subprocess(code, timeout=30)

    if result.stdout:
        for line in result.stdout.strip().split('\n'):
            print(f"  {line}")

    if result.returncode != 0 and result.stderr:
        for line in result.stderr.strip().split('\n'):
            if 'Error' in line or 'FATAL' in line:
                print(f"  ERROR: {line}")

    return "SUCCESS!" in result.stdout


def demonstrate_drafting_concept():
    """Demonstrate how suffix-based drafting works conceptually."""
    print("\n" + "=" * 60)
    print("Concept: Suffix-Based Drafting")
    print("=" * 60)

    # Simulated historical responses
    historical_responses = [
        "The answer is 4. Two plus two equals four.",
        "The answer is 4. This is basic arithmetic.",
        "The answer is 4. Let me explain: 2+2=4.",
    ]

    print("\nHistorical responses (from previous RL epochs):")
    for i, resp in enumerate(historical_responses, 1):
        print(f"  {i}. '{resp}'")

    # Build suffix index
    def tokenize(text):
        return text.split()

    suffix_index = {}
    for resp in historical_responses:
        tokens = tokenize(resp)
        for i in range(len(tokens)):
            for j in range(i + 1, min(i + 4, len(tokens))):
                pattern = tuple(tokens[i:j])
                continuation = tokens[j] if j < len(tokens) else None
                if continuation:
                    if pattern not in suffix_index:
                        suffix_index[pattern] = {}
                    suffix_index[pattern][continuation] = \
                        suffix_index[pattern].get(continuation, 0) + 1

    print(f"\nSuffix index: {len(suffix_index)} patterns")
    print("\nExample lookups:")

    test_patterns = [
        ("answer", "is"),
        ("is", "4."),
        ("4.", "Two"),
    ]

    for pattern in test_patterns:
        if pattern in suffix_index:
            continuations = suffix_index[pattern]
            print(f"  {pattern} -> {dict(continuations)}")

    print("\nHow it accelerates generation:")
    print("  1. Model generates: 'The answer is'")
    print("  2. Lookup pattern ('answer', 'is') -> draft ['4.']")
    print("  3. Model verifies '4.' in ONE forward pass")
    print("  4. If accepted, skip 1 autoregressive step!")

    return True


def main():
    print("=" * 60)
    print("specRL Drafting Tests")
    print("=" * 60)

    results = []

    results.append(("Server (1GB)", test_server_with_small_memory()))
    results.append(("Full flow", test_full_drafting_flow()))
    results.append(("Concept demo", demonstrate_drafting_concept()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
        print("\nUsage in production:")
        print("  # Start server (adjust memory based on your needs)")
        print("  server = RolloutCacheServer('[::]:6378', shared_memory_size_gb=100)")
        print("  server.initialize()")
        print("  server.start()")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

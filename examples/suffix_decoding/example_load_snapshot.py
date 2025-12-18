"""
Simple example: Load suffix tree snapshot into vLLM.

This example demonstrates how to:
1. Create a suffix tree cache and add patterns
2. Initialize vLLM with suffix decoding via monkey patches
3. Load the snapshot into vLLM
4. Run generation with the loaded patterns

Usage:
    python example_load_snapshot.py
"""

import numpy as np

# Apply monkey patches BEFORE importing vLLM
from verl.workers.rollout.vllm_rollout.patches import apply_all_patches
apply_all_patches()

from vllm import LLM, SamplingParams
from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

# 1. Create cache and add patterns
cache = ParallelSuffixDecodingCache(max_tree_depth=64)
cache.start_request("req0", np.array([1, 2, 3, 4, 5], dtype=np.int32))
cache.add_tokens("req0", np.array([6, 7, 8, 9, 10], dtype=np.int32))

# 2. Create snapshot with hash mapping
snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)
total_bytes = sum(len(s[1]) for s in snapshots)
print(f"Snapshot: {len(snapshots)} trees, {total_bytes} bytes")

# 3. Initialize vLLM with suffix decoding
# The monkey patches enable speculative_config with method="suffix"
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 5,
    },
)

# 4. Load snapshot using the patched load_snapshot method
llm.load_snapshot(snapshots, hash_mapping)
print("Snapshot loaded successfully!")

# 5. Generate
outputs = llm.generate(
    ["What is the capital of France?"],
    SamplingParams(temperature=0.7, max_tokens=50),
)

print(outputs[0].outputs[0].text)

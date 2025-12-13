"""
Simple example: Load suffix tree snapshot into vLLM.
"""

import numpy as np
from vllm import LLM, SamplingParams
from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

# 1. Create cache and add patterns
cache = ParallelSuffixDecodingCache(max_tree_depth=64)
cache.start_request("req0", np.array([1, 2, 3, 4, 5], dtype=np.int32))
cache.add_tokens("req0", np.array([6, 7, 8, 9, 10], dtype=np.int32))

# 2. Create snapshot
snapshots = cache.create_snapshot()
snapshot_bytes = snapshots[0][1] if snapshots else b''
print(f"Snapshot: {len(snapshot_bytes)} bytes")

# 3. Initialize vLLM with suffix decoding
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "suffix",
        "num_speculative_tokens": 5,
    },
)

# 4. Load snapshot
llm.load_snapshot(snapshot_bytes)

# 5. Generate
outputs = llm.generate(
    ["What is the capital of France?"],
    SamplingParams(temperature=0.7, max_tokens=50),
)

print(outputs[0].outputs[0].text)

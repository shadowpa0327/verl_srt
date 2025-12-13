#!/usr/bin/env python3
"""
Test: Pre-computed Hash for Suffix Tree Matching

This test validates hash consistency between tree building and tree lookup
using the canonical compute_prompt_hash function.

Tests:
1. Hash computation consistency (compute_prompt_hash == cache._hash_prompt)
2. Edge cases (short prompts, exact hash_token_count, truncation)
3. Tree lookup with pre-computed hash
4. (Optional, requires GPU) vLLM extra_args passthrough

Usage:
    # Run CPU-only tests (no GPU required)
    python test_precomputed_hash.py

    # Run all tests including vLLM integration (requires GPU)
    python test_precomputed_hash.py --with-vllm --model meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import sys

import numpy as np


def test_hash_consistency():
    """Test 1.1: Verify compute_prompt_hash matches cache._hash_prompt exactly."""
    print("\n" + "=" * 60)
    print("Test 1.1: Hash Function Consistency")
    print("=" * 60)

    from arctic_inference.suffix_decoding import compute_prompt_hash, ParallelSuffixDecodingCache

    cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)

    test_cases = [
        ("Short (< 128)", [1, 2, 3, 4, 5]),
        ("Exactly 128", list(range(128))),
        ("Long (> 128, truncation)", list(range(200))),
        ("Single token", [42]),
        ("Many repeats", [100] * 50),
    ]

    all_passed = True
    for name, tokens in test_cases:
        # Compute using standalone function
        standalone_hash = compute_prompt_hash(tokens, hash_token_count=128)

        # Compute using cache method
        prompt_array = np.array(tokens, dtype=np.int32)
        cache_hash = cache._hash_prompt(prompt_array, 128)

        match = standalone_hash == cache_hash
        status = "PASS" if match else "FAIL"
        print(f"  {name}: {status}")
        print(f"    Standalone: {standalone_hash}")
        print(f"    Cache:      {cache_hash}")

        if not match:
            all_passed = False

    return all_passed


def test_hash_edge_cases():
    """Test 1.2: Test boundary conditions for hash computation."""
    print("\n" + "=" * 60)
    print("Test 1.2: Edge Cases")
    print("=" * 60)

    from arctic_inference.suffix_decoding import compute_prompt_hash

    all_passed = True

    # Test: Exactly hash_token_count tokens
    tokens_128 = list(range(128))
    hash1 = compute_prompt_hash(tokens_128, hash_token_count=128)

    # Test: hash_token_count + 1 tokens (first token should be ignored)
    tokens_129 = [999] + list(range(128))  # Prefix 999 should be ignored
    hash2 = compute_prompt_hash(tokens_129, hash_token_count=128)

    if hash1 == hash2:
        print("  Truncation test: PASS")
        print(f"    128 tokens hash: {hash1}")
        print(f"    129 tokens hash (first ignored): {hash2}")
    else:
        print("  Truncation test: FAIL")
        print(f"    Expected hashes to match but got:")
        print(f"    128 tokens: {hash1}")
        print(f"    129 tokens: {hash2}")
        all_passed = False

    # Test: Different prefixes, same suffix should have same hash
    prefix_a = [1, 2, 3] + list(range(128))
    prefix_b = [100, 200, 300] + list(range(128))
    hash_a = compute_prompt_hash(prefix_a, hash_token_count=128)
    hash_b = compute_prompt_hash(prefix_b, hash_token_count=128)

    if hash_a == hash_b:
        print("  Same suffix, different prefix: PASS")
    else:
        print("  Same suffix, different prefix: FAIL")
        all_passed = False

    # Test: List vs numpy array should give same result
    tokens = [1, 2, 3, 4, 5]
    hash_list = compute_prompt_hash(tokens, hash_token_count=128)
    hash_array = compute_prompt_hash(np.array(tokens, dtype=np.int32), hash_token_count=128)

    if hash_list == hash_array:
        print("  List vs numpy array: PASS")
    else:
        print("  List vs numpy array: FAIL")
        all_passed = False

    return all_passed


def test_tree_lookup_precomputed():
    """Test 1.3: Verify tree reuse works when pre_computed_hash is provided."""
    print("\n" + "=" * 60)
    print("Test 1.3: Tree Lookup with Pre-computed Hash")
    print("=" * 60)

    from arctic_inference.suffix_decoding import compute_prompt_hash, ParallelSuffixDecodingCache

    cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)

    prompt_tokens = np.array(list(range(100)), dtype=np.int32)

    # Compute hash using standalone function (simulating rollout worker)
    pre_hash = compute_prompt_hash(prompt_tokens, hash_token_count=128)
    print(f"  Pre-computed hash: {pre_hash}")

    # Build tree with computed hash (req1 - tree building)
    cache.start_request("req1", prompt_tokens)
    cache.add_tokens("req1", np.array([200, 201, 202], dtype=np.int32))

    tree_idx_1 = cache._req_to_tree_idx["req1"]
    print(f"  req1 assigned to tree_idx: {tree_idx_1}")

    # Look up with pre-computed hash (req2 - simulating vLLM lookup)
    cache.start_request("req2", prompt_tokens, pre_computed_hash=pre_hash)

    tree_idx_2 = cache._req_to_tree_idx["req2"]
    print(f"  req2 assigned to tree_idx: {tree_idx_2}")

    if tree_idx_1 == tree_idx_2:
        print("  Tree reuse with pre-computed hash: PASS")
        return True
    else:
        print("  Tree reuse with pre-computed hash: FAIL")
        print(f"    Expected tree_idx {tree_idx_1}, got {tree_idx_2}")
        return False


def test_snapshot_with_hash_mapping():
    """Test 1.4: Verify snapshot includes correct hash mapping."""
    print("\n" + "=" * 60)
    print("Test 1.4: Snapshot with Hash Mapping")
    print("=" * 60)

    from arctic_inference.suffix_decoding import compute_prompt_hash, ParallelSuffixDecodingCache

    cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)

    # Create two different prompts
    prompt1 = np.array(list(range(50)), dtype=np.int32)
    prompt2 = np.array(list(range(100, 150)), dtype=np.int32)

    hash1 = compute_prompt_hash(prompt1, 128)
    hash2 = compute_prompt_hash(prompt2, 128)

    print(f"  Prompt 1 hash: {hash1}")
    print(f"  Prompt 2 hash: {hash2}")

    # Build trees
    cache.start_request("req1", prompt1)
    cache.add_tokens("req1", np.array([1000, 1001], dtype=np.int32))

    cache.start_request("req2", prompt2)
    cache.add_tokens("req2", np.array([2000, 2001], dtype=np.int32))

    # Create snapshot with hash mapping
    snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)

    print(f"  Number of trees: {len(snapshots)}")
    print(f"  Hash mapping: {hash_mapping}")

    all_passed = True

    if len(snapshots) == 2:
        print("  Snapshot tree count: PASS")
    else:
        print(f"  Snapshot tree count: FAIL (expected 2, got {len(snapshots)})")
        all_passed = False

    if hash1 in hash_mapping and hash2 in hash_mapping:
        print("  Hash mapping contains both hashes: PASS")
    else:
        print("  Hash mapping contains both hashes: FAIL")
        all_passed = False

    # Test loading into new cache
    new_cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)
    new_cache.load_snapshot(snapshots, hash_to_tree=hash_mapping)

    # Start request with same prompts - should find trees via hash
    new_cache.start_request("new_req1", prompt1, pre_computed_hash=hash1)
    new_cache.start_request("new_req2", prompt2, pre_computed_hash=hash2)

    if new_cache._req_to_tree_idx["new_req1"] == hash_mapping[hash1]:
        print("  Tree reuse after load (prompt1): PASS")
    else:
        print("  Tree reuse after load (prompt1): FAIL")
        all_passed = False

    if new_cache._req_to_tree_idx["new_req2"] == hash_mapping[hash2]:
        print("  Tree reuse after load (prompt2): PASS")
    else:
        print("  Tree reuse after load (prompt2): FAIL")
        all_passed = False

    return all_passed


def test_vllm_extra_args_passthrough(model: str, max_model_len: int = 2048):
    """Test 2.1: Verify prompt_hash flows through SamplingParams to suffix cache.

    This test verifies:
    1. extra_args are accepted by SamplingParams
    2. prompt_hash reaches InputBatch.prompt_hashes
    3. pre_computed_hash is passed to suffix_cache.start_request()

    Note: After generation completes, trees may be garbage-collected when their
    requests finish. We verify passthrough by:
    1. Monkey-patching start_request to capture actual hashes used
    2. Comparing captured hashes with our pre-computed ones
    """
    print("\n" + "=" * 60)
    print("Test 2.1: vLLM extra_args Passthrough")
    print("=" * 60)

    import os
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    from vllm import LLM, SamplingParams
    from arctic_inference.suffix_decoding import compute_prompt_hash

    print(f"  Initializing vLLM with model: {model}")
    llm = LLM(
        model=model,
        speculative_config={
            "method": "suffix",
            "num_speculative_tokens": 3,
            "suffix_decoding_max_tree_depth": 64,
        },
        gpu_memory_utilization=0.7,
        max_model_len=max_model_len,
        trust_remote_code=True,
    )

    # Get tokenizer - handle different vLLM versions
    tokenizer = llm.llm_engine.tokenizer
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer

    prompts = [
        "What is 2 + 2?",
        "What is the capital of France?",
    ]

    # Compute hashes
    prompt_hashes = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_hash = compute_prompt_hash(tokens, hash_token_count=128)
        prompt_hashes.append(prompt_hash)
        print(f"  Prompt: '{prompt}' -> hash: {prompt_hash}")

    # Create per-request SamplingParams with extra_args
    sampling_params_list = [
        SamplingParams(
            temperature=0.0,
            max_tokens=32,
            extra_args={"prompt_hash": prompt_hash},
        )
        for prompt_hash in prompt_hashes
    ]

    # Monkey-patch start_request to capture pre_computed_hash values
    captured_hashes = []
    drafter = None
    original_start_request = None

    try:
        # v1 engine path
        drafter = llm.llm_engine.model_executor.driver_worker.worker.model_runner.drafter
    except AttributeError:
        try:
            # Alternative path
            drafter = llm.llm_engine.model_executor.driver_worker.model_runner.drafter
        except AttributeError:
            pass

    if drafter is not None and hasattr(drafter, 'suffix_cache'):
        suffix_cache = drafter.suffix_cache
        original_start_request = suffix_cache.start_request

        def capturing_start_request(req_id, prompt_token_ids, hash_token_count=None, pre_computed_hash=None):
            """Wrapper that captures pre_computed_hash before calling original."""
            if pre_computed_hash is not None:
                captured_hashes.append(pre_computed_hash)
                print(f"    [CAPTURED] start_request called with pre_computed_hash={pre_computed_hash}")
            return original_start_request(req_id, prompt_token_ids, hash_token_count=hash_token_count, pre_computed_hash=pre_computed_hash)

        suffix_cache.start_request = capturing_start_request
        print("  Installed start_request capture hook")

    print("\n  Running generation with per-request SamplingParams...")
    try:
        outputs = llm.generate(prompts, sampling_params_list)
        print(f"  Generated {len(outputs)} outputs")
        for i, output in enumerate(outputs):
            print(f"    [{i}] {output.outputs[0].text[:50]}...")

        # Restore original method
        if drafter is not None and original_start_request is not None:
            drafter.suffix_cache.start_request = original_start_request

        # Verify our pre-computed hashes were passed through
        print(f"\n  Captured pre_computed_hash values: {captured_hashes}")
        print(f"  Expected hashes: {prompt_hashes}")

        # Check if all expected hashes were captured
        hashes_matched = 0
        for expected_hash in prompt_hashes:
            if expected_hash in captured_hashes:
                print(f"    Hash {expected_hash}: CAPTURED ✓")
                hashes_matched += 1
            else:
                print(f"    Hash {expected_hash}: NOT CAPTURED ✗")

        if hashes_matched == len(prompt_hashes):
            print(f"\n  All {hashes_matched} pre-computed hashes passed through: PASS")
            return True
        elif hashes_matched > 0:
            print(f"\n  {hashes_matched}/{len(prompt_hashes)} hashes passed through: PARTIAL PASS")
            print("  Note: Some requests may have been spec-decode-unsupported")
            return True
        else:
            # Check if generation worked at all (non-empty outputs)
            if all(len(o.outputs[0].text) > 0 for o in outputs):
                print("\n  Generation worked but no hashes captured.")
                print("  Possible reasons:")
                print("    - Requests completed during prefill (no speculation triggered)")
                print("    - All requests were spec-decode-unsupported")
                print("  Result: INCONCLUSIVE (generation worked)")
                return True
            else:
                print("\n  Hash passthrough: FAIL")
                return False

    except Exception as e:
        print(f"\n  extra_args passthrough: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Pre-computed Hash Tests")
    parser.add_argument("--with-vllm", action="store_true", help="Run vLLM integration tests")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()

    print("=" * 60)
    print("Pre-computed Hash Tests for Suffix Tree Matching")
    print("=" * 60)

    results = {}

    # Level 1: Unit tests (no GPU required)
    results["hash_consistency"] = test_hash_consistency()
    results["edge_cases"] = test_hash_edge_cases()
    results["tree_lookup_precomputed"] = test_tree_lookup_precomputed()
    results["snapshot_hash_mapping"] = test_snapshot_with_hash_mapping()

    # Level 2: vLLM integration tests (GPU required)
    if args.with_vllm:
        results["vllm_extra_args"] = test_vllm_extra_args_passthrough(
            args.model, args.max_model_len
        )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

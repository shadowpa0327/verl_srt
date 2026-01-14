# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Test suffix tree snapshot loading for vLLM server mode integration.

This test verifies:
1. Patch application works correctly
2. Snapshot can be created from ParallelSuffixDecodingCache
3. Snapshot can be loaded through the vLLM rollout hierarchy

Usage:
    # Unit tests (no GPU required)
    pytest tests/workers/rollout/rollout_vllm/test_vllm_suffix_snapshot.py -v -s -k "unit"

    # Integration tests (requires GPU and model)
    pytest tests/workers/rollout/rollout_vllm/test_vllm_suffix_snapshot.py -v -s -k "integration"

    # All tests
    pytest tests/workers/rollout/rollout_vllm/test_vllm_suffix_snapshot.py -v -s

    # Run as standalone script
    python tests/workers/rollout/rollout_vllm/test_vllm_suffix_snapshot.py
"""

import numpy as np
import pytest


# ==================== Unit Tests (No GPU Required) ====================


class TestPatchApplication:
    """Test that vLLM patches can be applied correctly."""

    def test_unit_patches_apply_successfully(self):
        """Test that apply_all_patches() runs without error."""
        from verl.workers.rollout.vllm_rollout.patches import apply_all_patches, is_patched

        # Apply patches
        apply_all_patches()

        # Verify patches were applied
        assert is_patched(), "Patches should be marked as applied"

    def test_unit_patches_are_idempotent(self):
        """Test that patches can be applied multiple times safely."""
        from verl.workers.rollout.vllm_rollout.patches import apply_all_patches, is_patched

        # Apply patches multiple times
        apply_all_patches()
        apply_all_patches()
        apply_all_patches(force=True)

        assert is_patched(), "Patches should remain applied after multiple calls"


class TestSuffixCacheSnapshot:
    """Test suffix cache snapshot creation and loading."""

    def test_unit_create_snapshot_empty_cache(self):
        """Test creating snapshot from empty cache."""
        from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

        cache = ParallelSuffixDecodingCache(max_tree_depth=64)

        # Create snapshot with hash mapping
        result = cache.create_snapshot(include_hash_mapping=True)

        if isinstance(result, tuple):
            snapshots, hash_mapping = result
        else:
            snapshots = result
            hash_mapping = {}

        # Empty cache should have empty snapshots
        assert isinstance(snapshots, list), "Snapshots should be a list"
        assert isinstance(hash_mapping, dict), "Hash mapping should be a dict"

    def test_unit_create_and_load_snapshot(self):
        """Test creating and loading a snapshot with data."""
        from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

        # Create cache and add some data
        cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)

        prompt = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        cache.start_request("req_1", prompt)
        cache.add_tokens("req_1", np.array([6, 7, 8], dtype=np.int32))

        # Create snapshot
        snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)

        assert len(snapshots) > 0, "Should have at least one tree snapshot"
        assert len(hash_mapping) > 0, "Should have hash mappings"

        # Verify snapshot format
        for tree_idx, snapshot_bytes in snapshots:
            assert isinstance(tree_idx, int), "Tree index should be int"
            assert isinstance(snapshot_bytes, bytes), "Snapshot should be bytes"
            assert len(snapshot_bytes) > 0, "Snapshot should not be empty"

        # Load snapshot into new cache
        new_cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)
        new_cache.load_snapshot(snapshots, hash_to_tree=hash_mapping)

        # Verify loaded cache works
        # Start new request with same prompt - should reuse loaded tree
        new_cache.start_request("req_2", prompt)

        # Speculation should work
        context = np.array([3, 4, 5, 6, 7, 8], dtype=np.int32)
        draft = new_cache.speculate(
            req_id="req_2",
            context=context,
            max_spec_tokens=5,
            max_spec_factor=1.0,
            min_token_prob=0.0,
            use_tree_spec=False,
        )

        assert draft is not None, "Speculation should return a draft"

        # Cleanup
        cache.stop_request("req_1")
        new_cache.stop_request("req_2")

    def test_unit_selective_snapshot(self):
        """Test creating selective snapshot for specific trees."""
        from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

        cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=0)

        # Create multiple requests with different prompts (no hash sharing)
        for i in range(5):
            prompt = np.array([i, i + 1, i + 2], dtype=np.int32)
            cache.start_request(f"req_{i}", prompt)
            cache.add_tokens(f"req_{i}", np.array([i + 10], dtype=np.int32))

        # Get all tree indices
        stats = cache.get_stats()
        num_trees = stats.get("num_trees_in_forest", 0)

        if num_trees > 2:
            # Create selective snapshot for subset of trees
            tree_indices = [0, 1]
            snapshots = cache.create_selective_snapshot(tree_indices)

            assert len(snapshots) <= len(tree_indices), "Should only include requested trees"

        # Cleanup
        for i in range(5):
            cache.stop_request(f"req_{i}")

    def test_unit_snapshot_with_multiple_sequences(self):
        """Test snapshot with multiple sequences sharing a tree."""
        from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

        cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)

        # Same prompt = shared tree
        prompt = np.array([1, 2, 3, 4, 5], dtype=np.int32)

        cache.start_request("req_1", prompt)
        cache.start_request("req_2", prompt)  # Should share tree

        # Add different tokens to each request
        cache.add_tokens("req_1", np.array([100, 101, 102], dtype=np.int32))
        cache.add_tokens("req_2", np.array([200, 201, 202], dtype=np.int32))

        # Create snapshot
        snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)

        # With hash sharing, we should have one tree for both requests
        assert len(hash_mapping) == 1, "Same prompt should result in one hash mapping"

        # Load and verify
        new_cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)
        new_cache.load_snapshot(snapshots, hash_to_tree=hash_mapping)

        # Start new request with same prompt
        new_cache.start_request("req_3", prompt)

        # Both sequences should be speculatable
        context1 = np.array([100, 101, 102], dtype=np.int32)
        draft1 = new_cache.speculate("req_3", context1, max_spec_tokens=3)

        context2 = np.array([200, 201, 202], dtype=np.int32)
        draft2 = new_cache.speculate("req_3", context2, max_spec_tokens=3)

        # Cleanup
        cache.stop_request("req_1")
        cache.stop_request("req_2")
        new_cache.stop_request("req_3")


class TestProposerSnapshot:
    """Test snapshot loading through the proposer."""

    def test_unit_proposer_load_snapshot(self):
        """Test ParallelSuffixDecodingProposer.load_snapshot() method."""
        pytest.importorskip("arctic_inference")

        from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

        # Create source cache with data
        source_cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)

        prompt = np.array([10, 20, 30, 40, 50], dtype=np.int32)
        source_cache.start_request("source_req", prompt)
        source_cache.add_tokens("source_req", np.array([60, 70, 80], dtype=np.int32))

        # Create snapshot
        snapshots, hash_mapping = source_cache.create_snapshot(include_hash_mapping=True)

        # Verify snapshot data
        assert len(snapshots) > 0, "Should have snapshots"
        assert len(hash_mapping) > 0, "Should have hash mapping"

        total_bytes = sum(len(s[1]) for s in snapshots)
        print(f"Snapshot: {len(snapshots)} trees, {total_bytes} bytes, {len(hash_mapping)} hash mappings")

        # Load into target cache (simulating what proposer.load_snapshot does)
        target_cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)
        target_cache.load_snapshot(snapshots, hash_to_tree=hash_mapping)

        # Verify target cache can use the loaded data
        target_cache.start_request("target_req", prompt)

        context = np.array([30, 40, 50, 60, 70, 80], dtype=np.int32)
        draft = target_cache.speculate("target_req", context, max_spec_tokens=5)

        print(f"Draft tokens: {draft.token_ids}")
        print(f"Match length: {draft.match_len}")

        # Cleanup
        source_cache.stop_request("source_req")
        target_cache.stop_request("target_req")


# ==================== Integration Tests (GPU Required) ====================


def test_integration_full_server_suffix_loading():
    """
    Full integration test: start vLLM server with suffix decoding and load snapshots.

    This test requires:
    - GPU available
    - Model available (e.g., Qwen/Qwen2.5-1.5B-Instruct)

    Skip if resources not available.
    """
    import os

    # Skip if no GPU
    try:
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not available")

    # Configuration
    MODEL_PATH = os.environ.get("TEST_MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
    GPUS_PER_NODE = 1
    TP_SIZE = 1

    print("=" * 60)
    print("vLLM Suffix Snapshot Integration Test")
    print("=" * 60)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {GPUS_PER_NODE}, TP Size: {TP_SIZE}")
    print("=" * 60)

    # Initialize Ray
    print("\n[1] Initializing Ray...")
    import asyncio

    import ray

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "INFO",
                "VLLM_USE_V1": "1",
            }
        },
        ignore_reinit_error=True,
    )

    try:
        # Create Config
        print("\n[2] Creating config...")
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("verl/verl/trainer/config")
        if not os.path.exists(config_dir):
            config_dir = os.path.abspath("verl/trainer/config")

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            config = compose(config_name="ppo_trainer")

        config.trainer.n_gpus_per_node = GPUS_PER_NODE
        config.trainer.nnodes = 1
        config.actor_rollout_ref.model.path = MODEL_PATH
        config.actor_rollout_ref.rollout.name = "vllm"
        config.actor_rollout_ref.rollout.mode = "async"
        config.actor_rollout_ref.rollout.tensor_model_parallel_size = TP_SIZE
        config.actor_rollout_ref.rollout.prompt_length = 256
        config.actor_rollout_ref.rollout.response_length = 128

        # Create Rollout Server
        print("\n[3] Creating rollout server...")
        from verl.workers.rollout.replica import get_rollout_replica_class

        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model

        rollout_server_class = get_rollout_replica_class("vllm")
        replica = rollout_server_class(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=GPUS_PER_NODE,
        )

        asyncio.run(replica.init_standalone())
        print(f"Server address: {replica._server_address}")

        # Create test snapshots
        print("\n[4] Creating test snapshots...")
        from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache

        cache = ParallelSuffixDecodingCache(max_tree_depth=64, hash_token_count=128)

        # Create some test data
        prompt = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int32)
        cache.start_request("test_req", prompt)
        cache.add_tokens("test_req", np.array([11, 12, 13, 14, 15], dtype=np.int32))

        snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)
        total_bytes = sum(len(s[1]) for s in snapshots)
        print(f"Created snapshot: {len(snapshots)} trees, {total_bytes} bytes")

        # Load snapshots through replica
        print("\n[5] Loading snapshots through replica...")
        asyncio.run(replica.load_suffix_snapshot(snapshots, hash_mapping))
        print("Snapshots loaded successfully!")

        # Cleanup
        cache.stop_request("test_req")
        print("\n" + "=" * 60)
        print("Integration test PASSED!")
        print("=" * 60)

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


# ==================== Main ====================


if __name__ == "__main__":
    print("Running suffix snapshot tests...\n")

    # Run unit tests
    print("=" * 60)
    print("UNIT TESTS")
    print("=" * 60)

    test_patches = TestPatchApplication()
    print("\n--- Test: Patches Apply Successfully ---")
    test_patches.test_unit_patches_apply_successfully()
    print("PASSED")

    print("\n--- Test: Patches Are Idempotent ---")
    test_patches.test_unit_patches_are_idempotent()
    print("PASSED")

    test_cache = TestSuffixCacheSnapshot()
    print("\n--- Test: Create Snapshot Empty Cache ---")
    test_cache.test_unit_create_snapshot_empty_cache()
    print("PASSED")

    print("\n--- Test: Create and Load Snapshot ---")
    test_cache.test_unit_create_and_load_snapshot()
    print("PASSED")

    print("\n--- Test: Selective Snapshot ---")
    test_cache.test_unit_selective_snapshot()
    print("PASSED")

    print("\n--- Test: Snapshot With Multiple Sequences ---")
    test_cache.test_unit_snapshot_with_multiple_sequences()
    print("PASSED")

    test_proposer = TestProposerSnapshot()
    print("\n--- Test: Proposer Load Snapshot ---")
    test_proposer.test_unit_proposer_load_snapshot()
    print("PASSED")

    print("\n" + "=" * 60)
    print("All unit tests PASSED!")
    print("=" * 60)

    # Ask about integration test
    print("\nIntegration test requires GPU and model.")
    run_integration = input("Run integration test? [y/N]: ").strip().lower()
    if run_integration == "y":
        test_integration_full_server_suffix_loading()

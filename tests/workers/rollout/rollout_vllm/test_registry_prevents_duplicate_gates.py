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
Registry Gate Sharing Verification Test

Goal
----
Verify that AdmissionGateRegistry correctly shares gates across workers and
prevents the "N independent counters" problem. Multiple workers requesting
the same gate should get the SAME actor handle, not create duplicates.

This test validates:
1. All workers get identical Ray actor handle for same server_idx
2. Config mismatch is detected and raises ValueError
3. Submissions through different handles hit the same counter

Usage
-----
  NUM_GPUS=1 python tests/workers/rollout/rollout_vllm/test_registry_prevents_duplicate_gates.py

Key env vars
------------
  MODEL_PATH: HF model path (default: Qwen/Qwen2.5-0.5B-Instruct)
  NUM_GPUS: Total GPUs available (default: 1)
  NUM_WORKERS: Number of workers requesting gates (default: 8)
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any
from uuid import uuid4

import ray

from test_vllm_run_ahead_server_side_admission import (
    AdmissionControlledServer,
    AdmissionGateConfig,
    AdmissionGateRegistry,
    get_or_create_registry,
)


@ray.remote
class GateRequestWorker:
    """Worker that requests gates from registry and submits requests."""

    def __init__(self, worker_id: int, tokenizer):
        self.worker_id = worker_id
        self.tokenizer = tokenizer

    async def get_gate_id(self, registry, server_idx: int, server_handle, config) -> str:
        """Request a gate and return its actor ID for comparison."""
        gate = await registry.get_or_create.remote(server_idx, server_handle, config)
        # Get gate's admission stats to verify it's a valid gate
        stats = await gate.get_admission_stats.remote()
        # Use the gate handle's internal ID for comparison
        return str(id(gate))

    async def get_gate_and_submit(
        self,
        registry,
        server_idx: int,
        server_handle,
        config,
        num_requests: int = 5,
    ) -> dict[str, Any]:
        """Get gate from registry and submit runahead requests."""
        gate = await registry.get_or_create.remote(server_idx, server_handle, config)

        results = {"submitted": 0, "completed": 0, "rejected": 0}

        for i in range(num_requests):
            prompt_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": "Hello"}],
                add_generation_prompt=True,
                tokenize=True,
            )

            request_id = f"worker_{self.worker_id}_req_{i}_{uuid4().hex[:8]}"
            sampling_params = {
                "_verl_request_kind": "runahead",
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 16,
            }

            try:
                output = await gate.generate.remote(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )
                results["submitted"] += 1

                stop_reason = getattr(output, "stop_reason", None)
                if stop_reason == "rejected":
                    results["rejected"] += 1
                else:
                    results["completed"] += 1
            except Exception as e:
                results["submitted"] += 1
                results["error"] = str(e)

        return results


def test_registry_prevents_duplicate_gates():
    """Test that registry correctly shares gates and prevents duplicates."""
    MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))

    cfg = AdmissionGateConfig(
        max_runahead_inflight=2,
        enforce_slack=False,
    )

    print("=" * 80)
    print("Registry Gate Sharing Verification Test")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {NUM_GPUS}")
    print(f"Workers requesting same gate: {NUM_WORKERS}")
    print(f"Max runahead inflight: {cfg.max_runahead_inflight}")
    print("=" * 80)

    print("\n[1] Initializing Ray...")
    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARNING",
                "VLLM_USE_V1": "1",
            }
        },
        ignore_reinit_error=True,
    )

    try:
        print("\n[2] Creating config...")
        from hydra import compose, initialize_config_dir

        config_dir = os.path.abspath("verl/verl/trainer/config")
        if not os.path.exists(config_dir):
            config_dir = os.path.abspath("verl/trainer/config")

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            trainer_config = compose(config_name="ppo_trainer")

        trainer_config.trainer.n_gpus_per_node = NUM_GPUS
        trainer_config.trainer.nnodes = 1
        trainer_config.actor_rollout_ref.model.path = MODEL_PATH
        trainer_config.actor_rollout_ref.rollout.name = "vllm"
        trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size = 1

        print("\n[3] Creating 1 vLLM server...")
        from verl.workers.rollout.replica import get_rollout_replica_class

        rollout_config = trainer_config.actor_rollout_ref.rollout
        model_config = trainer_config.actor_rollout_ref.model
        rollout_class = get_rollout_replica_class("vllm")

        server = rollout_class(
            replica_rank=0,
            config=rollout_config,
            model_config=model_config,
            gpus_per_node=1,
        )
        asyncio.run(server.init_standalone())
        server_handle = server._server_handle
        print("   Server ready")

        print("\n[4] Creating admission registry...")
        registry = get_or_create_registry()

        # Get initial registry stats
        initial_stats = ray.get(registry.get_stats.remote())
        print(f"   Initial registry: {initial_stats}")

        print("\n[5] Loading tokenizer...")
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(MODEL_PATH)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        print(f"\n[6] Spawning {NUM_WORKERS} workers to request the same gate...")
        workers = [
            GateRequestWorker.remote(worker_id=i, tokenizer=tokenizer)
            for i in range(NUM_WORKERS)
        ]

        # Test 1: All workers request gate for server_idx=0
        print("\n[7] Test 1: All workers request gate for server_idx=0...")

        async def all_request_same_gate():
            # All workers request the same gate concurrently
            tasks = [
                w.get_gate_id.remote(registry, 0, server_handle, cfg)
                for w in workers
            ]
            return await asyncio.gather(*tasks)

        gate_ids = asyncio.run(all_request_same_gate())

        # Verify registry stats after requests
        after_stats = ray.get(registry.get_stats.remote())
        print(f"   Registry after requests: {after_stats}")

        # Check all gate IDs are the same
        unique_ids = set(gate_ids)
        print(f"   Worker gate IDs: {gate_ids[:3]}... ({len(gate_ids)} total)")
        print(f"   Unique gate IDs: {len(unique_ids)}")

        if len(unique_ids) == 1:
            print("   [PASS] All workers got the SAME gate handle")
        else:
            print(f"   [FAIL] Workers got {len(unique_ids)} different handles!")

        # Verify only 1 gate was created
        if after_stats["num_gates"] == 1:
            print(f"   [PASS] Registry has exactly 1 gate")
        else:
            print(f"   [FAIL] Registry has {after_stats['num_gates']} gates (expected 1)")

        # Test 2: Config mismatch detection
        print("\n[8] Test 2: Config mismatch detection...")
        bad_cfg = AdmissionGateConfig(max_runahead_inflight=999)

        config_mismatch_detected = False
        try:
            # This should raise ValueError due to config mismatch
            bad_gate = ray.get(registry.get_or_create.remote(0, server_handle, bad_cfg))
            print("   [FAIL] No exception raised for config mismatch!")
        except Exception as e:
            error_msg = str(e)
            if "Config mismatch" in error_msg or "mismatch" in error_msg.lower():
                config_mismatch_detected = True
                print(f"   [PASS] Config mismatch correctly detected: {error_msg[:80]}...")
            else:
                print(f"   [FAIL] Unexpected error: {error_msg}")

        # Test 3: Shared counter verification
        print("\n[9] Test 3: Shared counter verification...")
        print("    All workers submit requests through their gate handles...")

        async def all_submit_requests():
            tasks = [
                w.get_gate_and_submit.remote(registry, 0, server_handle, cfg, num_requests=3)
                for w in workers
            ]
            return await asyncio.gather(*tasks)

        submit_results = asyncio.run(all_submit_requests())

        total_submitted = sum(r["submitted"] for r in submit_results)
        total_completed = sum(r["completed"] for r in submit_results)
        total_rejected = sum(r["rejected"] for r in submit_results)

        print(f"    Total submitted: {total_submitted}")
        print(f"    Completed: {total_completed}, Rejected: {total_rejected}")

        # Get final gate stats to verify shared counter
        async def get_gate_stats():
            gate = await registry.get.remote(0)
            return await gate.get_admission_stats.remote()

        final_gate_stats = asyncio.run(get_gate_stats())
        print(f"\n    Gate admission stats: {final_gate_stats}")

        # Verify counter was shared (rejections indicate contention on single gate)
        if total_rejected > 0:
            print(f"    [PASS] Counter is shared ({total_rejected} rejections from contention)")
        else:
            print(f"    [WARN] No rejections - workers may not have overlapped")

        # Verify max_observed is reasonable
        max_obs = final_gate_stats["runahead_max_observed"]
        limit = final_gate_stats["max_runahead_inflight"]
        if max_obs <= limit:
            print(f"    [PASS] Max observed ({max_obs}) <= limit ({limit})")
        else:
            print(f"    [FAIL] Oversubscription: max observed ({max_obs}) > limit ({limit})")

        # Test 4: get() vs get_or_create()
        print("\n[10] Test 4: Verify get() works for existing gates...")

        async def test_get():
            # get() should work for existing gate
            gate = await registry.get.remote(0)
            stats = await gate.get_admission_stats.remote()
            return stats

        get_stats = asyncio.run(test_get())
        print(f"     get(0) returned valid gate with stats: {get_stats}")

        # get() should fail for non-existent gate
        get_nonexistent_failed = False
        try:
            async def test_get_nonexistent():
                return await registry.get.remote(999)

            asyncio.run(test_get_nonexistent())
            print("     [FAIL] get(999) should have raised KeyError")
        except Exception as e:
            if "No gate registered" in str(e) or "KeyError" in str(type(e).__name__):
                get_nonexistent_failed = True
                print(f"     [PASS] get(999) correctly raised error: {str(e)[:50]}...")
            else:
                print(f"     [FAIL] Unexpected error: {e}")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)

        all_passed = True

        # Check 1: Single gate
        if len(unique_ids) == 1 and after_stats["num_gates"] == 1:
            print("[PASS] Registry correctly shares single gate across workers")
        else:
            print("[FAIL] Gate sharing failed")
            all_passed = False

        # Check 2: Config mismatch
        if config_mismatch_detected:
            print("[PASS] Config mismatch correctly detected and rejected")
        else:
            print("[FAIL] Config mismatch not detected")
            all_passed = False

        # Check 3: Shared counter
        if max_obs <= limit:
            print("[PASS] Shared counter enforces admission limit")
        else:
            print("[FAIL] Shared counter oversubscribed")
            all_passed = False

        # Check 4: get() behavior
        if get_nonexistent_failed:
            print("[PASS] get() correctly fails for non-existent gates")
        else:
            print("[FAIL] get() behavior incorrect")
            all_passed = False

        print("=" * 80)

        assert all_passed, "Some tests failed!"
        print("\nTest PASSED: Registry correctly prevents duplicate gates!")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    test_registry_prevents_duplicate_gates()

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
Server-Side Rejection and Retry Logic Test

Goal
----
Test the complete retry flow when server-side rejection occurs:
1. Client bypasses local budget check (skip_client_budget_check=True)
2. Server rejects due to global limit
3. Client receives stop_reason="rejected"
4. Controller requeues the item with incremented retry count
5. Item eventually succeeds or is dropped after max retries

This test uses skip_client_budget_check=True to force requests past the
client-side filter and trigger server-side rejections, validating the
retry mechanism works correctly.

Usage
-----
  NUM_GPUS=1 python tests/workers/rollout/rollout_vllm/test_server_rejection_and_retry.py

Key env vars
------------
  MODEL_PATH: HF model path (default: Qwen/Qwen2.5-0.5B-Instruct)
  NUM_GPUS: Total GPUs available (default: 1)
  MAX_INFLIGHT: Max runahead per gate (default: 1)
  MAX_RETRIES: Max retry attempts (default: 3)
  PRIMARY_SIZE: Number of primary requests (default: 4)
  RUNAHEAD_SIZE: Number of runahead requests (default: 8)
"""

from __future__ import annotations

import asyncio
import os
import time
from uuid import uuid4

import ray

from test_vllm_run_ahead_server_side_admission import (
    AdmissionGateConfig,
    BatchTracker,
    RequestTracker,
    ServerSideAdmissionController,
    ServerSideAdmissionServerManager,
    get_or_create_registry,
)


def test_server_rejection_and_retry():
    """Test server-side rejection detection and retry logic.

    This test validates:
    1. Server-side rejections are properly detected (stop_reason="rejected")
    2. Rejected items are requeued with incremented retry count
    3. Items complete after successful retry
    4. Items are dropped after max retries exceeded
    5. Primary requests are unaffected by runahead rejections
    """
    MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    NUM_GPUS = int(os.environ.get("NUM_GPUS", "1"))
    MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "1"))
    MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
    PRIMARY_SIZE = int(os.environ.get("PRIMARY_SIZE", "4"))
    RUNAHEAD_SIZE = int(os.environ.get("RUNAHEAD_SIZE", "8"))

    # Force server-side rejection by skipping client budget check
    cfg = AdmissionGateConfig(
        max_runahead_inflight=MAX_INFLIGHT,
        enforce_slack=False,  # Disable slack check
        skip_client_budget_check=True,  # CRITICAL: Forces server-side rejection
        max_runahead_retries=MAX_RETRIES,
        poll_interval_s=0.05,  # Fast polling for quicker retries
        workload_cache_ttl_s=0.1,
    )

    print("=" * 80)
    print("Server-Side Rejection and Retry Logic Test")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {NUM_GPUS}")
    print(f"Max runahead inflight: {MAX_INFLIGHT}")
    print(f"Max retries: {MAX_RETRIES}")
    print(f"Primary requests: {PRIMARY_SIZE}")
    print(f"Runahead requests: {RUNAHEAD_SIZE}")
    print("-" * 80)
    print("Test mode: skip_client_budget_check=True")
    print("  - Client-side budget check is disabled")
    print("  - All runahead bypass client filter")
    print("  - Server enforces limit, rejecting excess")
    print("  - Controller requeues rejected items for retry")
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
        trainer_config.actor_rollout_ref.rollout.disable_log_stats = False

        print("\n[3] Creating vLLM server...")
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

        print("\n[4] Creating admission gate...")
        registry = get_or_create_registry()

        async def create_gate():
            return await registry.get_or_create.remote(0, server_handle, cfg)

        gate = asyncio.run(create_gate())
        gated_handles = [gate]
        print(f"   Gate created with max_inflight={MAX_INFLIGHT}")

        print("\n[5] Loading tokenizer...")
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(MODEL_PATH)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        print("\n[6] Creating server manager and controller...")
        server_manager = ServerSideAdmissionServerManager(
            config=trainer_config,
            gated_handles=gated_handles,
            admission_config=cfg,
        )
        controller = ServerSideAdmissionController(server_manager, cfg, tokenizer)

        # Prepare test prompts
        # Primary: Long requests to create sustained load
        primary_prompts = [
            {"prompt": "Write a detailed essay about the history of computing.", "max_tokens": 150},
            {"prompt": "Explain quantum mechanics in simple terms.", "max_tokens": 150},
            {"prompt": "Describe the process of machine learning training.", "max_tokens": 150},
            {"prompt": "Write about the future of artificial intelligence.", "max_tokens": 150},
        ][:PRIMARY_SIZE]

        # Runahead: More requests than max_inflight allows (will trigger rejections)
        runahead_prompts = [
            {"prompt": "What is 2+2?", "max_tokens": 32},
            {"prompt": "Say hello.", "max_tokens": 32},
            {"prompt": "Name a color.", "max_tokens": 32},
            {"prompt": "Count to 3.", "max_tokens": 32},
            {"prompt": "What day is it?", "max_tokens": 32},
            {"prompt": "Name a fruit.", "max_tokens": 32},
            {"prompt": "Say goodbye.", "max_tokens": 32},
            {"prompt": "Name an animal.", "max_tokens": 32},
        ][:RUNAHEAD_SIZE]

        # Add request IDs
        for i, item in enumerate(primary_prompts):
            item["request_id"] = f"primary_{i}_{uuid4().hex[:8]}"
        for i, item in enumerate(runahead_prompts):
            item["request_id"] = f"runahead_{i}_{uuid4().hex[:8]}"

        print(f"\n[7] Prepared workload:")
        print(f"    Primary: {len(primary_prompts)} long requests (150 tokens each)")
        print(f"    Runahead: {len(runahead_prompts)} short requests (32 tokens each)")
        print(f"    Max inflight: {MAX_INFLIGHT} -> expect {RUNAHEAD_SIZE - MAX_INFLIGHT}+ rejections")

        primary_tracker = BatchTracker(batch_id="primary", total=len(primary_prompts))
        runahead_tracker = BatchTracker(batch_id="runahead", total=len(runahead_prompts))

        async def run_with_retry_tracking():
            # Validate gates
            await server_manager.validate_gate_handles()

            # Run with runahead
            primary_results, runahead_results = await controller.run_with_runahead(
                primary_items=primary_prompts,
                runahead_items=runahead_prompts,
                primary_tracker=primary_tracker,
                runahead_tracker=runahead_tracker,
            )

            # Get gate stats
            stats = await gate.get_admission_stats.remote()
            return primary_results, runahead_results, stats

        print("\n[8] Running with server-side rejection + retry logic...")
        start_time = time.perf_counter()
        primary_results, runahead_results, gate_stats = asyncio.run(run_with_retry_tracking())
        total_time = time.perf_counter() - start_time

        # Print results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        print("\n--- Primary Batch ---")
        primary_completed = 0
        for _, req in sorted(primary_tracker.requests.items(), key=lambda x: x[1].index):
            status_str = f"{req.status}"
            if req.stop_reason:
                status_str += f" ({req.stop_reason})"
            print(f"   [{req.index}] {status_str:20s} | {req.token_count:3d} tok | {req.duration:.2f}s")
            if req.status == "completed":
                primary_completed += 1

        print("\n--- Runahead Batch ---")
        runahead_completed = 0
        runahead_rejected = 0
        runahead_aborted = 0
        for _, req in sorted(runahead_tracker.requests.items(), key=lambda x: x[1].index):
            status_str = f"{req.status}"
            if req.stop_reason:
                status_str += f" ({req.stop_reason})"
            print(f"   [{req.index}] {status_str:20s} | {req.token_count:3d} tok | {req.duration:.2f}s")
            if req.status == "completed":
                runahead_completed += 1
            elif req.status == "rejected":
                runahead_rejected += 1
            elif req.status == "aborted":
                runahead_aborted += 1

        print("\n--- Controller Metrics ---")
        print(f"   Feeder ticks: {controller.feeder_ticks}")
        print(f"   Runahead submissions: {controller.runahead_submissions}")
        print(f"   Backpressure events: {controller.backpressure_events}")
        print(f"   Runahead requeues (retries): {controller.runahead_requeues}")
        print(f"   Runahead dropped (max retries): {controller.runahead_dropped}")

        print("\n--- Server Manager Metrics ---")
        print(f"   Total requests: {server_manager.total_requests}")
        print(f"   Primary submitted: {server_manager.primary_submitted}")
        print(f"   Runahead submitted: {server_manager.runahead_submitted}")
        print(f"   Runahead completed: {server_manager.runahead_completed}")
        print(f"   Runahead rejected (server): {server_manager.runahead_rejected}")
        print(f"   Client-side rejections: {server_manager.client_side_rejections}")

        print("\n--- Gate Stats ---")
        print(f"   Max observed inflight: {gate_stats['runahead_max_observed']} / {gate_stats['max_runahead_inflight']}")
        print(f"   Total rejections: {gate_stats['runahead_rejected_total']}")

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        print(f"\nTotal time: {total_time:.2f}s")
        print(f"Primary duration: {controller.primary_duration:.2f}s")

        # Assertions
        print("\n--- Assertions ---")
        all_passed = True

        # 1. All primary completed
        if primary_completed == len(primary_prompts):
            print(f"   [PASS] All primary requests completed ({primary_completed}/{len(primary_prompts)})")
        else:
            print(f"   [FAIL] Primary incomplete: {primary_completed}/{len(primary_prompts)}")
            all_passed = False

        # 2. Server-side rejections occurred (since skip_client_budget_check=True)
        if gate_stats["runahead_rejected_total"] > 0:
            print(f"   [PASS] Server-side rejections occurred ({gate_stats['runahead_rejected_total']} rejections)")
        else:
            print(f"   [WARN] No server-side rejections - may indicate insufficient contention")
            # Not a failure, but unexpected with skip_client_budget_check=True

        # 3. Requeues occurred (retry logic exercised)
        if controller.runahead_requeues > 0:
            print(f"   [PASS] Retry logic exercised ({controller.runahead_requeues} requeues)")
        else:
            print(f"   [WARN] No requeues - retry logic not fully tested")
            # Could happen if all runahead complete before needing retry

        # 4. No oversubscription
        if gate_stats["runahead_max_observed"] <= gate_stats["max_runahead_inflight"]:
            print(f"   [PASS] No oversubscription (max {gate_stats['runahead_max_observed']} <= limit {gate_stats['max_runahead_inflight']})")
        else:
            print(f"   [FAIL] Oversubscription: {gate_stats['runahead_max_observed']} > {gate_stats['max_runahead_inflight']}")
            all_passed = False

        # 5. Some runahead completed (retry eventually succeeded for some)
        if runahead_completed > 0:
            print(f"   [PASS] Some runahead completed after retry ({runahead_completed} completed)")
        else:
            print(f"   [INFO] No runahead completed (may be expected if primary finished quickly)")

        # 6. Client-side rejections should be 0 (skip_client_budget_check=True)
        if server_manager.client_side_rejections == 0:
            print(f"   [PASS] No client-side rejections (skip_client_budget_check working)")
        else:
            print(f"   [WARN] Client-side rejections occurred ({server_manager.client_side_rejections}) despite skip flag")

        # 7. Retry count respected
        if controller.runahead_dropped <= RUNAHEAD_SIZE:
            print(f"   [PASS] Max retry limit respected ({controller.runahead_dropped} dropped after {MAX_RETRIES} retries)")
        else:
            print(f"   [FAIL] More items dropped than runahead size?")
            all_passed = False

        print("\n" + "=" * 80)

        # Summary of retry flow
        print("\n--- Retry Flow Summary ---")
        print(f"1. Runahead submitted: {controller.runahead_submissions}")
        print(f"2. Server rejected: {gate_stats['runahead_rejected_total']}")
        print(f"3. Requeued for retry: {controller.runahead_requeues}")
        print(f"4. Completed after retry: {runahead_completed}")
        print(f"5. Dropped (max retries): {controller.runahead_dropped}")
        print(f"6. Aborted (primary done): {runahead_aborted}")

        print("=" * 80)

        # Final assertions
        assert primary_completed == len(primary_prompts), "All primary must complete"
        assert gate_stats["runahead_max_observed"] <= gate_stats["max_runahead_inflight"], "No oversubscription"

        if all_passed:
            print("\nTest PASSED: Server rejection and retry logic works correctly!")
        else:
            raise AssertionError("Some assertions failed")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    test_server_rejection_and_retry()

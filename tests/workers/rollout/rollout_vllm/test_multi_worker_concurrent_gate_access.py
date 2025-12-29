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
Multi-Worker Concurrent Gate Access Test

Goal
----
Test the core racing scenario where multiple Ray actor workers concurrently
access the same admission gates. This validates that server-side admission
control correctly enforces global limits even when multiple workers race to
submit runahead requests.

This test spawns N ConcurrentWorker Ray actors, each submitting M concurrent
runahead requests to random gates. The key assertion is that no gate ever
exceeds its max_runahead_inflight limit.

Usage
-----
  NUM_GPUS=2 python tests/workers/rollout/rollout_vllm/test_multi_worker_concurrent_gate_access.py

Key env vars
------------
  MODEL_PATH: HF model path (default: Qwen/Qwen2.5-0.5B-Instruct)
  NUM_GPUS: Total GPUs available (default: 2)
  TP_SIZE: Tensor parallel size (default: 1)
  NUM_WORKERS: Number of concurrent worker actors (default: 8)
  REQUESTS_PER_WORKER: Requests each worker submits (default: 10)
  MAX_INFLIGHT: Max concurrent runahead per gate (default: 2)
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Optional
from uuid import uuid4

import ray

# Import admission control components from the main test file
from test_vllm_run_ahead_server_side_admission import (
    AdmissionControlledServer,
    AdmissionGateConfig,
    AdmissionGateRegistry,
    get_or_create_registry,
)


@ray.remote
class ConcurrentWorker:
    """Simulates an AgentLoopWorker that hammers admission gates concurrently.

    Each worker submits multiple runahead requests to random gates simultaneously,
    testing the racing scenario where multiple workers compete for admission slots.
    """

    def __init__(self, worker_id: int, tokenizer):
        self.worker_id = worker_id
        self.tokenizer = tokenizer
        self.results = {
            "submitted": 0,
            "completed": 0,
            "rejected": 0,
            "errors": 0,
        }

    def _tokenize(self, prompt: str) -> list[int]:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    async def hammer_gates(
        self,
        gate_handles: list,
        num_requests: int = 10,
        max_tokens: int = 32,
    ) -> dict[str, Any]:
        """Submit many concurrent runahead requests to random gates.

        Args:
            gate_handles: List of AdmissionControlledServer actor handles
            num_requests: Number of concurrent requests to submit
            max_tokens: Max tokens per request

        Returns:
            Dict with submission statistics
        """
        prompts = [
            "What is 2+2?",
            "Say hello.",
            "Count to 5.",
            "Name a color.",
            "What day is it?",
        ]

        async def submit_one(request_idx: int) -> dict:
            """Submit a single runahead request to a random gate."""
            gate_idx = random.randint(0, len(gate_handles) - 1)
            gate = gate_handles[gate_idx]
            prompt = random.choice(prompts)
            request_id = f"worker_{self.worker_id}_req_{request_idx}_{uuid4().hex[:8]}"

            prompt_ids = self._tokenize(prompt)
            sampling_params = {
                "_verl_request_kind": "runahead",
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": max_tokens,
            }

            try:
                output = await gate.generate.remote(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )

                stop_reason = getattr(output, "stop_reason", None)
                if stop_reason == "rejected":
                    return {"status": "rejected", "gate_idx": gate_idx}
                else:
                    token_count = len(getattr(output, "token_ids", []))
                    return {"status": "completed", "gate_idx": gate_idx, "tokens": token_count}

            except Exception as e:
                return {"status": "error", "gate_idx": gate_idx, "error": str(e)}

        # Submit all requests concurrently
        tasks = [submit_one(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        stats = {
            "worker_id": self.worker_id,
            "submitted": num_requests,
            "completed": 0,
            "rejected": 0,
            "errors": 0,
            "total_tokens": 0,
            "per_gate": {},
        }

        for r in results:
            if isinstance(r, Exception):
                stats["errors"] += 1
            elif isinstance(r, dict):
                gate_idx = r.get("gate_idx", -1)
                if gate_idx not in stats["per_gate"]:
                    stats["per_gate"][gate_idx] = {"completed": 0, "rejected": 0}

                if r["status"] == "completed":
                    stats["completed"] += 1
                    stats["total_tokens"] += r.get("tokens", 0)
                    stats["per_gate"][gate_idx]["completed"] += 1
                elif r["status"] == "rejected":
                    stats["rejected"] += 1
                    stats["per_gate"][gate_idx]["rejected"] += 1
                else:
                    stats["errors"] += 1

        return stats


def test_multi_worker_concurrent_gate_access():
    """Test multiple workers concurrently accessing admission gates.

    This is the core racing test that verifies:
    1. Multiple concurrent workers can share gates without oversubscription
    2. Server-side admission correctly enforces global limits
    3. Racing to acquire slots results in proper rejection, not corruption
    """
    # Configuration
    MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    NUM_GPUS = int(os.environ.get("NUM_GPUS", "2"))
    TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
    DP_SIZE = NUM_GPUS // TP_SIZE
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
    REQUESTS_PER_WORKER = int(os.environ.get("REQUESTS_PER_WORKER", "10"))
    MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "2"))

    cfg = AdmissionGateConfig(
        max_runahead_inflight=MAX_INFLIGHT,
        enforce_slack=False,  # Disable slack check to maximize contention
        workload_cache_ttl_s=0.1,
    )

    print("=" * 80)
    print("Multi-Worker Concurrent Gate Access Test")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {NUM_GPUS} | TP: {TP_SIZE} | DP: {DP_SIZE}")
    print(f"Workers: {NUM_WORKERS} | Requests/worker: {REQUESTS_PER_WORKER}")
    print(f"Max runahead inflight per gate: {MAX_INFLIGHT}")
    print(f"Total concurrent requests: {NUM_WORKERS * REQUESTS_PER_WORKER}")
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
        trainer_config.actor_rollout_ref.rollout.tensor_model_parallel_size = TP_SIZE
        trainer_config.actor_rollout_ref.rollout.disable_log_stats = False

        print(f"\n[3] Creating {DP_SIZE} vLLM server(s)...")
        from verl.workers.rollout.replica import get_rollout_replica_class

        rollout_config = trainer_config.actor_rollout_ref.rollout
        model_config = trainer_config.actor_rollout_ref.model
        rollout_class = get_rollout_replica_class("vllm")

        servers = []
        server_handles = []
        for dp_rank in range(DP_SIZE):
            print(f"   Creating server {dp_rank}...")
            server = rollout_class(
                replica_rank=dp_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=TP_SIZE,
            )
            asyncio.run(server.init_standalone())
            servers.append(server)
            server_handles.append(server._server_handle)
            print(f"   Server {dp_rank} ready")

        print("\n[4] Creating admission registry and gates...")
        registry = get_or_create_registry()

        async def create_gates():
            gates = []
            for i, h in enumerate(server_handles):
                gate = await registry.get_or_create.remote(i, h, cfg)
                gates.append(gate)
            return gates

        gated_handles = asyncio.run(create_gates())
        print(f"   Created {len(gated_handles)} gates via registry")

        # Verify registry stats
        registry_stats = ray.get(registry.get_stats.remote())
        print(f"   Registry stats: {registry_stats}")
        assert registry_stats["num_gates"] == DP_SIZE, \
            f"Expected {DP_SIZE} gates, got {registry_stats['num_gates']}"

        print("\n[5] Loading tokenizer...")
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(MODEL_PATH)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        print(f"\n[6] Spawning {NUM_WORKERS} concurrent workers...")
        workers = [
            ConcurrentWorker.remote(worker_id=i, tokenizer=tokenizer)
            for i in range(NUM_WORKERS)
        ]

        print(f"\n[7] Running concurrent access test...")
        print(f"    Each worker submitting {REQUESTS_PER_WORKER} requests to random gates...")
        start_time = time.perf_counter()

        # All workers hammer gates simultaneously
        worker_results = ray.get([
            w.hammer_gates.remote(gated_handles, REQUESTS_PER_WORKER)
            for w in workers
        ])

        total_time = time.perf_counter() - start_time

        print(f"\n[8] Collecting admission stats from gates...")
        gate_stats = []
        for i, gate in enumerate(gated_handles):
            stats = ray.get(gate.get_admission_stats.remote())
            gate_stats.append(stats)

        # Print results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        print("\n--- Worker Results ---")
        total_submitted = 0
        total_completed = 0
        total_rejected = 0
        total_errors = 0
        total_tokens = 0

        for r in worker_results:
            print(f"   Worker {r['worker_id']}: "
                  f"submitted={r['submitted']}, completed={r['completed']}, "
                  f"rejected={r['rejected']}, errors={r['errors']}, tokens={r['total_tokens']}")
            total_submitted += r["submitted"]
            total_completed += r["completed"]
            total_rejected += r["rejected"]
            total_errors += r["errors"]
            total_tokens += r["total_tokens"]

        print(f"\n   TOTAL: submitted={total_submitted}, completed={total_completed}, "
              f"rejected={total_rejected}, errors={total_errors}, tokens={total_tokens}")

        print("\n--- Gate Admission Stats ---")
        oversubscribed = False
        for i, stats in enumerate(gate_stats):
            max_obs = stats["runahead_max_observed"]
            limit = stats["max_runahead_inflight"]
            rejected = stats["runahead_rejected_total"]
            status = "OK" if max_obs <= limit else "OVERSUBSCRIBED!"

            if max_obs > limit:
                oversubscribed = True

            print(f"   Gate {i}: max_observed={max_obs}/{limit} {status}, rejected={rejected}")

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)
        print(f"Total time: {total_time:.2f}s")
        print(f"Throughput: {total_submitted / total_time:.1f} requests/s")
        print(f"Completion rate: {total_completed / total_submitted * 100:.1f}%")
        print(f"Rejection rate: {total_rejected / total_submitted * 100:.1f}%")

        # Key assertions
        print("\n--- Assertions ---")

        # 1. No oversubscription
        if oversubscribed:
            print("   [FAIL] Oversubscription detected!")
            for i, stats in enumerate(gate_stats):
                if stats["runahead_max_observed"] > stats["max_runahead_inflight"]:
                    print(f"          Gate {i}: {stats['runahead_max_observed']} > {stats['max_runahead_inflight']}")
        else:
            print("   [PASS] No oversubscription (max_observed <= max_inflight for all gates)")

        # 2. Contention occurred (some rejections)
        total_gate_rejections = sum(s["runahead_rejected_total"] for s in gate_stats)
        if total_gate_rejections > 0:
            print(f"   [PASS] Contention occurred ({total_gate_rejections} server-side rejections)")
        else:
            print(f"   [WARN] No server-side rejections - may indicate insufficient contention")

        # 3. Registry integrity
        if registry_stats["num_gates"] == DP_SIZE:
            print(f"   [PASS] Registry created exactly {DP_SIZE} gates (no duplicates)")
        else:
            print(f"   [FAIL] Registry gate count mismatch: {registry_stats['num_gates']} != {DP_SIZE}")

        # 4. All requests accounted for
        accounted = total_completed + total_rejected + total_errors
        if accounted == total_submitted:
            print(f"   [PASS] All {total_submitted} requests accounted for")
        else:
            print(f"   [FAIL] Request accounting mismatch: {accounted} != {total_submitted}")

        print("\n" + "=" * 80)

        # Final assertions
        assert not oversubscribed, "Racing detected: gate oversubscribed!"
        assert registry_stats["num_gates"] == DP_SIZE, "Registry gate count mismatch"
        assert total_errors == 0, f"Unexpected errors: {total_errors}"

        print("\nTest PASSED: No racing/oversubscription detected!")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    test_multi_worker_concurrent_gate_access()

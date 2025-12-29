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
Herding Oversubscription Prevention Test

Goal
----
Stress-test the admission control under worst-case "herding" conditions where
many workers simultaneously detect slack and try to submit runahead requests.
This simulates the scenario where all workers check workload metrics at the
same time, all see slack, and all attempt to submit.

With max_runahead_inflight=1 and 16 concurrent workers, this creates maximum
contention and validates that the admission control never allows oversubscription.

Usage
-----
  NUM_GPUS=2 python tests/workers/rollout/rollout_vllm/test_herding_oversubscription_prevention.py

Key env vars
------------
  MODEL_PATH: HF model path (default: Qwen/Qwen2.5-0.5B-Instruct)
  NUM_GPUS: Total GPUs available (default: 2)
  NUM_WORKERS: Number of concurrent herding workers (default: 16)
  DURATION_S: Test duration in seconds (default: 30)
  MAX_INFLIGHT: Max runahead per gate (default: 1, very restrictive)
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import ray

from test_vllm_run_ahead_server_side_admission import (
    AdmissionControlledServer,
    AdmissionGateConfig,
    get_or_create_registry,
)


@dataclass
class HerdingStats:
    """Track herding attack statistics."""

    worker_id: int
    submitted: int = 0
    completed: int = 0
    rejected: int = 0
    errors: int = 0
    total_tokens: int = 0
    submissions_per_gate: dict = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else 0.0

    @property
    def throughput(self) -> float:
        return self.submitted / self.duration if self.duration > 0 else 0.0


@ray.remote
class HerdingWorker:
    """Worker that aggressively submits runahead during perceived slack.

    This worker intentionally bypasses any client-side slack checking and
    submits as fast as possible to create maximum herding pressure.
    """

    def __init__(self, worker_id: int, tokenizer):
        self.worker_id = worker_id
        self.tokenizer = tokenizer

    def _tokenize(self, prompt: str) -> list[int]:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    async def herd_attack(
        self,
        gate_handles: list,
        duration_s: float = 30.0,
        max_tokens: int = 16,
    ) -> dict[str, Any]:
        """Aggressively submit runahead for specified duration.

        Args:
            gate_handles: List of admission gate handles
            duration_s: How long to run the attack
            max_tokens: Max tokens per request (keep small for speed)

        Returns:
            Statistics about the herding attack
        """
        stats = HerdingStats(worker_id=self.worker_id)
        stats.start_time = time.perf_counter()

        prompts = ["Hi", "Hello", "Hey", "Count", "Name"]
        request_counter = 0

        while time.perf_counter() - stats.start_time < duration_s:
            # Pick a random gate
            gate_idx = random.randint(0, len(gate_handles) - 1)
            gate = gate_handles[gate_idx]

            # Track per-gate submissions
            if gate_idx not in stats.submissions_per_gate:
                stats.submissions_per_gate[gate_idx] = {"submitted": 0, "rejected": 0, "completed": 0}

            prompt = random.choice(prompts)
            request_id = f"herd_{self.worker_id}_{request_counter}_{uuid4().hex[:6]}"
            request_counter += 1

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

                stats.submitted += 1
                stats.submissions_per_gate[gate_idx]["submitted"] += 1

                stop_reason = getattr(output, "stop_reason", None)
                if stop_reason == "rejected":
                    stats.rejected += 1
                    stats.submissions_per_gate[gate_idx]["rejected"] += 1
                else:
                    stats.completed += 1
                    stats.total_tokens += len(getattr(output, "token_ids", []))
                    stats.submissions_per_gate[gate_idx]["completed"] += 1

            except asyncio.CancelledError:
                # Graceful shutdown
                break
            except Exception as e:
                stats.submitted += 1
                stats.errors += 1

            # Small yield to prevent complete CPU starvation
            # but keep it minimal to maximize herding pressure
            if request_counter % 10 == 0:
                await asyncio.sleep(0.001)

        stats.end_time = time.perf_counter()

        return {
            "worker_id": stats.worker_id,
            "submitted": stats.submitted,
            "completed": stats.completed,
            "rejected": stats.rejected,
            "errors": stats.errors,
            "total_tokens": stats.total_tokens,
            "duration": stats.duration,
            "throughput": stats.throughput,
            "submissions_per_gate": dict(stats.submissions_per_gate),
        }


def test_herding_oversubscription_prevention():
    """Test that admission control prevents oversubscription under herding.

    This is a stress test that creates maximum contention by:
    1. Using max_runahead_inflight=1 (very restrictive)
    2. Spawning many concurrent workers
    3. Having workers submit as fast as possible without client-side filtering

    The key assertion is that no gate ever exceeds its limit, even under
    this extreme herding pressure.
    """
    MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    NUM_GPUS = int(os.environ.get("NUM_GPUS", "2"))
    TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
    DP_SIZE = NUM_GPUS // TP_SIZE
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "16"))
    DURATION_S = float(os.environ.get("DURATION_S", "30"))
    MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "1"))

    cfg = AdmissionGateConfig(
        max_runahead_inflight=MAX_INFLIGHT,
        enforce_slack=False,  # Disable to maximize herding pressure
        workload_cache_ttl_s=0.05,  # Short TTL for fast updates
    )

    print("=" * 80)
    print("Herding Oversubscription Prevention Test")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {NUM_GPUS} | TP: {TP_SIZE} | DP: {DP_SIZE}")
    print(f"Herding workers: {NUM_WORKERS}")
    print(f"Duration: {DURATION_S}s")
    print(f"Max runahead inflight per gate: {MAX_INFLIGHT} (very restrictive)")
    print("=" * 80)
    print("\nThis test creates MAXIMUM herding pressure:")
    print(f"  - {NUM_WORKERS} workers all submitting as fast as possible")
    print(f"  - Only {MAX_INFLIGHT} slot(s) per gate")
    print("  - No client-side slack filtering")
    print("  - Expect HIGH rejection rate (>50%)")
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
        print(f"   Created {len(gated_handles)} gates with max_inflight={MAX_INFLIGHT}")

        print("\n[5] Loading tokenizer...")
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(MODEL_PATH)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        print(f"\n[6] Spawning {NUM_WORKERS} herding workers...")
        workers = [
            HerdingWorker.remote(worker_id=i, tokenizer=tokenizer)
            for i in range(NUM_WORKERS)
        ]

        print(f"\n[7] Starting herding attack for {DURATION_S}s...")
        print("    (Workers will submit as fast as possible to create maximum contention)")
        start_time = time.perf_counter()

        # All workers attack simultaneously
        worker_results = ray.get([
            w.herd_attack.remote(gated_handles, DURATION_S)
            for w in workers
        ])

        total_time = time.perf_counter() - start_time

        print(f"\n[8] Collecting gate admission stats...")
        gate_stats = []
        for i, gate in enumerate(gated_handles):
            stats = ray.get(gate.get_admission_stats.remote())
            gate_stats.append(stats)

        # Aggregate results
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)

        print("\n--- Worker Results (top 5 by throughput) ---")
        sorted_workers = sorted(worker_results, key=lambda x: x["throughput"], reverse=True)
        for r in sorted_workers[:5]:
            print(f"   Worker {r['worker_id']:2d}: "
                  f"submitted={r['submitted']:5d}, "
                  f"completed={r['completed']:4d}, "
                  f"rejected={r['rejected']:4d}, "
                  f"throughput={r['throughput']:.1f}/s")

        total_submitted = sum(r["submitted"] for r in worker_results)
        total_completed = sum(r["completed"] for r in worker_results)
        total_rejected = sum(r["rejected"] for r in worker_results)
        total_errors = sum(r["errors"] for r in worker_results)
        total_tokens = sum(r["total_tokens"] for r in worker_results)

        print(f"\n   AGGREGATE: submitted={total_submitted}, completed={total_completed}, "
              f"rejected={total_rejected}, errors={total_errors}")

        print("\n--- Gate Admission Stats ---")
        oversubscribed = False
        max_oversubscription = 0

        for i, stats in enumerate(gate_stats):
            max_obs = stats["runahead_max_observed"]
            limit = stats["max_runahead_inflight"]
            rejected = stats["runahead_rejected_total"]

            if max_obs > limit:
                oversubscribed = True
                max_oversubscription = max(max_oversubscription, max_obs - limit)
                status = f"OVERSUBSCRIBED by {max_obs - limit}!"
            else:
                status = "OK"

            print(f"   Gate {i}: max_observed={max_obs}/{limit} {status}, rejected={rejected}")

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        print(f"\nTotal wall time: {total_time:.2f}s")
        print(f"Total submissions: {total_submitted}")
        print(f"Aggregate throughput: {total_submitted / total_time:.1f} submissions/s")
        print(f"Total tokens generated: {total_tokens}")

        rejection_rate = total_rejected / total_submitted if total_submitted > 0 else 0
        print(f"\nRejection rate: {rejection_rate * 100:.1f}%")

        # Key assertions
        print("\n--- Assertions ---")

        # 1. No oversubscription (THE KEY TEST)
        if oversubscribed:
            print(f"   [FAIL] OVERSUBSCRIPTION DETECTED!")
            print(f"          Maximum oversubscription: {max_oversubscription}")
            for i, stats in enumerate(gate_stats):
                if stats["runahead_max_observed"] > stats["max_runahead_inflight"]:
                    print(f"          Gate {i}: observed {stats['runahead_max_observed']} > limit {stats['max_runahead_inflight']}")
        else:
            print(f"   [PASS] No oversubscription! All gates stayed within limit={MAX_INFLIGHT}")

        # 2. High rejection rate expected
        if rejection_rate > 0.5:
            print(f"   [PASS] High rejection rate ({rejection_rate * 100:.1f}%) confirms herding pressure")
        elif rejection_rate > 0.2:
            print(f"   [WARN] Moderate rejection rate ({rejection_rate * 100:.1f}%) - herding may be insufficient")
        else:
            print(f"   [WARN] Low rejection rate ({rejection_rate * 100:.1f}%) - herding pressure may be too low")

        # 3. No errors
        if total_errors == 0:
            print(f"   [PASS] No errors during herding attack")
        else:
            print(f"   [WARN] {total_errors} errors occurred during herding")

        # 4. Some completions (system is functional)
        if total_completed > 0:
            print(f"   [PASS] System remained functional ({total_completed} completions)")
        else:
            print(f"   [FAIL] No completions - system may have deadlocked")

        # 5. Distribution across gates
        total_gate_rejections = sum(s["runahead_rejected_total"] for s in gate_stats)
        print(f"\n   Total server-side rejections across all gates: {total_gate_rejections}")

        print("\n" + "=" * 80)

        # Final verdict
        if not oversubscribed and total_completed > 0 and total_errors == 0:
            print("\nTest PASSED: Admission control prevented oversubscription under herding!")
        else:
            if oversubscribed:
                raise AssertionError(f"RACING DETECTED: Gate oversubscribed by {max_oversubscription}")
            if total_completed == 0:
                raise AssertionError("System deadlocked - no completions")
            if total_errors > 0:
                raise AssertionError(f"Errors during herding: {total_errors}")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    test_herding_oversubscription_prevention()

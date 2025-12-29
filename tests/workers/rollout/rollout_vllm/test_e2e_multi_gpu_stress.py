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
End-to-End Multi-GPU Stress Test

Goal
----
Comprehensive end-to-end stress test combining all admission control components
with realistic workloads across multiple GPUs. This test runs multiple rounds
of primary + runahead batches through multiple concurrent workers, validating
system stability, throughput, and correctness under sustained load.

Usage
-----
  NUM_GPUS=4 python tests/workers/rollout/rollout_vllm/test_e2e_multi_gpu_stress.py

Key env vars
------------
  MODEL_PATH: HF model path (default: Qwen/Qwen2.5-0.5B-Instruct)
  NUM_GPUS: Total GPUs available (default: 4)
  TP_SIZE: Tensor parallel size (default: 1)
  NUM_WORKERS: Number of concurrent controller workers (default: 8)
  ROUNDS_PER_WORKER: Rounds each worker runs (default: 3)
  MAX_INFLIGHT: Max runahead per gate (default: 4)
  PRIMARY_SIZE: Primary requests per round (default: 4)
  RUNAHEAD_SIZE: Runahead requests per round (default: 8)
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

import ray

from test_vllm_run_ahead_server_side_admission import (
    AdmissionGateConfig,
    BatchTracker,
    ServerSideAdmissionController,
    ServerSideAdmissionServerManager,
    get_or_create_registry,
)


@dataclass
class RoundResult:
    """Result from a single round of primary + runahead."""

    worker_id: int
    round_idx: int
    primary_completed: int
    primary_total: int
    runahead_completed: int
    runahead_rejected: int
    runahead_aborted: int
    primary_tokens: int
    runahead_tokens: int
    duration: float
    requeues: int
    errors: list = field(default_factory=list)


@dataclass
class WorkerResult:
    """Aggregate result from a worker across all rounds."""

    worker_id: int
    rounds: list[RoundResult] = field(default_factory=list)
    total_duration: float = 0.0

    @property
    def total_primary_tokens(self) -> int:
        return sum(r.primary_tokens for r in self.rounds)

    @property
    def total_runahead_tokens(self) -> int:
        return sum(r.runahead_tokens for r in self.rounds)

    @property
    def total_primary_completed(self) -> int:
        return sum(r.primary_completed for r in self.rounds)

    @property
    def total_runahead_completed(self) -> int:
        return sum(r.runahead_completed for r in self.rounds)

    @property
    def total_requeues(self) -> int:
        return sum(r.requeues for r in self.rounds)


@ray.remote
class StressWorker:
    """Worker that runs multiple rounds of primary + runahead batches."""

    def __init__(self, worker_id: int, tokenizer, config):
        self.worker_id = worker_id
        self.tokenizer = tokenizer
        self.config = config

    def _generate_prompts(self, count: int, prompt_type: str, max_tokens_range: tuple) -> list[dict]:
        """Generate random prompts with varying token counts."""
        base_prompts = [
            "Explain the concept of",
            "Write about",
            "Describe in detail",
            "What is the history of",
            "How does one understand",
            "Analyze the importance of",
            "Compare and contrast",
            "Summarize the key points of",
        ]

        topics = [
            "artificial intelligence",
            "machine learning",
            "quantum computing",
            "climate change",
            "space exploration",
            "renewable energy",
            "biotechnology",
            "economics",
            "philosophy",
            "mathematics",
        ]

        prompts = []
        for i in range(count):
            prompt = f"{random.choice(base_prompts)} {random.choice(topics)}."
            max_tokens = random.randint(max_tokens_range[0], max_tokens_range[1])
            prompts.append({
                "prompt": prompt,
                "max_tokens": max_tokens,
                "request_id": f"{prompt_type}_{self.worker_id}_{i}_{uuid4().hex[:8]}",
            })
        return prompts

    async def run_stress_rounds(
        self,
        gated_handles: list,
        num_rounds: int,
        primary_size: int,
        runahead_size: int,
        trainer_config,
        admission_config: AdmissionGateConfig,
    ) -> dict[str, Any]:
        """Run multiple rounds of primary + runahead batches."""
        start_time = time.perf_counter()
        rounds: list[RoundResult] = []

        # Create server manager for this worker
        server_manager = ServerSideAdmissionServerManager(
            config=trainer_config,
            gated_handles=gated_handles,
            admission_config=admission_config,
        )

        # Validate gates once
        await server_manager.validate_gate_handles()

        for round_idx in range(num_rounds):
            round_start = time.perf_counter()

            # Generate fresh prompts for this round
            primary_prompts = self._generate_prompts(
                primary_size,
                f"primary_w{self.worker_id}_r{round_idx}",
                (100, 200),  # Primary: 100-200 tokens
            )
            runahead_prompts = self._generate_prompts(
                runahead_size,
                f"runahead_w{self.worker_id}_r{round_idx}",
                (32, 64),  # Runahead: 32-64 tokens
            )

            # Create trackers
            primary_tracker = BatchTracker(
                batch_id=f"primary_w{self.worker_id}_r{round_idx}",
                total=len(primary_prompts),
            )
            runahead_tracker = BatchTracker(
                batch_id=f"runahead_w{self.worker_id}_r{round_idx}",
                total=len(runahead_prompts),
            )

            # Create controller for this round
            controller = ServerSideAdmissionController(
                server_manager,
                admission_config,
                self.tokenizer,
            )

            # Run the round
            try:
                primary_results, runahead_results = await controller.run_with_runahead(
                    primary_items=primary_prompts,
                    runahead_items=runahead_prompts,
                    primary_tracker=primary_tracker,
                    runahead_tracker=runahead_tracker,
                )

                # Collect round stats
                primary_completed = sum(1 for r in primary_tracker.requests.values() if r.status == "completed")
                runahead_completed = sum(1 for r in runahead_tracker.requests.values() if r.status == "completed")
                runahead_rejected = sum(1 for r in runahead_tracker.requests.values() if r.status == "rejected")
                runahead_aborted = sum(1 for r in runahead_tracker.requests.values() if r.status == "aborted")

                primary_tokens = sum(r.token_count for r in primary_tracker.requests.values())
                runahead_tokens = sum(r.token_count for r in runahead_tracker.requests.values())

                round_result = RoundResult(
                    worker_id=self.worker_id,
                    round_idx=round_idx,
                    primary_completed=primary_completed,
                    primary_total=len(primary_prompts),
                    runahead_completed=runahead_completed,
                    runahead_rejected=runahead_rejected,
                    runahead_aborted=runahead_aborted,
                    primary_tokens=primary_tokens,
                    runahead_tokens=runahead_tokens,
                    duration=time.perf_counter() - round_start,
                    requeues=controller.runahead_requeues,
                )
                rounds.append(round_result)

            except Exception as e:
                rounds.append(RoundResult(
                    worker_id=self.worker_id,
                    round_idx=round_idx,
                    primary_completed=0,
                    primary_total=len(primary_prompts),
                    runahead_completed=0,
                    runahead_rejected=0,
                    runahead_aborted=0,
                    primary_tokens=0,
                    runahead_tokens=0,
                    duration=time.perf_counter() - round_start,
                    requeues=0,
                    errors=[str(e)],
                ))

        total_duration = time.perf_counter() - start_time

        # Return serializable result
        return {
            "worker_id": self.worker_id,
            "num_rounds": len(rounds),
            "total_duration": total_duration,
            "rounds": [
                {
                    "round_idx": r.round_idx,
                    "primary_completed": r.primary_completed,
                    "primary_total": r.primary_total,
                    "runahead_completed": r.runahead_completed,
                    "runahead_rejected": r.runahead_rejected,
                    "runahead_aborted": r.runahead_aborted,
                    "primary_tokens": r.primary_tokens,
                    "runahead_tokens": r.runahead_tokens,
                    "duration": r.duration,
                    "requeues": r.requeues,
                    "errors": r.errors,
                }
                for r in rounds
            ],
            "total_primary_tokens": sum(r.primary_tokens for r in rounds),
            "total_runahead_tokens": sum(r.runahead_tokens for r in rounds),
            "total_primary_completed": sum(r.primary_completed for r in rounds),
            "total_runahead_completed": sum(r.runahead_completed for r in rounds),
            "total_requeues": sum(r.requeues for r in rounds),
            "total_errors": sum(len(r.errors) for r in rounds),
        }


def test_e2e_multi_gpu_stress():
    """End-to-end multi-GPU stress test.

    This test validates:
    1. System stability under sustained multi-worker load
    2. No oversubscription across all servers
    3. Consistent behavior across multiple rounds
    4. Reasonable throughput with runahead slack utilization
    """
    MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
    NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
    TP_SIZE = int(os.environ.get("TP_SIZE", "1"))
    DP_SIZE = NUM_GPUS // TP_SIZE
    NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))
    ROUNDS_PER_WORKER = int(os.environ.get("ROUNDS_PER_WORKER", "3"))
    MAX_INFLIGHT = int(os.environ.get("MAX_INFLIGHT", "4"))
    PRIMARY_SIZE = int(os.environ.get("PRIMARY_SIZE", "4"))
    RUNAHEAD_SIZE = int(os.environ.get("RUNAHEAD_SIZE", "8"))

    cfg = AdmissionGateConfig(
        max_runahead_inflight=MAX_INFLIGHT,
        enforce_slack=True,  # Enable slack checking
        waiting_threshold=2,
        kv_cache_threshold=0.85,
        poll_interval_s=0.1,
        workload_cache_ttl_s=0.2,
        max_runahead_retries=3,
    )

    total_rounds = NUM_WORKERS * ROUNDS_PER_WORKER
    total_primary = total_rounds * PRIMARY_SIZE
    total_runahead = total_rounds * RUNAHEAD_SIZE

    print("=" * 80)
    print("End-to-End Multi-GPU Stress Test")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {NUM_GPUS} | TP: {TP_SIZE} | DP: {DP_SIZE}")
    print(f"Workers: {NUM_WORKERS} | Rounds/worker: {ROUNDS_PER_WORKER}")
    print(f"Primary/round: {PRIMARY_SIZE} | Runahead/round: {RUNAHEAD_SIZE}")
    print(f"Max runahead inflight per gate: {MAX_INFLIGHT}")
    print("-" * 80)
    print(f"Total rounds: {total_rounds}")
    print(f"Total primary requests: {total_primary}")
    print(f"Total runahead requests: {total_runahead}")
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
        print(f"   Created {len(gated_handles)} gates with max_inflight={MAX_INFLIGHT}")

        print("\n[5] Loading tokenizer...")
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(MODEL_PATH)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        print(f"\n[6] Spawning {NUM_WORKERS} stress workers...")
        workers = [
            StressWorker.remote(worker_id=i, tokenizer=tokenizer, config=cfg)
            for i in range(NUM_WORKERS)
        ]

        print(f"\n[7] Starting stress test ({ROUNDS_PER_WORKER} rounds per worker)...")
        print("    This may take a while...")
        start_time = time.perf_counter()

        # All workers run concurrently
        worker_results = ray.get([
            w.run_stress_rounds.remote(
                gated_handles=gated_handles,
                num_rounds=ROUNDS_PER_WORKER,
                primary_size=PRIMARY_SIZE,
                runahead_size=RUNAHEAD_SIZE,
                trainer_config=trainer_config,
                admission_config=cfg,
            )
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

        print("\n--- Per-Worker Summary ---")
        aggregate = {
            "total_primary_tokens": 0,
            "total_runahead_tokens": 0,
            "total_primary_completed": 0,
            "total_runahead_completed": 0,
            "total_requeues": 0,
            "total_errors": 0,
            "total_rounds": 0,
        }

        for r in worker_results:
            print(f"   Worker {r['worker_id']:2d}: "
                  f"rounds={r['num_rounds']}, "
                  f"primary_tok={r['total_primary_tokens']:5d}, "
                  f"runahead_tok={r['total_runahead_tokens']:5d}, "
                  f"requeues={r['total_requeues']:3d}, "
                  f"time={r['total_duration']:.1f}s")

            aggregate["total_primary_tokens"] += r["total_primary_tokens"]
            aggregate["total_runahead_tokens"] += r["total_runahead_tokens"]
            aggregate["total_primary_completed"] += r["total_primary_completed"]
            aggregate["total_runahead_completed"] += r["total_runahead_completed"]
            aggregate["total_requeues"] += r["total_requeues"]
            aggregate["total_errors"] += r["total_errors"]
            aggregate["total_rounds"] += r["num_rounds"]

        print(f"\n   AGGREGATE:")
        print(f"      Rounds completed: {aggregate['total_rounds']}/{total_rounds}")
        print(f"      Primary completed: {aggregate['total_primary_completed']}/{total_primary}")
        print(f"      Primary tokens: {aggregate['total_primary_tokens']:,}")
        print(f"      Runahead completed: {aggregate['total_runahead_completed']}")
        print(f"      Runahead tokens: {aggregate['total_runahead_tokens']:,}")
        print(f"      Total requeues: {aggregate['total_requeues']}")
        print(f"      Total errors: {aggregate['total_errors']}")

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

        total_gate_rejections = sum(s["runahead_rejected_total"] for s in gate_stats)

        print("\n" + "=" * 80)
        print("ANALYSIS")
        print("=" * 80)

        print(f"\nTotal wall time: {total_time:.2f}s")
        total_tokens = aggregate["total_primary_tokens"] + aggregate["total_runahead_tokens"]
        print(f"Total tokens generated: {total_tokens:,}")
        print(f"Throughput: {total_tokens / total_time:.0f} tokens/s")
        print(f"Primary completion rate: {aggregate['total_primary_completed'] / total_primary * 100:.1f}%")

        if aggregate["total_runahead_completed"] > 0:
            runahead_rate = aggregate["total_runahead_tokens"] / aggregate["total_primary_tokens"] * 100
            print(f"Runahead/Primary token ratio: {runahead_rate:.1f}%")
        else:
            print("Runahead tokens: 0 (no runahead completed)")

        print(f"\nServer-side rejections: {total_gate_rejections}")
        print(f"Client requeues: {aggregate['total_requeues']}")

        # Assertions
        print("\n--- Assertions ---")
        all_passed = True

        # 1. No oversubscription
        if oversubscribed:
            print(f"   [FAIL] OVERSUBSCRIPTION DETECTED! Max: {max_oversubscription}")
            all_passed = False
        else:
            print(f"   [PASS] No oversubscription across all {DP_SIZE} gates")

        # 2. All rounds completed
        if aggregate["total_rounds"] == total_rounds:
            print(f"   [PASS] All {total_rounds} rounds completed")
        else:
            print(f"   [FAIL] Only {aggregate['total_rounds']}/{total_rounds} rounds completed")
            all_passed = False

        # 3. All primary completed
        if aggregate["total_primary_completed"] == total_primary:
            print(f"   [PASS] All {total_primary} primary requests completed")
        else:
            print(f"   [FAIL] Only {aggregate['total_primary_completed']}/{total_primary} primary completed")
            all_passed = False

        # 4. No errors
        if aggregate["total_errors"] == 0:
            print(f"   [PASS] No errors during stress test")
        else:
            print(f"   [WARN] {aggregate['total_errors']} errors occurred")
            # Not a hard failure, but concerning

        # 5. Some runahead completed (slack utilization)
        if aggregate["total_runahead_completed"] > 0:
            print(f"   [PASS] Runahead slack utilized ({aggregate['total_runahead_completed']} completed)")
        else:
            print(f"   [INFO] No runahead completed (may indicate high primary load)")

        # 6. System stability (reasonable throughput)
        if total_tokens > 0 and total_time < 600:  # Should complete within 10 min
            print(f"   [PASS] System stable (completed in {total_time:.1f}s)")
        elif total_time >= 600:
            print(f"   [WARN] Test took too long ({total_time:.1f}s)")

        print("\n" + "=" * 80)

        # Final verdict
        if all_passed:
            print("\nTest PASSED: E2E multi-GPU stress test completed successfully!")
            print(f"Summary: {total_tokens:,} tokens in {total_time:.1f}s = {total_tokens/total_time:.0f} tok/s")
        else:
            raise AssertionError("Some assertions failed - see above for details")

    finally:
        print("\nShutting down Ray...")
        ray.shutdown()


if __name__ == "__main__":
    test_e2e_multi_gpu_stress()

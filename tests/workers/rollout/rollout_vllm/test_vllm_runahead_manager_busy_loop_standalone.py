# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
#
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
End-to-end vLLM test for AgentLoopManager.generate_sequences_with_runahead().

This runs REAL vLLM servers (GPU) behind AgentLoopManager and exercises the
manager-level Ray-native busy loop (ray.wait + drip-feed).

Usage:
  MODEL_PATH=~/models/Qwen/Qwen2.5-0.5B-Instruct \\
    NUM_GPUS=1 TP_SIZE=1 \\
    PRIMARY_SIZE=4 RUNAHEAD_SIZE=16 NUM_WORKERS=2 \\
    python tests/workers/rollout/rollout_vllm/test_vllm_runahead_manager_busy_loop_standalone.py

Notes:
  - Requires local model weights (avoid HF download in restricted envs).
  - Keep PRIMARY_SIZE divisible by NUM_WORKERS (DataProto.chunk requirement).
  - Optional kv-cache-aware admission (workload polling):
      USE_KV_CACHE_ADMISSION=1 KV_CACHE_THRESHOLD=0.85 \\
      WORKLOAD_POLL_INTERVAL_S=0.5 WORKLOAD_STALENESS_THRESHOLD_S=2.0 \\
      REQUIRE_FRESH_WORKLOAD=0
"""

from __future__ import annotations

import os
import time

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.runahead import RunaheadConfig
from verl.protocol import DataProto
from verl.utils import hf_tokenizer


def _compose_config(model_path: str, num_gpus: int, tp_size: int, num_workers: int) -> DictConfig:
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath("verl/verl/trainer/config")
    if not os.path.exists(config_dir):
        config_dir = os.path.abspath("verl/trainer/config")

    with initialize_config_dir(config_dir=config_dir, version_base=None):
        config = compose(config_name="ppo_trainer")

    config.trainer.n_gpus_per_node = num_gpus
    config.trainer.nnodes = 1

    config.actor_rollout_ref.model.path = model_path
    config.actor_rollout_ref.rollout.name = "vllm"
    config.actor_rollout_ref.rollout.mode = "async"
    config.actor_rollout_ref.rollout.tensor_model_parallel_size = tp_size
    config.actor_rollout_ref.rollout.data_parallel_size = 1
    config.actor_rollout_ref.rollout.pipeline_model_parallel_size = 1

    # Keep bounds small for test runtime.
    config.actor_rollout_ref.rollout.prompt_length = int(os.getenv("PROMPT_LENGTH", "256"))
    config.actor_rollout_ref.rollout.response_length = int(os.getenv("RESPONSE_LENGTH", "256"))

    config.actor_rollout_ref.rollout.agent.num_workers = num_workers

    # Disable reward for this smoke test.
    if hasattr(config, "reward_model"):
        config.reward_model.enable = False
        config.reward_model.use_reward_loop = False
        config.reward_model.enable_resource_pool = False

    return config


def _build_primary_dataproto(primary_size: int) -> tuple[DataProto, list[list[dict]]]:
    raw_prompts: list[list[dict]] = []
    for i in range(primary_size):
        raw_prompts.append([{"role": "user", "content": f"Primary {i}: What is {i}+{i}?"}])

    dp = DataProto(
        non_tensor_batch={
            "raw_prompt": np.array(raw_prompts, dtype=object),
            "agent_name": np.array(["single_turn_agent"] * primary_size, dtype=object),
            "data_source": np.array(["unit_test"] * primary_size, dtype=object),
            "reward_model": np.array([{}] * primary_size, dtype=object),
        },
    )
    return dp, raw_prompts


def _build_secondary_dataproto(tokenizer, runahead_size: int) -> tuple[DataProto, list[list[dict]]]:
    raw_prompts: list[list[dict]] = []
    for i in range(runahead_size):
        # Make secondaries "wordier" so some are likely still running when primaries finish.
        raw_prompts.append(
            [
                {
                    "role": "user",
                    "content": f"Secondary {i}: Write a detailed explanation (at least 5 sentences) "
                    f"about why {i}+{i}={2*i}.",
                }
            ]
        )

    prompt_ids_list = [
        tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True) for messages in raw_prompts
    ]
    padded = tokenizer.pad(
        {"input_ids": prompt_ids_list},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    batch = TensorDict(
        {
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
        },
        batch_size=(runahead_size,),
    )
    dp = DataProto(batch=batch)
    return dp, raw_prompts


def test_vllm_runahead_manager_busy_loop_standalone() -> None:
    model_path = os.path.expanduser(os.getenv("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct"))
    num_gpus = int(os.getenv("NUM_GPUS", "1"))
    tp_size = int(os.getenv("TP_SIZE", "1"))
    num_workers = int(os.getenv("NUM_WORKERS", "2"))

    primary_size = int(os.getenv("PRIMARY_SIZE", str(num_workers * 2)))
    runahead_size = int(os.getenv("RUNAHEAD_SIZE", "16"))

    if primary_size % num_workers != 0:
        raise ValueError(f"PRIMARY_SIZE ({primary_size}) must be divisible by NUM_WORKERS ({num_workers})")

    runahead_cfg = RunaheadConfig(
        enabled=True,
        load_threshold=int(os.getenv("LOAD_THRESHOLD", "32")),
        poll_interval_s=float(os.getenv("POLL_INTERVAL_S", "0.05")),
        max_retries=int(os.getenv("MAX_RETRIES", "0")),
        max_secondary_concurrent=int(os.getenv("MAX_SECONDARY_CONCURRENT", "4")),
        use_kv_cache_admission=os.getenv("USE_KV_CACHE_ADMISSION", "0").strip() not in ("0", "false", "False"),
        kv_cache_threshold=float(os.getenv("KV_CACHE_THRESHOLD", "0.85")),
        workload_poll_interval_s=float(os.getenv("WORKLOAD_POLL_INTERVAL_S", "0.5")),
        workload_staleness_threshold_s=float(os.getenv("WORKLOAD_STALENESS_THRESHOLD_S", "2.0")),
        require_fresh_workload=os.getenv("REQUIRE_FRESH_WORKLOAD", "0").strip() not in ("0", "false", "False"),
    )

    print("=" * 80)
    print("vLLM Runahead Manager Busy-Loop E2E")
    print("=" * 80)
    print(f"MODEL_PATH: {model_path}")
    print(f"NUM_GPUS: {num_gpus} | TP_SIZE: {tp_size} | NUM_WORKERS: {num_workers}")
    print(f"PRIMARY_SIZE: {primary_size} | RUNAHEAD_SIZE: {runahead_size}")
    print(
        "RunaheadConfig:",
        {
            "load_threshold": runahead_cfg.load_threshold,
            "poll_interval_s": runahead_cfg.poll_interval_s,
            "max_retries": runahead_cfg.max_retries,
            "max_secondary_concurrent": runahead_cfg.max_secondary_concurrent,
            "use_kv_cache_admission": runahead_cfg.use_kv_cache_admission,
            "kv_cache_threshold": runahead_cfg.kv_cache_threshold,
            "workload_poll_interval_s": runahead_cfg.workload_poll_interval_s,
            "workload_staleness_threshold_s": runahead_cfg.workload_staleness_threshold_s,
            "require_fresh_workload": runahead_cfg.require_fresh_workload,
        },
    )
    print("=" * 80)

    if ray.is_initialized():
        ray.shutdown()

    ray.init(
        runtime_env={
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": os.getenv("VLLM_LOGGING_LEVEL", "INFO"),
                "VLLM_USE_V1": os.getenv("VLLM_USE_V1", "1"),
            }
        },
        ignore_reinit_error=True,
    )

    try:
        print("\n[1] Build config + AgentLoopManager (this may take a while)...")
        config = _compose_config(model_path=model_path, num_gpus=num_gpus, tp_size=tp_size, num_workers=num_workers)
        manager = AgentLoopManager(config)

        print("\n[2] Build primary + secondary inputs...")
        primary_dp, primary_raw_prompts = _build_primary_dataproto(primary_size)
        tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
        secondary_dp, _secondary_raw_prompts = _build_secondary_dataproto(tokenizer, runahead_size)

        print("\n[3] Run generate_sequences_with_runahead()...")
        t0 = time.perf_counter()
        result = manager.generate_sequences_with_runahead(primary_dp, secondary_dp, runahead_cfg)
        dt = time.perf_counter() - t0

        primary_out = result.primary_outputs
        assert primary_out is not None
        assert len(primary_out) == primary_size

        # Verify deterministic primary order by comparing preserved raw_prompt field.
        out_raw_prompts = primary_out.non_tensor_batch["raw_prompt"]
        for i in range(primary_size):
            expected = primary_raw_prompts[i][0]["content"]
            got = out_raw_prompts[i][0]["content"]
            assert got == expected, (i, got, expected)

        print("\n[4] Results")
        print(f"Primary completed: {len(primary_out)} samples in {dt:.2f}s")
        print("Runahead metrics:", result.metrics)

        completed = sum(1 for s in result.secondary_outputs if s.status == "completed")
        aborted = sum(1 for s in result.secondary_outputs if s.status == "aborted")
        rejected = sum(1 for s in result.secondary_outputs if s.status == "rejected")
        print(f"Secondary: {completed} completed, {aborted} aborted, {rejected} rejected")

        if runahead_cfg.use_kv_cache_admission:
            try:
                snapshot = ray.get(manager.router.refresh_workload_cache.remote())
                print("\nRouter workload snapshot (post-run):", snapshot)
            except Exception as e:
                print(f"\nRouter workload snapshot unavailable: {e}")

        # Print a couple of decoded primary responses for sanity.
        resp = primary_out.batch["responses"]
        resp_mask = primary_out.batch["response_mask"]
        for i in range(min(2, primary_size)):
            token_ids = resp[i][resp_mask[i].bool()].tolist()
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            print(f"\nPrimary[{i}] response (truncated): {text[:200]!r}")

        print("\n[OK] vLLM runahead manager busy-loop e2e completed")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    test_vllm_runahead_manager_busy_loop_standalone()

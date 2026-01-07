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
  - Mixed secondary max_tokens:
      SECONDARY_LONG_COUNT=4 SECONDARY_LONG_MAX_TOKENS=256 SECONDARY_SHORT_MAX_TOKENS=32
  - Router workload debug (prints router.get_server_state(poll_workload=True)):
      PRINT_SERVER_WORKLOAD=1 PRINT_SERVER_WORKLOAD_DURING_RUN=1 SERVER_WORKLOAD_PRINT_INTERVAL_S=1.0
  - Heavier workload defaults:
      HEAVY_WORKLOAD=1
  - Abort behavior visibility:
      ABORT_GRACE_S=1.0 PRINT_ABORTED_SECONDARIES=1
"""

from __future__ import annotations

import os
import time
from collections import Counter
from threading import Thread

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from verl.experimental.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.runahead import RunaheadConfig
from verl.protocol import DataProto
from verl.utils import hf_tokenizer


def _compose_config(
    model_path: str,
    num_gpus: int,
    tp_size: int,
    num_workers: int,
    *,
    prompt_length: int,
    response_length: int,
) -> DictConfig:
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
    config.actor_rollout_ref.rollout.prompt_length = int(prompt_length)
    config.actor_rollout_ref.rollout.response_length = int(response_length)

    config.actor_rollout_ref.rollout.agent.num_workers = num_workers

    # Disable reward for this smoke test.
    if hasattr(config, "reward_model"):
        config.reward_model.enable = False
        config.reward_model.use_reward_loop = False
        config.reward_model.enable_resource_pool = False

    return config


def _build_primary_dataproto(primary_size: int, *, heavy: bool) -> tuple[DataProto, list[list[dict]]]:
    raw_prompts: list[list[dict]] = []
    for i in range(primary_size):
        if heavy:
            raw_prompts.append(
                [
                    {
                        "role": "user",
                        "content": f"Primary {i}: Explain in detail (at least 20 sentences) why {i}+{i}={2*i}. "
                        "Include examples and reasoning steps.",
                    }
                ]
            )
        else:
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


def _build_secondary_dataproto(
    tokenizer,
    runahead_size: int,
    *,
    response_length: int,
    heavy: bool,
) -> tuple[DataProto, list[list[dict]]]:
    raw_prompts: list[list[dict]] = []
    sampling_params: list[dict] = []
    default_long_divisor = 2 if heavy else 4
    long_count = int(os.getenv("SECONDARY_LONG_COUNT", str(max(1, runahead_size // default_long_divisor))))
    long_count = max(0, min(runahead_size, long_count))
    long_max_tokens = int(os.getenv("SECONDARY_LONG_MAX_TOKENS", str(response_length)))
    short_max_tokens = int(os.getenv("SECONDARY_SHORT_MAX_TOKENS", "64" if heavy else "32"))
    for i in range(runahead_size):
        # Make secondaries "wordier" so some are likely still running when primaries finish.
        # This increases the chance we exercise the abort + partial-output collection path.
        raw_prompts.append(
            [
                {
                    "role": "user",
                    "content": f"Secondary {i}: Write a very detailed explanation (at least 15 sentences) "
                    f"about why {i}+{i}={2*i}. Include examples and edge cases.",
                }
            ]
        )
        if i < long_count:
            sampling_params.append({"max_tokens": long_max_tokens})
        else:
            sampling_params.append({"max_tokens": short_max_tokens})

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
    dp = DataProto(batch=batch, non_tensor_batch={"sampling_params": np.array(sampling_params, dtype=object)})
    return dp, raw_prompts


def test_vllm_runahead_manager_busy_loop_standalone() -> None:
    model_path = os.path.expanduser(os.getenv("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct"))
    num_gpus = int(os.getenv("NUM_GPUS", "1"))
    tp_size = int(os.getenv("TP_SIZE", "1"))
    num_workers = int(os.getenv("NUM_WORKERS", "2"))

    heavy = os.getenv("HEAVY_WORKLOAD", "0").strip() not in ("0", "false", "False")

    primary_size = int(os.getenv("PRIMARY_SIZE", str(num_workers * (8 if heavy else 2))))
    runahead_size = int(os.getenv("RUNAHEAD_SIZE", "64" if heavy else "16"))

    if primary_size % num_workers != 0:
        raise ValueError(f"PRIMARY_SIZE ({primary_size}) must be divisible by NUM_WORKERS ({num_workers})")

    prompt_length = int(os.getenv("PROMPT_LENGTH", "512" if heavy else "256"))
    response_length = int(os.getenv("RESPONSE_LENGTH", "512" if heavy else "256"))

    runahead_cfg = RunaheadConfig(
        enabled=True,
        load_threshold=int(os.getenv("LOAD_THRESHOLD", "32")),
        admit_loop_poll_s=float(os.getenv("POLL_INTERVAL_S", "0.05" if heavy else "0.05")),
        max_secondary_concurrent=int(os.getenv("MAX_SECONDARY_CONCURRENT", "8" if heavy else "4")),
        use_kv_cache_admission=os.getenv("USE_KV_CACHE_ADMISSION", "0").strip() not in ("0", "false", "False"),
        kv_cache_threshold=float(os.getenv("KV_CACHE_THRESHOLD", "0.85")),
        workload_poll_interval_s=float(os.getenv("WORKLOAD_POLL_INTERVAL_S", "0.5")),
        workload_staleness_threshold_s=float(os.getenv("WORKLOAD_STALENESS_THRESHOLD_S", "2.0")),
        require_fresh_workload=os.getenv("REQUIRE_FRESH_WORKLOAD", "0").strip() not in ("0", "false", "False"),
        abort_grace_s=float(os.getenv("ABORT_GRACE_S", "1.0")),
    )

    print("=" * 80)
    print("vLLM Runahead Manager Busy-Loop E2E")
    print("=" * 80)
    print(f"MODEL_PATH: {model_path}")
    print(f"NUM_GPUS: {num_gpus} | TP_SIZE: {tp_size} | NUM_WORKERS: {num_workers}")
    print(f"PRIMARY_SIZE: {primary_size} | RUNAHEAD_SIZE: {runahead_size}")
    print(f"PROMPT_LENGTH: {prompt_length} | RESPONSE_LENGTH: {response_length} | HEAVY_WORKLOAD: {heavy}")
    print(
        "RunaheadConfig:",
        {
            "load_threshold": runahead_cfg.load_threshold,
            "admit_loop_poll_s": runahead_cfg.admit_loop_poll_s,
            "max_secondary_concurrent": runahead_cfg.max_secondary_concurrent,
            "use_kv_cache_admission": runahead_cfg.use_kv_cache_admission,
            "kv_cache_threshold": runahead_cfg.kv_cache_threshold,
            "workload_poll_interval_s": runahead_cfg.workload_poll_interval_s,
            "workload_staleness_threshold_s": runahead_cfg.workload_staleness_threshold_s,
            "require_fresh_workload": runahead_cfg.require_fresh_workload,
            "abort_grace_s": runahead_cfg.abort_grace_s,
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
        config = _compose_config(
            model_path=model_path,
            num_gpus=num_gpus,
            tp_size=tp_size,
            num_workers=num_workers,
            prompt_length=prompt_length,
            response_length=response_length,
        )
        manager = AgentLoopManager(config)

        print("\n[2] Build primary + secondary inputs...")
        primary_dp, primary_raw_prompts = _build_primary_dataproto(primary_size, heavy=heavy)
        tokenizer = hf_tokenizer(model_path, trust_remote_code=True)
        secondary_dp, _secondary_raw_prompts = _build_secondary_dataproto(
            tokenizer, runahead_size, response_length=response_length, heavy=heavy
        )
        long_count = int(os.getenv("SECONDARY_LONG_COUNT", str(max(1, runahead_size // 4))))
        long_count = max(0, min(runahead_size, long_count))
        print(
            "Secondary sampling mix:",
            {
                "long_count": long_count,
                "long_max_tokens": int(os.getenv("SECONDARY_LONG_MAX_TOKENS", os.getenv("RESPONSE_LENGTH", "256"))),
                "short_max_tokens": int(os.getenv("SECONDARY_SHORT_MAX_TOKENS", "32")),
            },
        )

        def _print_server_state(tag: str) -> None:
            try:
                state = ray.get(manager.router.get_server_state.remote(poll_workload=True))
                print(f"\n[{tag}] Router get_server_state(poll_workload=True):", state, flush=True)
            except Exception as e:
                print(f"\n[{tag}] Router get_server_state unavailable: {e}", flush=True)

        print_server_workload = os.getenv("PRINT_SERVER_WORKLOAD", "1").strip() not in ("0", "false", "False")
        if print_server_workload:
            _print_server_state("pre-run")

        print("\n[3] Run generate_sequences_with_runahead()...")
        t0 = time.perf_counter()
        result_box: dict[str, object] = {}
        error_box: dict[str, BaseException] = {}

        def _run_runahead() -> None:
            try:
                result_box["result"] = manager.generate_sequences_with_runahead(primary_dp, secondary_dp, runahead_cfg)
            except BaseException as e:
                error_box["error"] = e

        monitor = os.getenv("PRINT_SERVER_WORKLOAD_DURING_RUN", "1" if print_server_workload else "0").strip() not in (
            "0",
            "false",
            "False",
        )
        monitor_interval_s = float(os.getenv("SERVER_WORKLOAD_PRINT_INTERVAL_S", "0.5" if heavy else "1.0"))
        if monitor:
            t = Thread(target=_run_runahead, daemon=True)
            t.start()
            while t.is_alive():
                _print_server_state("during-run")
                time.sleep(max(0.1, monitor_interval_s))
            t.join()
        else:
            _run_runahead()

        if "error" in error_box:
            raise error_box["error"]
        result = result_box["result"]
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

        if os.getenv("PRINT_ABORTED_SECONDARIES", "1").strip() not in ("0", "false", "False"):
            aborted_outputs = [s for s in result.secondary_outputs if s.status == "aborted"]
            aborted_with_output = [s for s in aborted_outputs if s.output is not None]
            aborted_without_output = [s for s in aborted_outputs if s.output is None]
            print("\n[5] Abort behavior (secondary)")
            print(
                "Aborted outputs:",
                {
                    "total": len(aborted_outputs),
                    "with_output": len(aborted_with_output),
                    "without_output": len(aborted_without_output),
                },
            )
            if aborted_with_output:
                stop_reason_counts = Counter(getattr(s.output, "stop_reason", None) for s in aborted_with_output)
                token_counts = [s.tokens_generated for s in aborted_with_output]
                p = np.percentile(token_counts, [0, 50, 90, 100]).tolist()
                print("stop_reason counts:", dict(stop_reason_counts))
                print("tokens_generated percentiles (min/p50/p90/max):", [int(x) for x in p])

                # Show a couple of partial decodes for sanity (truncated).
                for i, s in enumerate(aborted_with_output[:2]):
                    text = tokenizer.decode(s.output.token_ids, skip_special_tokens=True)
                    print(
                        f"\nAborted secondary[{i}] sample_id={s.sample_id} tokens_generated={s.tokens_generated} "
                        f"stop_reason={getattr(s.output, 'stop_reason', None)!r}"
                    )
                    print(f"  text (truncated): {text[:200]!r}")

        if runahead_cfg.use_kv_cache_admission:
            try:
                snapshot = ray.get(manager.router.refresh_workload_cache.remote())
                print("\nRouter workload snapshot (post-run):", snapshot)
            except Exception as e:
                print(f"\nRouter workload snapshot unavailable: {e}")

        if print_server_workload:
            try:
                state = ray.get(manager.router.get_server_state.remote(poll_workload=True))
                print("\nRouter per-server state (post-run):", state, flush=True)
            except Exception as e:
                print(f"\nRouter per-server state unavailable: {e}", flush=True)

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

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
Standalone test for CentralRouter with real vLLM server.

This test validates the CentralRouter with an actual vLLM server running on GPU,
instead of mock servers. It tests:
1. Basic token generation through the router
2. Multiple concurrent requests
3. Sticky sessions (prefix caching)
4. RouterAdapter integration
5. Comparison with AsyncLLMServerManager (deterministic output matching)

Usage:
    python tests/workers/rollout/rollout_vllm/test_central_router_standalone.py

Requirements:
    - 1 GPU available
    - Model: Qwen/Qwen2.5-0.5B-Instruct (or set MODEL_PATH env var)

Environment variables:
    MODEL_PATH: HuggingFace model path (default: Qwen/Qwen2.5-0.5B-Instruct)
    NUM_GPUS: Number of GPUs (default: 1)
    TP_SIZE: Tensor parallel size (default: 1)
"""

from __future__ import annotations

import asyncio
import os
import time
from uuid import uuid4

import ray


# =============================================================================
# Configuration
# =============================================================================

MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen2.5-0.5B-Instruct")
NUM_GPUS = int(os.getenv("NUM_GPUS", "1"))
TP_SIZE = int(os.getenv("TP_SIZE", "1"))
DP_SIZE = NUM_GPUS // TP_SIZE


# =============================================================================
# Global state (initialized in main)
# =============================================================================

trainer_config = None
server_handles = []
tokenizer = None
router = None


# =============================================================================
# Test Functions
# =============================================================================


async def test_basic_generation():
    """Test CentralRouter generates tokens with real vLLM server."""
    from verl.experimental.agent_loop.router import RouterAdapter

    adapter = RouterAdapter(router)

    # Create a simple prompt
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2?"}],
        add_generation_prompt=True,
        tokenize=True,
    )

    output = await adapter.generate(
        request_id=f"basic_{uuid4().hex[:8]}",
        prompt_ids=prompt_ids,
        sampling_params={"max_tokens": 20, "temperature": 0.7},
    )

    assert output is not None
    assert len(output.token_ids) > 0
    # stop_reason can be: "completed", "length", "stop", or other vLLM finish reasons
    assert output.stop_reason is not None

    # Decode and print the response
    response_text = tokenizer.decode(output.token_ids, skip_special_tokens=True)
    print(f"   Response: {response_text[:100]}...")
    print(f"   Tokens generated: {len(output.token_ids)}")
    print(f"   Stop reason: {output.stop_reason}")

    print("test_basic_generation PASSED")


async def test_concurrent_requests():
    """Test multiple concurrent requests are handled correctly."""
    from verl.experimental.agent_loop.router import RouterAdapter

    adapter = RouterAdapter(router)

    prompts = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the speed of light?",
        "How many planets in the solar system?",
        "What is the largest ocean?",
    ]

    # Create tasks for all prompts
    async def generate_one(idx: int, prompt: str):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
        )
        output = await adapter.generate(
            request_id=f"concurrent_{idx}_{uuid4().hex[:8]}",
            prompt_ids=prompt_ids,
            sampling_params={"max_tokens": 30, "temperature": 0.7},
        )
        return idx, output

    start_time = time.time()
    tasks = [generate_one(i, p) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time

    # Verify all outputs
    for idx, output in results:
        assert output is not None
        assert len(output.token_ids) > 0, f"Request {idx} produced no tokens"

    print(f"   Completed {len(prompts)} requests in {elapsed:.2f}s")
    print(f"   Average: {elapsed / len(prompts):.2f}s per request")

    print("test_concurrent_requests PASSED")


async def test_sticky_sessions():
    """Test same request_id routes to same server (validates routing logic)."""
    from verl.experimental.agent_loop.router import RouterAdapter

    adapter = RouterAdapter(router)

    # Simulate a multi-turn conversation with same request_id
    conversation_id = f"conversation_{uuid4().hex[:8]}"
    turns = [
        "Hello, how are you?",
        "What's your favorite color?",
        "Tell me a joke.",
    ]

    outputs = []
    for turn_idx, content in enumerate(turns):
        # Build conversation history (simplified - just current turn for prompt)
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            add_generation_prompt=True,
            tokenize=True,
        )

        output = await adapter.generate(
            request_id=conversation_id,  # Same ID for all turns
            prompt_ids=prompt_ids,
            sampling_params={"max_tokens": 30, "temperature": 0.7},
        )

        assert output is not None
        assert len(output.token_ids) > 0
        outputs.append(output)
        print(f"   Turn {turn_idx + 1}: {len(output.token_ids)} tokens")

    # With 1 server, all go to same server (validates routing doesn't break)
    assert len(outputs) == len(turns)

    print("test_sticky_sessions PASSED")


async def test_router_adapter():
    """Test RouterAdapter provides same interface as AsyncLLMServerManager."""
    from verl.experimental.agent_loop.router import RouterAdapter

    adapter = RouterAdapter(router)

    # Test that adapter has the expected generate method
    assert hasattr(adapter, "generate")
    assert callable(adapter.generate)

    # Test generation
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hello."}],
        add_generation_prompt=True,
        tokenize=True,
    )

    output = await adapter.generate(
        request_id=f"adapter_{uuid4().hex[:8]}",
        prompt_ids=prompt_ids,
        sampling_params={"max_tokens": 10},
    )

    assert output is not None
    assert len(output.token_ids) > 0

    print("test_router_adapter PASSED")


async def test_compare_with_legacy():
    """Compare CentralRouter output with AsyncLLMServerManager."""
    from verl.experimental.agent_loop.agent_loop import AsyncLLMServerManager
    from verl.experimental.agent_loop.router import CentralRouter, RouterAdapter

    # Create fresh router for comparison
    comparison_router = CentralRouter.remote(server_handles)
    adapter = RouterAdapter(comparison_router)

    # Create legacy manager with same server handles
    legacy_manager = AsyncLLMServerManager(trainer_config, server_handles)

    # Use deterministic sampling (temperature=0) for identical outputs
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 2+2? Answer with just the number."}],
        add_generation_prompt=True,
        tokenize=True,
    )
    sampling_params = {"max_tokens": 10, "temperature": 0.0}

    # Run through CentralRouter
    output_router = await adapter.generate(
        request_id=f"compare_router_{uuid4().hex[:8]}",
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
    )

    # Run through legacy AsyncLLMServerManager
    output_legacy = await legacy_manager.generate(
        request_id=f"compare_legacy_{uuid4().hex[:8]}",
        prompt_ids=prompt_ids,
        sampling_params=sampling_params,
    )

    # Decode outputs
    text_router = tokenizer.decode(output_router.token_ids, skip_special_tokens=True)
    text_legacy = tokenizer.decode(output_legacy.token_ids, skip_special_tokens=True)

    print(f"   CentralRouter output: {text_router}")
    print(f"   Legacy manager output: {text_legacy}")

    # With temperature=0, outputs should be identical
    assert output_router.token_ids == output_legacy.token_ids, (
        f"Outputs differ!\n"
        f"  Router: {output_router.token_ids}\n"
        f"  Legacy: {output_legacy.token_ids}"
    )

    print("test_compare_with_legacy PASSED - Outputs are identical!")


async def test_load_tracking():
    """Test that router's load tracking works correctly."""
    # Check loads (may not be exactly 0 due to async timing)
    loads = await router.get_server_loads.remote()
    print(f"   Current loads: {loads}")

    # Check total requests counter - should be > 0 after previous tests
    total = await router.get_total_requests.remote()
    print(f"   Total requests processed: {total}")
    assert total > 0, "Expected total requests to be > 0 after running tests"

    print("test_load_tracking PASSED")


# =============================================================================
# Main
# =============================================================================


async def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Test 1: Basic Generation")
    print("-" * 70)
    await test_basic_generation()

    print("\n" + "=" * 70)
    print("Test 2: Concurrent Requests")
    print("-" * 70)
    await test_concurrent_requests()

    print("\n" + "=" * 70)
    print("Test 3: Sticky Sessions")
    print("-" * 70)
    await test_sticky_sessions()

    print("\n" + "=" * 70)
    print("Test 4: RouterAdapter Interface")
    print("-" * 70)
    await test_router_adapter()

    print("\n" + "=" * 70)
    print("Test 5: Compare with AsyncLLMServerManager")
    print("-" * 70)
    await test_compare_with_legacy()

    print("\n" + "=" * 70)
    print("Test 6: Load Tracking")
    print("-" * 70)
    await test_load_tracking()

    print("\n" + "=" * 70)
    print("All tests PASSED!")
    print("=" * 70 + "\n")


def main():
    global trainer_config, server_handles, tokenizer, router

    print("=" * 70)
    print("CentralRouter Standalone Test with Real vLLM Server")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"GPUs: {NUM_GPUS} | TP: {TP_SIZE} | DP: {DP_SIZE}")
    print("=" * 70)

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
        if hasattr(trainer_config, "reward_model"):
            trainer_config.reward_model.use_reward_loop = False

        print(f"\n[3] Creating {DP_SIZE} vLLM server(s)...")
        from verl.workers.rollout.replica import get_rollout_replica_class

        rollout_config = trainer_config.actor_rollout_ref.rollout
        model_config = trainer_config.actor_rollout_ref.model
        rollout_class = get_rollout_replica_class("vllm")

        servers = []
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

        print("\n[4] Loading tokenizer...")
        from verl.utils import hf_tokenizer
        from verl.utils.fs import copy_to_local

        local_path = copy_to_local(MODEL_PATH)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=True)

        print("\n[5] Creating CentralRouter...")
        from verl.experimental.agent_loop.router import CentralRouter

        router = CentralRouter.remote(server_handles)
        print("   CentralRouter created")

        print("\n[6] Running tests...")
        asyncio.run(run_all_tests())

    finally:
        print("\n[7] Cleanup: Shutting down Ray...")
        ray.shutdown()
        print("   Done")


if __name__ == "__main__":
    main()

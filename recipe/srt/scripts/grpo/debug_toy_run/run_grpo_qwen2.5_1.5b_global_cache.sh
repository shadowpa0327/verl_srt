#!/bin/bash
# SRT (Speculative Rollout with Tree-Structured Cache) PPO Training Script
# Global Cache Mode: Each vLLM worker maintains its own independent suffix cache.
#
# In global_cache mode:
#   - Workers build caches from in-flight tokens only (no snapshot pushing from trainer)
#   - No runahead generation (cache is local to each worker)
#   - Simplest mode - caches are self-contained within each vLLM worker process
#
# SRT config options:
#   enable_srt: Enable suffix tree management
#   srt_cache_mode: "global_cache" for worker-local caches
#   srt_max_tree_depth: Maximum tree depth for pattern collection (default: 64)
#   srt_num_speculative_tokens: Max tokens to speculate per step (default: 24)

set -x

# Allow CUDA visibility in Ray actors that don't request GPUs
# (needed for vLLM module imports that check CUDA capabilities)
export VERL_LOGGING_LEVEL=INFO
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

python3 -m recipe.srt.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k/train.parquet \
    data.val_files=$HOME/data/gsm8k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_srt' \
    trainer.experiment_name='qwen2.5_1.5b_srt_global_cache' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=200 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    +actor_rollout_ref.rollout.enable_srt=true \
    +actor_rollout_ref.rollout.srt_cache_mode=global_cache \
    +actor_rollout_ref.rollout.srt_max_tree_depth=64 \
    +actor_rollout_ref.rollout.srt_num_speculative_tokens=24 \
    $@

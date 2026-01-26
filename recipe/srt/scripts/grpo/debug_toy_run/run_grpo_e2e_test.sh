#!/bin/bash
# SRT E2E Test Script (Standard Mode, No Runahead)
#
# Minimal test to verify SRT integration works end-to-end.
# Uses small batch size and limited steps for quick validation.

set -x

# Allow CUDA visibility in Ray actors
export VERL_LOGGING_LEVEL=INFO
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

# Use snapshot mode (default) - not shared_memory
# To test shared_memory mode, add: +actor_rollout_ref.rollout.srt_cache_mode=shared_memory

python3 -m recipe.srt.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/ubuntu/data/gsm8k/train.parquet \
    data.val_files=/home/ubuntu/data/gsm8k/test.parquet \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='srt_e2e_test' \
    trainer.experiment_name='qwen0.5b_srt_standard' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=2 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False \
    +actor_rollout_ref.rollout.enable_srt=true \
    +actor_rollout_ref.rollout.srt_max_tree_depth=32 \
    +actor_rollout_ref.rollout.srt_hash_token_count=64 \
    +actor_rollout_ref.rollout.srt_num_speculative_tokens=16 \
    $@

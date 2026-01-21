#!/bin/bash
# SRT E2E Test Script (Runahead Mode)
#
# Tests SRT with runahead enabled - uses GPU bubbles to pre-generate
# future batches while current batch is training.
#
# Requirements:
# - Async/server mode rollout (mode=async)
# - Multiple GPUs recommended (runahead benefits from GPU bubbles)
#
# Key differences from standard mode:
# - Uses generate_sequences_with_runahead() instead of generate_sequences()
# - Secondary outputs populate suffix cache for next tick
# - Logs runahead metrics: secondary_started, secondary_completed, etc.
#
# Secondary batch strategy options (via +trainer.runahead.secondary_n):
# - Not set (default): Uses primary rollout.n - "depth" strategy
# - secondary_n=1: More unique prompts, 1 sample each - "breadth" strategy
# - secondary_n=N: Balanced approach - "hybrid" strategy

set -x

# Allow CUDA visibility in Ray actors
export VERL_LOGGING_LEVEL=INFO
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

python3 -m recipe.srt.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/ubuntu/data/gsm8k/train.parquet \
    data.val_files=/home/ubuntu/data/gsm8k/test.parquet \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.train_max_samples=500 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='srt_e2e_test_runahead' \
    trainer.experiment_name='qwen0.5b_srt_runahead' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=2 \
    trainer.total_epochs=15 \
    trainer.val_before_train=False \
    +trainer.enable_runahead=true \
    +trainer.runahead.load_threshold=32 \
    +trainer.runahead.max_queue_size=256 \
    +trainer.runahead.secondary_priority=10 \
    +trainer.runahead.abort_grace_s=1.0 \
    +trainer.runahead.secondary_n=1 \
    +actor_rollout_ref.rollout.enable_srt=true \
    +actor_rollout_ref.rollout.srt_max_tree_depth=32 \
    +actor_rollout_ref.rollout.srt_hash_token_count=64 \
    +actor_rollout_ref.rollout.srt_num_speculative_tokens=16 \
    $@

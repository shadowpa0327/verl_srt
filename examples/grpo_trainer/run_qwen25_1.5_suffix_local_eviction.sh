#!/bin/bash
# Test script for suffix tree local mode with sequence eviction enabled.
#
# This script uses LOCAL suffix decoding mode (server_mode=false) with:
# - Sequence eviction: max_sequences_per_tree=10 (keeps last 10 sequences per tree)
# - Trainer maintains SuffixTreeManager with accumulated Q/A patterns
# - Snapshots are pushed to vLLM workers before each rollout
# - No external gRPC server needed
#
# Compare with run_qwen25_1.5_suffix_local.sh which has eviction disabled.

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/home/cc2869/repositories/verl_srt/data/gsm8k/train.parquet \
    data.val_files=/home/cc2869/repositories/verl_srt/data/gsm8k/test.parquet \
    data.train_max_samples=500 \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
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
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.suffix_decoding.enable=true \
    actor_rollout_ref.rollout.suffix_decoding.server_mode=false \
    actor_rollout_ref.rollout.suffix_decoding.num_speculative_tokens=5 \
    actor_rollout_ref.rollout.suffix_decoding.max_tree_depth=64 \
    actor_rollout_ref.rollout.suffix_decoding.hash_token_count=128 \
    actor_rollout_ref.rollout.suffix_decoding.max_sequences_per_tree=10 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console", "wandb"]' \
    trainer.project_name='verl_grpo_suffix_decoding' \
    trainer.experiment_name='qwen2.5_1.5b_suffix_local_eviction' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=5000 \
    trainer.test_freq=5000 \
    trainer.total_epochs=15 $@

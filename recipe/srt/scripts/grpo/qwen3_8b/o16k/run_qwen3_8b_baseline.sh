#!/usr/bin/env bash
# GRPO Script with DAPO-aligned settings
#
# Based on verl GRPO example, but with:
# - Batch size, sequence lengths from DAPO
# - GPU memory utilization from DAPO
# - FSDP offload settings from DAPO
# - Parallel settings (TP, SP, FSDP) from DAPO
# - Async rollout mode from DAPO
# - SRT (Speculative) configurations from DAPO

set -xeuo pipefail

# ============================================
# Project Configuration
# ============================================
project_name='DAPO_SRT_Qwen3-8B'
exp_name='Qwen3_8B-16k-GRPO-DAPO17k-filtered-data'

# ============================================
# Paths
# ============================================
MODEL_PATH=/mnt/hdfs/ccchang_hldy/Qwen3-8B-Base
CKPTS_DIR=/mnt/hdfs/ccchang_hldy/ckpts/${project_name}/${exp_name}
TRAIN_FILE=/mnt/hdfs/ccchang_hldy/data/dapo-math-17k-unique-fixed.parquet
TEST_FILE=/mnt/hdfs/ccchang_hldy/data/aime-2024.parquet

# ============================================
# Ray / Cluster Configuration
# ============================================
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# ============================================
# Algorithm Configuration
# ============================================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=True
kl_loss_coef=0.001
kl_loss_type=low_var_kl

# ============================================
# Sequence Length Configuration
# ============================================
max_prompt_length=$((1024 * 2))    # 2048
max_response_length=$((1024 * 16)) # 12288

# ============================================
# Batch Size Configuration
# ============================================
train_prompt_bsz=128
n_resp_per_prompt=5
train_prompt_mini_bsz=32

# ============================================
# Parallel & Performance Configuration
# ============================================
sp_size=4
gen_tp=1
fsdp_size=8

use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=True
gpu_memory_utilization=0.80

# ============================================
# Sampling Parameters
# ============================================
temperature=1.0
top_p=1.0
top_k=-1

# ============================================
# Training Configuration (GRPO style)
# ============================================
lr=1e-6
grad_clip=1.0
entropy_coeff=0

total_epochs=15
save_freq=20
test_freq=5

# ============================================
# Logging Configuration
# ============================================
export VERL_LOGGING_LEVEL=INFO

# ============================================
# Main Training Script
# ============================================

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='left' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=${kl_loss_type} \
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff} \
    actor_rollout_ref.actor.grad_clip=${grad_clip} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.cudagraph_capture_sizes=[1,2,4,8,16,32,64,128,192,256,320,384,448,512,768,896] \
    +actor_rollout_ref.rollout.enable_srt=false \
    +actor_rollout_ref.rollout.srt_cache_mode=snapshot \
    +actor_rollout_ref.rollout.srt_max_tree_depth=32 \
    +actor_rollout_ref.rollout.srt_hash_token_count=64 \
    +actor_rollout_ref.rollout.srt_num_speculative_tokens=5 \
    +actor_rollout_ref.rollout.srt_enable_in_flight_updates=true \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=${save_freq} \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    "$@"
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in the `verl/trainer/ppo/` directory.

## Overview

This directory implements the PPO (Proximal Policy Optimization) trainer and related algorithms for reinforcement learning on LLMs. The trainer uses Ray for distributed orchestration and supports multiple RL algorithms beyond PPO including GRPO, RLOO, REINFORCE++, ReMax, and more.

## File Structure

```
verl/trainer/ppo/
├── ray_trainer.py      # Main RayPPOTrainer orchestrator
├── core_algos.py       # Algorithm implementations (advantage estimators, policy losses)
├── reward.py           # Reward function loading and computation
├── utils.py            # Role definitions and helper utilities
├── metric_utils.py     # Training and validation metrics computation
└── rollout_corr_helper.py  # Rollout correction (importance sampling, rejection sampling)
```

## Key Components

### RayPPOTrainer (ray_trainer.py)

The main orchestrator class that coordinates distributed training:

```
RayPPOTrainer.fit() training loop:
    for each batch:
        1. generate_sequences()    → Rollout (vLLM/SGLang)
        2. compute_log_prob()      → Actor (recompute old_log_probs)
        3. compute_ref_log_prob()  → Reference model (if KL enabled)
        4. compute_values()        → Critic (if GAE)
        5. compute_reward()        → Reward function/model
        6. compute_advantage()     → Driver process (lightweight)
        7. update_critic()         → Critic training step
        8. update_actor()          → Actor training step
```

Key classes:
- `RayPPOTrainer`: Main trainer class
- `ResourcePoolManager`: Manages Ray GPU resource pools for worker placement

### Advantage Estimators (core_algos.py)

Registry-based system for advantage computation. Use `@register_adv_est(name)` decorator:

| Estimator | Config Value | Description |
|-----------|--------------|-------------|
| GAE | `gae` | Generalized Advantage Estimation (requires critic) |
| GRPO | `grpo` | Group-wise normalization (no critic needed) |
| RLOO | `rloo` | Leave-one-out baseline |
| REINFORCE++ | `reinforce_plus_plus` | With whitening |
| ReMax | `remax` | Requires greedy baseline generation |
| OPO | `opo` | Length-weighted baseline |
| GPG | `gpg` | Gradient Policy Gradient |

```python
# To add a new estimator:
@register_adv_est("my_estimator")
def compute_my_advantage(token_level_rewards, response_mask, index, config=None):
    # Return (advantages, returns) tensors
    return advantages, returns
```

### Policy Loss Functions (core_algos.py)

Registry-based policy losses. Use `@register_policy_loss(name)` decorator:

| Loss | Config Value | Description |
|------|--------------|-------------|
| vanilla | `vanilla` | Standard PPO clipped loss |
| gspo | `gspo` | Sequence-level importance ratio |
| gpg | `gpg` | Pure policy gradient (no clipping) |
| geo_mean | `geo_mean` | GMPO geometric mean aggregation |
| clip_cov | `clip_cov` | Covariance-based clipping |
| kl_cov | `kl_cov` | KL penalty for high-covariance tokens |
| rollout_correction | `rollout_correction` | IS/RS for rollout-training mismatch |

```python
# To add a new policy loss:
@register_policy_loss("my_loss")
def compute_my_loss(old_log_prob, log_prob, advantages, response_mask,
                    loss_agg_mode, config, rollout_is_weights=None):
    # Return (loss, metrics_dict)
    return pg_loss, {"actor/my_metric": value}
```

### Role System (utils.py)

Worker roles for distributed training:

```python
class Role(Enum):
    Actor = 0           # Policy model training
    Rollout = 1         # Sequence generation
    ActorRollout = 2    # Fused actor + rollout
    Critic = 3          # Value function (GAE only)
    RefPolicy = 4       # Reference policy for KL
    RewardModel = 5     # Model-based reward
    ActorRolloutRef = 6 # Fused actor + rollout + ref
```

### Reward Computation (reward.py)

- `load_reward_manager()`: Loads reward manager based on config
- `get_custom_reward_fn()`: Loads user-defined reward from external file
- `compute_reward()`: Computes reward tensor for batch
- Supports: `naive`, `prime`, `batch`, `dapo` reward managers

### KL Controllers (core_algos.py)

- `AdaptiveKLController`: Adjusts KL coefficient based on current KL
- `FixedKLController`: Static KL coefficient

## Configuration

The trainer uses Hydra config at `verl/trainer/config/ppo_trainer.yaml`. Key sections:

```yaml
algorithm:
  adv_estimator: grpo        # Advantage estimator type
  gamma: 1.0                 # Discount factor
  lam: 1.0                   # GAE lambda
  use_kl_in_reward: false    # Add KL penalty to rewards
  kl_penalty: kl             # KL penalty type (kl, abs, mse, low_var_kl)

actor_rollout_ref:
  actor:
    strategy: fsdp2          # Training backend
    clip_ratio: 0.2          # PPO clip range
    policy_loss:
      loss_mode: vanilla     # Policy loss function
    loss_agg_mode: token-mean  # Loss aggregation

  rollout:
    name: vllm               # Inference engine (vllm, sglang)
    n: 5                     # Number of samples per prompt
    temperature: 1.0

critic:
  enable: null               # Auto-detect based on adv_estimator
```

## Loss Aggregation Modes

The `loss_agg_mode` parameter controls how losses are aggregated:

| Mode | Description |
|------|-------------|
| `token-mean` | Average over all valid tokens |
| `seq-mean-token-sum` | Sum tokens per sequence, average sequences |
| `seq-mean-token-mean` | Average tokens per sequence, average sequences |
| `seq-mean-token-sum-norm` | Normalized by max sequence length (DrGRPO) |

## Metrics

### Training Metrics (metric_utils.py)
- `critic/score/*`: Reward scores (mean, max, min)
- `critic/advantages/*`: Advantage statistics
- `actor/ppo_kl`: KL divergence from old policy
- `actor/pg_clipfrac`: Fraction of clipped policy gradients
- `perf/throughput`: Tokens per second per GPU

### Validation Metrics
- `val-core/{data_source}/{var}/mean@N`: Mean metric over N samples
- `val-core/{data_source}/{var}/best@N/mean`: Best-of-N with bootstrap

## Common Development Tasks

### Adding a New Advantage Estimator
1. Add enum to `AdvantageEstimator` in `core_algos.py` (optional)
2. Implement function with `@register_adv_est(name)` decorator
3. Function signature: `(token_level_rewards, response_mask, ...) -> (advantages, returns)`

### Adding a New Policy Loss
1. Implement function with `@register_policy_loss(name)` decorator
2. Function signature matches `PolicyLossFn` type alias
3. Return `(loss_tensor, metrics_dict)`

### Adding a Custom Reward Function
1. Create Python file with reward function
2. Configure in yaml:
```yaml
custom_reward_function:
  path: /path/to/reward.py
  name: my_reward_fn
  reward_kwargs:
    custom_arg: value
```

### Debugging Training Loop
Key points in `ray_trainer.py:fit()`:
- Line ~1050: Generation timing
- Line ~1100: Reward computation
- Line ~1200: Advantage computation
- Line ~1220: Critic update
- Line ~1230: Actor update

## DataProto Fields

The `DataProto` batch contains these key fields during training:

| Field | Shape | Description |
|-------|-------|-------------|
| `input_ids` | (B, seq_len) | Tokenized prompt + response |
| `attention_mask` | (B, seq_len) | Valid token mask |
| `responses` | (B, resp_len) | Generated response tokens |
| `response_mask` | (B, resp_len) | Valid response token mask |
| `old_log_probs` | (B, resp_len) | Log probs from old policy |
| `ref_log_prob` | (B, resp_len) | Log probs from reference |
| `token_level_scores` | (B, resp_len) | Per-token reward scores |
| `token_level_rewards` | (B, resp_len) | Scores with KL penalty |
| `advantages` | (B, resp_len) | Computed advantages |
| `returns` | (B, resp_len) | Computed returns |
| `values` | (B, resp_len) | Critic predictions (GAE only) |

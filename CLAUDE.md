# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning) is a flexible, efficient, and production-ready RL training library for large language models (LLMs). It implements a hybrid-controller programming model that decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks.

## Common Commands

### Installation
```bash
# Development install with test dependencies
pip install -e .[test,vllm]  # or .[test,sglang]

# Install pre-commit hooks
pip install pre-commit && pre-commit install
```

### Code Quality
```bash
# Run pre-commit on staged changes
pre-commit run

# Run on all files
pre-commit run --all-files

# Run specific hook
pre-commit run --all-files --show-diff-on-failure --color=always ruff
pre-commit run --all-files --show-diff-on-failure --color=always autogen-trainer-cfg
```

### Testing
```bash
# CPU unit tests (files matching *_on_cpu.py)
pytest -s -x tests/  # with pytest.ini setting python_files = *_on_cpu.py

# GPU unit tests
pytest -s -x --ignore-glob="*on_cpu.py" --ignore-glob="tests/special*" tests/

# Distributed tests (multi-GPU)
torchrun --standalone --nnodes=1 --nproc-per-node=2 tests/workers/actor/test_special_dp_actor.py
```

### Running Training
```bash
# PPO training with vLLM rollout
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.rollout.name=vllm \
    ...

# GRPO training
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    ...

# SFT training
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    -m verl.trainer.fsdp_sft_trainer \
    ...
```

### Data Preprocessing
```bash
# Example data preprocessing scripts
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
python3 examples/data_preprocess/math_dataset.py --local_dir ~/data/math
```

## Architecture

### Core Components

```
verl/
├── trainer/           # Training orchestration
│   ├── main_ppo.py    # PPO/GRPO entry point
│   ├── fsdp_sft_trainer.py  # SFT trainer
│   ├── ppo/
│   │   ├── ray_trainer.py   # RayPPOTrainer - main orchestrator
│   │   └── core_algos.py    # Algorithm implementations (GAE, GRPO, RLOO, etc.)
│   └── config/        # Hydra configuration schemas
├── workers/           # Distributed workers
│   ├── roles/         # Actor, Critic, RolloutRef workers
│   ├── engine/        # Training backends
│   │   ├── fsdp/      # FSDP engine (PyTorch native)
│   │   └── megatron/  # Megatron-LM engine (3D parallelism)
│   ├── rollout/       # Inference engines
│   │   ├── vllm_rollout/   # vLLM integration
│   │   └── sglang_rollout/ # SGLang integration
│   ├── fsdp_workers.py     # FSDP-based workers
│   └── megatron_workers.py # Megatron-based workers
├── single_controller/ # Ray-based distributed coordination
│   ├── base/          # Worker, WorkerGroup abstractions
│   └── ray/           # RayResourcePool, RayWorkerGroup
├── protocol.py        # DataProto - universal data container
├── models/            # Model-specific implementations
└── utils/             # Utilities (checkpoint, dataset, profiler)
```

### Key Design Patterns

1. **DataProto Protocol**: Universal data container using TensorDict for passing data between workers (actor, critic, rollout, reward model).

2. **Engine Abstraction**: `BaseEngine` defines the interface for training backends. Swap between FSDP and Megatron via config:
   ```yaml
   actor_rollout_ref.actor.strategy=fsdp2  # or megatron
   ```

3. **Rollout Integration**: `BaseRollout` interface for inference engines. Choose between vLLM and SGLang:
   ```yaml
   actor_rollout_ref.rollout.name=vllm  # or sglang
   ```

4. **Ray Dispatch**: Workers use `@register(dispatch_mode=...)` decorators for distributed method dispatch (ONE_TO_ALL, BROADCAST, etc.).

5. **Algorithm Registry**: Pluggable advantage estimators in `core_algos.py`:
   - GAE (Generalized Advantage Estimation) for PPO
   - GRPO, REINFORCE++, RLOO, REMAX, OPO for various RL algorithms

### Training Flow (PPO/GRPO)

```
RayPPOTrainer (orchestrator)
  ├── ActorRolloutRefWorker
  │   ├── ActorWorker → Engine (FSDP/Megatron)
  │   ├── RefWorker (reference policy)
  │   └── BaseRollout (vLLM/SGLang)
  ├── CriticWorker → Engine
  └── RewardModel (optional)
```

## Test Organization

- `tests/<module>/` - Tests for corresponding `verl/<module>/`
- `tests/special_distributed/` - Multi-GPU tests
- `tests/special_e2e/` - End-to-end training tests
- `*_on_cpu.py` - CPU-only tests (no GPU required)

## Configuration

Training uses Hydra for configuration. Key config groups:
- `algorithm` - RL algorithm settings (adv_estimator, kl_coef)
- `data` - Dataset and batching settings
- `actor_rollout_ref` - Actor, rollout, reference model settings
- `critic` - Critic model settings
- `reward_model` - Reward model settings
- `trainer` - Training loop settings (epochs, logging, checkpointing)

Config files are in `verl/trainer/config/`. Auto-generated configs should not be manually edited.

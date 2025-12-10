# verl Architecture

## Core Components

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

## Key Design Patterns

### 1. DataProto Protocol
Universal data container using TensorDict for passing data between workers (actor, critic, rollout, reward model).

### 2. Engine Abstraction
`BaseEngine` defines the interface for training backends. Swap between FSDP and Megatron via config:
```yaml
actor_rollout_ref.actor.strategy=fsdp2  # or megatron
```

### 3. Rollout Integration
`BaseRollout` interface for inference engines. Choose between vLLM and SGLang:
```yaml
actor_rollout_ref.rollout.name=vllm  # or sglang
```

### 4. Ray Dispatch
Workers use `@register(dispatch_mode=...)` decorators for distributed method dispatch (ONE_TO_ALL, BROADCAST, etc.).

### 5. Algorithm Registry
Pluggable advantage estimators in `core_algos.py`:
- GAE (Generalized Advantage Estimation) for PPO
- GRPO, REINFORCE++, RLOO, REMAX, OPO for various RL algorithms

## Training Flow (PPO/GRPO)

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

# verl Documentation Index

## Core Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Core components, design patterns, training flow |
| [Suffix Tree Speculation](suffix_tree_speculation.md) | vLLM speculation integration for faster inference |

## Third-Party Components

Detailed documentation for third-party integrations is in [`third_party/claude_docs/`](../third_party/claude_docs/INDEX.md):

| Component | Description |
|-----------|-------------|
| vLLM | Inference engine with speculative decoding |
| ArcticInference | Suffix tree implementation |

## Quick Links

| Task | Location |
|------|----------|
| Add vLLM rollout | `verl/workers/rollout/vllm_rollout/` |
| Modify training loop | `verl/trainer/ppo/ray_trainer.py` |
| Add new algorithm | `verl/trainer/ppo/core_algos.py` |
| Configure training | `verl/trainer/config/` |

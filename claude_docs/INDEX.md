# verl Documentation Index

## Core Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Core components, design patterns, training flow |
| [Suffix Tree Speculation](suffix_tree_speculation.md) | vLLM speculation integration for faster inference |

## Task Plans

| Document | Description |
|----------|-------------|
| [Suffix Tree VERL Integration](task_plans/suffix_tree_verl_integration.md) | Integration plan for SuffixTreeManager in trainer |
| [Suffix Tree Code Review](task_plans/suffix_tree_code_review.md) | Code quality analysis and recommendations |
| [Precomputed Hash Suffix Tree](task_plans/precomputed_hash_suffix_tree.md) | Hash-based tree mapping implementation |

## Future Enhancements

| Document | Description |
|----------|-------------|
| [Suffix Tree Async Options](future/suffix_tree_async_options.md) | Async update options (ThreadPool vs Ray Actor) |
| [Suffix Tree Memory Optimizations](future/suffix_tree_memory_optimizations.md) | Memory concerns and optimization strategies |

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

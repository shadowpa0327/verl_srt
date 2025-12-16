# verl Documentation Index

## Core Documentation

| Document | Description |
|----------|-------------|
| [Architecture](architecture.md) | Core components, design patterns, training flow |
| [Suffix Tree Speculation](suffix_tree_speculation.md) | vLLM speculation integration overview |

## Skills (Implemented, Reusable)

| Document | Description |
|----------|-------------|
| [Suffix Tree VERL Integration](skills/suffix_tree_verl_integration.md) | How SuffixTreeManager integrates with trainer |
| [Selective Snapshot Distribution](skills/selective_snapshot_distribution.md) | Batch-specific tree transfer optimization |

## Task Plans (Pending/In-Progress)

| Document | Status | Description |
|----------|--------|-------------|
| [Per-Worker Snapshot Analysis](task_plans/per_worker_snapshot_analysis.md) | Analysis | gRPC-based per-worker tree distribution |
| [Spec Decode Metrics Aggregation](task_plans/spec_decode_metrics_aggregation.md) | Ready | Multi-GPU metrics reduction |

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
| SuffixTreeManager | `verl/trainer/ppo/suffix_tree_manager.py` |

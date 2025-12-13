# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

verl (Volcano Engine Reinforcement Learning) is a flexible, efficient, and production-ready RL training library for large language models (LLMs). It implements a hybrid-controller programming model that decouples computation and data dependencies, enabling seamless integration with existing LLM frameworks.

## Documentation

| Document | Description |
|----------|-------------|
| [`claude_docs/INDEX.md`](claude_docs/INDEX.md) | Full documentation index |
| [`claude_docs/architecture.md`](claude_docs/architecture.md) | Core components & design patterns |
| [`claude_docs/suffix_tree_speculation.md`](claude_docs/suffix_tree_speculation.md) | vLLM speculation integration |
| [`third_party/CLAUDE.md`](third_party/CLAUDE.md) | Third-party components (vLLM, ArcticInference) |

## Common Commands

```bash
# Install
pip install -e .[test,vllm]

# Code quality
pre-commit run --all-files

# Tests
pytest -s -x tests/                    # CPU tests
pytest -s -x --ignore-glob="*on_cpu.py" tests/  # GPU tests

# Training
python3 -m verl.trainer.main_ppo algorithm.adv_estimator=gae actor_rollout_ref.rollout.name=vllm ...
```

## Current Tasks
<!--
INSTRUCTIONS FOR CLAUDE:
- Update this section as tasks progress
- Mark completed tasks with [x]
- Add new tasks as they emerge
- Last updated: 2025-12-13
-->

### Active
- [ ] Test suffix tree integration end-to-end with training run
- [ ] Implement spec decode metrics aggregation across workers
  - Add `reduce_spec_decode_metrics()` to `verl/utils/profiler/performance.py`
  - Call in `fsdp_workers.py:generate_sequences()` after timing reduction
  - Details: [`claude_docs/task_plans/spec_decode_metrics_aggregation.md`](claude_docs/task_plans/spec_decode_metrics_aggregation.md)
- [ ] Implement age-based sequence eviction (preferred over tree-level LRU)
  - Track alive req_ids with `self._alive_requests: Dict[str, int]`
  - Periodically call `stop_request()` on old sequences
  - Details: [`claude_docs/future/suffix_tree_memory_optimizations.md`](claude_docs/future/suffix_tree_memory_optimizations.md)
- [ ] Implement selective tree loading per worker (multi-GPU optimization)
  - Each worker should only load trees for prompts it will process
  - Details: [`claude_docs/task_plans/selective_tree_loading.md`](claude_docs/task_plans/selective_tree_loading.md) (to be created)
- [ ] Add incremental snapshot transfer (only send changed trees)

### Completed
- [x] Integrate suffix tree speculation into verl rollout worker
  - Created `verl/trainer/ppo/suffix_tree_manager.py` (SuffixTreeManager class)
  - Updated `verl/trainer/ppo/ray_trainer.py` (training loop integration)
  - Updated `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py` (load_suffix_snapshot)
  - Updated `verl/workers/fsdp_workers.py` (worker dispatch)
  - Details: [`claude_docs/task_plans/suffix_tree_verl_integration.md`](claude_docs/task_plans/suffix_tree_verl_integration.md)
- [x] Implement hash-based tree mapping in ArcticInference
- [x] Add direct access `llm.load_snapshot()` API to vLLM
- [x] Remove legacy RPC chain and global tree mode from vLLM
- [x] Fix seq_id initialization and add protected tree indices
- [x] Add per-rollout spec decode metrics with delta calculation
  - Created `verl/workers/rollout/vllm_rollout/spec_decode_metrics.py`
  - Added wandb logging for acceptance rate metrics
- [x] Code review of suffix tree integration
  - Details: [`claude_docs/task_plans/suffix_tree_code_review.md`](claude_docs/task_plans/suffix_tree_code_review.md)
  - Memory analysis: [`claude_docs/future/suffix_tree_memory_optimizations.md`](claude_docs/future/suffix_tree_memory_optimizations.md)

### References
| Task | Details |
|------|---------|
| Suffix tree integration | [`third_party/claude_docs/skills/vllm_suffix_tree_integration.md`](third_party/claude_docs/skills/vllm_suffix_tree_integration.md) |
| Code review | [`claude_docs/task_plans/suffix_tree_code_review.md`](claude_docs/task_plans/suffix_tree_code_review.md) |
| Memory optimizations | [`claude_docs/future/suffix_tree_memory_optimizations.md`](claude_docs/future/suffix_tree_memory_optimizations.md) |
| Metrics aggregation | [`claude_docs/task_plans/spec_decode_metrics_aggregation.md`](claude_docs/task_plans/spec_decode_metrics_aggregation.md) |

## Key Directories
```
verl/
├── trainer/          # Training orchestration (main_ppo.py, ray_trainer.py)
├── workers/          # Distributed workers (actor, critic, rollout)
│   └── rollout/vllm_rollout/  # vLLM integration point
├── single_controller/  # Ray-based coordination
└── protocol.py       # DataProto - universal data container

third_party/
├── vllm/             # Modified vLLM with suffix decoding
├── ArcticInference_srt/  # Suffix tree implementation
└── claude_docs/      # Third-party documentation
```

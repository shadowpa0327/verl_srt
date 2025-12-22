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
- Last updated: 2025-12-21
-->

### Active
- [ ] Test suffix tree integration end-to-end with training run

### Completed
- [x] Selective suffix-tree snapshot distribution
- [x] Suffix tree VERL integration
- [x] Hash-based tree mapping in ArcticInference
- [x] Direct access `llm.load_snapshot()` API in vLLM
- [x] Per-rollout spec decode metrics
- [x] Proposer C++ optimization (`propose_from_batch` zero-copy API)
- [x] Spec decode metrics aggregation across workers

### Skills (Implemented Reference)

| Skill | Description |
|-------|-------------|
| [Suffix Tree Speculation](claude_docs/skills/suffix_tree_speculation.md) | Complete speculation system (VERL + vLLM + ArcticInference) |
| [Selective Snapshot Distribution](claude_docs/skills/selective_snapshot_distribution.md) | Batch-specific tree transfer |

### Future

| Enhancement | Details |
|-------------|---------|
| Memory optimizations | [`claude_docs/future/suffix_tree_memory_optimizations.md`](claude_docs/future/suffix_tree_memory_optimizations.md) |
| Async update options | [`claude_docs/future/suffix_tree_async_options.md`](claude_docs/future/suffix_tree_async_options.md) |

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

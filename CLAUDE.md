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
- Last updated: 2024-12-10
-->

### Active
- [ ] Integrate suffix tree speculation into verl rollout worker
  - Integration point: `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
  - Details: [`claude_docs/suffix_tree_speculation.md`](claude_docs/suffix_tree_speculation.md)

### Completed
- [x] Implement hash-based tree mapping in ArcticInference
- [x] Add direct access `llm.load_snapshot()` API to vLLM
- [x] Remove legacy RPC chain and global tree mode from vLLM
- [x] Fix seq_id initialization and add protected tree indices

### References
| Task | Details |
|------|---------|
| Suffix tree integration | [`third_party/claude_docs/skills/vllm_suffix_tree_integration.md`](third_party/claude_docs/skills/vllm_suffix_tree_integration.md) |

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

# verl_srt

A fork of [verl](https://github.com/volcengine/verl) with suffix tree speculation support for accelerated LLM inference during RL training.

## Overview

This repository extends verl with suffix tree-based speculative decoding, enabling faster rollout generation by leveraging repeated token patterns across training batches.

Key additions:
- **Suffix Tree Integration**: ArcticInference suffix tree for token pattern matching
- **Modified vLLM**: Custom vLLM fork with suffix decode support
- **Selective Snapshot Distribution**: Efficient tree state transfer to rollout workers

## Installation

### Prerequisites

- Python 3.11
- CUDA 12.8+
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Start

1. **Clone with submodules**:
   ```bash
   git clone --recurse-submodules https://github.com/shadowpa0327/verl_srt.git
   cd verl_srt
   ```

   If you already cloned without submodules:
   ```bash
   git submodule update --init --recursive
   ```

2. **Run the installation script**:
   ```bash
   ./scripts/install_deps.sh
   ```

   This will:
   - Create a virtual environment at `.venv/`
   - Install PyTorch 2.8.0 with CUDA 12.8
   - Install Flash Attention 2 and FlashInfer
   - Install vLLM (editable mode from `third_party/vllm`)
   - Install ArcticInference (editable mode from `third_party/ArcticInference_srt`)
   - Install verl in editable mode

3. **Activate the environment**:
   ```bash
   source .venv/bin/activate
   ```

### Custom Environment Location

To use a different venv location:
```bash
VENV_DIR=/path/to/your/venv ./scripts/install_deps.sh
```

## Project Structure

```
verl_srt/
├── verl/                    # Core verl library
│   ├── trainer/             # Training orchestration
│   ├── workers/             # Distributed workers (actor, critic, rollout)
│   └── utils/suffix_tree/   # Suffix tree manager
├── third_party/
│   ├── vllm/                # Modified vLLM with suffix decoding (submodule)
│   └── ArcticInference_srt/ # Suffix tree implementation (submodule)
├── scripts/
│   └── install_deps.sh      # Installation script
└── docs/
    └── README_upstream.md   # Original verl README
```

## Submodules

| Component | Branch | Description |
|-----------|--------|-------------|
| `third_party/vllm` | `suffix_decode` | vLLM fork with suffix speculation APIs |
| `third_party/ArcticInference_srt` | `dev/grpc_server` | Suffix tree implementation |

To update submodules to latest:
```bash
git submodule update --remote
```

## Documentation

- [Architecture Overview](claude_docs/architecture.md)
- [Suffix Tree Speculation](claude_docs/suffix_tree_speculation.md)
- [Third-party Components](third_party/CLAUDE.md)

## Upstream

For the original verl documentation and features, see [docs/README_upstream.md](docs/README_upstream.md) or visit [verl on GitHub](https://github.com/volcengine/verl).

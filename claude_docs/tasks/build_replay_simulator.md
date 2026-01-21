# Context

We have collected the primary rollout and secondary rollouts. We used the secondary to filled the trees and perform speculative decoding for the next steps.
Now, we have stored some rollout datas. Explore `/home/ubuntu/verl_srt/rollout_datas/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead` for examples.

I want to build a simulator, that read from collected rollout datas. Use the SRT data flow logics to fill the cache and perform speculation and simulate real decoding.
Particularily, in the rollout data we got the completed response. What I want you to do is for every sequects start from its promts, perform speculation got the speculated tokens, check with the response to verify the matchness untill the total responses generated.

---

# Status: ✅ COMPLETE

## Implementation Summary

The replay simulator is fully implemented in `recipe/srt/replay_simulator.py`.

### Key Features

1. **Dual Mode Support**:
   - `shm` mode (default): Multi-process architecture with shared memory
   - `parallel` mode: Single-process using `ParallelSuffixDecodingCache`

2. **SHM Mode (Multi-process)** - Handles protobuf conflicts between `SuffixCacheUpdater` and `SuffixCache` by using separate subprocesses:
   - Cache server subprocess (`RolloutCacheServer`)
   - Cache population subprocess (`SuffixCacheUpdater`)
   - Simulation subprocess (`SuffixCache` client)

3. **Parallel Mode (In-process)** - Simpler single-process approach:
   - Uses `ParallelSuffixDecodingCache` directly
   - No shared memory or gRPC server required
   - Supports hash-based tree sharing for efficient lookups

4. **Dual Data Format Support**:
   - `"secondary"` format: `prompt`/`response` keys (default, for SRT workflow)
   - `"rollout"` format: `input`/`output` keys

5. **Cache Verification** - Verifies cache population before simulation with retry logic (shm mode)

6. **Comprehensive Metrics**:
   - Per-request: acceptance rate, tokens/step, prompt/response lengths
   - Aggregated: mean acceptance rate, mean tokens/step, total steps/tokens
   - Statistical summary: min/max/median for acceptance rate and tokens/step

### Usage

**SHM Mode (default):**
```bash
python recipe/srt/replay_simulator.py \
    --model_path /path/to/model \
    --data_dir /path/to/rollout_datas/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead \
    --mode shm \
    --cache_tick 1 \
    --cache_source secondary \
    --sim_tick 2 \
    --verbose
```

**Parallel Mode:**
```bash
python recipe/srt/replay_simulator.py \
    --model_path /path/to/model \
    --data_dir /path/to/rollout_datas/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead \
    --mode parallel \
    --cache_tick 1 \
    --cache_source secondary \
    --sim_tick 2 \
    --hash_token_count 128 \
    --verbose
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | shm | Simulation mode: `"shm"` or `"parallel"` |
| `--cache_tick` | 1 | Tick to populate cache from |
| `--cache_source` | secondary | `"secondary"` or `"rollout"` |
| `--sim_tick` | 2 | Tick to simulate speculation for |
| `--spec_start_len` | 2 | Initial speculation length |
| `--spec_max_len` | 16 | Maximum speculation length |
| `--spec_prefix_len` | 7 | Pattern length for lookup |
| `--min_token_prob` | 0.1 | Minimum token probability threshold |
| `--max_samples` | 0 | Max samples (0=all) |

**Parallel Mode Specific Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--max_tree_depth` | 64 | Maximum tree depth |
| `--num_threads` | -1 | Number of threads (-1=auto) |
| `--parallel_threshold` | 4 | Min batch size for parallelization |
| `--hash_token_count` | 128 | Trailing tokens for tree hash sharing |

### Related Scripts

Analysis scripts added in `recipe/srt/scripts/rollot_analysis/`:
- `analyze_topk_diagnostic.py` - Diagnostic analysis
- `analyze_topk_single.py` - Single-run analysis
- `analyze_topk_sweep.py` - Parameter sweep analysis

Data organization script: `scripts/organize_rollout_by_prompt.py`

## Recent Commits

- `ba91f494` - Added cache verification and improved format handling (+57 lines)
- `049c75c2` - Added rollout data analysis scripts
- `e5c9c73d` - Reorganized scripts into `rollot_analysis/` folder 


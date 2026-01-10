#!/bin/bash
# =============================================================================
# Benchmark Parameter Sweep Script
# =============================================================================
# Iterates over combinations of benchmark parameters and saves results to JSON.
#
# Usage:
#   ./scripts/run_benchmark_sweep.sh                    # Run full sweep
#   ./scripts/run_benchmark_sweep.sh --dry-run          # Preview commands only
#   ./scripts/run_benchmark_sweep.sh --resume [DIR]     # Resume from DIR (or latest if not specified)
#   ./scripts/run_benchmark_sweep.sh --baseline-only    # Run only baseline configs
#   ./scripts/run_benchmark_sweep.sh --runahead-only    # Run only runahead configs
#
# =============================================================================

# Don't use set -e as ((current++)) returns 1 when current=0

# Default settings
DRY_RUN=false
RESUME=false
BASELINE_ONLY=false
RUNAHEAD_ONLY=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --resume)
            RESUME=true
            # Check if next argument is a path (not another flag)
            if [[ -n "$2" && "$2" != --* ]]; then
                RESUME_DIR="$2"
                shift
            fi
            shift
            ;;
        --baseline-only)
            BASELINE_ONLY=true
            shift
            ;;
        --runahead-only)
            RUNAHEAD_ONLY=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run         Preview commands without executing"
            echo "  --resume [DIR]    Resume from DIR, skipping existing results"
            echo "  --baseline-only   Run only baseline configurations"
            echo "  --runahead-only   Run only runahead configurations"
            echo "  --help, -h        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Configuration - Customize these arrays as needed
# =============================================================================

# Parameter arrays
PRIMARY_SIZES=(2048)
LONG_TAIL_RATIOS=(0.05)
LOAD_THRESHOLDS=(16 32 64)
MAX_SECONDARY_CONCURRENT=(64 128 256)

# Hardware settings (adjust for your setup)
NUM_GPUS=8
TP_SIZE=1
NUM_WORKERS=8

# Model settings
MODEL_PATH="/opt/tiger/verl_srt/Qwen3-8B"

# Multi-round settings
NUM_ROUNDS=3
WARMUP_ROUNDS=1

# Prompt length settings
MAX_PROMPT_LENGTH=512

# Output directory
if [[ -n "$RESUME_DIR" ]]; then
    # Validate directory exists
    if [[ ! -d "$RESUME_DIR" ]]; then
        echo "ERROR: Resume directory does not exist: $RESUME_DIR"
        exit 1
    fi
    # Convert to absolute path
    OUTPUT_DIR="$(cd "$RESUME_DIR" && pwd)"
else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="results/benchmark_sweep_${TIMESTAMP}"
fi

# Benchmark script location
BENCHMARK_SCRIPT="tests/workers/rollout/rollout_vllm/benchmark_agentloop_runahead.py"

# =============================================================================
# Helper Functions
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_benchmark() {
    local mode=$1
    local primary_size=$2
    local long_tail_ratio=$3
    local load_threshold=$4
    local max_secondary=$5

    # Build output filename
    local output_file
    if [[ "$mode" == "baseline" ]]; then
        output_file="${OUTPUT_DIR}/baseline_ps${primary_size}_ltr${long_tail_ratio}.json"
    else
        output_file="${OUTPUT_DIR}/runahead_ps${primary_size}_ltr${long_tail_ratio}_lt${load_threshold}_msc${max_secondary}.json"
    fi

    # Check for resume
    if [[ "$RESUME" == "true" && -f "$output_file" ]]; then
        log "SKIP: $output_file already exists"
        return 0
    fi

    # Build command
    local cmd="python $BENCHMARK_SCRIPT"
    cmd+=" --mode $mode"
    cmd+=" --model-path $MODEL_PATH"
    cmd+=" --num-gpus $NUM_GPUS"
    cmd+=" --tp-size $TP_SIZE"
    cmd+=" --num-workers $NUM_WORKERS"
    cmd+=" --primary-size $primary_size"
    cmd+=" --long-tail-ratio $long_tail_ratio"
    cmd+=" --num-rounds $NUM_ROUNDS"
    cmd+=" --warmup-rounds $WARMUP_ROUNDS"
    cmd+=" --max-prompt-length $MAX_PROMPT_LENGTH"
    cmd+=" --output-file $output_file"

    if [[ "$mode" == "runahead" ]]; then
        cmd+=" --load-threshold $load_threshold"
        cmd+=" --max-secondary-concurrent $max_secondary"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY-RUN: $cmd"
    else
        log "RUN: $cmd"
        mkdir -p "$OUTPUT_DIR"
        # Run with unbuffered Python output
        python -u ${cmd#python } 2>&1 | tee -a "${OUTPUT_DIR}/sweep.log"
        local exit_code=${PIPESTATUS[0]}
        if [[ $exit_code -eq 0 ]]; then
            log "DONE: Saved to $output_file"
        else
            log "ERROR: Command failed with exit code $exit_code"
        fi
    fi
    # Always return 0 so that failures don't stop the sweep
    return 0
}

# =============================================================================
# Main Execution
# =============================================================================

log "=========================================="
log "Benchmark Parameter Sweep"
log "=========================================="
log "Output directory: $OUTPUT_DIR"
log "Dry run: $DRY_RUN"
log "Resume: $RESUME"
log ""
log "Parameter space:"
log "  Primary sizes: ${PRIMARY_SIZES[*]}"
log "  Long-tail ratios: ${LONG_TAIL_RATIOS[*]}"
log "  Load thresholds: ${LOAD_THRESHOLDS[*]}"
log "  Max secondary concurrent: ${MAX_SECONDARY_CONCURRENT[*]}"
log "  Max prompt length: $MAX_PROMPT_LENGTH"
log ""

# Count configurations
baseline_count=0
runahead_count=0

if [[ "$RUNAHEAD_ONLY" != "true" ]]; then
    baseline_count=$((${#PRIMARY_SIZES[@]} * ${#LONG_TAIL_RATIOS[@]}))
fi
if [[ "$BASELINE_ONLY" != "true" ]]; then
    runahead_count=$((${#PRIMARY_SIZES[@]} * ${#LONG_TAIL_RATIOS[@]} * ${#LOAD_THRESHOLDS[@]} * ${#MAX_SECONDARY_CONCURRENT[@]}))
fi
total_count=$((baseline_count + runahead_count))

log "Total configurations: $total_count"
log "  Baseline: $baseline_count"
log "  Runahead: $runahead_count"
log "=========================================="
log ""

# Counter
current=0

# Run baseline configurations
if [[ "$RUNAHEAD_ONLY" != "true" ]]; then
    log "--- BASELINE CONFIGURATIONS ---"
    for primary_size in "${PRIMARY_SIZES[@]}"; do
        for long_tail_ratio in "${LONG_TAIL_RATIOS[@]}"; do
            ((current++))
            log "[$current/$total_count] baseline ps=$primary_size ltr=$long_tail_ratio"
            run_benchmark "baseline" "$primary_size" "$long_tail_ratio" "" "" || true
            echo ""
        done
    done
fi

# Run runahead configurations
if [[ "$BASELINE_ONLY" != "true" ]]; then
    log "--- RUNAHEAD CONFIGURATIONS ---"
    for primary_size in "${PRIMARY_SIZES[@]}"; do
        for long_tail_ratio in "${LONG_TAIL_RATIOS[@]}"; do
            for load_threshold in "${LOAD_THRESHOLDS[@]}"; do
                for max_secondary in "${MAX_SECONDARY_CONCURRENT[@]}"; do
                    ((current++))
                    log "[$current/$total_count] runahead ps=$primary_size ltr=$long_tail_ratio lt=$load_threshold msc=$max_secondary"
                    run_benchmark "runahead" "$primary_size" "$long_tail_ratio" "$load_threshold" "$max_secondary" || true
                    echo ""
                done
            done
        done
    done
fi

log "=========================================="
log "Sweep complete!"
log "Results saved to: $OUTPUT_DIR"
log "=========================================="

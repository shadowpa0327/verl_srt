#!/bin/bash
#
# 1-Click Reproduction Script for SRT Speculation Analysis
#
# This script reproduces the complete SRT speculation analysis including:
# - Running simulation sweeps across training ticks (3 modes)
# - Generating all 8 analysis figures
# - Creating a summary report
#
# Usage:
#   ./reproduce.sh                           # Use defaults
#   ./reproduce.sh /path/to/data             # Custom data directory
#   ./reproduce.sh /path/to/data ./output    # Custom data and output directories
#
# Default data directory:
#   /home/ubuntu/verl_srt/rollout_datas_0119/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead
#
# Output structure:
#   <output_dir>/
#   ├── sweep_summary.json       # Aggregated metrics by mode
#   ├── per_request_data.csv     # Per-request detailed data
#   ├── figures/                 # All 8 visualization figures
#   │   ├── three_mode_comparison.png
#   │   ├── three_mode_bars.png
#   │   ├── speedup_decomposition.png
#   │   ├── draft_contribution_over_ticks.png
#   │   ├── hit_vs_acceptance_tradeoff.png
#   │   ├── metrics_by_length.png
#   │   ├── long_seq_heatmap.png
#   │   └── online_update_insight.png
#   └── ANALYSIS_REPORT.md       # Summary report
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/home/ubuntu/verl_srt"
DEFAULT_DATA_DIR="${PROJECT_ROOT}/rollout_datas_0119/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead"
DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/draft_contribution_sweep"

# Parse arguments
DATA_DIR="${1:-$DEFAULT_DATA_DIR}"
OUTPUT_DIR="${2:-$DEFAULT_OUTPUT_DIR}"

# Analysis parameters (matching original reproduce_analysis.sh)
MODEL_PATH="Qwen/Qwen2.5-7B"
TICK_START=1
TICK_END=46
TICK_STEP=10
MIN_TOKEN_PROB=0.3
MIN_RESPONSE_LEN=4000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}==============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}==============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    print_error "Virtual environment not found at ${PROJECT_ROOT}/.venv"
    exit 1
fi

print_header "SRT Speculation Analysis - 1-Click Reproduction"

echo ""
echo "Configuration:"
echo "  Data directory:     $DATA_DIR"
echo "  Output directory:   $OUTPUT_DIR"
echo "  Model:              $MODEL_PATH"
echo "  Tick range:         $TICK_START to $TICK_END (step=$TICK_STEP)"
echo "  Min token prob:     $MIN_TOKEN_PROB"
echo "  Min response len:   $MIN_RESPONSE_LEN (for analysis focus)"

# Verify data directory exists
if [ ! -d "$DATA_DIR" ]; then
    print_error "Data directory not found: $DATA_DIR"
    exit 1
fi

# Show data info
print_header "Step 0: Verifying Data Directory"
python -m recipe.srt.scripts.rollout_analysis.srt_analyze info "$DATA_DIR" --detailed

# Run full analysis
print_header "Step 1: Running Full Analysis Pipeline"
echo ""
echo "This will:"
echo "  1. Run simulation sweep across ticks (3 modes: prefill_only, online_only, prefill_plus_online)"
echo "  2. Generate all 8 analysis figures"
echo "  3. Create summary report"
echo ""

python -m recipe.srt.scripts.rollout_analysis.srt_analyze full \
    "$DATA_DIR" \
    -o "$OUTPUT_DIR" \
    --model "$MODEL_PATH" \
    --tick-start "$TICK_START" \
    --tick-end "$TICK_END" \
    --tick-step "$TICK_STEP" \
    --min-token-prob "$MIN_TOKEN_PROB" \
    --min-response-len "$MIN_RESPONSE_LEN"

# Print summary statistics
print_header "Summary Statistics (Long Sequences >= ${MIN_RESPONSE_LEN} tokens)"

python3 << EOF
import pandas as pd
import json

output_dir = "$OUTPUT_DIR"
min_len = $MIN_RESPONSE_LEN

# Load data
df = pd.read_csv(f"{output_dir}/per_request_data.csv")
df_4k = df[df['response_len'] >= min_len]

print(f"Total samples: {len(df)}")
print(f"Long sequences (>= {min_len}): {len(df_4k)}")
print()

# Mode statistics
mode_labels = {
    'prefill_only': 'Prefill Only',
    'online_only': 'Online Only',
    'prefill_plus_online': 'Prefill + Online'
}

print("Mode Comparison:")
print("-" * 70)
print(f"{'Mode':<20} {'Speedup':>10} {'Hit Rate':>10} {'Accept':>10} {'Draft %':>10}")
print("-" * 70)

for mode, label in mode_labels.items():
    subset = df_4k[df_4k['mode'] == mode]
    if len(subset) > 0:
        tps = subset['tokens_per_step'].mean()
        hr = subset['hit_rate'].mean()
        ar = subset['acceptance_rate'].mean()
        dc = subset['draft_contribution'].mean()
        print(f"{label:<20} {tps:>9.2f}x {hr:>9.1%} {ar:>9.1%} {dc:>9.1%}")

print("-" * 70)
EOF

# List generated files
print_header "Generated Files"

echo ""
echo "Data files:"
if [ -f "${OUTPUT_DIR}/sweep_summary.json" ]; then
    print_success "sweep_summary.json"
fi
if [ -f "${OUTPUT_DIR}/per_request_data.csv" ]; then
    print_success "per_request_data.csv ($(wc -l < "${OUTPUT_DIR}/per_request_data.csv") rows)"
fi

echo ""
echo "Figures:"
FIGURES_DIR="${OUTPUT_DIR}/figures"
if [ -d "$FIGURES_DIR" ]; then
    for fig in three_mode_comparison three_mode_bars speedup_decomposition \
               draft_contribution_over_ticks hit_vs_acceptance_tradeoff \
               metrics_by_length long_seq_heatmap online_update_insight; do
        if [ -f "${FIGURES_DIR}/${fig}.png" ]; then
            print_success "${fig}.png"
        else
            print_warning "${fig}.png (not generated)"
        fi
    done
fi

echo ""
echo "Report:"
if [ -f "${OUTPUT_DIR}/ANALYSIS_REPORT.md" ]; then
    print_success "ANALYSIS_REPORT.md"
fi

print_header "Analysis Complete!"

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Quick commands:"
echo "  # View figures"
echo "  ls ${FIGURES_DIR}/*.png"
echo ""
echo "  # Read report"
echo "  cat ${OUTPUT_DIR}/ANALYSIS_REPORT.md"
echo ""
echo "  # Load data in Python"
echo "  import pandas as pd"
echo "  df = pd.read_csv('${OUTPUT_DIR}/per_request_data.csv')"
echo ""

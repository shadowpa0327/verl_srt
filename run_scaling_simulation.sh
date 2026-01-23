#!/bin/bash
# Repro script for scaling simulation
# Usage: ./run_scaling_simulation.sh <step> [max_samples]
#
# This script measures whether expanded_rollouts from step N can accelerate
# the actual rollout generation at step N.

STEP=$1
MAX_SAMPLES=${2:-100}

# Absolute path to the repository root
REPO_ROOT="/home/ubuntu/verl_srt"
cd $REPO_ROOT

if [ -z "$STEP" ]; then
    echo "Usage: $0 <step> [max_samples]"
    echo "Example: $0 100 100"
    echo ""
    echo "Available Steps (both expanded_rollouts and rollout_datas):"
    # Find steps that exist in both directories
    for step_dir in expanded_rollouts/step_*; do
        step=$(basename $step_dir | sed 's/step_//')
        if [ -f "rollout_datas/DAPO-Qwen3-8B-SRT-Runahead/rollout/${step}.jsonl" ]; then
            echo -n "$step "
        fi
    done
    echo -e "\n"
    exit 1
fi

MODEL_PATH="${REPO_ROOT}/Qwen3-8B-Base"
SCALING_DATA="expanded_rollouts/step_${STEP}/scaling_rollouts.jsonl"
SIM_DATA="rollout_datas/DAPO-Qwen3-8B-SRT-Runahead/rollout/${STEP}.jsonl"
OUTPUT_DIR="scaling_simulation_results/step_${STEP}"

# Verify both files exist
if [ ! -f "$SCALING_DATA" ]; then
    echo "ERROR: Scaling data not found: $SCALING_DATA"
    exit 1
fi

if [ ! -f "$SIM_DATA" ]; then
    echo "ERROR: Simulation data not found: $SIM_DATA"
    exit 1
fi

mkdir -p $OUTPUT_DIR

echo "--------------------------------------------------------"
echo "Running Scaling Simulation for Step ${STEP}"
echo "Scaling Data (tree filling): $SCALING_DATA"
echo "Simulation Data (replay):    $SIM_DATA"
echo "Max Samples: $MAX_SAMPLES"
echo "--------------------------------------------------------"

PYTHONPATH=recipe/srt ${REPO_ROOT}/.venv/bin/python recipe/srt/scaling_replay_simulator.py \
    --model_path $MODEL_PATH \
    --scaling_data $SCALING_DATA \
    --simulation_data $SIM_DATA \
    --k_values 1,2,4,8,16,32 \
    --max_samples $MAX_SAMPLES \
    --output "${OUTPUT_DIR}/results.json" \
    --verbose

echo "--------------------------------------------------------"
echo "Simulation Complete. Results saved to ${OUTPUT_DIR}/results.json"
echo "--------------------------------------------------------"

#!/bin/bash
# Sweep scaling simulation with ONLINE UPDATES enabled
# Usage: ./run_scaling_sweep_online.sh [max_samples]

MAX_SAMPLES=${1:-100}

REPO_ROOT="/home/ubuntu/verl_srt"
cd $REPO_ROOT

OUTPUT_BASE="scaling_simulation_results_online"
mkdir -p $OUTPUT_BASE

# Find all steps with matching data
STEPS=()
for step_dir in expanded_rollouts/step_*; do
    step=$(basename $step_dir | sed 's/step_//')
    if [ -f "rollout_datas/DAPO-Qwen3-8B-SRT-Runahead/rollout/${step}.jsonl" ]; then
        STEPS+=($step)
    fi
done

# Sort steps numerically
IFS=$'\n' SORTED_STEPS=($(sort -n <<<"${STEPS[*]}")); unset IFS

echo "========================================================"
echo "Scaling Simulation Sweep (WITH ONLINE UPDATES)"
echo "========================================================"
echo "Steps to process: ${SORTED_STEPS[*]}"
echo "Max samples per step: $MAX_SAMPLES"
echo "Output directory: $OUTPUT_BASE"
echo "========================================================"

MODEL_PATH="${REPO_ROOT}/Qwen3-8B-Base"

for i in "${!SORTED_STEPS[@]}"; do
    step=${SORTED_STEPS[$i]}
    echo ""
    echo "[$(($i + 1))/${#SORTED_STEPS[@]}] Processing step $step (online updates)..."
    
    SCALING_DATA="expanded_rollouts/step_${step}/scaling_rollouts.jsonl"
    SIM_DATA="rollout_datas/DAPO-Qwen3-8B-SRT-Runahead/rollout/${step}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/step_${step}"
    mkdir -p $OUTPUT_DIR

    PYTHONPATH=recipe/srt ${REPO_ROOT}/.venv/bin/python recipe/srt/scaling_replay_simulator.py \
        --model_path $MODEL_PATH \
        --scaling_data $SCALING_DATA \
        --simulation_data $SIM_DATA \
        --k_values 1,2,4,8,16,32 \
        --max_samples $MAX_SAMPLES \
        --online_update \
        --output "${OUTPUT_DIR}/results.json" \
        --verbose
done

echo ""
echo "========================================================"
echo "Generating consolidated report..."
echo "========================================================"

# Generate report
PYTHONPATH=recipe/srt ${REPO_ROOT}/.venv/bin/python - <<'PYTHON_SCRIPT'
import json
from pathlib import Path

output_base = "scaling_simulation_results_online"
report = {"config": {"k_values": [1, 2, 4, 8, 16, 32], "online_update": True}, "steps": {}, "summary_by_k": {}}

step_dirs = sorted([d for d in Path(output_base).iterdir() if d.is_dir() and d.name.startswith("step_")],
                   key=lambda x: int(x.name.split("_")[1]))

for step_dir in step_dirs:
    step = int(step_dir.name.split("_")[1])
    results_file = step_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        if "scaling_results" in data:
            report["steps"][step] = {"results_by_k": {}}
            for result in data["scaling_results"]:
                k = result["k"]
                report["steps"][step]["results_by_k"][k] = {
                    "acceptance_rate": result.get("mean_acceptance_rate", 0),
                    "hit_rate": result.get("mean_hit_rate", 0),
                    "tokens_per_step": result.get("mean_tokens_per_step", 0),
                    "draft_contribution": result.get("mean_draft_contribution", 0),
                }

k_values = [1, 2, 4, 8, 16, 32]
for k in k_values:
    vals = {"acc": [], "hit": [], "toks": [], "draft": []}
    for step, step_data in report["steps"].items():
        if k in step_data.get("results_by_k", {}):
            r = step_data["results_by_k"][k]
            vals["acc"].append(r["acceptance_rate"])
            vals["hit"].append(r["hit_rate"])
            vals["toks"].append(r["tokens_per_step"])
            vals["draft"].append(r["draft_contribution"])
    if vals["acc"]:
        report["summary_by_k"][k] = {
            "mean_acceptance_rate": sum(vals["acc"]) / len(vals["acc"]),
            "mean_hit_rate": sum(vals["hit"]) / len(vals["hit"]),
            "mean_tokens_per_step": sum(vals["toks"]) / len(vals["toks"]),
            "mean_draft_contribution": sum(vals["draft"]) / len(vals["draft"]),
        }

with open(f"{output_base}/sweep_report.json", "w") as f:
    json.dump(report, f, indent=2)

print("\n## Average Metrics by K (WITH ONLINE UPDATES)")
print("-" * 70)
print(f"{'K':>4} | {'Accept Rate':>12} | {'Hit Rate':>10} | {'Tok/Step':>10} | {'Draft Contrib':>13}")
print("-" * 70)
for k in k_values:
    if k in report["summary_by_k"]:
        s = report["summary_by_k"][k]
        print(f"{k:>4} | {s['mean_acceptance_rate']:>11.1%} | {s['mean_hit_rate']:>9.1%} | {s['mean_tokens_per_step']:>10.2f} | {s['mean_draft_contribution']:>12.1%}")

print(f"\nReport saved to: {output_base}/sweep_report.json")
PYTHON_SCRIPT

echo ""
echo "Done! Results in ${OUTPUT_BASE}/"

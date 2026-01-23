#!/bin/bash
# Sweep scaling simulation over all available steps
# Usage: ./run_scaling_sweep.sh [max_samples]
#
# This script runs scaling simulation for all steps that have matching data
# in both expanded_rollouts/ and rollout_datas/, then generates a summary report.

MAX_SAMPLES=${1:-100}

REPO_ROOT="/home/ubuntu/verl_srt"
cd $REPO_ROOT

OUTPUT_BASE="scaling_simulation_results"
REPORT_FILE="${OUTPUT_BASE}/sweep_report.json"

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
echo "Scaling Simulation Sweep"
echo "========================================================"
echo "Steps to process: ${SORTED_STEPS[*]}"
echo "Max samples per step: $MAX_SAMPLES"
echo "Output directory: $OUTPUT_BASE"
echo "========================================================"
echo ""

# Track results
SUCCESSFUL_STEPS=()
FAILED_STEPS=()

# Run simulation for each step
for i in "${!SORTED_STEPS[@]}"; do
    step=${SORTED_STEPS[$i]}
    echo ""
    echo "[$(($i + 1))/${#SORTED_STEPS[@]}] Processing step $step..."
    echo "--------------------------------------------------------"

    # Run the simulation
    ./run_scaling_simulation.sh $step $MAX_SAMPLES

    if [ $? -eq 0 ] && [ -f "${OUTPUT_BASE}/step_${step}/results.json" ]; then
        SUCCESSFUL_STEPS+=($step)
        echo "✓ Step $step completed successfully"
    else
        FAILED_STEPS+=($step)
        echo "✗ Step $step failed"
    fi
done

echo ""
echo "========================================================"
echo "Sweep Complete!"
echo "========================================================"
echo "Successful: ${#SUCCESSFUL_STEPS[@]} steps (${SUCCESSFUL_STEPS[*]})"
echo "Failed: ${#FAILED_STEPS[@]} steps (${FAILED_STEPS[*]})"
echo ""

# Generate consolidated report
echo "Generating consolidated report..."

PYTHONPATH=recipe/srt ${REPO_ROOT}/.venv/bin/python - <<'PYTHON_SCRIPT'
import json
import os
from pathlib import Path

output_base = "scaling_simulation_results"
report = {
    "config": {
        "k_values": [1, 2, 4, 8, 16, 32],
    },
    "steps": {},
    "summary_by_k": {},
}

# Collect results from all steps
step_dirs = sorted([d for d in Path(output_base).iterdir() if d.is_dir() and d.name.startswith("step_")],
                   key=lambda x: int(x.name.split("_")[1]))

for step_dir in step_dirs:
    step = int(step_dir.name.split("_")[1])
    results_file = step_dir / "results.json"

    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)

        # Extract summary for this step
        if "scaling_results" in data:
            report["steps"][step] = {
                "num_prompts": data.get("data_stats", {}).get("scaling_unique_prompts", 0),
                "num_simulation_samples": data.get("data_stats", {}).get("simulation_total_samples", 0),
                "results_by_k": {}
            }
            for result in data["scaling_results"]:
                k = result["k"]
                report["steps"][step]["results_by_k"][k] = {
                    "acceptance_rate": result.get("mean_acceptance_rate", 0),
                    "hit_rate": result.get("mean_hit_rate", 0),
                    "tokens_per_step": result.get("mean_tokens_per_step", 0),
                    "draft_contribution": result.get("mean_draft_contribution", 0),
                }

# Build summary by k (average across all steps)
k_values = [1, 2, 4, 8, 16, 32]
for k in k_values:
    acceptance_rates = []
    hit_rates = []
    tokens_per_step = []
    draft_contributions = []

    for step, step_data in report["steps"].items():
        if k in step_data.get("results_by_k", {}):
            result = step_data["results_by_k"][k]
            acceptance_rates.append(result["acceptance_rate"])
            hit_rates.append(result["hit_rate"])
            tokens_per_step.append(result["tokens_per_step"])
            draft_contributions.append(result["draft_contribution"])

    if acceptance_rates:
        report["summary_by_k"][k] = {
            "mean_acceptance_rate": sum(acceptance_rates) / len(acceptance_rates),
            "mean_hit_rate": sum(hit_rates) / len(hit_rates),
            "mean_tokens_per_step": sum(tokens_per_step) / len(tokens_per_step),
            "mean_draft_contribution": sum(draft_contributions) / len(draft_contributions),
            "num_steps": len(acceptance_rates),
        }

# Save report
report_file = os.path.join(output_base, "sweep_report.json")
with open(report_file, "w") as f:
    json.dump(report, f, indent=2)

# Print summary tables
print("\n" + "=" * 70)
print("SUMMARY REPORT")
print("=" * 70)

# Table 1: Summary by K (averaged across all steps)
print("\n## Average Metrics by K (across all steps)")
print("-" * 70)
print(f"{'K':>4} | {'Accept Rate':>12} | {'Hit Rate':>10} | {'Tok/Step':>10} | {'Draft Contrib':>13}")
print("-" * 70)
for k in k_values:
    if k in report["summary_by_k"]:
        s = report["summary_by_k"][k]
        print(f"{k:>4} | {s['mean_acceptance_rate']:>11.1%} | {s['mean_hit_rate']:>9.1%} | {s['mean_tokens_per_step']:>10.2f} | {s['mean_draft_contribution']:>12.1%}")

# Table 2: Acceptance rate by step and K
print("\n## Acceptance Rate by Step and K")
print("-" * 70)
header = f"{'Step':>6} |" + "".join([f" k={k:>2} |" for k in k_values])
print(header)
print("-" * 70)
for step in sorted(report["steps"].keys()):
    row = f"{step:>6} |"
    for k in k_values:
        if k in report["steps"][step].get("results_by_k", {}):
            val = report["steps"][step]["results_by_k"][k]["acceptance_rate"]
            row += f" {val:>4.1%} |"
        else:
            row += "   N/A |"
    print(row)

# Table 3: Tokens per step by step and K
print("\n## Tokens per Step by Step and K")
print("-" * 70)
header = f"{'Step':>6} |" + "".join([f" k={k:>2} |" for k in k_values])
print(header)
print("-" * 70)
for step in sorted(report["steps"].keys()):
    row = f"{step:>6} |"
    for k in k_values:
        if k in report["steps"][step].get("results_by_k", {}):
            val = report["steps"][step]["results_by_k"][k]["tokens_per_step"]
            row += f" {val:>4.2f} |"
        else:
            row += "   N/A |"
    print(row)

print("\n" + "=" * 70)
print(f"Full report saved to: {report_file}")
print("=" * 70)
PYTHON_SCRIPT

echo ""
echo "Done! Results in ${OUTPUT_BASE}/"

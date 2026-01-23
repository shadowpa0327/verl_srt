#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Scaling Experiment Sweep Script.

This script sweeps over multiple training steps to collect scaling_rollouts.jsonl
for suffix tree scaling analysis. It automatically:
1. Finds matching checkpoint and rollout data for each step
2. Runs the scaling experiment
3. Deletes the merged model after each step to avoid disk exhaustion
4. Collects all scaling_rollouts.jsonl into a common output directory

Usage:
    python recipe/srt/scripts/run_scaling_sweep.py \
        --steps 10,20,50,100 \
        --output-dir /path/to/scaling_sweep_results

    # Or use step ranges
    python recipe/srt/scripts/run_scaling_sweep.py \
        --step-range 10:110:10 \
        --output-dir /path/to/scaling_sweep_results
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Default paths
DEFAULT_CKPT_BASE = "/home/ubuntu/verl_srt/ckpts/DAPO/DAPO-Qwen3-8B-SRT-Runahead"
DEFAULT_ROLLOUT_BASE = "/home/ubuntu/verl_srt/rollout_datas/DAPO-Qwen3-8B-SRT-Runahead/rollout"


def find_checkpoint_path(step: int, ckpt_base: str) -> Optional[str]:
    """Find checkpoint path for a given step."""
    ckpt_path = os.path.join(ckpt_base, f"global_step_{step}", "actor")
    if os.path.exists(ckpt_path):
        return ckpt_path
    return None


def find_rollout_path(step: int, rollout_base: str) -> Optional[str]:
    """Find rollout data path for a given step."""
    rollout_path = os.path.join(rollout_base, f"{step}.jsonl")
    if os.path.exists(rollout_path):
        return rollout_path
    return None


def parse_steps(steps_str: str) -> List[int]:
    """Parse comma-separated steps."""
    return [int(s.strip()) for s in steps_str.split(",") if s.strip()]


def parse_step_range(range_str: str) -> List[int]:
    """Parse step range in format start:end:step."""
    parts = range_str.split(":")
    if len(parts) == 2:
        start, end = int(parts[0]), int(parts[1])
        step = 1
    elif len(parts) == 3:
        start, end, step = int(parts[0]), int(parts[1]), int(parts[2])
    else:
        raise ValueError(f"Invalid step range format: {range_str}. Use start:end or start:end:step")
    return list(range(start, end + 1, step))


def get_disk_usage_gb(path: str) -> float:
    """Get disk usage of a path in GB."""
    if not os.path.exists(path):
        return 0.0
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.exists(fp):
                total += os.path.getsize(fp)
    return total / (1024 ** 3)


def run_scaling_experiment(
    step: int,
    ckpt_path: str,
    rollout_path: str,
    output_dir: str,
    k_rollouts: int = 32,
    top_percentile: float = 0.3,
    max_prompts: int = 16,
    temperature: float = 1.0,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    delete_model: bool = True,
    verbose: bool = True,
) -> Tuple[bool, Optional[str]]:
    """
    Run scaling experiment for a single step.

    Returns:
        Tuple of (success, scaling_rollouts_path)
    """
    step_output_dir = os.path.join(output_dir, f"step_{step}")
    os.makedirs(step_output_dir, exist_ok=True)

    merged_model_dir = os.path.join(step_output_dir, "merged_model")
    scaling_rollouts_path = os.path.join(step_output_dir, "scaling_rollouts.jsonl")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Step {step}")
        print(f"{'='*60}")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Rollout data: {rollout_path}")
        print(f"  Output: {step_output_dir}")

    # Build command
    cmd = [
        sys.executable,
        "-m", "recipe.srt.scripts.suffix_tree_scaling_experiment",
        "--checkpoint-path", ckpt_path,
        "--rollout-data", rollout_path,
        "--output-dir", step_output_dir,
        "--k-rollouts", str(k_rollouts),
        "--top-percentile", str(top_percentile),
        "--max-prompts", str(max_prompts),
        "--temperature", str(temperature),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]

    if verbose:
        print(f"  Running: {' '.join(cmd[:6])} ...")

    try:
        # Run the experiment
        result = subprocess.run(
            cmd,
            cwd="/home/ubuntu/verl_srt",
            capture_output=not verbose,
            text=True,
        )

        if result.returncode != 0:
            print(f"  ERROR: Experiment failed with code {result.returncode}")
            if result.stderr:
                print(f"  stderr: {result.stderr[:500]}")
            return False, None

        # Verify output exists
        if not os.path.exists(scaling_rollouts_path):
            print(f"  ERROR: scaling_rollouts.jsonl not created")
            return False, None

        # Count samples
        with open(scaling_rollouts_path) as f:
            sample_count = sum(1 for _ in f)
        print(f"  SUCCESS: Generated {sample_count} samples")

    except Exception as e:
        print(f"  ERROR: {e}")
        return False, None

    finally:
        # Clean up merged model to save disk space
        if delete_model and os.path.exists(merged_model_dir):
            model_size = get_disk_usage_gb(merged_model_dir)
            if verbose:
                print(f"  Deleting merged model ({model_size:.2f} GB)...")
            shutil.rmtree(merged_model_dir, ignore_errors=True)

            # Force garbage collection
            gc.collect()

    return True, scaling_rollouts_path


def main():
    parser = argparse.ArgumentParser(
        description="Sweep scaling experiments over multiple training steps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Step selection (mutually exclusive)
    step_group = parser.add_mutually_exclusive_group(required=True)
    step_group.add_argument(
        "--steps",
        type=str,
        help="Comma-separated list of steps (e.g., 10,20,50,100)"
    )
    step_group.add_argument(
        "--step-range",
        type=str,
        help="Step range in format start:end or start:end:step (e.g., 10:100:10)"
    )

    # Paths
    parser.add_argument(
        "--ckpt-base",
        type=str,
        default=DEFAULT_CKPT_BASE,
        help="Base directory for checkpoints"
    )
    parser.add_argument(
        "--rollout-base",
        type=str,
        default=DEFAULT_ROLLOUT_BASE,
        help="Base directory for rollout data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for sweep results"
    )

    # Experiment parameters
    parser.add_argument(
        "--k-rollouts",
        type=int,
        default=32,
        help="Number of samples per prompt"
    )
    parser.add_argument(
        "--top-percentile",
        type=float,
        default=0.3,
        help="Top percentile of prompts by length"
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=16,
        help="Maximum prompts to select"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )

    # Hardware
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization"
    )

    # Options
    parser.add_argument(
        "--keep-models",
        action="store_true",
        help="Keep merged models (warning: uses lots of disk space)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip steps that already have scaling_rollouts.jsonl"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without running"
    )

    args = parser.parse_args()

    # Parse steps
    if args.steps:
        steps = parse_steps(args.steps)
    else:
        steps = parse_step_range(args.step_range)

    print(f"Scaling Experiment Sweep")
    print(f"========================")
    print(f"Steps to process: {steps}")
    print(f"Checkpoint base: {args.ckpt_base}")
    print(f"Rollout base: {args.rollout_base}")
    print(f"Output directory: {args.output_dir}")
    print(f"K rollouts: {args.k_rollouts}")
    print(f"Top percentile: {args.top_percentile}")
    print(f"Max prompts: {args.max_prompts}")
    print(f"Keep models: {args.keep_models}")

    # Validate steps have both checkpoint and rollout data
    valid_steps = []
    missing_steps = []

    for step in steps:
        ckpt_path = find_checkpoint_path(step, args.ckpt_base)
        rollout_path = find_rollout_path(step, args.rollout_base)

        if ckpt_path and rollout_path:
            valid_steps.append((step, ckpt_path, rollout_path))
        else:
            missing = []
            if not ckpt_path:
                missing.append("checkpoint")
            if not rollout_path:
                missing.append("rollout")
            missing_steps.append((step, missing))

    print(f"\nValid steps: {len(valid_steps)}/{len(steps)}")
    if missing_steps:
        print(f"Missing data for steps: {[(s, m) for s, m in missing_steps[:5]]}")
        if len(missing_steps) > 5:
            print(f"  ... and {len(missing_steps) - 5} more")

    if not valid_steps:
        print("ERROR: No valid steps to process!")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save sweep config
    config = {
        "steps": [s for s, _, _ in valid_steps],
        "ckpt_base": args.ckpt_base,
        "rollout_base": args.rollout_base,
        "k_rollouts": args.k_rollouts,
        "top_percentile": args.top_percentile,
        "max_prompts": args.max_prompts,
        "temperature": args.temperature,
        "started_at": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "sweep_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if args.dry_run:
        print("\n[DRY RUN] Would process the following steps:")
        for step, ckpt_path, rollout_path in valid_steps:
            print(f"  Step {step}:")
            print(f"    Checkpoint: {ckpt_path}")
            print(f"    Rollout: {rollout_path}")
        sys.exit(0)

    # Run experiments
    results = []
    for i, (step, ckpt_path, rollout_path) in enumerate(valid_steps):
        print(f"\n[{i+1}/{len(valid_steps)}] Processing step {step}...")

        # Check if already done
        step_output_dir = os.path.join(args.output_dir, f"step_{step}")
        scaling_rollouts_path = os.path.join(step_output_dir, "scaling_rollouts.jsonl")

        if args.skip_existing and os.path.exists(scaling_rollouts_path):
            print(f"  SKIP: scaling_rollouts.jsonl already exists")
            results.append({"step": step, "status": "skipped", "path": scaling_rollouts_path})
            continue

        success, output_path = run_scaling_experiment(
            step=step,
            ckpt_path=ckpt_path,
            rollout_path=rollout_path,
            output_dir=args.output_dir,
            k_rollouts=args.k_rollouts,
            top_percentile=args.top_percentile,
            max_prompts=args.max_prompts,
            temperature=args.temperature,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            delete_model=not args.keep_models,
        )

        results.append({
            "step": step,
            "status": "success" if success else "failed",
            "path": output_path,
        })

    # Save results summary
    summary = {
        "completed_at": datetime.now().isoformat(),
        "results": results,
        "success_count": sum(1 for r in results if r["status"] == "success"),
        "failed_count": sum(1 for r in results if r["status"] == "failed"),
        "skipped_count": sum(1 for r in results if r["status"] == "skipped"),
    }
    with open(os.path.join(args.output_dir, "sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("Sweep Complete!")
    print(f"{'='*60}")
    print(f"  Success: {summary['success_count']}")
    print(f"  Failed: {summary['failed_count']}")
    print(f"  Skipped: {summary['skipped_count']}")
    print(f"\nOutput directory: {args.output_dir}")
    print(f"  - sweep_config.json: experiment configuration")
    print(f"  - sweep_summary.json: results summary")
    print(f"  - step_N/scaling_rollouts.jsonl: per-step results")

    # List successful outputs
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        print(f"\nSuccessful scaling_rollouts.jsonl files:")
        for r in successful[:10]:
            print(f"  - {r['path']}")
        if len(successful) > 10:
            print(f"  ... and {len(successful) - 10} more")


if __name__ == "__main__":
    main()

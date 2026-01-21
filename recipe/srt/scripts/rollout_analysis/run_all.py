#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Run all rollout analysis scripts.

This script runs:
1. Data organization by prompt (optional)
2. Output length analysis
3. Runahead prediction analysis

Usage:
    python -m recipe.srt.scripts.rollout_analysis.run_all \\
        --data_dir /path/to/rollout_data \\
        --output_dir /path/to/results \\
        --tokenizer Qwen/Qwen2.5-7B-Instruct

Example:
    python -m recipe.srt.scripts.rollout_analysis.run_all \\
        --data_dir rollout_datas_0119/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead \\
        --output_dir rollout_datas_0119/DAPO/DAPO-Qwen2.5-7b-MATH-SRT-Runahead/analysis_results
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from .organize_by_prompt import organize_by_step, organize_by_prompt, create_summary
from .analyze_lengths import (
    load_primary_data,
    load_secondary_data,
    analyze_primary_lengths,
    analyze_secondary_lengths,
)
from .analyze_runahead_prediction import (
    load_data_by_step,
    analyze_same_step_correlation,
    analyze_runahead_correlation,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run all rollout analysis scripts."
    )

    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to rollout data directory (containing rollout/ and secondary/ subdirs)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to output directory for analysis results"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Tokenizer model name or path"
    )
    parser.add_argument(
        "--organize", action="store_true",
        help="Also organize data by prompt (creates organized_by_prompt/ subdirs)"
    )
    parser.add_argument(
        "--min_samples", type=int, default=8,
        help="Minimum samples per prompt for length analysis"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print detailed progress"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ROLLOUT ANALYSIS SUITE")
    print("=" * 80)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print()

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # Step 1: Organize data (optional)
    if args.organize:
        print("\n" + "=" * 80)
        print("STEP 1: ORGANIZING DATA BY PROMPT")
        print("=" * 80)

        # Primary by prompt
        primary_org_dir = output_dir / "organized_primary_by_prompt"
        print(f"\nOrganizing primary data -> {primary_org_dir}")
        stats = organize_by_prompt(data_dir, primary_org_dir, "rollout", verbose=args.verbose)
        create_summary(primary_org_dir, "rollout", "by_prompt", stats)
        print(f"  Done: {stats['unique_prompts']} prompts, {stats['total_samples']} samples")

        # Secondary by prompt
        if (data_dir / "secondary").exists():
            secondary_org_dir = output_dir / "organized_secondary_by_prompt"
            print(f"\nOrganizing secondary data -> {secondary_org_dir}")
            stats = organize_by_prompt(data_dir, secondary_org_dir, "secondary", verbose=args.verbose)
            create_summary(secondary_org_dir, "secondary", "by_prompt", stats)
            print(f"  Done: {stats['unique_prompts']} prompts, {stats['total_samples']} samples")

    # Step 2: Length analysis
    print("\n" + "=" * 80)
    print("STEP 2: OUTPUT LENGTH ANALYSIS")
    print("=" * 80)

    print("\nLoading primary data...")
    primary_data = load_primary_data(data_dir, tokenizer, args.verbose)

    print("Analyzing primary lengths...")
    primary_analysis = analyze_primary_lengths(primary_data, args.min_samples)

    secondary_analysis = None
    if (data_dir / "secondary").exists():
        print("\nLoading secondary data...")
        secondary_data = load_secondary_data(data_dir, tokenizer, args.verbose)

        print("Analyzing secondary lengths...")
        secondary_analysis = analyze_secondary_lengths(secondary_data, args.min_samples)

    # Print length analysis results
    dist = primary_analysis['distribution']
    print(f"\nPrimary output token distribution ({dist['total_prompts']} prompts):")
    print(f"  10th percentile: {dist['p10']:,.0f} tokens")
    print(f"  Median:          {dist['median']:,.0f} tokens")
    print(f"  90th percentile: {dist['p90']:,.0f} tokens")

    print("\nAccuracy by output length:")
    for b in primary_analysis['accuracy_by_bucket']:
        print(f"  {b['bucket']:<12} {b['count']:>4} prompts, {b['accuracy']*100:>5.1f}% accuracy")

    # Save length analysis
    length_results = {
        'primary': {
            'distribution': primary_analysis['distribution'],
            'accuracy_by_bucket': primary_analysis['accuracy_by_bucket'],
            'accuracy_comparison': primary_analysis['accuracy_comparison'],
        }
    }
    if secondary_analysis:
        length_results['secondary'] = {
            'distribution': secondary_analysis['distribution'],
            'completion_by_bucket': secondary_analysis['completion_by_bucket'],
        }

    length_output = output_dir / "length_analysis.json"
    with open(length_output, 'w') as f:
        json.dump(length_results, f, indent=2)
    print(f"\nLength analysis saved to: {length_output}")

    # Step 3: Runahead prediction analysis
    if (data_dir / "secondary").exists():
        print("\n" + "=" * 80)
        print("STEP 3: RUNAHEAD PREDICTION ANALYSIS")
        print("=" * 80)

        print("\nLoading data by step...")
        primary_by_step, secondary_by_step = load_data_by_step(data_dir, tokenizer, args.verbose)

        print("Analyzing same-step correlation...")
        same_step = analyze_same_step_correlation(primary_by_step, secondary_by_step, min_samples=8)

        print("Analyzing runahead prediction (secondary[N] -> primary[N+1])...")
        runahead = analyze_runahead_correlation(primary_by_step, secondary_by_step, min_samples=4)

        # Print runahead results
        if 'error' not in runahead:
            print(f"\nRunahead prediction results:")
            print(f"  Pearson correlation:  {runahead['pearson_correlation']:.4f}")
            print(f"  Spearman correlation: {runahead['spearman_correlation']:.4f}")
            print(f"\n  Quartile accuracy:")
            for q, data in runahead['quartile_accuracy'].items():
                print(f"    {q}: {data['match_rate']*100:.1f}%")
            print(f"\n  Long output prediction (top 25%):")
            print(f"    Precision: {runahead['prediction_accuracy']['top25']['precision']*100:.1f}%")
            print(f"    Recall:    {runahead['prediction_accuracy']['top25']['recall']*100:.1f}%")

        # Save runahead analysis
        runahead_results = {
            'same_step': {k: v for k, v in same_step.items() if k != 'error'},
            'runahead': {k: v for k, v in runahead.items() if k not in ['error', 'step_correlations']},
            'step_correlations': runahead.get('step_correlations', []),
        }

        runahead_output = output_dir / "runahead_prediction.json"
        with open(runahead_output, 'w') as f:
            json.dump(runahead_results, f, indent=2)
        print(f"\nRunahead analysis saved to: {runahead_output}")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {output_dir / 'length_analysis.json'}")
    if (data_dir / "secondary").exists():
        print(f"  - {output_dir / 'runahead_prediction.json'}")
    if args.organize:
        print(f"  - {output_dir / 'organized_primary_by_prompt/'}")
        if (data_dir / "secondary").exists():
            print(f"  - {output_dir / 'organized_secondary_by_prompt/'}")


if __name__ == "__main__":
    main()

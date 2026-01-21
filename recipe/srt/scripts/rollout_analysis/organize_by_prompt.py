#!/usr/bin/env python3
# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""
Organize rollout data by prompts for each step.

This script reads dumped rollout data and reorganizes it by prompt,
creating a directory structure that groups all responses for each prompt.

Data Format:
- Rollout (primary): {input, output, gts, score, step, acc, pred}
- Secondary: {sample_id, status, tokens_generated, prompt_hash, step, prompt,
              prompt_length, response, response_length, stop_reason, is_partial}

Output Structure (by_step mode):
    output_dir/step_N/prompt_{hash}.jsonl

Output Structure (by_prompt mode):
    output_dir/prompt_{hash}/step_N.jsonl + metadata.json

Usage:
    python organize_by_prompt.py \\
        --data_dir /path/to/rollout_data \\
        --output_dir /path/to/organized \\
        --source rollout \\
        --mode by_step
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def compute_prompt_hash(prompt_text: str) -> str:
    """Compute SHA256 hash of prompt text (first 16 hex chars)."""
    return hashlib.sha256(prompt_text.encode('utf-8')).hexdigest()[:16]


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load a JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], filepath: Path):
    """Save data to a JSONL file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def get_prompt_key(item: Dict, source: str) -> Optional[Tuple[str, str]]:
    """Extract prompt text and compute hash key.

    Returns:
        Tuple of (prompt_text, prompt_hash_key), or None if prompt is missing.
    """
    if source == 'secondary':
        if 'prompt' not in item:
            return None
        prompt_text = item['prompt']
        if 'prompt_hash' in item:
            hash_key = f"{item['prompt_hash']:016x}"
        else:
            hash_key = compute_prompt_hash(prompt_text)
    else:  # rollout
        if 'input' not in item:
            return None
        prompt_text = item['input']
        hash_key = compute_prompt_hash(prompt_text)

    return prompt_text, hash_key


def organize_by_step(
    data_dir: Path,
    output_dir: Path,
    source: str,
    steps: Optional[List[int]] = None,
    verbose: bool = False
) -> Dict:
    """Organize data into step directories with per-prompt files.

    Output structure: output_dir/step_N/prompt_{hash}.jsonl
    """
    source_dir = data_dir / source
    stats = {
        'total_files': 0,
        'total_samples': 0,
        'unique_prompts': set(),
        'steps_processed': [],
        'skipped_samples': 0,
    }

    step_files = sorted(source_dir.glob("*.jsonl"), key=lambda x: int(x.stem))

    for step_file in step_files:
        step = int(step_file.stem)

        if steps and step not in steps:
            continue

        stats['total_files'] += 1
        stats['steps_processed'].append(step)

        if verbose:
            print(f"Processing {source}/step {step}...")

        prompt_groups: Dict[str, List[Dict]] = defaultdict(list)
        data = load_jsonl(step_file)

        skipped = 0
        for item in data:
            result = get_prompt_key(item, source)
            if result is None:
                skipped += 1
                continue
            prompt_text, hash_key = result

            if source == 'rollout' and 'prompt_text' not in item:
                item['prompt_text'] = prompt_text

            prompt_groups[hash_key].append(item)
            stats['unique_prompts'].add(hash_key)
            stats['total_samples'] += 1

        stats['skipped_samples'] += skipped

        step_dir = output_dir / f"step_{step}"
        for hash_key, items in prompt_groups.items():
            output_file = step_dir / f"prompt_{hash_key}.jsonl"
            save_jsonl(items, output_file)

        if verbose:
            skip_msg = f" (skipped {skipped})" if skipped > 0 else ""
            print(f"  -> {len(prompt_groups)} prompts, {len(data)} samples{skip_msg}")

    stats['unique_prompts'] = len(stats['unique_prompts'])
    return stats


def organize_by_prompt(
    data_dir: Path,
    output_dir: Path,
    source: str,
    steps: Optional[List[int]] = None,
    verbose: bool = False
) -> Dict:
    """Organize data into prompt directories with per-step files.

    Output structure: output_dir/prompt_{hash}/step_N.jsonl + metadata.json
    """
    source_dir = data_dir / source

    prompt_data: Dict[str, Dict[int, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    prompt_texts: Dict[str, str] = {}

    stats = {
        'total_files': 0,
        'total_samples': 0,
        'steps_processed': [],
        'skipped_samples': 0,
    }

    step_files = sorted(source_dir.glob("*.jsonl"), key=lambda x: int(x.stem))

    for step_file in step_files:
        step = int(step_file.stem)

        if steps and step not in steps:
            continue

        stats['total_files'] += 1
        stats['steps_processed'].append(step)

        if verbose:
            print(f"Processing {source}/step {step}...")

        data = load_jsonl(step_file)

        for item in data:
            result = get_prompt_key(item, source)
            if result is None:
                stats['skipped_samples'] += 1
                continue
            prompt_text, hash_key = result

            if hash_key not in prompt_texts:
                prompt_texts[hash_key] = prompt_text

            if source == 'rollout' and 'prompt_text' not in item:
                item['prompt_text'] = prompt_text

            prompt_data[hash_key][step].append(item)
            stats['total_samples'] += 1

    for hash_key, step_items in prompt_data.items():
        prompt_dir = output_dir / f"prompt_{hash_key}"

        metadata = {
            'prompt_hash': hash_key,
            'prompt_text': prompt_texts[hash_key],
            'steps': sorted(step_items.keys()),
            'total_responses': sum(len(items) for items in step_items.values()),
        }
        metadata_file = prompt_dir / "metadata.json"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        for step, items in step_items.items():
            output_file = prompt_dir / f"step_{step}.jsonl"
            save_jsonl(items, output_file)

    stats['unique_prompts'] = len(prompt_data)
    return stats


def create_summary(output_dir: Path, source: str, mode: str, stats: Dict):
    """Create a summary file for the organized data."""
    summary = {
        'source': source,
        'mode': mode,
        'statistics': {
            'files_processed': stats['total_files'],
            'samples_processed': stats['total_samples'],
            'unique_prompts': stats['unique_prompts'],
            'steps_processed': stats['steps_processed'],
            'skipped_samples': stats.get('skipped_samples', 0),
        }
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Organize rollout data by prompts for each step."
    )

    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to rollout data directory (containing rollout/ and secondary/ subdirs)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to output directory for organized data"
    )
    parser.add_argument(
        "--source", type=str, default="rollout",
        choices=["rollout", "secondary"],
        help="Data source: 'rollout' (primary) or 'secondary' (runahead). Default: rollout"
    )
    parser.add_argument(
        "--mode", type=str, default="by_step",
        choices=["by_step", "by_prompt"],
        help="Organization mode. Default: by_step"
    )
    parser.add_argument(
        "--steps", type=str, default=None,
        help="Comma-separated list of steps to process (e.g., '1,2,3,10'). Default: all"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print verbose progress information"
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    steps = None
    if args.steps:
        steps = [int(s.strip()) for s in args.steps.split(',')]

    source_dir = data_dir / args.source
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Organizing {args.source} data from: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {args.mode}")
    if steps:
        print(f"Steps filter: {steps}")
    print()

    if args.mode == "by_step":
        stats = organize_by_step(data_dir, output_dir, args.source, steps, args.verbose)
    else:
        stats = organize_by_prompt(data_dir, output_dir, args.source, steps, args.verbose)

    create_summary(output_dir, args.source, args.mode, stats)

    print(f"\nOrganization complete!")
    print(f"  Files processed: {stats['total_files']}")
    print(f"  Samples processed: {stats['total_samples']}")
    print(f"  Unique prompts: {stats['unique_prompts']}")
    print(f"  Steps processed: {len(stats['steps_processed'])}")
    if stats.get('skipped_samples', 0) > 0:
        print(f"  Skipped samples: {stats['skipped_samples']}")
    print(f"\nSummary written to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

"""
Example: Suffix Tree Speculation with vLLM Integration

This example demonstrates the realistic verl workflow:
1. Create a single LLM instance with suffix decoding
2. Run baseline inference (cold start, no patterns)
3. Collect response patterns through multiple rounds (simulating RL rollouts)
4. Build per-prompt suffix trees with hash-based sharing
5. Load snapshot into the SAME LLM instance using load_snapshot()
6. Run enhanced inference and measure acceptance length improvement

KEY INSIGHT: We use a SINGLE LLM instance throughout, which reflects
the realistic verl use case. Metrics are cumulative Prometheus counters,
so we track deltas between phases to measure improvement.

Usage:
    # Run the example
    python example_suffix_tree_speculation.py --model meta-llama/Llama-3.1-8B-Instruct

    # Run with fewer samples for testing
    python example_suffix_tree_speculation.py --model meta-llama/Llama-3.1-8B-Instruct --num-rounds 2 --num-prompts 5
"""

import argparse
import gc
import os

import numpy as np
from typing import Dict, List, Tuple

# Disable multiprocessing before any vLLM imports
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

# Apply monkey patches BEFORE importing vLLM
from verl.workers.rollout.vllm_rollout.patches import apply_all_patches
apply_all_patches()

from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

# Lazy import for ArcticInference
ParallelSuffixDecodingCache = None


def get_cache():
    """Lazy import of ParallelSuffixDecodingCache."""
    global ParallelSuffixDecodingCache
    if ParallelSuffixDecodingCache is None:
        from arctic_inference.suffix_decoding import ParallelSuffixDecodingCache as Cache
        ParallelSuffixDecodingCache = Cache
    return ParallelSuffixDecodingCache


# Sample Q/A prompts for demonstration
QA_PROMPTS = [
    # Math questions
    "What is 25 + 37?",
    "Calculate 42 + 18.",
    "What is the sum of 15 and 29?",
    "Add 33 and 27 together.",
    "What is 56 plus 44?",

    # Definition questions
    "What is machine learning?",
    "Define artificial intelligence.",
    "What is deep learning?",
    "Explain what neural networks are.",
    "What is reinforcement learning?",

    # Explanation questions
    "How does photosynthesis work?",
    "Explain the water cycle.",
    "How do vaccines work?",
    "Explain how computers process data.",
    "How does the internet work?",

    # Factual questions
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Japan?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",

    # Comparison questions
    "Compare Python and JavaScript.",
    "What are the differences between TCP and UDP?",
    "Compare supervised and unsupervised learning.",
    "What is the difference between HTTP and HTTPS?",
    "Compare REST and GraphQL.",
]


def extract_acceptance_metrics(metrics: list) -> dict:
    """Extract speculative decoding metrics from vLLM metrics."""
    num_drafts = 0
    num_draft_tokens = 0
    num_accepted_tokens = 0
    acceptance_counts = []

    for metric in metrics:
        if metric.name == "vllm:spec_decode_num_drafts":
            assert isinstance(metric, Counter)
            num_drafts += metric.value
        elif metric.name == "vllm:spec_decode_num_draft_tokens":
            assert isinstance(metric, Counter)
            num_draft_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens":
            assert isinstance(metric, Counter)
            num_accepted_tokens += metric.value
        elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            assert isinstance(metric, Vector)
            if not acceptance_counts:
                acceptance_counts = [0] * len(metric.values)
            for pos in range(len(metric.values)):
                acceptance_counts[pos] += metric.values[pos]

    mean_acceptance_length = 1.0 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1.0

    per_pos_rates = []
    if acceptance_counts and num_drafts > 0:
        per_pos_rates = [count / num_drafts for count in acceptance_counts]

    return {
        "num_drafts": num_drafts,
        "num_draft_tokens": num_draft_tokens,
        "num_accepted_tokens": num_accepted_tokens,
        "mean_acceptance_length": mean_acceptance_length,
        "per_position_acceptance_rates": per_pos_rates,
    }


def print_acceptance_stats(stats: dict, label: str = ""):
    """Pretty print acceptance statistics."""
    print("-" * 60)
    if label:
        print(f"Acceptance Statistics: {label}")
    else:
        print("Acceptance Statistics:")
    print("-" * 60)
    print(f"  Number of drafts: {stats['num_drafts']}")
    print(f"  Draft tokens proposed: {stats['num_draft_tokens']}")
    print(f"  Draft tokens accepted: {stats['num_accepted_tokens']}")
    print(f"  Mean acceptance length: {stats['mean_acceptance_length']:.3f}")

    if stats['per_position_acceptance_rates']:
        print("  Per-position acceptance rates:")
        for i, rate in enumerate(stats['per_position_acceptance_rates']):
            print(f"    Position {i}: {rate:.3f}")
    print("-" * 60)


def build_suffix_forest(
    prompts: list[str],
    responses: list[str],
    tokenizer,
    max_tree_depth: int = 64,
    hash_token_count: int = 128,
) -> Tuple[List[Tuple[int, bytes]], Dict[str, int]]:
    """
    Build a SuffixForest with one tree per unique prompt, using hash-based sharing.

    Multiple responses for the same prompt are consolidated into the same tree.
    This is the pattern used in RL training where the same question is asked
    multiple times.

    IMPORTANT: We use add_special_tokens=True to match vLLM's tokenization,
    which includes the BOS token. This ensures the hash computed during tree
    building matches the hash computed during inference.

    Args:
        prompts: Original prompts (may have duplicates)
        responses: Generated responses
        tokenizer: Tokenizer for encoding text
        max_tree_depth: Maximum tree depth for context matching
        hash_token_count: Number of tokens to hash for tree lookup

    Returns:
        Tuple of (snapshots, hash_mapping)
    """
    Cache = get_cache()
    cache = Cache(
        max_tree_depth=max_tree_depth,
        hash_token_count=hash_token_count,
        num_threads=-1,
        parallel_threshold=4
    )

    seen_prompts = set()

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        # CRITICAL: Use add_special_tokens=True to match vLLM's tokenization
        # vLLM adds BOS token to prompts, so we must do the same for hash matching
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
        prompt_array = np.array(prompt_tokens, dtype=np.int32)

        req_id = f"req_{i}"
        cache.start_request(req_id, prompt_array)

        # For response tokens, we don't need special tokens since they're appended
        full_text = prompt + response
        full_tokens = tokenizer.encode(full_text, add_special_tokens=True)
        response_tokens = full_tokens[len(prompt_tokens):]

        if len(response_tokens) > 0:
            response_array = np.array(response_tokens, dtype=np.int32)
            cache.add_tokens(req_id, response_array)

        seen_prompts.add(prompt)

    stats = cache.get_stats()
    print(f"Built suffix forest:")
    print(f"  Unique prompts: {len(seen_prompts)}")
    print(f"  Total responses: {len(prompts)}")
    print(f"  Trees in forest: {stats.get('num_trees_in_forest', 'N/A')}")

    snapshots, hash_mapping = cache.create_snapshot(include_hash_mapping=True)
    total_bytes = sum(len(s[1]) for s in snapshots)
    print(f"  Snapshot: {len(snapshots)} trees, {total_bytes} bytes total")
    print(f"  Hash mappings: {len(hash_mapping)}")

    return snapshots, hash_mapping


def create_llm(args) -> LLM:
    """Create a fresh LLM instance with suffix decoding."""
    print("\nInitializing fresh vLLM instance with suffix decoding...")
    llm = LLM(
        model=args.model,
        speculative_config={
            "method": "suffix",
            "num_speculative_tokens": args.num_spec_tokens,
            "suffix_decoding_max_tree_depth": args.max_tree_depth,
            "suffix_decoding_max_spec_factor": 1.0,
            "suffix_decoding_min_token_prob": 0.1,
        },
        gpu_memory_utilization=0.8,
        max_model_len=args.max_model_len,
        disable_log_stats=False,
        trust_remote_code=True,
    )
    return llm


def cleanup_llm(llm: LLM):
    """Clean up LLM instance to free GPU memory."""
    print("Cleaning up LLM instance...")
    del llm
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass
    print("Cleanup complete")


def run_inference_and_collect_stats(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    label: str = "",
) -> tuple[list[str], dict]:
    """Run inference and collect acceptance metrics."""
    print(f"\n{'='*60}")
    print(f"Running inference: {label}")
    print(f"{'='*60}")
    print("Prompts:", prompts)
    outputs = llm.generate(prompts, sampling_params)
    responses = [output.outputs[0].text for output in outputs]
    try:
        metrics = llm.get_metrics()
        stats = extract_acceptance_metrics(metrics)
    except (AssertionError, AttributeError) as e:
        print(f"Warning: Could not get metrics ({e})")
        stats = {
            "num_drafts": 0,
            "num_draft_tokens": 0,
            "num_accepted_tokens": 0,
            "mean_acceptance_length": 0.0,
            "per_position_acceptance_rates": [],
        }

    return responses, stats


def main(args):
    """
    Main example workflow using a SINGLE LLM instance.

    This reflects the realistic verl use case where we:
    1. Create one LLM instance
    2. Run inference to collect patterns (simulating RL rollouts)
    3. Build suffix trees from collected patterns
    4. Load snapshot into the SAME instance
    5. Continue inference with improved speculation

    Note: Metrics are cumulative, so we track deltas between phases.
    """
    print("=" * 60)
    print("Suffix Tree Speculation Example")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Num speculative tokens: {args.num_spec_tokens}")
    print(f"Num prompts per round: {args.num_prompts}")
    print(f"Num training rounds: {args.num_rounds}")
    print("=" * 60)

    prompts = QA_PROMPTS[:args.num_prompts]
    if len(prompts) < args.num_prompts:
        prompts = (prompts * (args.num_prompts // len(prompts) + 1))[:args.num_prompts]

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # =========================================================================
    # PHASE 1: Initialize LLM
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Initialize LLM with Suffix Decoding")
    print("=" * 60)

    llm = create_llm(args)

    # =========================================================================
    # PHASE 2: Baseline - Cold Start Inference
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Baseline Inference (Cold Start, No Patterns)")
    print("=" * 60)

    baseline_responses, baseline_stats = run_inference_and_collect_stats(
        llm, prompts, sampling_params, "Baseline (no historical patterns)"
    )
    print_acceptance_stats(baseline_stats, "Baseline")

    if args.print_output:
        print("\nSample Baseline Outputs:")
        for i, (prompt, response) in enumerate(zip(prompts[:3], baseline_responses[:3])):
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Response: {response[:200]}...")

    # =========================================================================
    # PHASE 3: Collect Training Data (Simulating RL Rollouts)
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Collecting Training Data (Simulating RL Rollouts)")
    print("=" * 60)

    all_prompts = list(prompts)
    all_responses = list(baseline_responses)

    for round_idx in range(args.num_rounds - 1):
        print(f"\nTraining round {round_idx + 2}/{args.num_rounds}")
        responses, _ = run_inference_and_collect_stats(
            llm, prompts, sampling_params,
            f"Training Round {round_idx + 2}"
        )
        all_prompts.extend(prompts)
        all_responses.extend(responses)

    # =========================================================================
    # PHASE 4: Build Suffix Trees with Hash Mapping
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 4: Building Suffix Trees with Hash Mapping")
    print("=" * 60)

    snapshots, hash_mapping = build_suffix_forest(
        all_prompts, all_responses, tokenizer, args.max_tree_depth
    )

    # =========================================================================
    # PHASE 5: Load Snapshot into SAME LLM Instance
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 5: Loading Snapshot (verl SPMD Style)")
    print("=" * 60)
    print("Loading into the SAME LLM instance - this is the realistic verl pattern.")

    # Load snapshot using direct access (verl SPMD style)
    print(f"Loading {len(snapshots)} trees with {len(hash_mapping)} hash mappings...")
    llm.load_snapshot(snapshots, hash_mapping)
    print("Snapshot loaded successfully!")

    # =========================================================================
    # PHASE 6: Enhanced Inference with Loaded Patterns
    # =========================================================================
    print("\n" + "=" * 60)
    print("PHASE 6: Enhanced Inference (With Loaded Patterns)")
    print("=" * 60)

    # Get metrics before enhanced run
    try:
        pre_metrics = llm.get_metrics()
        pre_stats = extract_acceptance_metrics(pre_metrics)
    except:
        pre_stats = {"num_drafts": 0, "num_accepted_tokens": 0}

    enhanced_responses, post_stats = run_inference_and_collect_stats(
        llm, prompts, sampling_params,
        "With pre-loaded historical patterns"
    )

    # Calculate delta (metrics are cumulative)
    enhanced_stats = {
        "num_drafts": post_stats["num_drafts"] - pre_stats.get("num_drafts", 0),
        "num_draft_tokens": post_stats["num_draft_tokens"] - pre_stats.get("num_draft_tokens", 0),
        "num_accepted_tokens": post_stats["num_accepted_tokens"] - pre_stats.get("num_accepted_tokens", 0),
        "mean_acceptance_length": 0.0,
        "per_position_acceptance_rates": [],
    }
    if enhanced_stats["num_drafts"] > 0:
        enhanced_stats["mean_acceptance_length"] = 1.0 + (
            enhanced_stats["num_accepted_tokens"] / enhanced_stats["num_drafts"]
        )

    print_acceptance_stats(enhanced_stats, "With Patterns (delta)")

    if args.print_output:
        print("\nSample Enhanced Outputs:")
        for i, (prompt, response) in enumerate(zip(prompts[:3], enhanced_responses[:3])):
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Response: {response[:200]}...")

    # =========================================================================
    # PHASE 7: Compare Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)

    baseline_al = baseline_stats['mean_acceptance_length']
    enhanced_al = enhanced_stats['mean_acceptance_length']
    improvement = ((enhanced_al - baseline_al) / baseline_al * 100) if baseline_al > 0 else 0

    print(f"Baseline mean acceptance length: {baseline_al:.3f}")
    print(f"Enhanced mean acceptance length: {enhanced_al:.3f}")
    print(f"Improvement: {improvement:+.1f}%")

    print("\nDetailed comparison:")
    print(f"  Baseline drafts: {baseline_stats['num_drafts']}")
    print(f"  Enhanced drafts: {enhanced_stats['num_drafts']}")
    print(f"  Baseline accepted: {baseline_stats['num_accepted_tokens']}")
    print(f"  Enhanced accepted: {enhanced_stats['num_accepted_tokens']}")

    if enhanced_al > baseline_al:
        print("\n[SUCCESS] Suffix tree patterns improved speculation accuracy!")
    elif enhanced_al < baseline_al:
        print("\n[NOTE] Acceptance length decreased.")
    else:
        print("\n[NOTE] No significant change in acceptance length")

    cleanup_llm(llm)

    print("\nExample completed successfully!")
    return baseline_stats, enhanced_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Suffix Tree Speculation Example")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--num-spec-tokens", type=int, default=5)
    parser.add_argument("--max-tree-depth", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=10)
    parser.add_argument("--num-rounds", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--print-output", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

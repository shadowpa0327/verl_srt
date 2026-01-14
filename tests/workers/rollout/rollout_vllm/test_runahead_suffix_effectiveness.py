"""
Test: Runahead -> SuffixCache Effectiveness

Measures whether secondary (runahead) outputs improve speculative decoding
quality when the same batch becomes primary in the next tick.

Test Flow:
1. Prepare 3 batches (b1, b2, b3) of prompts from DAPO dataset
2. Tick 1: rollout(primary=b1, secondary=b2) -> feed b2 runahead to SuffixCache
3. Tick 2: rollout(primary=b2, secondary=b3) -> measure speculation for b2
4. Baseline: rollout(primary=b2) without cache -> compare metrics

Expected Result:
- Tick 2's b2 should have higher speculation acceptance than baseline
- This validates the SRT optimization path
"""

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Optional

import datasets
import numpy as np
import ray
import torch
from tensordict import TensorDict

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

# Note: SRT patches are applied by the worker_extension module when vLLM loads it
# via worker_extension_cls config. The patches are applied lazily in the GPU worker
# process to avoid CUDA initialization issues in non-GPU processes.

from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.experimental.agent_loop.runahead import RunaheadConfig
from verl.protocol import DataProto
from verl.utils import hf_tokenizer
from recipe.srt.suffix_tree_manager import SuffixTreeManager, SuffixTreeManagerConfig
from recipe.srt.vllm_server import SRTvLLMReplica


class SRTAgentLoopManager(AgentLoopManager):
    """AgentLoopManager with SRT suffix tree support.

    Uses SRTvLLMReplica which has load_suffix_snapshot() method.
    """

    def __init__(self, config, worker_group=None, rm_resource_pool=None):
        # Set the replica class BEFORE calling parent __init__
        self.rollout_replica_class = SRTvLLMReplica
        super().__init__(config, worker_group, rm_resource_pool)


# =============================================================================
# Constants
# =============================================================================

DAPO_DATASET_NAME = "BytedTsinghua-SIA/DAPO-Math-17K"


# =============================================================================
# Test Configuration
# =============================================================================

@dataclass
class TestConfig:
    """Configuration for the runahead suffix effectiveness test."""
    # Model
    model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"

    # Hardware
    num_gpus: int = 1
    tp_size: int = 1
    num_workers: int = 1

    # Batch sizes
    batch_size: int = 16  # Samples per batch (b1, b2, b3 each have this many)

    # Long-tail distribution (key for seeing runahead benefit)
    long_tail_ratio: float = 0.2  # 20% of requests are "long"
    short_max_tokens: int = 512    # Short requests generate up to 64 tokens
    long_max_tokens: int = 8192    # Long requests generate up to 256 tokens
    max_prompt_length: int = 512  # Max input prompt length

    # Runahead settings
    load_threshold: int = 32

    # Reproducibility
    seed: int = 42
    dataset_cache_dir: Optional[str] = None


# =============================================================================
# Dataset Loading
# =============================================================================

def load_dapo_dataset(cache_dir: Optional[str] = None) -> datasets.Dataset:
    """Load the DAPO math dataset from HuggingFace."""
    return datasets.load_dataset(DAPO_DATASET_NAME, cache_dir=cache_dir)["train"]


@dataclass
class DAPOSample:
    """Single sample from DAPO dataset."""
    prompt: list[dict]  # Message list
    reward_model: dict  # Contains ground_truth and style
    data_source: str


def sample_prompts(
    dataset: datasets.Dataset,
    n: int,
    seed: Optional[int] = None
) -> list[DAPOSample]:
    """Sample n prompts from the dataset.

    Returns:
        List of DAPOSample objects containing prompt, reward_model, and data_source.
    """
    if seed is not None:
        random.seed(seed)
    indices = random.sample(range(len(dataset)), min(n, len(dataset)))
    samples = []
    for idx in indices:
        row = dataset[idx]
        samples.append(DAPOSample(
            prompt=row["prompt"],
            reward_model=row["reward_model"],
            data_source=row.get("data_source", "dapo"),
        ))
    return samples


def filter_prompts_by_length(
    samples: list[DAPOSample],
    tokenizer,
    max_length: int,
    min_required: int,
) -> list[DAPOSample]:
    """Filter samples to only include those within max_length tokens.

    Args:
        samples: List of DAPOSample objects.
        tokenizer: Tokenizer for length calculation.
        max_length: Maximum token length.
        min_required: Minimum number of samples needed.

    Returns:
        Filtered list of DAPOSample objects.
    """
    filtered = []
    for sample in samples:
        # Use chat template to get actual tokenized length
        tokens = tokenizer.apply_chat_template(
            sample.prompt, add_generation_prompt=True, tokenize=True
        )
        if len(tokens) <= max_length:
            filtered.append(sample)
        if len(filtered) >= min_required:
            break
    return filtered


# =============================================================================
# DataProto Builders with Long-Tail Distribution
# =============================================================================

def build_dataproto_with_longtail(
    config: TestConfig,
    samples: list[DAPOSample],
    batch_name: str,
    shuffle_seed: Optional[int] = None,
    sampling_seed_base: Optional[int] = None,
) -> DataProto:
    """
    Build DataProto with long-tail distribution for testing.

    Long-tail distribution means:
    - Most requests (1 - long_tail_ratio) have short max_tokens
    - Some requests (long_tail_ratio) have long max_tokens

    This creates GPU bubbles when short requests finish early,
    which runahead can fill with secondary work.

    Args:
        config: Test configuration
        samples: List of DAPOSample objects
        batch_name: Name for logging (b1, b2, b3)
        shuffle_seed: Seed for reproducible shuffle
        sampling_seed_base: Base seed for per-sample vLLM sampling

    Returns:
        DataProto with raw_prompt and max_tokens configured for long-tail
    """
    batch_size = config.batch_size
    num_long = max(1, int(batch_size * config.long_tail_ratio))
    num_short = batch_size - num_long

    raw_prompts = []
    reward_models = []
    data_sources = []
    max_tokens_list = []

    # Short requests (most of them)
    for i in range(num_short):
        sample = samples[i % len(samples)]
        raw_prompts.append(sample.prompt)
        reward_models.append(sample.reward_model)
        data_sources.append(sample.data_source)
        max_tokens_list.append(config.short_max_tokens)

    # Long requests (create the "tail")
    for i in range(num_long):
        sample = samples[(num_short + i) % len(samples)]
        raw_prompts.append(sample.prompt)
        reward_models.append(sample.reward_model)
        data_sources.append(sample.data_source)
        max_tokens_list.append(config.long_max_tokens)

    # Shuffle to distribute long tasks randomly (important for realistic simulation)
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
    combined = list(zip(raw_prompts, reward_models, data_sources, max_tokens_list))
    random.shuffle(combined)
    raw_prompts, reward_models, data_sources, max_tokens_list = zip(*combined)
    raw_prompts = list(raw_prompts)
    reward_models = list(reward_models)
    data_sources = list(data_sources)
    max_tokens_list = list(max_tokens_list)

    # Build non_tensor_batch
    non_tensor_batch = {
        "raw_prompt": np.array(raw_prompts, dtype=object),
        "agent_name": np.array(["single_turn_agent"] * batch_size, dtype=object),
        "data_source": np.array(data_sources, dtype=object),
        "reward_model": np.array(reward_models, dtype=object),
        "max_tokens": np.array(max_tokens_list, dtype=object),
    }

    # Add per-sample seeds for deterministic vLLM sampling
    if sampling_seed_base is not None:
        sampling_seeds = [sampling_seed_base + i for i in range(batch_size)]
        non_tensor_batch["sampling_seed"] = np.array(sampling_seeds, dtype=object)

    print(f"    {batch_name}: {num_short} short ({config.short_max_tokens} tok) + "
          f"{num_long} long ({config.long_max_tokens} tok)")

    return DataProto(non_tensor_batch=non_tensor_batch)


def prepare_three_batches(
    config: TestConfig,
    tokenizer,
) -> tuple[DataProto, DataProto, DataProto]:
    """
    Prepare 3 batches (b1, b2, b3) from DAPO dataset with long-tail distribution.

    Each batch uses different prompts but same long-tail structure.
    This simulates a real training scenario where batches have similar
    statistical properties but different content.

    Returns:
        (b1, b2, b3) - Three DataProto batches
    """
    print("\n[1] Loading DAPO dataset...")
    dataset = load_dapo_dataset(cache_dir=config.dataset_cache_dir)
    print(f"    Dataset loaded: {len(dataset)} samples available")

    # Sample prompts for each batch (3x batch_size, then filter)
    sample_multiplier = 3
    total_needed = config.batch_size * 3 * sample_multiplier

    print(f"\n[2] Sampling and filtering samples (max {config.max_prompt_length} tokens)...")

    # Sample all data at once, then filter by length
    all_samples_raw = sample_prompts(dataset, total_needed, seed=config.seed)
    all_samples = filter_prompts_by_length(
        all_samples_raw, tokenizer, config.max_prompt_length, config.batch_size * 3
    )

    if len(all_samples) < config.batch_size * 3:
        print(f"    Warning: Only {len(all_samples)} samples after filtering, "
              f"need {config.batch_size * 3}. Recycling samples.")
        while len(all_samples) < config.batch_size * 3:
            all_samples = all_samples + all_samples[:config.batch_size]

    # Split into 3 batches
    samples_b1 = all_samples[0:config.batch_size]
    samples_b2 = all_samples[config.batch_size:config.batch_size*2]
    samples_b3 = all_samples[config.batch_size*2:config.batch_size*3]

    print(f"    Split into 3 batches of {config.batch_size} samples each")

    # Build DataProto for each batch with long-tail distribution
    print(f"\n[3] Building DataProto with long-tail distribution "
          f"(ratio={config.long_tail_ratio})...")

    b1 = build_dataproto_with_longtail(
        config, samples_b1, "b1",
        shuffle_seed=config.seed,
        sampling_seed_base=config.seed * 1000,
    )

    b2 = build_dataproto_with_longtail(
        config, samples_b2, "b2",
        shuffle_seed=config.seed + 1,
        sampling_seed_base=config.seed * 1000 + 100,
    )

    b3 = build_dataproto_with_longtail(
        config, samples_b3, "b3",
        shuffle_seed=config.seed + 2,
        sampling_seed_base=config.seed * 1000 + 200,
    )

    return b1, b2, b3


# =============================================================================
# Metrics
# =============================================================================

@dataclass
class SpeculationMetrics:
    """Metrics for speculation quality measurement."""
    draft_acceptance_rate: float = 0.0
    num_accepted_tokens: int = 0
    num_draft_tokens: int = 0
    generation_time_s: float = 0.0
    tokens_generated: int = 0
    throughput_tokens_per_s: float = 0.0


@dataclass
class TestResult:
    """Result of a single test run."""
    tick: str  # "tick1", "tick2", "baseline"
    batch_name: str  # "b1", "b2", "b3"
    role: str  # "primary", "secondary"
    metrics: SpeculationMetrics
    suffix_cache_stats: dict = field(default_factory=dict)


def extract_spec_metrics(gen_output: DataProto, elapsed_time: float) -> SpeculationMetrics:
    """Extract speculation metrics from generation output."""
    spec_metrics = gen_output.meta_info.get("spec_decode_metrics", {})

    # Count total tokens from response_mask
    total_tokens = 0
    if gen_output.batch is not None and "response_mask" in gen_output.batch:
        for i in range(len(gen_output)):
            total_tokens += gen_output.batch["response_mask"][i].sum().item()

    return SpeculationMetrics(
        draft_acceptance_rate=spec_metrics.get("spec_decode/acceptance_rate", 0.0),
        num_accepted_tokens=int(spec_metrics.get("spec_decode/num_accepted_tokens", 0)),
        num_draft_tokens=int(spec_metrics.get("spec_decode/num_draft_tokens", 0)),
        generation_time_s=elapsed_time,
        tokens_generated=total_tokens,
        throughput_tokens_per_s=total_tokens / elapsed_time if elapsed_time > 0 else 0,
    )


# =============================================================================
# Test Harness
# =============================================================================

class RunaheadSuffixEffectivenessTest:
    """Test harness for measuring runahead -> suffix cache effectiveness."""

    def __init__(self, config: TestConfig):
        self.config = config
        self.manager: Optional[AgentLoopManager] = None
        self.suffix_tree_manager: Optional[SuffixTreeManager] = None
        self.tokenizer = None
        self.results: list[TestResult] = []

    def _compose_hydra_config(self):
        """Compose Hydra config for AgentLoopManager."""
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        config_dir = os.path.abspath("verl/trainer/config")
        if not os.path.exists(config_dir):
            # Try from project root
            project_root = os.path.join(os.path.dirname(__file__), "../../../..")
            config_dir = os.path.join(project_root, "verl/trainer/config")
            config_dir = os.path.abspath(config_dir)

        with initialize_config_dir(config_dir=config_dir, version_base=None):
            hydra_config = compose(config_name="ppo_trainer")

        # Hardware settings
        hydra_config.trainer.n_gpus_per_node = self.config.num_gpus
        hydra_config.trainer.nnodes = 1

        # Model settings
        hydra_config.actor_rollout_ref.model.path = self.config.model_path
        hydra_config.actor_rollout_ref.rollout.name = "vllm"
        hydra_config.actor_rollout_ref.rollout.mode = "async"
        hydra_config.actor_rollout_ref.rollout.tensor_model_parallel_size = self.config.tp_size
        hydra_config.actor_rollout_ref.rollout.scheduling_policy = "priority"

        # Token length bounds
        hydra_config.actor_rollout_ref.rollout.prompt_length = self.config.max_prompt_length
        hydra_config.actor_rollout_ref.rollout.response_length = self.config.long_max_tokens

        # Agent workers
        hydra_config.actor_rollout_ref.rollout.agent.num_workers = self.config.num_workers

        # vLLM settings
        hydra_config.actor_rollout_ref.rollout.gpu_memory_utilization = 0.9
        hydra_config.actor_rollout_ref.rollout.enable_prefix_caching = False  # Disable to test suffix speculation
        hydra_config.actor_rollout_ref.rollout.disable_log_stats = False  # Enable for spec decode metrics

        # SRT: Configure engine_kwargs.vllm for suffix decoding
        # This sets: worker_extension_cls and speculative_config
        # Need to disable struct mode to add new keys
        OmegaConf.set_struct(hydra_config, False)

        if not hasattr(hydra_config.actor_rollout_ref.rollout, "engine_kwargs"):
            hydra_config.actor_rollout_ref.rollout.engine_kwargs = OmegaConf.create({})
        if not hasattr(hydra_config.actor_rollout_ref.rollout.engine_kwargs, "vllm"):
            hydra_config.actor_rollout_ref.rollout.engine_kwargs.vllm = OmegaConf.create({})

        vllm_kwargs = hydra_config.actor_rollout_ref.rollout.engine_kwargs.vllm

        # Inject worker extension for suffix snapshot loading
        vllm_kwargs.worker_extension_cls = (
            "recipe.srt.vllm_plugin.worker_extension.SuffixTreeWorkerExtension"
        )

        # Configure speculative decoding with suffix method
        # Note: The SRT config patches extend SpeculativeConfig to accept these fields
        vllm_kwargs.speculative_config = {
            "method": "suffix",
            "num_speculative_tokens": 8,  # Required by vLLM
            "suffix_decoding_max_tree_depth": 64,
            "suffix_decoding_use_parallel": True,  # Use parallel proposer for better hash matching
        }

        # Re-enable struct mode
        OmegaConf.set_struct(hydra_config, True)

        # Disable reward model
        if hasattr(hydra_config, "reward_model"):
            hydra_config.reward_model.enable = False

        return hydra_config

    def setup(self):
        """Initialize AgentLoopManager and SuffixTreeManager."""
        print("\n" + "="*60)
        print("SETUP: Initializing test environment")
        print("="*60)

        # Initialize Ray if needed
        if not ray.is_initialized():
            print("\n[0] Initializing Ray...")
            ray.init()

        # Load tokenizer first (needed for filtering prompts by length)
        print("\n[1] Loading tokenizer...")
        self.tokenizer = hf_tokenizer(self.config.model_path, trust_remote_code=True)
        print(f"    Loaded tokenizer for: {self.config.model_path}")

        # Create SRTAgentLoopManager (with SRTvLLMReplica for suffix snapshot loading)
        print("\n[2] Creating SRTAgentLoopManager...")
        hydra_config = self._compose_hydra_config()
        self.manager = SRTAgentLoopManager(hydra_config)
        print(f"    Model: {self.config.model_path}")
        print(f"    GPUs: {self.config.num_gpus}, TP: {self.config.tp_size}")

        # Create SuffixTreeManager
        print("\n[3] Creating SuffixTreeManager...")
        suffix_config = SuffixTreeManagerConfig(
            enable=True,
            max_tree_depth=64,
            hash_token_count=128,
        )
        self.suffix_tree_manager = SuffixTreeManager(
            suffix_config,
            self.tokenizer
        )
        print("    max_tree_depth: 64, hash_token_count: 128")

    def prepare_batches(self) -> tuple[DataProto, DataProto, DataProto]:
        """Prepare 3 batches (b1, b2, b3) from DAPO dataset with long-tail."""
        return prepare_three_batches(self.config, self.tokenizer)

    def run_tick1(self, b1: DataProto, b2: DataProto) -> TestResult:
        """
        Tick 1: rollout(primary=b1, secondary=b2)
        Feed b2 runahead results to SuffixCache.
        """
        runahead_config = RunaheadConfig(
            enabled=True,
            load_threshold=self.config.load_threshold,
        )

        t0 = time.perf_counter()
        result = self.manager.generate_sequences_with_runahead(
            b1, b2, runahead_config
        )
        elapsed = time.perf_counter() - t0

        # Feed secondary outputs to suffix cache
        secondary_tokens_added = 0
        for out in result.secondary_outputs:
            if out.status in ("completed", "aborted") and out.output and out.prompt_ids:
                self.suffix_tree_manager.add_sequence(
                    prompt_tokens=out.prompt_ids,
                    response_tokens=out.output.token_ids,
                )
                secondary_tokens_added += len(out.output.token_ids)

        metrics = extract_spec_metrics(result.primary_outputs, elapsed)

        return TestResult(
            tick="tick1",
            batch_name="b1",
            role="primary",
            metrics=metrics,
            suffix_cache_stats={
                "secondary_outputs_count": len(result.secondary_outputs),
                "secondary_completed": result.metrics.secondary_completed,
                "secondary_aborted": result.metrics.secondary_aborted,
                "secondary_tokens_added": secondary_tokens_added,
            }
        )

    def run_tick2(self, b2: DataProto, b3: DataProto) -> TestResult:
        """
        Tick 2: rollout(primary=b2, secondary=b3)
        b2 should benefit from suffix cache populated in tick1.
        """
        # Push suffix snapshot to workers
        snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()
        snapshot_loaded = False
        if snapshots:
            print(f"    Pushing {len(snapshots)} suffix trees to vLLM servers...")
            results = self.manager.load_suffix_snapshot(snapshots, hash_mapping)
            snapshot_loaded = len(results) > 0
            print(f"    Snapshot load results: {results}")

        runahead_config = RunaheadConfig(
            enabled=True,
            load_threshold=self.config.load_threshold,
        )

        t0 = time.perf_counter()
        result = self.manager.generate_sequences_with_runahead(
            b2, b3, runahead_config
        )
        elapsed = time.perf_counter() - t0

        metrics = extract_spec_metrics(result.primary_outputs, elapsed)

        return TestResult(
            tick="tick2",
            batch_name="b2",
            role="primary",
            metrics=metrics,
            suffix_cache_stats={
                "cache_trees": self.suffix_tree_manager.get_metrics().get("suffix_tree/num_trees", 0),
                "snapshot_loaded": snapshot_loaded,
            }
        )

    def run_baseline(self, b2: DataProto) -> TestResult:
        """
        Baseline: rollout(primary=b2) WITHOUT suffix cache.
        Used to compare against tick2 to measure improvement.
        """
        # Clear or disable suffix cache
        # Option 1: Create fresh SuffixTreeManager
        # Option 2: Don't push snapshot to workers

        t0 = time.perf_counter()
        result = self.manager.generate_sequences(b2)
        elapsed = time.perf_counter() - t0

        metrics = extract_spec_metrics(result, elapsed)

        return TestResult(
            tick="baseline",
            batch_name="b2",
            role="primary",
            metrics=metrics,
            suffix_cache_stats={"cache_enabled": False}
        )

    def run_test(self) -> dict:
        """Run the full test sequence."""
        self.setup()
        b1, b2, b3 = self.prepare_batches()

        print("\n" + "="*60)
        print("RUNNING TEST SEQUENCE")
        print("="*60)

        # Run baseline FIRST (before cache is populated)
        print("\n" + "-"*40)
        print("BASELINE: Running b2 without suffix cache")
        print("-"*40)
        baseline_result = self.run_baseline(b2)
        self.results.append(baseline_result)
        print(f"    Generation time: {baseline_result.metrics.generation_time_s:.2f}s")
        print(f"    Tokens generated: {baseline_result.metrics.tokens_generated}")
        print(f"    Throughput: {baseline_result.metrics.throughput_tokens_per_s:.1f} tok/s")

        # Run tick 1: populates cache with b2 patterns
        print("\n" + "-"*40)
        print("TICK 1: primary=b1, secondary=b2 (populates cache)")
        print("-"*40)
        tick1_result = self.run_tick1(b1, b2)
        self.results.append(tick1_result)
        print(f"    Generation time: {tick1_result.metrics.generation_time_s:.2f}s")
        print(f"    Secondary completed: {tick1_result.suffix_cache_stats.get('secondary_completed', 0)}")
        print(f"    Secondary aborted: {tick1_result.suffix_cache_stats.get('secondary_aborted', 0)}")
        print(f"    Secondary tokens added to cache: {tick1_result.suffix_cache_stats.get('secondary_tokens_added', 0)}")

        # Run tick 2: b2 should benefit from cache
        print("\n" + "-"*40)
        print("TICK 2: primary=b2, secondary=b3 (uses cache)")
        print("-"*40)
        tick2_result = self.run_tick2(b2, b3)
        self.results.append(tick2_result)
        print(f"    Generation time: {tick2_result.metrics.generation_time_s:.2f}s")
        print(f"    Tokens generated: {tick2_result.metrics.tokens_generated}")
        print(f"    Throughput: {tick2_result.metrics.throughput_tokens_per_s:.1f} tok/s")
        print(f"    Draft acceptance rate: {tick2_result.metrics.draft_acceptance_rate:.2%}")

        # Compute improvement
        improvement = self._compute_improvement(baseline_result, tick2_result)

        return {
            "config": asdict(self.config),
            "results": [asdict(r) for r in self.results],
            "improvement": improvement,
        }

    def _compute_improvement(self, baseline: TestResult, tick2: TestResult) -> dict:
        """Compute improvement metrics."""
        return {
            "acceptance_rate_baseline": baseline.metrics.draft_acceptance_rate,
            "acceptance_rate_with_cache": tick2.metrics.draft_acceptance_rate,
            "acceptance_rate_improvement": (
                tick2.metrics.draft_acceptance_rate - baseline.metrics.draft_acceptance_rate
            ),
            "throughput_baseline": baseline.metrics.throughput_tokens_per_s,
            "throughput_with_cache": tick2.metrics.throughput_tokens_per_s,
            "throughput_improvement_pct": (
                (tick2.metrics.throughput_tokens_per_s - baseline.metrics.throughput_tokens_per_s)
                / baseline.metrics.throughput_tokens_per_s * 100
                if baseline.metrics.throughput_tokens_per_s > 0 else 0
            ),
        }


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test runahead -> suffix cache effectiveness"
    )
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Model path")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Samples per batch (b1, b2, b3)")
    parser.add_argument("--long-tail-ratio", type=float, default=0.2,
                        help="Fraction of requests with long max_tokens")
    parser.add_argument("--short-max-tokens", type=int, default=64,
                        help="max_tokens for short requests")
    parser.add_argument("--long-max-tokens", type=int, default=256,
                        help="max_tokens for long requests (creates bubbles)")
    parser.add_argument("--load-threshold", type=int, default=32,
                        help="Runahead admission threshold")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="runahead_suffix_test_results.json")
    args = parser.parse_args()

    config = TestConfig(
        model_path=args.model,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        long_tail_ratio=args.long_tail_ratio,
        short_max_tokens=args.short_max_tokens,
        long_max_tokens=args.long_max_tokens,
        load_threshold=args.load_threshold,
        seed=args.seed,
    )

    print("="*60)
    print("RUNAHEAD -> SUFFIX CACHE EFFECTIVENESS TEST")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model:           {config.model_path}")
    print(f"  Batch size:      {config.batch_size}")
    print(f"  Long-tail ratio: {config.long_tail_ratio}")
    print(f"  Short tokens:    {config.short_max_tokens}")
    print(f"  Long tokens:     {config.long_max_tokens}")
    print(f"  Load threshold:  {config.load_threshold}")

    test = RunaheadSuffixEffectivenessTest(config)
    results = test.run_test()

    # Print summary
    print("\n" + "="*60)
    print("RUNAHEAD -> SUFFIX CACHE EFFECTIVENESS TEST RESULTS")
    print("="*60)

    improvement = results["improvement"]
    print(f"\nSpeculation Acceptance Rate:")
    print(f"  Baseline (no cache):  {improvement['acceptance_rate_baseline']:.2%}")
    print(f"  With runahead cache:  {improvement['acceptance_rate_with_cache']:.2%}")
    print(f"  Improvement:          {improvement['acceptance_rate_improvement']:.2%}")

    print(f"\nThroughput (tokens/sec):")
    print(f"  Baseline:             {improvement['throughput_baseline']:.1f}")
    print(f"  With cache:           {improvement['throughput_with_cache']:.1f}")
    print(f"  Improvement:          {improvement['throughput_improvement_pct']:.1f}%")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

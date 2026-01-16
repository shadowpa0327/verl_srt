"""
Test: Runahead -> SuffixCache Effectiveness

Measures whether secondary (runahead) outputs improve speculative decoding
quality when the same batch becomes primary in the next tick.

Supports two cache modes:
- snapshot (default): Uses SuffixTreeManager with snapshot-based loading
- shared_memory: Uses SharedMemoryCacheManager with SpecRL's zero-copy shared memory

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
import logging
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

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

logger = logging.getLogger(__name__)


# =============================================================================
# Cache Manager Interface (Abstraction for both modes)
# =============================================================================

class CacheManagerInterface(ABC):
    """Abstract interface for both cache modes (snapshot and shared_memory)."""

    @abstractmethod
    def add_sequence(self, prompt_tokens: List[int], response_tokens: List[int]) -> None:
        """Add a prompt/response pair to the cache."""
        pass

    @abstractmethod
    def push_to_workers(self, manager: AgentLoopManager) -> List[Dict[str, Any]]:
        """Push cache state to workers (snapshot) or flush pending updates (shm)."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics for logging."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Cleanup resources."""
        pass


class SnapshotCacheManager(CacheManagerInterface):
    """Wraps SuffixTreeManager for snapshot mode."""

    def __init__(self, tokenizer):
        self.suffix_tree_manager = SuffixTreeManager(
            SuffixTreeManagerConfig(enable=True, max_tree_depth=64, hash_token_count=128),
            tokenizer
        )
        self._tokens_added = 0

    def add_sequence(self, prompt_tokens: List[int], response_tokens: List[int]) -> None:
        self.suffix_tree_manager.add_sequence(
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
        )
        self._tokens_added += len(response_tokens)

    def push_to_workers(self, manager: AgentLoopManager) -> List[Dict[str, Any]]:
        snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()
        if snapshots:
            return manager.load_suffix_snapshot(snapshots, hash_mapping)
        return []

    def get_snapshot(self):
        """Get snapshot for direct use."""
        return self.suffix_tree_manager.get_snapshot()

    def get_metrics(self) -> Dict[str, Any]:
        metrics = self.suffix_tree_manager.get_metrics()
        metrics["cache_mode"] = "snapshot"
        metrics["tokens_added"] = self._tokens_added
        return metrics

    def shutdown(self) -> None:
        pass  # No cleanup needed


class SharedMemoryCacheManagerTest(CacheManagerInterface):
    """
    Wraps SharedMemoryCacheManager for shared memory mode (test version).

    Key differences from snapshot mode:
    - Cache servers are deployed on worker nodes via Ray
    - Updates are sent via gRPC AFTER tick (not before)
    - Workers read directly from shared memory (no snapshot loading)
    """

    def __init__(self, config, tokenizer):
        self._config = config
        self._tokenizer = tokenizer
        self._pending_sequences: List[tuple] = []
        self._cache_manager = None
        self._cache_updater = None
        self._cache_servers: List[dict] = []
        self._initialized = False

    def initialize_with_manager(self, manager: AgentLoopManager):
        """
        Initialize cache servers using manager's resource info.

        Deploys RolloutCacheServer on the same node as rollout workers.
        """
        try:
            from recipe.specRL.cache_manager import CacheWorker
            from specrl.cache_updater import SuffixCacheUpdater
        except ImportError as e:
            logger.error(f"Failed to import SpecRL modules: {e}")
            logger.error("Shared memory mode requires the specrl package to be installed.")
            raise

        port = self._config.shm_port

        # Deploy cache server - for testing, we deploy on the current node
        # In production, SRTRayPPOTrainer deploys on all GPU nodes
        server = CacheWorker.remote(port=port)

        # For local testing, use localhost instead of node's IPv6 link-local address
        # IPv6 link-local addresses (fe80::...) require scope ID which gRPC doesn't handle well
        # Use IPv4 localhost for single-node testing
        ip = "127.0.0.1"
        # For production, would use: ip = ray.get(server.get_node_ip.remote())

        self._cache_servers.append({
            "server": server,
            "ip": ip,
            "port": port,
        })

        # Create updater with server addresses (use IPv4 format without brackets)
        addresses = [f"{s['ip']}:{s['port']}" for s in self._cache_servers]
        self._cache_updater = SuffixCacheUpdater(server_addresses=addresses)
        self._initialized = True

        logger.info(f"SharedMemoryCacheManagerTest: Deployed cache server on {ip}:{port}")
        print(f"    SharedMemory: Deployed cache server on {ip}:{port}")

    def add_sequence(self, prompt_tokens: List[int], response_tokens: List[int]) -> None:
        """Queue sequence for batch update (shared memory sends after tick, not during)."""
        self._pending_sequences.append((list(prompt_tokens), list(response_tokens)))

    def flush_pending_sequences(self) -> Dict[str, Any]:
        """Send all pending sequences to cache servers via gRPC."""
        if not self._pending_sequences or not self._cache_updater:
            return {"shm_cache/sequences_flushed": 0}

        prompts = [s[0] for s in self._pending_sequences]
        responses = [s[1] for s in self._pending_sequences]
        prompt_lengths = [float(len(p)) for p in prompts]
        response_lengths = [float(len(r)) for r in responses]

        try:
            self._cache_updater.update_response_cache(
                prompts=prompts,
                responses=responses,
                prompt_lengths=prompt_lengths,
                response_lengths=response_lengths,
                responses_per_prompt=1,
            )
        except Exception as e:
            logger.warning(f"Failed to flush sequences to cache: {e}")
            return {"shm_cache/flush_error": str(e)}

        count = len(self._pending_sequences)
        self._pending_sequences.clear()
        return {"shm_cache/sequences_flushed": count}

    def push_to_workers(self, manager: AgentLoopManager) -> List[Dict[str, Any]]:
        """
        Shared memory mode: workers access cache directly, no push needed.
        But we do flush pending sequences.
        """
        flush_result = self.flush_pending_sequences()
        return [{
            "status": "skipped",
            "reason": "shared_memory_mode",
            "flush_result": flush_result,
        }]

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "cache_mode": "shared_memory",
            "pending_sequences": len(self._pending_sequences),
            "num_servers": len(self._cache_servers),
            "initialized": self._initialized,
        }

    def shutdown(self) -> None:
        if self._cache_servers:
            for server_info in self._cache_servers:
                try:
                    ray.get(server_info["server"].shutdown.remote(), timeout=5)
                except Exception as e:
                    logger.warning(f"Error shutting down cache server: {e}")
            self._cache_servers.clear()


class SRTAgentLoopManager(AgentLoopManager):
    """AgentLoopManager with SRT suffix tree support.

    Uses SRTvLLMReplica which has load_suffix_snapshot() method.
    Supports both snapshot and shared_memory cache modes.
    """

    def __init__(self, config, worker_group=None, rm_resource_pool=None, cache_mode: str = "snapshot"):
        self._cache_mode = cache_mode

        # Set environment variable for shared memory mode detection by worker processes
        if cache_mode == "shared_memory":
            os.environ["SRT_CACHE_MODE"] = "shared_memory"
            logger.info("SRTAgentLoopManager: Set SRT_CACHE_MODE=shared_memory for worker processes")
        else:
            os.environ.pop("SRT_CACHE_MODE", None)

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
    model_path: str = "Qwen/Qwen3-8B-Base"

    # Hardware
    num_gpus: int = 1
    tp_size: int = 1
    num_workers: int = 1

    # Batch sizes
    batch_size: int = 16  # Samples per batch (b1, b2, b3 each have this many)
    n_samples: int = 1  # Number of responses per prompt (GRPO/DAPO style)

    # Long-tail distribution (key for seeing runahead benefit)
    long_tail_ratio: float = 0.2  # 20% of requests are "long"
    short_max_tokens: int = 512    # Short requests generate up to 64 tokens
    long_max_tokens: int = 8192    # Long requests generate up to 256 tokens
    max_prompt_length: int = 512  # Max input prompt length

    # Runahead settings
    load_threshold: int = 32

    # Suffix decoding settings
    enable_in_flight_updates: bool = True  # Whether to add tokens to suffix tree during speculation

    # Cache mode toggle (NEW: supports snapshot or shared_memory)
    cache_mode: str = "snapshot"  # "snapshot" (default) or "shared_memory"
    shm_port: int = 6378  # Port for shared memory cache server (only for shared_memory mode)

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

    For GRPO/DAPO-style rollouts, each prompt is repeated n_samples times
    to generate multiple responses per prompt. The total batch size becomes
    batch_size * n_samples.

    Args:
        config: Test configuration
        samples: List of DAPOSample objects
        batch_name: Name for logging (b1, b2, b3)
        shuffle_seed: Seed for reproducible shuffle
        sampling_seed_base: Base seed for per-sample vLLM sampling

    Returns:
        DataProto with raw_prompt and max_tokens configured for long-tail
    """
    n_samples = config.n_samples
    num_unique_prompts = config.batch_size
    total_batch_size = num_unique_prompts * n_samples

    # Calculate long-tail distribution based on unique prompts
    num_long_prompts = max(1, int(num_unique_prompts * config.long_tail_ratio))
    num_short_prompts = num_unique_prompts - num_long_prompts

    raw_prompts = []
    reward_models = []
    data_sources = []
    max_tokens_list = []
    prompt_indices = []  # Track which unique prompt each sample belongs to

    # Short requests (most of them) - repeat each n_samples times
    for i in range(num_short_prompts):
        sample = samples[i % len(samples)]
        for _ in range(n_samples):
            raw_prompts.append(sample.prompt)
            reward_models.append(sample.reward_model)
            data_sources.append(sample.data_source)
            max_tokens_list.append(config.short_max_tokens)
            prompt_indices.append(i)

    # Long requests (create the "tail") - repeat each n_samples times
    for i in range(num_long_prompts):
        sample = samples[(num_short_prompts + i) % len(samples)]
        for _ in range(n_samples):
            raw_prompts.append(sample.prompt)
            reward_models.append(sample.reward_model)
            data_sources.append(sample.data_source)
            max_tokens_list.append(config.long_max_tokens)
            prompt_indices.append(num_short_prompts + i)

    # Shuffle to distribute long tasks randomly (important for realistic simulation)
    # Shuffle maintains the grouping of n_samples per prompt for better cache utilization
    if shuffle_seed is not None:
        random.seed(shuffle_seed)
    combined = list(zip(raw_prompts, reward_models, data_sources, max_tokens_list, prompt_indices))
    random.shuffle(combined)
    raw_prompts, reward_models, data_sources, max_tokens_list, prompt_indices = zip(*combined)
    raw_prompts = list(raw_prompts)
    reward_models = list(reward_models)
    data_sources = list(data_sources)
    max_tokens_list = list(max_tokens_list)
    prompt_indices = list(prompt_indices)

    # Build non_tensor_batch
    non_tensor_batch = {
        "raw_prompt": np.array(raw_prompts, dtype=object),
        "agent_name": np.array(["single_turn_agent"] * total_batch_size, dtype=object),
        "data_source": np.array(data_sources, dtype=object),
        "reward_model": np.array(reward_models, dtype=object),
        "max_tokens": np.array(max_tokens_list, dtype=object),
        "prompt_index": np.array(prompt_indices, dtype=object),  # Track which unique prompt
    }

    # Add per-sample seeds for deterministic vLLM sampling
    # Each sample gets a unique seed even for the same prompt (to generate diverse responses)
    if sampling_seed_base is not None:
        sampling_seeds = [sampling_seed_base + i for i in range(total_batch_size)]
        non_tensor_batch["sampling_seed"] = np.array(sampling_seeds, dtype=object)

    num_short_total = num_short_prompts * n_samples
    num_long_total = num_long_prompts * n_samples
    print(f"    {batch_name}: {num_unique_prompts} unique prompts x {n_samples} samples = {total_batch_size} total")
    print(f"           {num_short_prompts} short prompts ({num_short_total} samples @ {config.short_max_tokens} tok) + "
          f"{num_long_prompts} long prompts ({num_long_total} samples @ {config.long_max_tokens} tok)")

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

    For GRPO/DAPO-style rollouts with n_samples > 1:
    - Each batch has batch_size unique prompts
    - Each prompt is repeated n_samples times
    - Total samples per batch = batch_size * n_samples

    Returns:
        (b1, b2, b3) - Three DataProto batches
    """
    print("\n[1] Loading DAPO dataset...")
    dataset = load_dapo_dataset(cache_dir=config.dataset_cache_dir)
    print(f"    Dataset loaded: {len(dataset)} samples available")

    # We need batch_size unique prompts per batch (each will be repeated n_samples times)
    # Sample extra prompts to account for filtering
    sample_multiplier = 3
    unique_prompts_needed = config.batch_size * 3  # 3 batches
    total_needed = unique_prompts_needed * sample_multiplier

    print(f"\n[2] Sampling and filtering samples (max {config.max_prompt_length} tokens)...")
    print(f"    Need {unique_prompts_needed} unique prompts for 3 batches (n_samples={config.n_samples})")

    # Sample all data at once, then filter by length
    all_samples_raw = sample_prompts(dataset, total_needed, seed=config.seed)
    all_samples = filter_prompts_by_length(
        all_samples_raw, tokenizer, config.max_prompt_length, unique_prompts_needed
    )

    if len(all_samples) < unique_prompts_needed:
        print(f"    Warning: Only {len(all_samples)} samples after filtering, "
              f"need {unique_prompts_needed}. Recycling samples.")
        while len(all_samples) < unique_prompts_needed:
            all_samples = all_samples + all_samples[:config.batch_size]

    # Split into 3 batches (each batch gets batch_size unique prompts)
    samples_b1 = all_samples[0:config.batch_size]
    samples_b2 = all_samples[config.batch_size:config.batch_size*2]
    samples_b3 = all_samples[config.batch_size*2:config.batch_size*3]

    print(f"    Split into 3 batches of {config.batch_size} unique prompts each")
    if config.n_samples > 1:
        print(f"    Each prompt repeated {config.n_samples}x -> {config.batch_size * config.n_samples} samples per batch")

    # Build DataProto for each batch with long-tail distribution
    print(f"\n[3] Building DataProto with long-tail distribution "
          f"(ratio={config.long_tail_ratio}, n_samples={config.n_samples})...")

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
    """Metrics for speculation quality measurement.

    These are DELTA metrics computed per-tick (not accumulated).
    """
    # Core speculation metrics (per-tick delta)
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    mean_accepted_length: float = 0.0
    tokens_per_step: float = 0.0

    # Per-position acceptance rates (shows quality at each draft position)
    per_position_rates: dict = field(default_factory=dict)

    # Timing
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
    """Extract speculation metrics from generation output.

    The spec_decode_metrics in meta_info are DELTA metrics computed by
    AgentLoopManager (before/after snapshot subtraction), not accumulated values.
    """
    spec_metrics = gen_output.meta_info.get("spec_decode_metrics", {})

    # Count total tokens from response_mask
    total_tokens = 0
    if gen_output.batch is not None and "response_mask" in gen_output.batch:
        for i in range(len(gen_output)):
            total_tokens += gen_output.batch["response_mask"][i].sum().item()

    # Extract per-position acceptance rates
    per_position_rates = {}
    for key, value in spec_metrics.items():
        if key.startswith("spec_decode/acceptance_rate_pos_"):
            pos = int(key.split("_")[-1])
            per_position_rates[pos] = value

    return SpeculationMetrics(
        # Core metrics (per-tick delta)
        num_drafts=int(spec_metrics.get("spec_decode/num_drafts", 0)),
        num_draft_tokens=int(spec_metrics.get("spec_decode/num_draft_tokens", 0)),
        num_accepted_tokens=int(spec_metrics.get("spec_decode/num_accepted_tokens", 0)),
        acceptance_rate=spec_metrics.get("spec_decode/acceptance_rate", 0.0),
        mean_accepted_length=spec_metrics.get("spec_decode/mean_accepted_length", 0.0),
        tokens_per_step=spec_metrics.get("spec_decode/tokens_per_step", 0.0),
        per_position_rates=per_position_rates,
        # Timing
        generation_time_s=elapsed_time,
        tokens_generated=total_tokens,
        throughput_tokens_per_s=total_tokens / elapsed_time if elapsed_time > 0 else 0,
    )


# =============================================================================
# Test Harness
# =============================================================================

class RunaheadSuffixEffectivenessTest:
    """Test harness for measuring runahead -> suffix cache effectiveness.

    Supports two cache modes:
    - snapshot (default): Uses SuffixTreeManager with snapshot-based loading
    - shared_memory: Uses SharedMemoryCacheManager with SpecRL's zero-copy shared memory
    """

    def __init__(self, config: TestConfig):
        self.config = config
        self.manager: Optional[AgentLoopManager] = None
        self.cache_manager: Optional[CacheManagerInterface] = None  # Unified interface
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
            "recipe.srt.srt_plugin.worker_extension.SuffixTreeWorkerExtension"
        )

        # Configure speculative decoding with suffix method
        # Note: The SRT config patches extend SpeculativeConfig to accept these fields
        # srt_* prefixed keys are extracted by SRTSuffixConfig.extract_from_dict()
        vllm_kwargs.speculative_config = {
            "method": "suffix",
            "num_speculative_tokens": 5,  # Required by vLLM
            "suffix_decoding_max_tree_depth": 64,
            "suffix_decoding_use_parallel": True,  # Use parallel proposer for better hash matching
            "suffix_decoding_enable_in_flight_updates": self.config.enable_in_flight_updates,
            # SRT-specific config (srt_ prefix will be stripped by extract_from_dict)
            "srt_cache_mode": self.config.cache_mode,  # "snapshot" or "shared_memory"
        }

        # Re-enable struct mode
        OmegaConf.set_struct(hydra_config, True)

        # Disable reward model
        if hasattr(hydra_config, "reward_model"):
            hydra_config.reward_model.enable = False

        return hydra_config

    def setup(self):
        """Initialize AgentLoopManager and cache manager based on cache mode."""
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

        # For shared_memory mode, deploy cache server BEFORE creating manager
        # (vLLM workers will try to connect to SuffixCache during initialization)
        if self.config.cache_mode == "shared_memory":
            print("\n[2] Creating SharedMemoryCacheManager (shared_memory mode)...")
            print("    (Must be done before manager creation for SuffixCache connection)")
            self.cache_manager = SharedMemoryCacheManagerTest(self.config, self.tokenizer)
            # Deploy cache server now, before workers try to connect
            self._deploy_shared_memory_cache_server()
            print(f"    Cache server port: {self.config.shm_port}")

        # Create SRTAgentLoopManager (with SRTvLLMReplica for suffix snapshot loading)
        print("\n[3] Creating SRTAgentLoopManager...")
        print(f"    Cache mode: {self.config.cache_mode}")
        hydra_config = self._compose_hydra_config()
        self.manager = SRTAgentLoopManager(hydra_config, cache_mode=self.config.cache_mode)
        print(f"    Model: {self.config.model_path}")
        print(f"    GPUs: {self.config.num_gpus}, TP: {self.config.tp_size}")

        # For shared_memory mode, finalize cache manager initialization
        if self.config.cache_mode == "shared_memory":
            # Manager is created, no additional setup needed
            pass
        else:
            # Snapshot mode: create cache manager
            print("\n[4] Creating SuffixTreeManager (snapshot mode)...")
            self.cache_manager = SnapshotCacheManager(self.tokenizer)
            print("    max_tree_depth: 64, hash_token_count: 128")

    def _deploy_shared_memory_cache_server(self):
        """Deploy cache server for shared memory mode.

        This must be called BEFORE creating SRTAgentLoopManager because
        vLLM workers try to connect to SuffixCache during initialization.
        """
        try:
            from recipe.specRL.cache_manager import CacheWorker
            from specrl.cache_updater import SuffixCacheUpdater
        except ImportError as e:
            logger.error(f"Failed to import SpecRL modules: {e}")
            raise

        port = self.config.shm_port

        # Deploy cache server
        server = CacheWorker.remote(port=port)

        # Use localhost for single-node testing
        ip = "127.0.0.1"

        self.cache_manager._cache_servers.append({
            "server": server,
            "ip": ip,
            "port": port,
        })

        # Wait for server to start
        import time
        time.sleep(2)  # Give server time to initialize shared memory

        # Create updater with server addresses
        addresses = [f"{s['ip']}:{s['port']}" for s in self.cache_manager._cache_servers]
        self.cache_manager._cache_updater = SuffixCacheUpdater(server_addresses=addresses)
        self.cache_manager._initialized = True

        logger.info(f"Deployed cache server on {ip}:{port}")
        print(f"    Cache server deployed on {ip}:{port}")

    def prepare_batches(self) -> tuple[DataProto, DataProto, DataProto]:
        """Prepare 3 batches (b1, b2, b3) from DAPO dataset with long-tail."""
        return prepare_three_batches(self.config, self.tokenizer)

    def run_tick1(self, b1: DataProto, b2: DataProto) -> TestResult:
        """
        Tick 1: rollout(primary=b1, secondary=b2)
        Feed b2 runahead results to cache (mode-agnostic via CacheManagerInterface).
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

        # Feed secondary outputs to cache (mode-agnostic)
        secondary_tokens_added = 0
        for out in result.secondary_outputs:
            if out.status in ("completed", "aborted") and out.output and out.prompt_ids:
                self.cache_manager.add_sequence(
                    prompt_tokens=list(out.prompt_ids),
                    response_tokens=list(out.output.token_ids),
                )
                secondary_tokens_added += len(out.output.token_ids)

        # For shared memory mode: flush sequences to cache servers now
        flush_result = {}
        if self.config.cache_mode == "shared_memory" and hasattr(self.cache_manager, 'flush_pending_sequences'):
            flush_result = self.cache_manager.flush_pending_sequences()

        metrics = extract_spec_metrics(result.primary_outputs, elapsed)

        return TestResult(
            tick="tick1",
            batch_name="b1",
            role="primary",
            metrics=metrics,
            suffix_cache_stats={
                "cache_mode": self.config.cache_mode,
                "secondary_outputs_count": len(result.secondary_outputs),
                "secondary_completed": result.metrics.secondary_completed,
                "secondary_aborted": result.metrics.secondary_aborted,
                "secondary_rejected": result.metrics.secondary_rejected,
                "secondary_started": result.metrics.secondary_started,
                "secondary_tokens_added": secondary_tokens_added,
                "primary_time_s": result.metrics.primary_time_s,
                **flush_result,
            }
        )

    def run_tick2(self, b2: DataProto, b3: DataProto) -> TestResult:
        """
        Tick 2: rollout(primary=b2, secondary=b3)
        b2 should benefit from suffix cache populated in tick1.
        """
        # Push to workers (mode handles internally)
        push_stats = {}
        if self.config.cache_mode == "shared_memory":
            # Shared memory mode: workers access cache directly via shared memory
            # The flush happened in tick1, so cache is already populated
            print("    Shared memory mode: Workers access cache directly (no snapshot push needed)")
            push_stats["push_method"] = "shared_memory"
        else:
            # Snapshot mode: Push suffix snapshot to workers
            results = self.cache_manager.push_to_workers(self.manager)
            snapshot_loaded = len(results) > 0 and any(r.get("status") == "success" for r in results if isinstance(r, dict))
            if isinstance(self.cache_manager, SnapshotCacheManager):
                snapshots, _ = self.cache_manager.get_snapshot()
                print(f"    Pushing {len(snapshots)} suffix trees to vLLM servers...")
            print(f"    Push results: {results}")
            push_stats["push_method"] = "snapshot"
            push_stats["snapshot_loaded"] = snapshot_loaded

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

        cache_stats = self.cache_manager.get_metrics()
        cache_stats.update(push_stats)

        return TestResult(
            tick="tick2",
            batch_name="b2",
            role="primary",
            metrics=metrics,
            suffix_cache_stats=cache_stats,
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
        print(f"Cache mode: {self.config.cache_mode}")
        print("="*60)

        try:
            # Run baseline FIRST (before cache is populated)
            print("\n" + "-"*40)
            print("BASELINE: Running b2 without suffix cache")
            print("-"*40)
            baseline_result = self.run_baseline(b2)
            self.results.append(baseline_result)
            self._print_tick_metrics("baseline", baseline_result)

            # Run tick 1: populates cache with b2 patterns
            print("\n" + "-"*40)
            print(f"TICK 1: primary=b1, secondary=b2 (populates cache - {self.config.cache_mode} mode)")
            print("-"*40)
            tick1_result = self.run_tick1(b1, b2)
            self.results.append(tick1_result)
            self._print_tick_metrics("tick1", tick1_result)
            print(f"    --- Runahead Stats ---")
            print(f"    Cache mode: {tick1_result.suffix_cache_stats.get('cache_mode', 'unknown')}")
            print(f"    Primary time: {tick1_result.suffix_cache_stats.get('primary_time_s', 0):.2f}s")
            print(f"    Secondary started: {tick1_result.suffix_cache_stats.get('secondary_started', 0)}")
            print(f"    Secondary completed: {tick1_result.suffix_cache_stats.get('secondary_completed', 0)}")
            print(f"    Secondary aborted: {tick1_result.suffix_cache_stats.get('secondary_aborted', 0)}")
            print(f"    Secondary rejected: {tick1_result.suffix_cache_stats.get('secondary_rejected', 0)}")
            print(f"    Secondary tokens added to cache: {tick1_result.suffix_cache_stats.get('secondary_tokens_added', 0)}")
            if self.config.cache_mode == "shared_memory":
                print(f"    Sequences flushed to shm: {tick1_result.suffix_cache_stats.get('shm_cache/sequences_flushed', 0)}")

            # Run tick 2: b2 should benefit from cache
            print("\n" + "-"*40)
            print(f"TICK 2: primary=b2, secondary=b3 (uses cache - {self.config.cache_mode} mode)")
            print("-"*40)
            tick2_result = self.run_tick2(b2, b3)
            self.results.append(tick2_result)
            self._print_tick_metrics("tick2", tick2_result)

            # Compute improvement
            improvement = self._compute_improvement(baseline_result, tick2_result)

            return {
                "config": asdict(self.config),
                "results": [asdict(r) for r in self.results],
                "improvement": improvement,
            }
        finally:
            # Cleanup cache manager resources
            if self.cache_manager is not None:
                self.cache_manager.shutdown()

    def _print_tick_metrics(self, tick_name: str, result: TestResult):
        """Print detailed per-tick speculation metrics."""
        m = result.metrics
        print(f"    [DELTA METRICS for {tick_name}]")
        print(f"    Generation time: {m.generation_time_s:.2f}s")
        print(f"    Tokens generated: {m.tokens_generated}")
        print(f"    Throughput: {m.throughput_tokens_per_s:.1f} tok/s")

        if m.num_drafts > 0:
            print(f"    --- Speculation (this tick only) ---")
            print(f"    Num drafts: {m.num_drafts}")
            print(f"    Draft tokens: {m.num_draft_tokens}")
            print(f"    Accepted tokens: {m.num_accepted_tokens}")
            print(f"    Acceptance rate: {m.acceptance_rate:.2%}")
            print(f"    Mean accepted length: {m.mean_accepted_length:.2f}")
            print(f"    Tokens per step: {m.tokens_per_step:.2f}")

            if m.per_position_rates:
                pos_str = ", ".join(
                    f"pos{p}={r:.1%}"
                    for p, r in sorted(m.per_position_rates.items())[:5]
                )
                print(f"    Per-position rates: {pos_str}")
        else:
            print(f"    --- No speculation activity ---")

    def _compute_improvement(self, baseline: TestResult, tick2: TestResult) -> dict:
        """Compute improvement metrics."""
        return {
            # Acceptance rate comparison
            "acceptance_rate_baseline": baseline.metrics.acceptance_rate,
            "acceptance_rate_with_cache": tick2.metrics.acceptance_rate,
            "acceptance_rate_improvement": (
                tick2.metrics.acceptance_rate - baseline.metrics.acceptance_rate
            ),
            # Mean accepted length comparison
            "mean_accepted_length_baseline": baseline.metrics.mean_accepted_length,
            "mean_accepted_length_with_cache": tick2.metrics.mean_accepted_length,
            "mean_accepted_length_improvement": (
                tick2.metrics.mean_accepted_length - baseline.metrics.mean_accepted_length
            ),
            # Tokens per step comparison
            "tokens_per_step_baseline": baseline.metrics.tokens_per_step,
            "tokens_per_step_with_cache": tick2.metrics.tokens_per_step,
            # Throughput comparison
            "throughput_baseline": baseline.metrics.throughput_tokens_per_s,
            "throughput_with_cache": tick2.metrics.throughput_tokens_per_s,
            "throughput_improvement_pct": (
                (tick2.metrics.throughput_tokens_per_s - baseline.metrics.throughput_tokens_per_s)
                / baseline.metrics.throughput_tokens_per_s * 100
                if baseline.metrics.throughput_tokens_per_s > 0 else 0
            ),
            # Raw counts for verification
            "num_drafts_baseline": baseline.metrics.num_drafts,
            "num_drafts_with_cache": tick2.metrics.num_drafts,
            "num_draft_tokens_baseline": baseline.metrics.num_draft_tokens,
            "num_draft_tokens_with_cache": tick2.metrics.num_draft_tokens,
            "num_accepted_tokens_baseline": baseline.metrics.num_accepted_tokens,
            "num_accepted_tokens_with_cache": tick2.metrics.num_accepted_tokens,
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
                        help="Number of unique prompts per batch (b1, b2, b3)")
    parser.add_argument("--n-samples", type=int, default=1,
                        help="Number of responses per prompt (GRPO/DAPO style). "
                             "Total samples per batch = batch_size * n_samples")
    parser.add_argument("--long-tail-ratio", type=float, default=0.2,
                        help="Fraction of requests with long max_tokens")
    parser.add_argument("--short-max-tokens", type=int, default=64,
                        help="max_tokens for short requests")
    parser.add_argument("--long-max-tokens", type=int, default=256,
                        help="max_tokens for long requests (creates bubbles)")
    parser.add_argument("--load-threshold", type=int, default=32,
                        help="Runahead admission threshold")
    parser.add_argument("--enable-in-flight-updates", type=str, default="true",
                        choices=["true", "false"],
                        help="Whether to add tokens to suffix tree during speculation (default: true)")
    # Cache mode toggle
    parser.add_argument("--cache-mode", type=str, default="snapshot",
                        choices=["snapshot", "shared_memory"],
                        help="Cache mode: snapshot (default) or shared_memory. "
                             "snapshot uses SuffixTreeManager with snapshot-based loading. "
                             "shared_memory uses SpecRL's zero-copy shared memory.")
    parser.add_argument("--shm-port", type=int, default=6378,
                        help="Port for shared memory cache server (only for shared_memory mode)")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="runahead_suffix_test_results.json")
    args = parser.parse_args()

    config = TestConfig(
        model_path=args.model,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        long_tail_ratio=args.long_tail_ratio,
        short_max_tokens=args.short_max_tokens,
        long_max_tokens=args.long_max_tokens,
        load_threshold=args.load_threshold,
        enable_in_flight_updates=args.enable_in_flight_updates.lower() == "true",
        cache_mode=args.cache_mode,
        shm_port=args.shm_port,
        seed=args.seed,
    )

    print("="*60)
    print("RUNAHEAD -> SUFFIX CACHE EFFECTIVENESS TEST")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Model:           {config.model_path}")
    print(f"  Batch size:      {config.batch_size} unique prompts")
    print(f"  N-samples:       {config.n_samples} responses per prompt")
    print(f"  Total per batch: {config.batch_size * config.n_samples} samples")
    print(f"  Long-tail ratio: {config.long_tail_ratio}")
    print(f"  Short tokens:    {config.short_max_tokens}")
    print(f"  Long tokens:     {config.long_max_tokens}")
    print(f"  Load threshold:  {config.load_threshold}")
    print(f"  In-flight updates: {config.enable_in_flight_updates}")
    print(f"  Cache mode:      {config.cache_mode}")
    if config.cache_mode == "shared_memory":
        print(f"  SHM port:        {config.shm_port}")

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

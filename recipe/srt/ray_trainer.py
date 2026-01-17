# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
SRT (Speculative Rollout with Tree-Structured Cache) PPO Trainer.

This trainer extends RayPPOTrainer with suffix tree-based speculative decoding
support for accelerating on-policy RL rollout.
"""

import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from recipe.srt.suffix_tree_manager import SuffixTreeManager, SuffixTreeManagerConfig
from verl import DataProto

# Type hint for SharedMemoryCacheManager (imported conditionally)
SharedMemoryCacheManager = None
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayPPOTrainer,
    ResourcePoolManager,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip


class SRTRayPPOTrainer(RayPPOTrainer):
    """
    PPO Trainer with SRT (Speculative Rollout with Tree-Structured Cache) support.

    This trainer extends RayPPOTrainer to integrate suffix tree-based speculative
    decoding for accelerating rollout generation. It:

    1. Collects Q/A patterns from rollouts into a SuffixTreeManager
    2. Creates snapshots and pushes them to vLLM workers before generation
    3. Workers use the suffix trees for speculative decoding during generation

    The suffix trees learn from historical responses, so repeated prompts (common
    in RL training) benefit from cached token patterns.
    """

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict,
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        # Inject vLLM engine_kwargs for suffix decoding if SRT is enabled
        self._inject_srt_engine_kwargs(config)

        # Store references for SharedMemoryCacheManager initialization
        self._role_worker_mapping = role_worker_mapping
        self._resource_pool_manager = resource_pool_manager

        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn,
            val_reward_fn,
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler,
            device_name,
        )

        # Initialize cache manager based on mode
        cache_mode = self._srt_config.get("srt_cache_mode", "snapshot")

        if cache_mode == "shared_memory" and self._srt_config.get("enable_srt", False):
            # Shared memory mode: Initialize SharedMemoryCacheManager
            # (actual server deployment happens in init_workers after workers are up)
            from recipe.srt.shared_memory_cache_manager import SharedMemoryCacheManager

            shm_config = self._srt_config["srt_shared_memory"]
            self.shm_cache_manager = SharedMemoryCacheManager(
                config=self._srt_config,
                role_worker_mapping=self._role_worker_mapping,
                resource_pool_manager=self._resource_pool_manager,
                tokenizer=self.tokenizer,
                port=shm_config["port"],
                memory_size_gb=shm_config.get("memory_size_gb", 100),
                shared_memory_name=shm_config.get("shared_memory_name", ""),
            )
            # Create disabled placeholder for API compatibility
            self.suffix_tree_manager = SuffixTreeManager(
                SuffixTreeManagerConfig(enable=False), self.tokenizer
            )
            print("SRT: Using SharedMemoryCacheManager (shared_memory mode)")
        else:
            # Snapshot mode (default): Use SuffixTreeManager
            self.suffix_tree_manager = self._init_suffix_tree_manager()
            self.shm_cache_manager = None

    def _inject_srt_engine_kwargs(self, config):
        """Inject vLLM engine_kwargs for suffix decoding when SRT is enabled.

        This automatically configures vLLM to use the appropriate speculative
        decoding setup based on srt_cache_mode:
        - "snapshot" (default): Uses ParallelSuffixDecodingProposer with worker_extension_cls
        - "shared_memory": Uses SpecRL's GPUModelRunnerPatch for zero-copy shared memory

        Also stores SRT config values in self._srt_config and removes them from
        rollout_config to avoid RolloutConfig schema validation errors.
        """
        import os

        from omegaconf import OmegaConf, open_dict

        rollout_config = config.actor_rollout_ref.rollout
        enable_srt = rollout_config.get("enable_srt", False)

        # Get cache mode (snapshot or shared_memory)
        cache_mode = rollout_config.get("srt_cache_mode", "snapshot")

        # Get shared memory config (nested dict)
        shm_config = rollout_config.get("srt_shared_memory", {})
        if not isinstance(shm_config, dict):
            shm_config = OmegaConf.to_container(shm_config, resolve=True) if shm_config else {}

        # Store SRT config for later use (before removing from rollout_config)
        self._srt_config = {
            "enable_srt": enable_srt,
            "srt_cache_mode": cache_mode,
            "srt_max_tree_depth": rollout_config.get("srt_max_tree_depth", 64),
            "srt_hash_token_count": rollout_config.get("srt_hash_token_count", 128),
            "srt_num_speculative_tokens": rollout_config.get("srt_num_speculative_tokens", 24),
            "srt_shared_memory": {
                "port": shm_config.get("port", 6378),
                "memory_size_gb": shm_config.get("memory_size_gb", 100),
                "shared_memory_name": shm_config.get("shared_memory_name", ""),  # Empty = default "SUFFIX_CACHE"
                "spec_start_len": shm_config.get("spec_start_len", 2),  # Initial/min speculation length
                "spec_max_len": shm_config.get("spec_max_len", 16),  # Maximum speculation length
            },
        }

        # Remove SRT fields from rollout_config to avoid RolloutConfig schema errors
        srt_fields = [
            "enable_srt",
            "srt_cache_mode",
            "srt_max_tree_depth",
            "srt_hash_token_count",
            "srt_num_speculative_tokens",
            "srt_shared_memory",
        ]
        with open_dict(config):
            for field in srt_fields:
                if field in rollout_config:
                    del rollout_config[field]

        if not enable_srt:
            return

        # Get SRT config values
        max_tree_depth = self._srt_config["srt_max_tree_depth"]
        num_speculative_tokens = self._srt_config["srt_num_speculative_tokens"]

        # Use open_dict to allow adding new keys to the config
        with open_dict(config):
            # Ensure engine_kwargs.vllm exists
            engine_kwargs = rollout_config.get("engine_kwargs")
            if engine_kwargs is None:
                rollout_config.engine_kwargs = OmegaConf.create({})
                engine_kwargs = rollout_config.engine_kwargs

            vllm_kwargs = engine_kwargs.get("vllm")
            if vllm_kwargs is None:
                engine_kwargs.vllm = OmegaConf.create({})
                vllm_kwargs = engine_kwargs.vllm

            if cache_mode == "shared_memory":
                # Shared memory mode: Use SpecRL's GPUModelRunnerPatch
                # GPUModelRunner will be patched to use SuffixCache directly
                shm_config = self._srt_config["srt_shared_memory"]

                # Set env var for cache mode detection (used by runner_patches.py)
                os.environ["SRT_CACHE_MODE"] = "shared_memory"

                # Pass all SRT config through speculative_config dict
                # Workers read these in runner_patches.py to configure SuffixCache
                # Fields with srt_ prefix are extracted before vLLM validation
                speculative_config = {
                    "method": "suffix",
                    "num_speculative_tokens": num_speculative_tokens,
                    # SRT-specific params (extracted by SRTSuffixConfig.extract_from_dict)
                    "srt_max_tree_depth": max_tree_depth,
                    "srt_max_spec_factor": 1.0,
                    "srt_min_token_prob": 0.1,
                    "srt_enable_in_flight_updates": False,  # Disabled for shared_memory
                    "srt_cache_mode": "shared_memory",
                    # Shared memory specific params
                    "srt_shared_memory_name": shm_config.get("shared_memory_name", ""),
                    "srt_spec_start_len": shm_config.get("spec_start_len", 2),
                    "srt_spec_max_len": shm_config.get("spec_max_len", 16),
                }
                vllm_kwargs.speculative_config = speculative_config

                print(
                    f"SRT: Configured for shared memory mode "
                    f"(port={shm_config.get('port', 'default')}, "
                    f"name={shm_config.get('shared_memory_name') or 'SUFFIX_CACHE'}, "
                    f"spec_len={shm_config.get('spec_start_len', 2)}-{shm_config.get('spec_max_len', 16)})"
                )

            else:
                # Snapshot mode (default): Use worker_extension_cls and speculative_config
                # Use vLLM's --speculative-config JSON argument format
                # SRT params use srt_ prefix and are extracted by arg_utils_patches
                # before SpeculativeConfig validation. See config.py:SRTSuffixConfig
                speculative_config = {
                    "method": "suffix",
                    "num_speculative_tokens": num_speculative_tokens,
                    # SRT-specific params (extracted by SRTSuffixConfig.extract_from_dict)
                    "srt_max_tree_depth": max_tree_depth,
                    "srt_max_spec_factor": 1.0,
                    "srt_min_token_prob": 0.1,
                    "srt_enable_in_flight_updates": True,
                }

                # Merge with existing speculative_config if present
                existing_spec_config = vllm_kwargs.get("speculative_config")
                if existing_spec_config is not None:
                    if isinstance(existing_spec_config, dict):
                        speculative_config.update(existing_spec_config)

                vllm_kwargs.speculative_config = speculative_config

                # Add worker_extension_cls to inject load_suffix_snapshot method into workers
                # This enables collective_rpc calls from the server to load suffix tree snapshots
                vllm_kwargs.worker_extension_cls = (
                    "recipe.srt.srt_plugin.worker_extension.SuffixTreeWorkerExtension"
                )

                print(
                    f"SRT: Configured for snapshot mode with speculative_config: {speculative_config}, "
                    f"worker_extension_cls: recipe.srt.srt_plugin.worker_extension.SuffixTreeWorkerExtension"
                )

    def init_workers(self):
        """Initialize workers with cache infrastructure.

        Overrides parent to initialize shared memory cache servers after
        workers are up (for shared_memory mode).
        """
        super().init_workers()

        # Initialize shared memory cache servers after workers are ready
        if self.shm_cache_manager is not None:
            print("SRT: Initializing shared memory cache servers...")
            self.shm_cache_manager.initialize()

    def _init_suffix_tree_manager(self) -> SuffixTreeManager:
        """Initialize SuffixTreeManager for speculative decoding.

        Reads config from self._srt_config (populated by _inject_srt_engine_kwargs)
        and creates a SuffixTreeManager if SRT is enabled.
        """
        enable_srt = self._srt_config.get("enable_srt", False)

        if enable_srt:
            manager_config = SuffixTreeManagerConfig(
                enable=True,
                max_tree_depth=self._srt_config.get("srt_max_tree_depth", 64),
                hash_token_count=self._srt_config.get("srt_hash_token_count", 128),
            )
            print(
                f"SRT: Initializing SuffixTreeManager with max_tree_depth={manager_config.max_tree_depth}, "
                f"hash_token_count={manager_config.hash_token_count}"
            )
            return SuffixTreeManager(manager_config, self.tokenizer)

        # Return disabled manager if SRT not configured
        return SuffixTreeManager(SuffixTreeManagerConfig(enable=False), self.tokenizer)

    def _push_suffix_snapshots(self, gen_batch_output: DataProto, metrics: dict, timing_raw: dict):
        """Push suffix tree snapshots to workers before rollout.

        This method extracts batch hashes, creates selective snapshots, and
        distributes them to all rollout replicas.

        Note: In shared_memory mode, this is a no-op because workers access
        the cache directly via shared memory (populated by previous updates).

        Args:
            gen_batch_output: Batch to be generated (contains prompts)
            metrics: Metrics dict to update with transfer stats
            timing_raw: Timing dict for profiling
        """
        # Shared memory mode: no snapshot pushing needed
        if self.shm_cache_manager is not None:
            return

        if not self.suffix_tree_manager.enabled:
            return

        with marked_timer("push_suffix_snapshot", timing_raw):
            # Extract batch hashes for selective snapshot
            input_ids = gen_batch_output.batch.get("input_ids")
            attention_mask = gen_batch_output.batch.get("attention_mask")

            if input_ids is not None and attention_mask is not None:
                # Convert to numpy if tensors
                if hasattr(input_ids, "cpu"):
                    input_ids = input_ids.cpu().numpy()
                if hasattr(attention_mask, "cpu"):
                    attention_mask = attention_mask.cpu().numpy()

                batch_hashes = self.suffix_tree_manager.extract_batch_hashes(
                    input_ids, attention_mask
                )

                if batch_hashes:
                    # Use selective snapshot (only trees for this batch)
                    snapshots, hash_mapping = self.suffix_tree_manager.get_selective_snapshot(
                        hashes=batch_hashes
                    )
                else:
                    # Fallback to full snapshot if no hashes extracted
                    snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()
            else:
                # Fallback to full snapshot if batch data not available
                snapshots, hash_mapping = self.suffix_tree_manager.get_snapshot()

            # Push to workers
            if snapshots:
                self._load_suffix_snapshot_to_workers(snapshots, hash_mapping)

            # Log transfer metrics
            metrics["suffix_tree/trees_transferred"] = len(snapshots)
            metrics["suffix_tree/transfer_bytes"] = sum(len(s[1]) for s in snapshots)

    def _load_suffix_snapshot_to_workers(
        self,
        snapshots: list[tuple[int, bytes]],
        hash_mapping: dict[str, int],
    ):
        """Load suffix tree snapshots to all rollout workers.

        Handles both sync (actor_rollout_wg) and async (async_rollout_manager) modes.

        Args:
            snapshots: List of (tree_idx, snapshot_bytes) tuples
            hash_mapping: Dict mapping prompt_hash -> tree_idx
        """
        if not snapshots:
            return

        if self.async_rollout_mode and hasattr(self, "async_rollout_manager"):
            # Server mode: load through rollout replicas (sync API)
            for replica in self.async_rollout_manager.rollout_replicas:
                replica.load_suffix_snapshot(snapshots, hash_mapping)
        else:
            # Sync mode: load through worker group (if method exists)
            if hasattr(self.actor_rollout_wg, "load_suffix_snapshot"):
                self.actor_rollout_wg.load_suffix_snapshot(snapshots, hash_mapping)

    def _update_suffix_trees(self, batch: DataProto, metrics: dict, timing_raw: dict):
        """Update suffix trees with rollout results.

        Args:
            batch: DataProto containing prompts and responses
            metrics: Metrics dict to update with tree stats
            timing_raw: Timing dict for profiling
        """
        # Shared memory mode: send async gRPC updates
        if self.shm_cache_manager is not None:
            with marked_timer("update_cache_shm", timing_raw):
                responses_per_prompt = self.config.actor_rollout_ref.rollout.n
                stats = self.shm_cache_manager.update_from_rollout(batch, responses_per_prompt)
                metrics.update(stats)
            return

        if not self.suffix_tree_manager.enabled:
            return

        with marked_timer("update_suffix_tree", timing_raw):
            suffix_stats = self.suffix_tree_manager.update_from_rollout(batch)
            metrics.update(suffix_stats)

    def _update_suffix_trees_from_secondary(
        self,
        secondary_outputs: list,
        metrics: dict,
        timing_raw: dict,
    ) -> None:
        """Update suffix trees with secondary (runahead) outputs.

        Only processes completed and aborted outputs (which have partial responses).
        Rejected outputs have no tokens to add.

        This is the key SRT optimization - runahead outputs from tick N populate
        the cache for when batch N becomes primary in tick N+1.

        Args:
            secondary_outputs: List of SecondaryOutput from runahead.
            metrics: Metrics dict to update.
            timing_raw: Timing dict for profiling.
        """
        # Shared memory mode: use SharedMemoryCacheManager
        if self.shm_cache_manager is not None:
            with marked_timer("update_cache_shm_secondary", timing_raw):
                responses_per_prompt = self.config.actor_rollout_ref.rollout.n
                stats = self.shm_cache_manager.update_from_secondary(
                    secondary_outputs, responses_per_prompt
                )
                metrics.update(stats)
            return

        if not self.suffix_tree_manager.enabled:
            return

        # Filter to outputs with actual tokens
        usable_outputs = [
            out for out in secondary_outputs
            if out.status in ("completed", "aborted")
            and out.output is not None
            and len(out.output.token_ids) > 0
            and len(out.prompt_ids) > 0
        ]

        if not usable_outputs:
            return

        with marked_timer("update_suffix_tree_secondary", timing_raw):
            tokens_added = 0
            for out in usable_outputs:
                prompt_tokens = out.prompt_ids
                response_tokens = out.output.token_ids

                # Add to suffix tree
                self.suffix_tree_manager.add_sequence(
                    prompt_tokens=prompt_tokens,
                    response_tokens=response_tokens,
                )
                tokens_added += len(response_tokens)

            metrics["suffix_tree/secondary_outputs_processed"] = len(usable_outputs)
            metrics["suffix_tree/secondary_tokens_added"] = tokens_added

    def _validate(self):
        """Override _validate to update suffix trees with validation data."""
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # SRT: Update suffix trees with validation Q/A patterns
            if self.suffix_tree_manager.enabled:
                self.suffix_tree_manager.update_from_rollout(test_batch)

            test_batch.meta_info["validate"] = True

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _save_checkpoint(self):
        """Override _save_checkpoint to include suffix tree state."""
        super()._save_checkpoint()

        # Save suffix tree state
        if self.suffix_tree_manager.enabled:
            local_global_step_folder = os.path.join(
                self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
            )
            suffix_tree_path = os.path.join(local_global_step_folder, "suffix_tree")
            self.suffix_tree_manager.save(suffix_tree_path)

    def _load_checkpoint(self):
        """Override _load_checkpoint to restore suffix tree state."""
        super()._load_checkpoint()

        # Load suffix tree state
        if self.suffix_tree_manager.enabled:
            global_step_folder = os.path.join(
                self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
            )
            suffix_tree_path = os.path.join(global_step_folder, "suffix_tree")
            if os.path.exists(suffix_tree_path):
                if self.suffix_tree_manager.load(suffix_tree_path):
                    print(f"SRT: Loaded suffix tree state from {suffix_tree_path}")
            else:
                print(f"SRT: No suffix tree state found at {suffix_tree_path}, starting with empty trees")

    def fit(self):
        """Training entry point - dispatches to appropriate implementation."""
        if self._should_use_runahead():
            return self._fit_runahead()
        return self._fit_standard()

    def _should_use_runahead(self) -> bool:
        """Check if runahead mode should be used.

        Runahead requires:
        1. async_rollout_mode (server mode) for AgentLoopManager
        2. enable_runahead config flag set to True
        """
        return (
            self.async_rollout_mode
            and self.config.trainer.get("enable_runahead", False)
        )

    def _fit_standard(self):
        """
        The standard training loop of PPO with SRT integration.

        This extends the base fit() method to:
        1. Push suffix tree snapshots before rollout generation
        2. Update suffix trees after rollout generation
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # SRT: Push suffix tree snapshots to workers before rollout
                    self._push_suffix_snapshots(gen_batch_output, metrics, timing_raw)

                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                        # Extract spec decode metrics for wandb logging
                        if "spec_decode_metrics" in gen_batch_output.meta_info:
                            metrics.update(gen_batch_output.meta_info["spec_decode_metrics"])
                            gen_batch_output.meta_info.pop("spec_decode_metrics", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # SRT: Update suffix trees with rollout results
                    self._update_suffix_trees(batch, metrics, timing_raw)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        from verl.trainer.ppo.utils import Role

                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        # compute advantages
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check ESI expiration
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # Add suffix tree / shared memory cache metrics
                if self.shm_cache_manager is not None:
                    metrics.update(self.shm_cache_manager.get_metrics())
                elif self.suffix_tree_manager.enabled:
                    metrics.update(self.suffix_tree_manager.get_metrics())

                # this is experimental and may be changed/removed in the future
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    # Cleanup shared memory cache manager
                    if self.shm_cache_manager is not None:
                        self.shm_cache_manager.shutdown()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental
                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)

    def _fit_runahead(self):
        """
        Training loop with runahead sliding window.

        Uses generate_sequences_with_runahead() to overlap:
        - Primary batch (current step): full training, must complete
        - Secondary batch (next step): opportunistic pre-generation during GPU bubbles

        The sliding window pattern:
        - Tick 1: batch_1 (primary) + batch_2 (secondary)
        - Tick 2: batch_2 (primary) + batch_3 (secondary)
        - ...
        """
        from verl.experimental.agent_loop.runahead import RunaheadConfig
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # Build runahead config from trainer settings
        runahead_settings = self.config.trainer.get("runahead", {})
        runahead_config = RunaheadConfig(
            enabled=True,
            load_threshold=runahead_settings.get("load_threshold", 32),
            max_queue_size=runahead_settings.get("max_queue_size", 256),
            admit_loop_poll_s=runahead_settings.get("admit_loop_poll_s", 0.05),
            use_kv_cache_admission=runahead_settings.get("use_kv_cache_admission", False),
            kv_cache_threshold=runahead_settings.get("kv_cache_threshold", 0.85),
            abort_grace_s=runahead_settings.get("abort_grace_s", 1.0),
            wait_for_primary_start=runahead_settings.get("wait_for_primary_start", True),
            primary_priority=runahead_settings.get("primary_priority", 0),
            secondary_priority=runahead_settings.get("secondary_priority", 10),
        )

        print(f"SRT Runahead: Enabled with config: load_threshold={runahead_config.load_threshold}, "
              f"secondary_priority={runahead_config.secondary_priority}")

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training (Runahead)")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            # === SLIDING WINDOW ITERATION ===
            batch_iter = iter(self.train_dataloader)
            next_batch_dict = next(batch_iter, None)

            while next_batch_dict is not None:
                # Promote: secondary -> primary
                current_batch_dict = next_batch_dict
                next_batch_dict = next(batch_iter, None)

                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # === PREPARE PRIMARY BATCH ===
                batch: DataProto = DataProto.from_single_dict(current_batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                primary_prompts = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                # === PREPARE SECONDARY BATCH (if available) ===
                secondary_prompts = None
                if next_batch_dict is not None:
                    next_batch = DataProto.from_single_dict(next_batch_dict)
                    next_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                    next_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(next_batch.batch))], dtype=object
                    )
                    next_gen_batch = self._get_gen_batch(next_batch)
                    secondary_prompts = next_gen_batch.repeat(
                        repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # SRT: Push suffix tree snapshots to workers before rollout
                    self._push_suffix_snapshots(primary_prompts, metrics, timing_raw)

                    # === GENERATE WITH RUNAHEAD ===
                    with marked_timer("gen", timing_raw, color="red"):
                        if secondary_prompts is not None:
                            result = self.async_rollout_manager.generate_sequences_with_runahead(
                                primary_prompts, secondary_prompts, runahead_config
                            )
                            gen_batch_output = result.primary_outputs

                            # Log runahead metrics
                            runahead_metrics = result.metrics
                            metrics["runahead/primary_time_s"] = runahead_metrics.primary_time_s
                            metrics["runahead/secondary_started"] = runahead_metrics.secondary_started
                            metrics["runahead/secondary_completed"] = runahead_metrics.secondary_completed
                            metrics["runahead/secondary_aborted"] = runahead_metrics.secondary_aborted
                            metrics["runahead/secondary_rejected"] = runahead_metrics.secondary_rejected

                            # Compute utilization ratio
                            total_secondary = (runahead_metrics.secondary_completed +
                                               runahead_metrics.secondary_aborted +
                                               runahead_metrics.secondary_rejected)
                            if total_secondary > 0:
                                metrics["runahead/completion_rate"] = (
                                    runahead_metrics.secondary_completed / total_secondary
                                )

                            # SRT: Update suffix trees with secondary outputs
                            # This populates the cache for the NEXT tick when this batch becomes primary
                            self._update_suffix_trees_from_secondary(
                                result.secondary_outputs, metrics, timing_raw
                            )
                        else:
                            # Last batch - no secondary available
                            gen_batch_output = self.async_rollout_manager.generate_sequences(primary_prompts)

                        timing_raw.update(gen_batch_output.meta_info.get("timing", {}))
                        gen_batch_output.meta_info.pop("timing", None)

                        # Extract spec decode metrics for wandb logging
                        if "spec_decode_metrics" in gen_batch_output.meta_info:
                            metrics.update(gen_batch_output.meta_info["spec_decode_metrics"])
                            gen_batch_output.meta_info.pop("spec_decode_metrics", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    # SRT: Update suffix trees with rollout results
                    self._update_suffix_trees(batch, metrics, timing_raw)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        from verl.trainer.ppo.utils import Role

                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            metrics.update(is_metrics)

                        # compute advantages
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            rollout_config = self.config.actor_rollout_ref.rollout
                            batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                            batch.meta_info["temperature"] = rollout_config.temperature
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check ESI expiration
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # Add suffix tree / shared memory cache metrics
                if self.shm_cache_manager is not None:
                    metrics.update(self.shm_cache_manager.get_metrics())
                elif self.suffix_tree_manager.enabled:
                    metrics.update(self.suffix_tree_manager.get_metrics())

                # this is experimental and may be changed/removed in the future
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    # Cleanup shared memory cache manager
                    if self.shm_cache_manager is not None:
                        self.shm_cache_manager.shutdown()
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental
                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)

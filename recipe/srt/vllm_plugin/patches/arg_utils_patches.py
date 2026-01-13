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
AsyncEngineArgs patches for suffix decoding CLI arguments.

Adds CLI arguments:
- --speculative-method: Speculative decoding method (suffix, suffix_remote, etc.)
- --num-speculative-tokens: Number of speculative tokens
- --suffix-decoding-max-tree-depth: Maximum tree depth for suffix decoding
- --suffix-decoding-use-parallel: Use parallel suffix proposer

These arguments get merged into speculative_config for SpeculativeConfig creation.
"""

import logging

logger = logging.getLogger(__name__)

# Track if patches have been applied
_patches_applied = False


def apply_patches():
    """Apply AsyncEngineArgs patches for suffix decoding CLI arguments.

    This patches:
    1. AsyncEngineArgs.add_cli_args() to add suffix decoding arguments
    2. AsyncEngineArgs.create_speculative_config() to merge CLI args into speculative_config
    """
    global _patches_applied

    if _patches_applied:
        logger.debug("AsyncEngineArgs patches already applied, skipping")
        return

    from vllm.engine.arg_utils import AsyncEngineArgs

    # Store original methods
    _original_add_cli_args = AsyncEngineArgs.add_cli_args
    _original_create_speculative_config = AsyncEngineArgs.create_speculative_config

    @classmethod
    def patched_add_cli_args(cls, parser, async_args_only=False):
        """Extended add_cli_args that adds suffix decoding arguments."""
        # Call original first
        parser = _original_add_cli_args.__func__(cls, parser, async_args_only)

        # Add suffix decoding arguments
        suffix_group = parser.add_argument_group("Suffix Decoding Options")

        suffix_group.add_argument(
            "--speculative-method",
            type=str,
            default=None,
            help="Speculative decoding method: suffix, suffix_remote, ngram, eagle, etc. "
                 "When set, creates a speculative_config with this method.",
        )
        suffix_group.add_argument(
            "--num-speculative-tokens",
            type=int,
            default=None,
            help="Number of speculative tokens to generate per step.",
        )
        suffix_group.add_argument(
            "--suffix-decoding-max-tree-depth",
            type=int,
            default=None,
            help="Maximum depth of suffix trees for pattern matching.",
        )
        suffix_group.add_argument(
            "--suffix-decoding-use-parallel",
            action="store_true",
            default=False,
            help="Use parallel suffix proposer for batch operations.",
        )

        return parser

    def patched_create_speculative_config(
        self,
        target_model_config,
        target_parallel_config,
        enable_chunked_prefill,
        disable_log_stats,
    ):
        """Extended create_speculative_config that merges CLI args.

        If --speculative-method is set, creates speculative_config from CLI args.
        """
        # Check if we have CLI-based suffix decoding args
        speculative_method = getattr(self, 'speculative_method', None)
        num_speculative_tokens = getattr(self, 'num_speculative_tokens', None)
        suffix_decoding_max_tree_depth = getattr(self, 'suffix_decoding_max_tree_depth', None)
        suffix_decoding_use_parallel = getattr(self, 'suffix_decoding_use_parallel', False)

        # If --speculative-method is set, create/merge speculative_config
        if speculative_method is not None:
            if self.speculative_config is None:
                self.speculative_config = {}

            # Merge CLI args into speculative_config (CLI args take precedence)
            if 'method' not in self.speculative_config:
                self.speculative_config['method'] = speculative_method

            if num_speculative_tokens is not None and 'num_speculative_tokens' not in self.speculative_config:
                self.speculative_config['num_speculative_tokens'] = num_speculative_tokens

            if suffix_decoding_max_tree_depth is not None:
                self.speculative_config['suffix_decoding_max_tree_depth'] = suffix_decoding_max_tree_depth

            if suffix_decoding_use_parallel:
                self.speculative_config['suffix_decoding_use_parallel'] = suffix_decoding_use_parallel

            logger.info(f"Created speculative_config from CLI args: {self.speculative_config}")

        # Call original method
        return _original_create_speculative_config(
            self,
            target_model_config,
            target_parallel_config,
            enable_chunked_prefill,
            disable_log_stats,
        )

    # Apply patches
    AsyncEngineArgs.add_cli_args = patched_add_cli_args
    AsyncEngineArgs.create_speculative_config = patched_create_speculative_config

    # Add new attributes to AsyncEngineArgs class for CLI argument storage
    # These get populated by argparse when from_cli_args() is called
    if not hasattr(AsyncEngineArgs, 'speculative_method'):
        AsyncEngineArgs.speculative_method = None
    if not hasattr(AsyncEngineArgs, 'suffix_decoding_max_tree_depth'):
        AsyncEngineArgs.suffix_decoding_max_tree_depth = None
    if not hasattr(AsyncEngineArgs, 'suffix_decoding_use_parallel'):
        AsyncEngineArgs.suffix_decoding_use_parallel = False

    _patches_applied = True
    logger.info("Applied AsyncEngineArgs suffix decoding CLI patches")

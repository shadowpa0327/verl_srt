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
SpeculativeConfig patches for suffix decoding.

Adds:
- New method types: "suffix", "suffix_remote"
- Suffix decoding configuration fields
- Validation for suffix-specific params
"""

import logging
from typing import Literal, Optional, get_args, get_origin

from arctic_inference.patching import ArcticPatch
from vllm.config import SpeculativeConfig

logger = logging.getLogger(__name__)

# Track if patches have been applied
_patches_applied = False

# Store original methods for wrapping
_original_post_init = SpeculativeConfig.__post_init__
_original_repr = SpeculativeConfig.__repr__


def _patch_method_literal_type():
    """
    Patch the 'method' field's Literal type to include 'suffix' and 'suffix_remote'.

    This must be done before pydantic validates the field, so we:
    1. Extract the existing Literal values from the type annotation
    2. Add 'suffix' and 'suffix_remote' to the Literal
    3. Update the annotation
    4. Rebuild the pydantic dataclass validator
    """
    import dataclasses
    from typing import Union

    # Get current method annotation
    current_annotation = SpeculativeConfig.__annotations__.get('method')
    if current_annotation is None:
        logger.warning("SpeculativeConfig.method annotation not found")
        return

    # Extract Literal values from Optional[Literal[...]]
    # The annotation is Optional[Literal['ngram', 'eagle', ...]]
    origin = get_origin(current_annotation)
    if origin is Union:
        # Optional[X] is Union[X, None]
        args = get_args(current_annotation)
        literal_type = None
        for arg in args:
            if get_origin(arg) is Literal:
                literal_type = arg
                break
            elif arg is type(None):
                continue

        if literal_type is None:
            logger.warning("Could not find Literal type in method annotation")
            return

        existing_values = get_args(literal_type)
    elif origin is Literal:
        existing_values = get_args(current_annotation)
    else:
        logger.warning(f"Unexpected method annotation type: {current_annotation}")
        return

    # Add suffix methods if not already present
    new_values = list(existing_values)
    if 'suffix' not in new_values:
        new_values.append('suffix')
    if 'suffix_remote' not in new_values:
        new_values.append('suffix_remote')

    # Create new Literal type with extended values
    new_literal = Literal[tuple(new_values)]  # type: ignore

    # Wrap in Optional if original was Optional
    if origin is Union:
        new_annotation = Optional[new_literal]
    else:
        new_annotation = new_literal

    # Update annotation
    SpeculativeConfig.__annotations__['method'] = new_annotation

    # Update dataclass field if present
    if hasattr(SpeculativeConfig, '__dataclass_fields__') and 'method' in SpeculativeConfig.__dataclass_fields__:
        field = SpeculativeConfig.__dataclass_fields__['method']
        # Create new field with updated type
        SpeculativeConfig.__dataclass_fields__['method'] = dataclasses.field(
            default=field.default,
            default_factory=field.default_factory if field.default_factory is not dataclasses.MISSING else dataclasses.MISSING,
            init=field.init,
            repr=field.repr,
            hash=field.hash,
            compare=field.compare,
            metadata=field.metadata,
            kw_only=field.kw_only,
        )
        # Update the type attribute directly
        SpeculativeConfig.__dataclass_fields__['method'].type = new_annotation

    # Rebuild pydantic validator
    # For pydantic v2 dataclasses, we need to rebuild the validator
    if hasattr(SpeculativeConfig, '__pydantic_complete__'):
        try:
            from pydantic import TypeAdapter
            from pydantic.dataclasses import rebuild_dataclass

            # Rebuild the dataclass to regenerate pydantic schema
            rebuild_dataclass(SpeculativeConfig, force=True)
            logger.debug("Rebuilt pydantic dataclass validator for SpeculativeConfig")
        except ImportError:
            logger.warning("Could not import pydantic rebuild utilities")
        except Exception as e:
            logger.warning(f"Failed to rebuild pydantic validator: {e}")

    logger.debug(f"Patched method annotation to include suffix methods: {new_values}")


class SpeculativeConfigSuffixPatch(ArcticPatch[SpeculativeConfig]):
    """
    Patches SpeculativeConfig to add suffix decoding support.

    Adds fields:
    - suffix_decoding_max_tree_depth: int = 24
    - suffix_decoding_max_cached_requests: int = 10000
    - suffix_decoding_max_spec_factor: float = 1.0
    - suffix_decoding_min_token_prob: float = 0.1
    - suffix_decoding_use_parallel: bool = True
    - suffix_decoding_server_host: str = None
    - suffix_decoding_server_port: int = 50051

    Extends __post_init__ to handle "suffix" and "suffix_remote" methods.
    """

    # New fields with defaults
    suffix_decoding_max_tree_depth: int = 24
    suffix_decoding_max_cached_requests: int = 10000
    suffix_decoding_max_spec_factor: float = 1.0
    suffix_decoding_min_token_prob: float = 0.1
    suffix_decoding_use_parallel: bool = True
    suffix_decoding_server_host: Optional[str] = None
    suffix_decoding_server_port: int = 50051

    def _validate_suffix_decoding(self):
        """Validate suffix decoding configuration."""
        from recipe.srt.vllm_plugin.patches.vllm_utils import has_arctic_inference

        if not has_arctic_inference():
            raise ImportError(
                "Arctic Inference is required for suffix decoding. "
                "Install via `pip install arctic-inference`."
            )

        if self.num_speculative_tokens is None:
            # Suffix decoding decides the actual number of speculative tokens
            # dynamically and treats num_speculative_tokens as a maximum limit.
            self.num_speculative_tokens = self.suffix_decoding_max_tree_depth
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix decoding.",
                self.num_speculative_tokens,
            )

        # Validate values
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1"
            )

        if self.suffix_decoding_max_cached_requests < 0:
            raise ValueError(
                f"suffix_decoding_max_cached_requests="
                f"{self.suffix_decoding_max_cached_requests} must be >= 0"
            )

        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0"
            )

        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]"
            )

    def _validate_suffix_remote(self):
        """Validate configuration for remote suffix decoding."""
        from recipe.srt.vllm_plugin.patches.vllm_utils import has_arctic_inference

        if not has_arctic_inference():
            raise ImportError(
                "Arctic Inference is required for suffix_remote. "
                "Install via `pip install arctic-inference`."
            )

        if self.suffix_decoding_server_host is None:
            raise ValueError(
                "suffix_decoding_server_host must be specified when using "
                "suffix_remote method."
            )

        if self.num_speculative_tokens is None:
            self.num_speculative_tokens = self.suffix_decoding_max_tree_depth
            logger.warning(
                "Defaulted num_speculative_tokens to %s for suffix_remote.",
                self.num_speculative_tokens,
            )

        # Validate values (same as suffix decoding)
        if self.suffix_decoding_max_tree_depth < 1:
            raise ValueError(
                f"suffix_decoding_max_tree_depth="
                f"{self.suffix_decoding_max_tree_depth} must be >= 1"
            )

        if self.suffix_decoding_max_spec_factor < 0:
            raise ValueError(
                f"suffix_decoding_max_spec_factor="
                f"{self.suffix_decoding_max_spec_factor} must be >= 0"
            )

        if not 0 <= self.suffix_decoding_min_token_prob <= 1:
            raise ValueError(
                f"suffix_decoding_min_token_prob="
                f"{self.suffix_decoding_min_token_prob} must be in [0, 1]"
            )

    def __post_init__(self):
        """Extended __post_init__ that handles suffix methods before calling original."""
        # Set default values for suffix fields if not present
        if not hasattr(self, 'suffix_decoding_max_tree_depth'):
            self.suffix_decoding_max_tree_depth = 24
        if not hasattr(self, 'suffix_decoding_max_cached_requests'):
            self.suffix_decoding_max_cached_requests = 10000
        if not hasattr(self, 'suffix_decoding_max_spec_factor'):
            self.suffix_decoding_max_spec_factor = 1.0
        if not hasattr(self, 'suffix_decoding_min_token_prob'):
            self.suffix_decoding_min_token_prob = 0.1
        if not hasattr(self, 'suffix_decoding_use_parallel'):
            self.suffix_decoding_use_parallel = True
        if not hasattr(self, 'suffix_decoding_server_host'):
            self.suffix_decoding_server_host = None
        if not hasattr(self, 'suffix_decoding_server_port'):
            self.suffix_decoding_server_port = 50051

        # Handle suffix method detection before calling original __post_init__
        # This allows original to skip model loading for suffix methods
        if self.method == "suffix":
            self.model = "suffix"
        elif self.method == "suffix_remote":
            self.model = "suffix_remote"

        # Call original __post_init__
        # We need to handle the case where method is suffix/suffix_remote
        # Original will try to validate draft model which doesn't exist for suffix
        if self.method in ("suffix", "suffix_remote"):
            # For suffix methods, we do our own validation
            self._verify_args()
            if self.method == "suffix":
                self._validate_suffix_decoding()
            elif self.method == "suffix_remote":
                self._validate_suffix_remote()
        else:
            # For other methods, use original validation
            _original_post_init(self)

    def __repr__(self) -> str:
        """Extended __repr__ that handles suffix methods."""
        method = self.method
        if method in ("ngram", "suffix", "suffix_remote"):
            model = None
        elif self.draft_model_config is not None:
            model = self.draft_model_config.model
        else:
            model = None
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"


def apply_patches():
    """Apply SpeculativeConfig patches."""
    global _patches_applied

    if _patches_applied:
        logger.debug("SpeculativeConfig patches already applied, skipping")
        return

    # First, patch the method Literal type to accept suffix methods
    _patch_method_literal_type()

    # Then apply the ArcticPatch
    SpeculativeConfigSuffixPatch.apply_patch()

    _patches_applied = True
    logger.info("Applied SpeculativeConfig suffix decoding patches")

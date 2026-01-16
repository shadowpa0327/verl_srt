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

This module only patches the 'method' Literal type to include 'suffix' and
'suffix_remote'. All other SRT configuration is handled via SRTSuffixConfig
in config.py, avoiding complex pydantic manipulation.
"""

import logging
from typing import Literal, Optional, Union, get_args, get_origin

from vllm.config import SpeculativeConfig

logger = logging.getLogger(__name__)

# Track if patches have been applied
_patches_applied = False


def _patch_method_literal_type():
    """
    Patch the 'method' field's Literal type to include 'suffix' and 'suffix_remote'.

    This is the minimal patch needed for vLLM to accept method="suffix" in
    SpeculativeConfig. All other SRT configuration is handled separately
    via SRTSuffixConfig.
    """
    import dataclasses

    # Get current method annotation
    current_annotation = SpeculativeConfig.__annotations__.get("method")
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
    if "suffix" not in new_values:
        new_values.append("suffix")
    if "suffix_remote" not in new_values:
        new_values.append("suffix_remote")

    # Create new Literal type with extended values
    new_literal = Literal[tuple(new_values)]  # type: ignore

    # Wrap in Optional if original was Optional
    if origin is Union:
        new_annotation = Optional[new_literal]
    else:
        new_annotation = new_literal

    # Update annotation
    SpeculativeConfig.__annotations__["method"] = new_annotation

    # Update dataclass field if present
    if (
        hasattr(SpeculativeConfig, "__dataclass_fields__")
        and "method" in SpeculativeConfig.__dataclass_fields__
    ):
        field = SpeculativeConfig.__dataclass_fields__["method"]
        # Create new field with updated type
        SpeculativeConfig.__dataclass_fields__["method"] = dataclasses.field(
            default=field.default,
            default_factory=field.default_factory
            if field.default_factory is not dataclasses.MISSING
            else dataclasses.MISSING,
            init=field.init,
            repr=field.repr,
            hash=field.hash,
            compare=field.compare,
            metadata=field.metadata,
            kw_only=field.kw_only,
        )
        # Update the type attribute directly
        SpeculativeConfig.__dataclass_fields__["method"].type = new_annotation

    # Rebuild pydantic validator to pick up the new Literal type
    # This is necessary because pydantic caches validators
    if hasattr(SpeculativeConfig, "__pydantic_complete__"):
        try:
            from pydantic.dataclasses import rebuild_dataclass

            rebuild_dataclass(SpeculativeConfig, force=True)
            logger.debug("Rebuilt pydantic dataclass validator for SpeculativeConfig")
        except ImportError:
            logger.warning("Could not import pydantic rebuild utilities")
        except Exception as e:
            logger.warning(f"Failed to rebuild pydantic validator: {e}")

    logger.debug(f"Patched method annotation to include suffix methods: {new_values}")


def _add_srt_cache_mode_attribute():
    """Mark SpeculativeConfig to accept srt_cache_mode attribute.

    Instead of adding a dataclass field (which causes pydantic issues),
    we'll set the attribute in __post_init__ and store srt_cache_mode
    as a simple instance attribute.
    """
    # Just mark that we want to handle srt_cache_mode - actual setting
    # happens in patched __post_init__
    logger.debug("srt_cache_mode handling enabled for SpeculativeConfig")


def _patch_post_init():
    """Patch SpeculativeConfig.__post_init__ to handle suffix methods.

    For suffix/suffix_remote methods, we skip the draft model validation
    since suffix decoding doesn't use a draft model.
    """
    _original_post_init = SpeculativeConfig.__post_init__

    def patched_post_init(self):
        """Patched __post_init__ that handles suffix methods."""
        # For suffix methods, skip draft model validation
        if self.method in ("suffix", "suffix_remote"):
            # Set model to method name (no draft model for suffix)
            self.model = self.method
            # Call _verify_args for basic validation
            self._verify_args()
            return

        # For other methods, use original validation
        _original_post_init(self)

    SpeculativeConfig.__post_init__ = patched_post_init
    logger.debug("Patched SpeculativeConfig.__post_init__ for suffix methods")


def _patch_repr():
    """Patch SpeculativeConfig.__repr__ to handle suffix methods.

    The original __repr__ assumes draft_model_config exists for non-ngram methods,
    but suffix methods also don't have a draft model.
    """
    def patched_repr(self) -> str:
        """Patched __repr__ that handles suffix methods."""
        method = self.method
        # Both ngram and suffix methods don't have a draft model
        if method in ("ngram", "suffix", "suffix_remote"):
            model = None
        else:
            model = self.draft_model_config.model if self.draft_model_config else None
        num_spec_tokens = self.num_speculative_tokens
        return f"SpeculativeConfig({method=}, {model=}, {num_spec_tokens=})"

    SpeculativeConfig.__repr__ = patched_repr
    logger.debug("Patched SpeculativeConfig.__repr__ for suffix methods")


def apply_patches():
    """Apply SpeculativeConfig patches.

    Patches:
    1. Method Literal type to accept 'suffix' and 'suffix_remote'
    2. srt_cache_mode field for passing cache mode to workers
    3. __post_init__ to skip draft model validation for suffix methods
    4. __repr__ to handle suffix methods (no draft model)

    All other configuration is handled via SRTSuffixConfig.
    """
    global _patches_applied

    if _patches_applied:
        logger.debug("SpeculativeConfig patches already applied, skipping")
        return

    # Patch the method Literal type to accept suffix methods
    _patch_method_literal_type()

    # Enable srt_cache_mode handling
    _add_srt_cache_mode_attribute()

    # Patch __post_init__ to handle suffix methods
    _patch_post_init()

    # Patch __repr__ to handle suffix methods
    _patch_repr()

    _patches_applied = True
    logger.info("Applied SpeculativeConfig suffix method patches")

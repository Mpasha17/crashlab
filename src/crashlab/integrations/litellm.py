"""``crashlab.integrations.litellm.patch()`` / ``unpatch()``: record LiteLLM calls."""

from ..capture.litellm import is_patched, patch, unpatch

__all__ = ["is_patched", "patch", "unpatch"]

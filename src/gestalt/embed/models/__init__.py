"""Vendored foundation-model adapters from `platonic-universe`.

`HFAdapter` covers the HuggingFace vision families (vit, vit-mae, ijepa,
vjepa, clip, convnext); `VLMAdapter` covers vision-language models (llava);
`AstroptAdapter` wraps astropt. Importing this package registers all of
them in `gestalt.embed.models.registry._REGISTRY`.
"""

from gestalt.embed.models import astropt, hf  # noqa: F401 — side-effect registration
from gestalt.embed.models.base import ModelAdapter
from gestalt.embed.models.registry import get_adapter, register_adapter

__all__ = [
    "ModelAdapter",
    "get_adapter",
    "register_adapter",
]

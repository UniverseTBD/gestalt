"""Vendored foundation-model adapters from `platonic-universe`.

`HFAdapter` covers the HuggingFace vision families (vit, vit-mae, ijepa,
vjepa, clip, convnext); `VLMAdapter` covers vision-language models (llava);
`AstroptAdapter` wraps astropt. Importing this package registers all of
them in `bazaar.embed.models.registry._REGISTRY`.
"""
from bazaar.embed.models import astropt, hf  # noqa: F401 — side-effect registration
from bazaar.embed.models.base import ModelAdapter
from bazaar.embed.models.registry import get_adapter, list_adapters, register_adapter

__all__ = [
    "ModelAdapter",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]

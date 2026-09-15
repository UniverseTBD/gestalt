"""Vendored from `pu.models.astropt` — AstroPT adapter.

Re-points imports at `bazaar.embed.*`.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from astropt.model_utils import load_astropt

from bazaar.embed.models.base import ModelAdapter
from bazaar.embed.models.registry import register_adapter
from bazaar.embed.preprocess import PreprocessAstropt


class AstroptAdapter(ModelAdapter):
    """
    Adapter for astroPT models. Wraps `load_astropt` and uses `PreprocessAstropt`
    for preprocessing and the model's `generate_embeddings` for embedding.
    """

    def __init__(self, model_name: str, size: str, alias: str | None = None):
        super().__init__(model_name, size, alias)
        self.model: Any = None

    def load(self) -> None:
        self.model = load_astropt(self.model_name, path=f"astropt/{self.size}").to("cuda")
        self.model.eval()

    def get_preprocessor(
        self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"
    ):
        return PreprocessAstropt(
            self.model.modality_registry, modes, resize=resize, resize_mode=resize_mode
        )

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        inputs = {
            "images": batch[f"{mode}_images"].to("cuda"),
            "images_positions": batch[f"{mode}_positions"].to("cuda"),
        }
        with torch.no_grad():
            return self.model.generate_embeddings(inputs)["images"].detach()


register_adapter("astropt", AstroptAdapter)

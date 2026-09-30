"""Vendored from `pu.models.hf` — HuggingFace vision-model adapters.

Re-points imports at `gestalt.embed.*` and is otherwise faithful to upstream
so the same model checkpoints produce the same embeddings.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoVideoProcessor,
    CLIPModel,
    CLIPProcessor,
)

from gestalt.embed.models.base import ModelAdapter
from gestalt.embed.models.registry import register_adapter
from gestalt.embed.preprocess import PreprocessHF

_CLIP_FAMILY = {"clip"}


class HFAdapter(ModelAdapter):
    """
    Adapter for HuggingFace vision models using AutoModel + AutoImageProcessor.
    The adapter uses the 'alias' passed at construction to decide pooling:
      - 'vit' -> CLS excluded mean over tokens (last_hidden_state[:,1:].mean)
      - 'convnext' -> spatial mean over HxW (last_hidden_state.mean(dim=(2,3)))
      - 'ijepa' -> mean over token dim (last_hidden_state.mean(dim=1))
      - 'vjepa' -> mean over token dim (last_hidden_state.mean(dim=1))
      - 'vit-mae' -> CLS excluded mean over tokens (last_hidden_state[:,1:].mean)
      - 'clip' -> image features: final, projected visual embs that have been
            aligned with text (get_image_features(), shape [batch, embedding_dim])
    """

    def __init__(self, model_name: str, size: str, alias: str | None = None):
        super().__init__(model_name, size, alias)
        self.processor = None
        self.model: Any = None

    def load(self) -> None:
        if self.alias == "vjepa":
            self.processor = AutoVideoProcessor.from_pretrained(self.model_name)
        elif self.alias == "clip":
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
        else:
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)

        if self.alias == "clip":
            self.model = CLIPModel.from_pretrained(self.model_name).to("cuda").eval()  # pyright: ignore[reportArgumentType, reportCallIssue]  # transformers/torch stub quirk
        else:
            self.model = AutoModel.from_pretrained(self.model_name).to("cuda").eval()

    def get_preprocessor(
        self, modes: Iterable[str], resize: bool = False, resize_mode: str = "fill"
    ):
        return PreprocessHF(
            modes, self.processor, alias=self.alias, resize=resize, resize_mode=resize_mode
        )

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        inputs = batch[f"{mode}"].to("cuda")
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=self._use_amp, dtype=torch.float16):  # pyright: ignore[reportPrivateImportUsage]
                if self.alias == "clip":
                    outputs = self.model.get_image_features(pixel_values=inputs)
                    # transformers >=5.0 returns BaseModelOutputWithPooling here
                    # (projected image features live in `pooler_output`); older
                    # versions returned a plain tensor.
                    feats = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
                    return feats.float().detach()
                outputs = self.model(inputs).last_hidden_state
                if self.alias in ("vit", "vit-mae"):
                    emb = outputs[:, 1:].mean(dim=1)
                elif self.alias == "convnext":
                    emb = outputs.mean(dim=(2, 3))
                elif self.alias in ("dino", "dinov3"):
                    emb = outputs[:, 0]
                elif self.alias in ("ijepa", "vjepa"):
                    emb = outputs.mean(dim=1)
                else:
                    emb = outputs.mean(dim=1)
            emb = emb.float().detach()
        return emb


class VLMAdapter(HFAdapter):
    """
    Subclass of HFAdapter for vision-language models that need:
      - AutoProcessor instead of AutoImageProcessor
      - AutoModelForImageTextToText instead of AutoModel
      - pixel_values passed explicitly (with dtype cast) alongside a text prompt
      - last_hidden_state mean-pooled over the full sequence

    PaliGemma2 (all sizes) and LLaVA-1.5 and LLaVA-OneVision.
    """

    _PROMPTS = {
        "paligemma": "<image> ",
        "paligemma_3b": "<image> ",
        "paligemma_10b": "<image> ",
        "paligemma_28b": "<image> ",
        "llava_15": "USER: <image>\n ASSISTANT:",
        "llava_ov": "<image>",
    }

    def load(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            dtype=torch.bfloat16,
            device_map="balanced",
            low_cpu_mem_usage=True,
        ).eval()

    def get_preprocessor(
        self, modes: Iterable[str], resize: bool = True, resize_mode: str = "match"
    ):
        return PreprocessHF(
            modes, self.processor, alias=self.alias, resize=resize, resize_mode=resize_mode
        )

    def _prepare_vlm_inputs(self, batch, mode):
        """Prepare tokenized inputs for the VLM forward pass."""
        import warnings

        warnings.filterwarnings("ignore", message=".*PaliGemma.*")
        warnings.filterwarnings("ignore", message=".*PaliGemmaProcessor.*")
        warnings.filterwarnings("ignore", message=".*text prefix.*")
        warnings.filterwarnings("ignore", message=".*special image tokens.*")

        device = next(self.model.parameters()).device
        pv = batch[f"{mode}"].to(device)

        model_dtype = next(self.model.parameters()).dtype
        pv = pv.to(dtype=model_dtype)

        from PIL import Image

        B = pv.shape[0]
        prompt = self._PROMPTS.get(self.alias or "", " ")
        pv_cpu = pv.cpu().float()
        pv_cpu = (pv_cpu - pv_cpu.min()) / (pv_cpu.max() - pv_cpu.min() + 1e-8)
        pv_cpu = (pv_cpu * 255).byte()
        pil_images = [Image.fromarray(pv_cpu[i].permute(1, 2, 0).numpy()) for i in range(B)]
        enc = self.processor(
            images=pil_images,
            text=[prompt] * B,
            return_tensors="pt",
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)
        pv_enc = enc["pixel_values"].to(device, dtype=model_dtype)
        return input_ids, pv_enc, attn_mask

    @staticmethod
    def _masked_mean_pool(hs, attn_mask):
        m = attn_mask.to(hs.device).float().unsqueeze(-1)
        return ((hs * m).sum(1) / m.sum(1).clamp_min(1.0)).float().detach()

    def _last_lm_layer(self) -> Any:
        base = getattr(self.model, "model", self.model)
        lm = getattr(base, "language_model", base)
        layers = getattr(lm, "layers", None) or lm.model.layers
        return layers[-1]

    def embed_for_mode(self, batch: Dict[str, Any], mode: str):
        input_ids, pv, attn_mask = self._prepare_vlm_inputs(batch, mode)
        captured: Dict[str, torch.Tensor] = {}

        def _hook(_mod, _args, out):
            captured["hs"] = out[0] if isinstance(out, tuple) else out

        handle = self._last_lm_layer().register_forward_hook(_hook)
        try:
            with torch.no_grad():
                self.model(
                    input_ids=input_ids,
                    pixel_values=pv,
                    attention_mask=attn_mask,
                    return_dict=True,
                )
        finally:
            handle.remove()
        return self._masked_mean_pool(captured["hs"], attn_mask)


for alias in ("vit", "dino", "convnext", "ijepa", "vjepa", "vit-mae", "clip"):
    register_adapter(alias, HFAdapter)

for alias in ("paligemma", "paligemma_3b", "paligemma_10b", "paligemma_28b"):
    register_adapter(alias, VLMAdapter)

for alias in ("llava_15", "llava_15_7b", "llava_15_13b", "llava_ov", "llava_ov_7b"):
    register_adapter(alias, VLMAdapter)

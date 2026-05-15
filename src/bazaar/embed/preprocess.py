"""Vendored from `pu.preprocess`.

Converts raw galaxy flux blobs (the `{"flux": …, "band": …}` shape that both
HF crossmatched datasets and MMU HATS rows expose) into the input tensors
expected by the foundation-model adapters in `bazaar.embed.models`.

Differences from upstream:
- `PreprocessSAM2` removed (no SAM2 in the bazaar basket)
- Per-modality flux→RGB logic lives in `bazaar.modalities`; this module is the
  generic dispatcher + the two outer Preprocess classes that the basket uses.
"""
from functools import partial

import numpy as np
import torch
from astropt.local_datasets import GalaxyImageDataset
from torchvision import transforms

from bazaar.modalities import MODALITIES, get_modality


def flux_to_pil(
    blob,
    mode,
    modes,
    resize=True,
    norm_mode="arcsinh",
    resize_mode="match",
):
    """Dispatch to the modality's preprocess() — see `bazaar.modalities`."""
    mod = get_modality(mode)
    if mod is None:
        raise ValueError(f"unknown modality {mode!r}; registered: {sorted(MODALITIES)}")
    return mod.preprocess(blob, resize=resize, resize_mode=resize_mode, norm_mode=norm_mode)


class PreprocessHF:
    """Preprocessor that converts galaxy images to the format expected by Dino and ViT models"""

    # Processors that require images= as a keyword argument rather than positional
    _IMAGES_KWARG_ALIASES = {
        "clip",
        "paligemma", "paligemma_3b", "paligemma_10b", "paligemma_28b",
        "llava_15", "llava_15_7b", "llava_15_13b",
        "llava_ov", "llava_ov_7b",
    }

    def __init__(self, modes, autoproc, resize=True, resize_mode="match", alias=None):
        self.modes = modes
        self.autoproc = autoproc
        self.alias = alias
        self.f2p = partial(
            flux_to_pil, resize=resize, resize_mode=resize_mode
        )

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if (mode == "desi") or (mode == "sdss"):
                continue
            else:
                im = self.f2p(idx[f"{mode}_image"], mode, self.modes)
                if self.alias in ("llava_15", "llava_15_7b", "llava_15_13b",
                                     "llava_ov", "llava_ov_7b"):
                    proc_out = self.autoproc(
                        images=im, text="<image>",
                        return_tensors="pt", padding=True,
                    )
                elif self.alias in ("paligemma", "paligemma_3b",
                                    "paligemma_10b", "paligemma_28b"):
                    proc_out = self.autoproc(
                        images=im, text="<image> ",
                        return_tensors="pt", padding=True,
                    )
                elif self.alias in self._IMAGES_KWARG_ALIASES:
                    proc_out = self.autoproc(images=im, return_tensors="pt")
                else:
                    proc_out = self.autoproc(im, return_tensors="pt")
                if "pixel_values" in proc_out:
                    result[f"{mode}"] = proc_out["pixel_values"].squeeze().numpy()
                elif "pixel_values_videos" in proc_out:
                    result[f"{mode}"] = proc_out["pixel_values_videos"].repeat(
                        1, 16, 1, 1, 1
                    ).squeeze().numpy()
                else:
                    raise KeyError(
                        "autoproc does not have 'pixel_values' or "
                        "'pixel_values_videos' in its dict"
                    )
        return result


class PreprocessAstropt:
    """Preprocessor that converts galaxy images to the format expected by AstroPT models"""

    @staticmethod
    def normalise_for_astropt(x):
        std, mean = torch.std_mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + 1e-8)

    @classmethod
    def data_transforms(cls):
        return transforms.Compose([transforms.Lambda(cls.normalise_for_astropt)])

    def __init__(
        self,
        modality_registry,
        modes,
        resize=True,
        resize_mode="match",
    ):
        self.galproc = GalaxyImageDataset(
            None,
            spiral=True,
            transform={"images": self.data_transforms()},
            modality_registry=modality_registry,
        )
        self.modes = modes
        self.f2p = partial(
            flux_to_pil, resize=resize, resize_mode=resize_mode
        )

    def __call__(self, idx):
        result = {}
        for mode in self.modes:
            if (mode == "desi") or (mode == "sdss"):
                continue
            else:
                im = self.f2p(idx[f"{mode}_image"], mode, self.modes).swapaxes(0, 2)
                im = self.galproc.process_galaxy(
                    torch.from_numpy(im).to(torch.float)
                ).to(torch.float).numpy()
                result[f"{mode}_images"] = im
                result[f"{mode}_positions"] = np.arange(0, len(im), dtype=np.int64)

        return result

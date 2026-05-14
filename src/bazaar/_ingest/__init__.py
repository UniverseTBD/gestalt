"""Catalog ingest layer.

`iter_galaxies(input)` returns a `CatalogSource` whose `.rows()` iterator
yields `{f"{modality}_image": {"flux": ndarray, "band": [...]}}`-shaped dicts
that `bazaar.embed.preprocess.PreprocessHF` consumes directly.

Today only the HF-streaming adapter exists (HATS catalogs published on the
Hugging Face Hub, including all MultimodalUniverse imagery datasets). A
folder-of-PNGs or local-HATS adapter can drop in alongside without touching
the rest of the pipeline.
"""
from bazaar._ingest.galaxies import GalaxiesSource, galaxies_source
from bazaar._ingest.gz10 import GZ10Source, gz10_source
from bazaar._ingest.hf_streaming import CatalogSource, iter_galaxies

__all__ = [
    "CatalogSource",
    "GZ10Source",
    "GalaxiesSource",
    "galaxies_source",
    "gz10_source",
    "iter_galaxies",
]

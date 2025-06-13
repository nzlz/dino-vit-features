"""Deep ViT Features as Dense Visual Descriptors.

This is a vendored version of dino-vit-features from:
https://github.com/normandipalo/dino-vit-features
"""

from .extractor import ViTExtractor
from .keypoint_utils import (
    extract_descriptors,
    extract_desc_maps,
    extract_descriptor_nn,
    draw_keypoints,
)

__all__ = [
    "ViTExtractor",
    "extract_descriptors",
    "extract_desc_maps",
    "extract_descriptor_nn",
    "draw_keypoints",
]

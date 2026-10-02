"""
enhancement.py
==============
Optional, classic image enhancement operations. All functions take and
return RGB uint8 arrays and never modify the input in place.

These operations change how an image LOOKS. They do not add information
that the camera did not capture, and there is no evidence here that they
improve diagnostic accuracy.
"""

import io
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


def apply_clahe(rgb: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """
    CLAHE = Contrast Limited Adaptive Histogram Equalization.

    Instead of stretching the histogram of the whole image at once, CLAHE
    splits the image into a grid of tiles and equalizes each tile, so local
    detail in both dark and bright regions becomes visible. The clip limit
    caps how much any tile can be boosted, which stops noise from being
    amplified too much.

    It is applied only to the L (lightness) channel of the LAB colour space,
    so colours (important for tissue appearance) are not distorted.
    """
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    l_eq = clahe.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_eq, a_channel, b_channel)), cv2.COLOR_LAB2RGB)


def reduce_noise_gaussian(rgb: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian blur: replaces each pixel with a weighted average of its
    neighbours (closer pixels get more weight). Removes grainy sensor noise
    but also softens real detail, so use small kernels.
    The kernel size must be odd; even values are bumped up by one.
    """
    k = int(kernel_size)
    if k % 2 == 0:
        k += 1
    k = max(k, 1)
    return cv2.GaussianBlur(rgb, (k, k), float(sigma))


def sharpen(rgb: np.ndarray, amount: float = 1.0, sigma: float = 1.5) -> np.ndarray:
    """
    Unsharp masking:

        sharpened = original + amount * (original - blurred)

    (original - blurred) isolates the fine detail; adding it back
    exaggerates edges. Too much amount produces halos and boosts noise.
    """
    blurred = cv2.GaussianBlur(rgb, (0, 0), float(sigma))
    # addWeighted computes (1+amount)*original - amount*blurred, with clipping
    return cv2.addWeighted(rgb, 1.0 + float(amount), blurred, -float(amount), 0)


def adjust_brightness(rgb: np.ndarray, beta: float = 30, alpha: float = 1.0) -> np.ndarray:
    """
    Linear brightness/contrast adjustment:

        output = clip(alpha * input + beta, 0, 255)

    beta  > 0 brightens, beta  < 0 darkens.
    alpha > 1 increases contrast, alpha < 1 decreases it.
    Note: brightening cannot recover detail in pixels that were pure black.
    """
    return cv2.convertScaleAbs(rgb, alpha=float(alpha), beta=float(beta))


# Fixed processing order: denoise first (so noise is not sharpened),
# then fix brightness, then local contrast, then sharpen last.
PIPELINE_ORDER = ["Noise reduction", "Brightness adjustment", "CLAHE", "Sharpening"]


def enhance_pipeline(rgb: np.ndarray, steps: List[Tuple[str, dict]]) -> np.ndarray:
    """
    Apply several enhancement steps in PIPELINE_ORDER.

    steps : list of (step_name, params) pairs, e.g.
            [("CLAHE", {"clip_limit": 2.0, "tile_grid": 8})]
    """
    functions = {
        "Noise reduction": reduce_noise_gaussian,
        "Brightness adjustment": adjust_brightness,
        "CLAHE": apply_clahe,
        "Sharpening": sharpen,
    }
    requested = dict(steps)
    result = rgb.copy()
    for name in PIPELINE_ORDER:
        if name in requested:
            result = functions[name](result, **requested[name])
    return result


def to_png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an RGB array as PNG bytes (lossless) for downloading."""
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return buffer.getvalue()

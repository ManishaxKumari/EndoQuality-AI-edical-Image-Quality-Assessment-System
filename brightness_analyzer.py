"""
brightness_analyzer.py
======================
Brightness, overexposure and contrast measurements, plus detection of the
black frame that surrounds most endoscopy images.

Colour convention used in this whole project
--------------------------------------------
Images are kept in **RGB** order everywhere (that is what Pillow and
Streamlit use). OpenCV's default is BGR, so every OpenCV colour conversion in
this project explicitly uses an RGB constant such as cv2.COLOR_RGB2GRAY.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to 8-bit grayscale.

    OpenCV uses the standard luminance formula (ITU-R BT.601):
        Gray = 0.299 * R + 0.587 * G + 0.114 * B
    Green gets the largest weight because human eyes are most sensitive
    to green light.
    """
    if rgb.ndim == 2:                       # already grayscale
        return rgb.astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _masked_pixels(gray: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Return a flat array of the pixels we should measure."""
    if mask is None:
        return gray.ravel()
    pixels = gray[mask > 0]
    return pixels if pixels.size > 0 else gray.ravel()


# --------------------------------------------------------------------------
# Field-of-view (black frame) detection
# --------------------------------------------------------------------------
def compute_fov_mask(
    gray: np.ndarray,
    border_threshold: int = 15,
    min_coverage: float = 0.25,
    max_coverage: float = 0.98,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Find the illuminated field of view (FOV) inside the black frame.

    Steps
    -----
    1. Threshold: pixels brighter than `border_threshold` are "inside".
    2. Morphological closing joins small gaps caused by dark folds or noise.
    3. Keep the largest outer contour and FILL it, so dark areas *inside*
       the view (e.g. the lumen of the bowel) are still measured.

    Safety fallbacks (return None = "use the whole image")
    -----------------------------------------------------
    * coverage < min_coverage : the image is probably just very dark, and
      masking would hide that darkness. Measure everything instead.
    * coverage > max_coverage : there is no meaningful frame.

    Returns
    -------
    (mask or None, coverage) where coverage is the fraction of the image
    inside the FOV (1.0 when the whole image is used).
    """
    _, binary = cv2.threshold(gray, border_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, 1.0

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)

    coverage = float(np.count_nonzero(mask)) / mask.size
    if coverage < min_coverage or coverage > max_coverage:
        return None, 1.0
    return mask, coverage


# --------------------------------------------------------------------------
# Measurements
# --------------------------------------------------------------------------
def analyze_brightness(
    gray: np.ndarray,
    dark_threshold: float,
    bright_mean_threshold: float,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Mean brightness = average gray value (0 = black, 255 = white).

        mean = (1 / N) * sum(I(x, y))
    """
    pixels = _masked_pixels(gray, mask)
    mean = float(pixels.mean())
    return {
        "mean_brightness": mean,
        "is_too_dark": mean < dark_threshold,
        "is_too_bright": mean > bright_mean_threshold,
    }


def analyze_exposure(
    gray: np.ndarray,
    pixel_value: int,
    percent_limit: float,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Overexposure = percentage of "blown-out" (near-white) pixels.

        overexposed % = 100 * count(I >= pixel_value) / N

    Blown-out pixels have lost all detail: there is nothing left to
    recover, which is why this is measured separately from the mean.
    """
    pixels = _masked_pixels(gray, mask)
    percent = 100.0 * float(np.count_nonzero(pixels >= pixel_value)) / pixels.size
    return {
        "overexposed_percent": percent,
        "is_overexposed": percent > percent_limit,
    }


def analyze_contrast(
    gray: np.ndarray,
    low_contrast_threshold: float,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Contrast = standard deviation of gray values (RMS contrast).

        std = sqrt( (1 / N) * sum( (I - mean)^2 ) )

    A flat, washed-out image has all pixels close to the mean -> low std.
    """
    pixels = _masked_pixels(gray, mask)
    std = float(pixels.std())
    return {
        "contrast_std": std,
        "is_low_contrast": std < low_contrast_threshold,
    }

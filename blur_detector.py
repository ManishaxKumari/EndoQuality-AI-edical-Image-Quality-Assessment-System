"""
blur_detector.py
================
Blur detection with the *variance of the Laplacian*.

The idea in one paragraph
-------------------------
The Laplacian is a second-derivative filter: it responds strongly wherever
pixel intensity changes abruptly (edges, fine texture, vessel boundaries) and
gives values near zero in smooth regions. A sharp image has many strong edges,
so its Laplacian response contains both large positive and large negative
values -> HIGH variance. A blurry image has soft, smeared edges, so the
response stays close to zero everywhere -> LOW variance.

    L(x, y)   = d2I/dx2 + d2I/dy2                 (Laplacian)
    variance  = mean( (L - mean(L))^2 )           (spread of the response)

OpenCV's default 3x3 Laplacian kernel is:

        [ 0   1   0 ]
        [ 1  -4   1 ]
        [ 0   1   0 ]
"""

from typing import Optional

import cv2
import numpy as np


def compute_laplacian(gray: np.ndarray) -> np.ndarray:
    """Return the raw Laplacian response (float64, can be negative)."""
    # CV_64F keeps negative values; with uint8 they would be clipped to 0
    # and the variance would be wrong.
    return cv2.Laplacian(gray, cv2.CV_64F)


def compute_laplacian_variance(
    gray: np.ndarray, mask: Optional[np.ndarray] = None
) -> float:
    """
    Variance of the Laplacian, optionally only inside a mask.

    Parameters
    ----------
    gray : 2-D uint8 grayscale image.
    mask : optional 2-D uint8 array (255 = include pixel, 0 = ignore).

    Returns
    -------
    float : the Laplacian variance (higher = sharper).
    """
    laplacian = compute_laplacian(gray)

    if mask is not None:
        # Shrink the mask a few pixels so the hard edge between the black
        # frame and the tissue is NOT counted. That edge is always sharp
        # (it comes from the scope's optics), even when the tissue is blurred,
        # so including it would make blurry images look sharp.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        inner = cv2.erode(mask, kernel, iterations=2)
        values = laplacian[inner > 0]
        if values.size < 100:          # mask too small after erosion
            values = laplacian[mask > 0]
    else:
        values = laplacian.ravel()

    if values.size == 0:
        return 0.0
    return float(values.var())


def laplacian_edge_map(gray: np.ndarray) -> np.ndarray:
    """
    A displayable edge map (uint8) of the absolute Laplacian response.

    Bright areas = strong edges/texture. A blurry image gives a mostly
    dark map. Values are scaled with the 99th percentile so a few extreme
    pixels do not make everything else look black.
    """
    magnitude = np.abs(compute_laplacian(gray))
    top = np.percentile(magnitude, 99)
    if top <= 0:
        return np.zeros_like(gray, dtype=np.uint8)
    scaled = np.clip(magnitude / top * 255.0, 0, 255)
    return scaled.astype(np.uint8)


def detect_blur(
    gray: np.ndarray, threshold: float, mask: Optional[np.ndarray] = None
) -> dict:
    """
    Decide whether an image is blurry.

    Returns a dict with:
        laplacian_variance : float
        threshold          : float (the threshold that was used)
        is_blurry          : bool  (variance < threshold)
    """
    variance = compute_laplacian_variance(gray, mask)
    return {
        "laplacian_variance": variance,
        "threshold": float(threshold),
        "is_blurry": variance < threshold,
    }

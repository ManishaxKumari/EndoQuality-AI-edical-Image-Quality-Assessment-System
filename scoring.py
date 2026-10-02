"""
scoring.py
==========
Turns the raw measurements into a 0-100 heuristic quality score.

    Sharpness  : 40 points
    Brightness : 25 points
    Exposure   : 20 points
    Contrast   : 15 points
                 ---------
                 100 points

Each component is a simple, transparent formula that maps a measurement to a
fraction between 0 and 1, which is then multiplied by its weight. There is no
learning and no randomness: the same image and the same thresholds always
give the same score.

This score is a HEURISTIC. It has not been validated against expert
judgement or diagnostic outcomes.
"""

import numpy as np

from config import SCORE_BANDS, SCORE_WEIGHTS, QualityThresholds


def _clip01(value: float) -> float:
    """Keep a value inside [0, 1]."""
    return float(np.clip(value, 0.0, 1.0))


def sharpness_fraction(laplacian_variance: float, t: QualityThresholds) -> float:
    """
    Full points once the Laplacian variance reaches the blur threshold;
    below it, points shrink proportionally.

        fraction = min(1, variance / blur_threshold)
    """
    if t.blur_threshold <= 0:
        return 1.0
    return _clip01(laplacian_variance / t.blur_threshold)


def brightness_fraction(mean: float, t: QualityThresholds) -> float:
    """
    Full points inside the ideal range [ideal_min, ideal_max].
    Below the range: falls linearly to 0 at black (0).
    Above the range: falls linearly to 0 at white (255).
    """
    lo, hi = t.ideal_brightness_min, t.ideal_brightness_max
    if lo <= mean <= hi:
        return 1.0
    if mean < lo:
        return _clip01(mean / lo) if lo > 0 else 0.0
    return _clip01((255.0 - mean) / (255.0 - hi)) if hi < 255 else 0.0


def exposure_fraction(overexposed_percent: float, t: QualityThresholds) -> float:
    """
    Full points with no blown-out pixels; half points exactly at the limit;
    zero points at twice the limit.

        fraction = 1 - overexposed% / (2 * limit)
    """
    limit = t.overexposure_percent_limit
    if limit <= 0:
        return 1.0 if overexposed_percent == 0 else 0.0
    return _clip01(1.0 - overexposed_percent / (2.0 * limit))


def contrast_fraction(contrast_std: float, t: QualityThresholds) -> float:
    """
    Full points at or above `good_contrast_level`, proportional below.

        fraction = min(1, std / good_contrast_level)
    """
    if t.good_contrast_level <= 0:
        return 1.0
    return _clip01(contrast_std / t.good_contrast_level)


def score_band(total: float) -> str:
    """Map a numeric score to a word (Excellent / Good / Fair / Poor)."""
    for minimum, label in SCORE_BANDS:
        if total >= minimum:
            return label
    return SCORE_BANDS[-1][1]


def compute_quality_score(metrics: dict, t: QualityThresholds) -> dict:
    """
    Compute every component score and the total.

    Parameters
    ----------
    metrics : dict with keys laplacian_variance, mean_brightness,
              overexposed_percent, contrast_std.
    t       : thresholds in use.

    Returns
    -------
    dict with 'components' (name -> {score, max, fraction}), 'total', 'band'.
    """
    fractions = {
        "sharpness": sharpness_fraction(metrics["laplacian_variance"], t),
        "brightness": brightness_fraction(metrics["mean_brightness"], t),
        "exposure": exposure_fraction(metrics["overexposed_percent"], t),
        "contrast": contrast_fraction(metrics["contrast_std"], t),
    }

    components = {}
    for name, fraction in fractions.items():
        weight = SCORE_WEIGHTS[name]
        components[name] = {
            "score": round(fraction * weight, 1),
            "max": weight,
            "fraction": fraction,
        }

    total = round(sum(c["score"] for c in components.values()), 1)
    return {"components": components, "total": total, "band": score_band(total)}

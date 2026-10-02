"""
visualization.py
================
Matplotlib charts used by the dashboard. Each function returns a Figure;
the caller is responsible for closing it (plt.close(fig)) after display so
memory does not build up in a long-running Streamlit session.
"""

from typing import Optional

import matplotlib

matplotlib.use("Agg")  # no GUI window needed (works on servers / CPU-only)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.brightness_analyzer import compute_fov_mask, to_grayscale  # noqa: E402

# Palette shared with the dashboard theme
BLUE = "#1F5FAE"
NAVY = "#14263D"
SKY = "#8DB7E8"
GRID = "#E3EAF3"
RED = "#C0392B"
AMBER = "#C7801E"
GREEN = "#1E8E5A"
MUTED = "#5B6B80"


def _style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=11, color=NAVY, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _pixels(gray: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None:
        return gray.ravel()
    values = gray[mask > 0]
    return values if values.size else gray.ravel()


def plot_grayscale_histogram(
    gray: np.ndarray, thresholds: dict, mask: Optional[np.ndarray] = None
) -> plt.Figure:
    """Histogram of gray values with the dark / overexposure thresholds marked."""
    values = _pixels(gray, mask)
    counts = np.bincount(values, minlength=256)
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.fill_between(np.arange(256), counts, color=SKY, alpha=0.6, step="mid")
    ax.plot(np.arange(256), counts, color=BLUE, linewidth=1)

    ax.axvline(values.mean(), color=NAVY, linestyle="-", linewidth=1.4,
               label=f"Mean = {values.mean():.1f}")
    ax.axvline(thresholds["dark_threshold"], color=AMBER, linestyle="--",
               linewidth=1.2, label=f"Too-dark limit ({thresholds['dark_threshold']:.0f})")
    ax.axvline(thresholds["overexposure_pixel_value"], color=RED, linestyle="--",
               linewidth=1.2, label=f"Blown-out level ({thresholds['overexposure_pixel_value']})")
    ax.set_xlim(0, 255)
    _style(ax, "Grayscale histogram", "Pixel intensity (0 = black, 255 = white)", "Pixel count")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig


def plot_brightness_distribution(
    rgb: np.ndarray, thresholds: dict, mask: Optional[np.ndarray] = None
) -> plt.Figure:
    """
    Two panels:
      left  - per-channel (R, G, B) intensity distributions
      right - cumulative brightness curve: the share of pixels at or below
              each gray level. The shaded zones show how much of the image
              is too dark or blown out.
    """
    fig, (ax_rgb, ax_cdf) = plt.subplots(1, 2, figsize=(10, 3.2))

    for idx, (name, color) in enumerate(zip("RGB", ("#D0453A", "#2E9E5B", BLUE))):
        channel = rgb[..., idx]
        values = channel[mask > 0] if mask is not None else channel.ravel()
        counts = np.bincount(values, minlength=256) / max(values.size, 1) * 100
        ax_rgb.plot(np.arange(256), counts, color=color, linewidth=1.2, label=name)
    ax_rgb.set_xlim(0, 255)
    _style(ax_rgb, "Colour channel distribution", "Intensity", "% of pixels")
    ax_rgb.legend(fontsize=8, frameon=False)

    gray = to_grayscale(rgb)
    values = _pixels(gray, mask)
    counts = np.bincount(values, minlength=256)
    cdf = np.cumsum(counts) / max(values.size, 1) * 100
    dark = thresholds["dark_threshold"]
    blown = thresholds["overexposure_pixel_value"]
    ax_cdf.axvspan(0, dark, color=AMBER, alpha=0.12, label="Too-dark zone")
    ax_cdf.axvspan(blown, 255, color=RED, alpha=0.12, label="Blown-out zone")
    ax_cdf.plot(np.arange(256), cdf, color=BLUE, linewidth=2)
    ax_cdf.set_xlim(0, 255)
    ax_cdf.set_ylim(0, 101)
    _style(ax_cdf, "Cumulative brightness", "Gray level", "% of pixels at or below")
    ax_cdf.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def plot_score_breakdown(score: dict) -> plt.Figure:
    """Horizontal bars: points earned vs. points available per component."""
    comps = score["components"]
    names = [n.capitalize() for n in comps]
    earned = [c["score"] for c in comps.values()]
    maximum = [c["max"] for c in comps.values()]
    fractions = [c["fraction"] for c in comps.values()]
    colors = [GREEN if f >= 0.8 else AMBER if f >= 0.5 else RED for f in fractions]

    fig, ax = plt.subplots(figsize=(7, 2.8))
    y = np.arange(len(names))[::-1]
    ax.barh(y, maximum, color=GRID, height=0.55)
    ax.barh(y, earned, color=colors, height=0.55)
    for yi, e, m in zip(y, earned, maximum):
        ax.text(m + 0.6, yi, f"{e:.1f} / {m}", va="center", fontsize=9, color=NAVY)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9, color=NAVY)
    ax.set_xlim(0, max(maximum) + 9)
    _style(ax, f"Score breakdown  ·  total {score['total']:.1f} / 100", "Points", "")
    ax.grid(False)
    fig.tight_layout()
    return fig


def plot_side_by_side(original: np.ndarray, enhanced: np.ndarray) -> plt.Figure:
    """Original and enhanced images next to each other."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    for ax, image, title in zip(axes, (original, enhanced), ("Original", "Enhanced")):
        ax.imshow(image)
        ax.set_title(title, fontsize=11, color=NAVY, fontweight="bold")
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_histogram_comparison(
    original: np.ndarray, enhanced: np.ndarray, border_threshold: int = 15
) -> plt.Figure:
    """
    Overlay of grayscale histograms before and after enhancement.
    The black frame (found on the ORIGINAL image) is excluded from both, so
    its huge spike at 0 does not flatten the rest of the chart.
    """
    mask, _ = compute_fov_mask(to_grayscale(original), border_threshold)
    fig, ax = plt.subplots(figsize=(7, 3))
    for image, label, color, alpha in (
        (original, "Original", MUTED, 0.35),
        (enhanced, "Enhanced", BLUE, 0.45),
    ):
        values = _pixels(to_grayscale(image), mask)
        counts = np.bincount(values, minlength=256) / values.size * 100
        ax.fill_between(np.arange(256), counts, color=color, alpha=alpha, label=label, step="mid")
    ax.set_xlim(0, 255)
    _style(ax, "Brightness histogram: before vs. after", "Pixel intensity", "% of pixels")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    return fig

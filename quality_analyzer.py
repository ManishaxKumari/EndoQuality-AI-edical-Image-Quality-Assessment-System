"""
quality_analyzer.py
===================
The main analysis pipeline. It ties the other modules together:

    upload bytes
        -> load_image()            validate + decode to RGB numpy array
        -> prepare_for_analysis()  resize to a standard size
        -> analyze_image()         blur, brightness, exposure, contrast
                                   -> score -> classification -> explanations
        -> QualityReport           exportable as CSV or a text report
"""

import csv
import io
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import (
    ALLOWED_EXTENSIONS,
    ALLOWED_PIL_FORMATS,
    ANALYSIS_MAX_SIDE,
    APP_NAME,
    APP_VERSION,
    CATEGORY_BLURRY,
    CATEGORY_CLEAR,
    CATEGORY_DARK,
    CATEGORY_OVEREXPOSED,
    DISCLAIMER,
    MAX_FILE_SIZE_MB,
    MIN_IMAGE_SIDE,
    QualityThresholds,
)
from src.blur_detector import detect_blur
from src.brightness_analyzer import (
    analyze_brightness,
    analyze_contrast,
    analyze_exposure,
    compute_fov_mask,
    to_grayscale,
)
from src.scoring import compute_quality_score


class ImageLoadError(ValueError):
    """Raised with a user-friendly message when an upload cannot be used."""


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------
def load_image(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Validate an uploaded file and decode it into an RGB uint8 array.

    Checks performed:
      * file extension is jpg / jpeg / png
      * file is not empty and not larger than MAX_FILE_SIZE_MB
      * the content is a real, uncorrupted JPEG or PNG (not just renamed)
      * the image is at least MIN_IMAGE_SIDE pixels on each side

    Also handles: EXIF rotation from cameras, transparent PNGs, grayscale
    and palette images (all converted to 3-channel RGB).
    """
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        raise ImageLoadError(
            f"'{filename}' is not a supported file type. "
            "Upload a JPG, JPEG or PNG image."
        )

    if not file_bytes:
        raise ImageLoadError(f"'{filename}' is empty. Upload a different file.")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ImageLoadError(
            f"'{filename}' is {size_mb:.1f} MB. The limit is {MAX_FILE_SIZE_MB} MB."
        )

    try:
        # verify() checks file integrity but leaves the object unusable,
        # so the image is opened a second time to actually read pixels.
        with Image.open(io.BytesIO(file_bytes)) as probe:
            probe.verify()
            detected_format = probe.format
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
    except Image.DecompressionBombError:
        raise ImageLoadError(
            f"'{filename}' has too many pixels to process safely."
        )
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageLoadError(
            f"'{filename}' could not be read as an image. The file may be "
            "corrupted or not a real JPG/PNG."
        ) from exc

    if detected_format not in ALLOWED_PIL_FORMATS:
        raise ImageLoadError(
            f"'{filename}' contains {detected_format} data, not JPG or PNG."
        )

    # Respect camera orientation metadata (otherwise photos can appear rotated)
    image = ImageOps.exif_transpose(image)

    # Transparent pixels are placed on black, matching an endoscope frame
    if image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        image = Image.alpha_composite(background, rgba)

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

    height, width = rgb.shape[:2]
    if min(height, width) < MIN_IMAGE_SIDE:
        raise ImageLoadError(
            f"'{filename}' is only {width}x{height} pixels. Images need to be "
            f"at least {MIN_IMAGE_SIDE}x{MIN_IMAGE_SIDE} pixels."
        )
    return rgb


def prepare_for_analysis(rgb: np.ndarray, max_side: int = ANALYSIS_MAX_SIDE) -> np.ndarray:
    """
    Resize so the longer side equals `max_side` (keeping the aspect ratio).

    Why always resize, even small images? Laplacian variance changes with
    resolution, so measuring every image at the same scale is what makes a
    single blur threshold meaningful. INTER_AREA is best for shrinking,
    INTER_CUBIC for enlarging. (Enlarging a tiny image adds no detail, so
    low-resolution uploads will honestly score as less sharp.)
    """
    height, width = rgb.shape[:2]
    scale = max_side / float(max(height, width))
    if abs(scale - 1.0) < 1e-6:
        return rgb.copy()
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(rgb, new_size, interpolation=interpolation)


# --------------------------------------------------------------------------
# Report object
# --------------------------------------------------------------------------
@dataclass
class QualityReport:
    """Everything the dashboard needs to display one analysis."""

    filename: str
    original_size: tuple            # (width, height) of the upload
    analysis_size: tuple            # (width, height) actually measured
    metrics: dict                   # raw measurements
    flags: dict                     # True/False problem flags
    score: dict                     # output of compute_quality_score
    category: str                   # one of the four classes
    problems: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    thresholds: dict = field(default_factory=dict)
    fov_coverage: float = 1.0
    fov_mask: Optional[np.ndarray] = field(default=None, repr=False)
    timestamp: str = field(
        default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    # ---- flat dictionary (one row of the CSV) ---------------------------
    def to_flat_dict(self) -> dict:
        row = {
            "timestamp": self.timestamp,
            "filename": self.filename,
            "original_width": self.original_size[0],
            "original_height": self.original_size[1],
            "analysis_width": self.analysis_size[0],
            "analysis_height": self.analysis_size[1],
            "category": self.category,
            "quality_score": self.score["total"],
            "score_band": self.score["band"],
        }
        for name, comp in self.score["components"].items():
            row[f"{name}_score"] = comp["score"]
            row[f"{name}_max"] = comp["max"]
        row.update({
            "laplacian_variance": round(self.metrics["laplacian_variance"], 2),
            "mean_brightness": round(self.metrics["mean_brightness"], 2),
            "overexposed_percent": round(self.metrics["overexposed_percent"], 3),
            "contrast_std": round(self.metrics["contrast_std"], 2),
            "field_of_view_coverage": round(self.fov_coverage, 3),
        })
        for name, value in self.flags.items():
            row[name] = value
        row["problems"] = " | ".join(self.problems) if self.problems else "None"
        for name, value in self.thresholds.items():
            row[f"threshold_{name}"] = value
        return row

    def to_csv_bytes(self) -> bytes:
        row = self.to_flat_dict()
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
        return buffer.getvalue().encode("utf-8")

    def to_text_report(self) -> str:
        line = "=" * 64
        comps = self.score["components"]
        t = self.thresholds
        out = [
            line,
            f"{APP_NAME} v{APP_VERSION} - Image Quality Report",
            line,
            f"Generated : {self.timestamp}",
            f"File      : {self.filename}",
            f"Size      : {self.original_size[0]} x {self.original_size[1]} px "
            f"(measured at {self.analysis_size[0]} x {self.analysis_size[1]} px)",
            "",
            "RESULT",
            "-" * 64,
            f"Classification : {self.category}",
            f"Quality score  : {self.score['total']:.1f} / 100 ({self.score['band']})",
            "",
            "SCORE BREAKDOWN",
            "-" * 64,
        ]
        for name, comp in comps.items():
            out.append(f"  {name.capitalize():<11}: {comp['score']:>5.1f} / {comp['max']}")
        out += [
            "",
            "MEASUREMENTS",
            "-" * 64,
            f"  Laplacian variance  : {self.metrics['laplacian_variance']:.2f}"
            f"   (blurry below {t['blur_threshold']})",
            f"  Mean brightness     : {self.metrics['mean_brightness']:.2f}"
            f"   (too dark below {t['dark_threshold']}, ideal "
            f"{t['ideal_brightness_min']}-{t['ideal_brightness_max']})",
            f"  Overexposed pixels  : {self.metrics['overexposed_percent']:.2f} %"
            f"   (limit {t['overexposure_percent_limit']} % at gray >= "
            f"{t['overexposure_pixel_value']})",
            f"  Contrast (std dev)  : {self.metrics['contrast_std']:.2f}"
            f"   (low below {t['low_contrast_threshold']})",
            f"  Field of view used  : {self.fov_coverage * 100:.1f} % of image",
            "",
            "DETECTED PROBLEMS",
            "-" * 64,
        ]
        out += [f"  - {p}" for p in self.problems] or ["  None detected."]
        if self.notes:
            out += ["", "NOTES", "-" * 64] + [f"  - {n}" for n in self.notes]
        out += ["", "DISCLAIMER", "-" * 64, f"  {DISCLAIMER}", line]
        return "\n".join(out)


# --------------------------------------------------------------------------
# Classification and explanations
# --------------------------------------------------------------------------
def classify(flags: dict) -> str:
    """
    Pick ONE of the four categories using a fixed priority order:

        1. Too Dark     2. Overexposed     3. Blurry     4. Clear

    Exposure problems come first because they distort the blur measurement:
    a very dark image has almost no visible edges, so it also gets a low
    Laplacian variance even if the camera was perfectly focused. Calling it
    "blurry" would point the user at the wrong fix.
    """
    if flags["is_too_dark"]:
        return CATEGORY_DARK
    if flags["is_overexposed"]:
        return CATEGORY_OVEREXPOSED
    if flags["is_blurry"]:
        return CATEGORY_BLURRY
    return CATEGORY_CLEAR


def explain(metrics: dict, flags: dict, t: QualityThresholds) -> tuple:
    """Build plain-language problem descriptions and extra notes."""
    problems, notes = [], []
    m = metrics

    if flags["is_too_dark"]:
        problems.append(
            f"Too dark: mean brightness is {m['mean_brightness']:.1f}, below the "
            f"threshold of {t.dark_threshold:.0f}. Details in shadowed areas may "
            "be hidden."
        )
    if flags["is_overexposed"]:
        reasons = []
        if flags["has_blown_out_pixels"]:
            reasons.append(
                f"{m['overexposed_percent']:.1f}% of pixels are blown out "
                f"(limit {t.overexposure_percent_limit:.0f}%)"
            )
        if flags["is_too_bright"]:
            reasons.append(
                f"mean brightness {m['mean_brightness']:.1f} is above "
                f"{t.bright_mean_threshold:.0f}"
            )
        problems.append(
            "Overexposed: " + " and ".join(reasons)
            + ". Blown-out regions contain no recoverable detail."
        )
    if flags["is_blurry"]:
        problems.append(
            f"Blurry: Laplacian variance is {m['laplacian_variance']:.1f}, below "
            f"the threshold of {t.blur_threshold:.0f}. Edges and fine texture "
            "look soft."
        )
    if flags["is_low_contrast"]:
        problems.append(
            f"Low contrast: standard deviation is {m['contrast_std']:.1f}, below "
            f"{t.low_contrast_threshold:.0f}. The image looks flat."
        )

    if flags["is_blurry"] and (flags["is_too_dark"] or flags["is_overexposed"]):
        notes.append(
            "The blur result is less reliable here because poor exposure also "
            "lowers the Laplacian variance. Fix the exposure first, then "
            "re-check sharpness."
        )
    if flags["is_blurry"] and flags["is_low_contrast"]:
        notes.append(
            "Low contrast also lowers the Laplacian variance (it scales with "
            "contrast squared), so this image may be flatter rather than "
            "truly out of focus."
        )
    if (
        not flags["is_too_dark"]
        and not flags["is_overexposed"]
        and not (t.ideal_brightness_min <= m["mean_brightness"] <= t.ideal_brightness_max)
    ):
        side = "darker" if m["mean_brightness"] < t.ideal_brightness_min else "brighter"
        notes.append(
            f"Brightness ({m['mean_brightness']:.1f}) is acceptable but {side} "
            f"than the ideal range ({t.ideal_brightness_min:.0f}-"
            f"{t.ideal_brightness_max:.0f}), so some brightness points were lost."
        )
    if 0 < m["overexposed_percent"] <= t.overexposure_percent_limit:
        notes.append(
            f"{m['overexposed_percent']:.1f}% of pixels are blown out. Small "
            "specular highlights like this are normal in endoscopy."
        )
    return problems, notes


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def analyze_image(
    rgb: np.ndarray,
    thresholds: Optional[QualityThresholds] = None,
    filename: str = "image",
) -> QualityReport:
    """
    Run the full quality assessment on one RGB image.

    Parameters
    ----------
    rgb        : H x W x 3 uint8 array in RGB order (from load_image()).
    thresholds : QualityThresholds; defaults are used if None.
    filename   : name shown in reports.
    """
    t = thresholds or QualityThresholds()
    original_size = (rgb.shape[1], rgb.shape[0])

    small = prepare_for_analysis(rgb)
    gray = to_grayscale(small)

    # Optional: exclude the black frame around the field of view
    mask, coverage = (None, 1.0)
    if t.ignore_black_border:
        mask, coverage = compute_fov_mask(gray, t.border_intensity_threshold)

    blur = detect_blur(gray, t.blur_threshold, mask)
    brightness = analyze_brightness(gray, t.dark_threshold, t.bright_mean_threshold, mask)
    exposure = analyze_exposure(gray, t.overexposure_pixel_value, t.overexposure_percent_limit, mask)
    contrast = analyze_contrast(gray, t.low_contrast_threshold, mask)

    metrics = {
        "laplacian_variance": blur["laplacian_variance"],
        "mean_brightness": brightness["mean_brightness"],
        "overexposed_percent": exposure["overexposed_percent"],
        "contrast_std": contrast["contrast_std"],
    }
    flags = {
        "is_blurry": blur["is_blurry"],
        "is_too_dark": brightness["is_too_dark"],
        "is_too_bright": brightness["is_too_bright"],
        "has_blown_out_pixels": exposure["is_overexposed"],
        # An image is "Overexposed" if too many pixels are blown out OR the
        # whole image is washed out (high mean), even below pure white.
        "is_overexposed": exposure["is_overexposed"] or brightness["is_too_bright"],
        "is_low_contrast": contrast["is_low_contrast"],
    }

    score = compute_quality_score(metrics, t)
    category = classify(flags)
    problems, notes = explain(metrics, flags, t)

    return QualityReport(
        filename=filename,
        original_size=original_size,
        analysis_size=(small.shape[1], small.shape[0]),
        metrics=metrics,
        flags=flags,
        score=score,
        category=category,
        problems=problems,
        notes=notes,
        thresholds=t.to_dict(),
        fov_coverage=coverage,
        fov_mask=mask,
    )

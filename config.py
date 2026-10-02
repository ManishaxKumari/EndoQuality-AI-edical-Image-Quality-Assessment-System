"""
config.py
=========
Central configuration for EndoQuality AI.

Every number that decides "good" vs "bad" lives here, so you can tune the
system without touching the algorithm code. The Streamlit sidebar also lets
you change the thresholds while the app is running.

IMPORTANT: The default thresholds are reasonable starting points chosen by
experimenting with sample images. They are NOT clinically validated. Different
endoscopes, light sources, resolutions and compression settings need
different values. Always calibrate on images from your own equipment.
"""

from dataclasses import dataclass, asdict

# --------------------------------------------------------------------------
# Application metadata
# --------------------------------------------------------------------------
APP_NAME = "EndoQuality AI"
APP_SUBTITLE = "Automated Endoscopy Image Quality Assessment"
APP_VERSION = "1.0.0"

DISCLAIMER = (
    "EndoQuality AI is a rule-based image quality tool built for learning and "
    "prototyping. Its scores are heuristics, not a medically validated metric, "
    "and it must not be used for clinical decisions."
)

# --------------------------------------------------------------------------
# File upload rules
# --------------------------------------------------------------------------
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png")
ALLOWED_PIL_FORMATS = ("JPEG", "PNG")   # the real file content must match
MAX_FILE_SIZE_MB = 20
MIN_IMAGE_SIDE = 32                     # smaller images are rejected

# --------------------------------------------------------------------------
# Analysis resolution
# --------------------------------------------------------------------------
# Laplacian variance depends heavily on image size: the same scene at 4000 px
# and at 400 px gives very different numbers. To make one threshold usable for
# every upload, each image is resized so its LONGER side equals this value
# before any metric is measured.
ANALYSIS_MAX_SIDE = 512


@dataclass
class QualityThresholds:
    """All tunable thresholds used by the analysis pipeline."""

    # Blur: Laplacian variance below this value => "Blurry"
    blur_threshold: float = 100.0

    # Brightness (mean grayscale intensity, 0-255)
    dark_threshold: float = 50.0          # mean below this => "Too Dark"
    ideal_brightness_min: float = 80.0    # full brightness points inside
    ideal_brightness_max: float = 180.0   #   [ideal_min, ideal_max]
    bright_mean_threshold: float = 200.0  # mean above this => "Overexposed"

    # Overexposure: a pixel is "blown out" if its gray value >= this ...
    overexposure_pixel_value: int = 240
    # ... and the image is "Overexposed" if more than this % of pixels are.
    # Endoscopy images almost always contain a few small specular highlights
    # (light reflecting off wet mucosa), so this must not be 0 %.
    overexposure_percent_limit: float = 10.0

    # Contrast (grayscale standard deviation)
    low_contrast_threshold: float = 30.0  # std below this => low contrast
    good_contrast_level: float = 55.0     # std at/above this => full points

    # Endoscopy images usually have a black frame around a circular or
    # octagonal field of view. Including that frame drags the mean brightness
    # down, and its hard edge inflates the blur metric. When enabled, the
    # frame is detected and excluded from every measurement.
    ignore_black_border: bool = True
    border_intensity_threshold: int = 15  # gray values <= this count as frame

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# Scoring (rule-based, 0-100)
# --------------------------------------------------------------------------
SCORE_WEIGHTS = {
    "sharpness": 40,
    "brightness": 25,
    "exposure": 20,
    "contrast": 15,
}

# Score bands shown next to the numeric score (checked top to bottom)
SCORE_BANDS = [
    (85, "Excellent"),
    (70, "Good"),
    (50, "Fair"),
    (0, "Poor"),
]

# --------------------------------------------------------------------------
# Classification labels
# --------------------------------------------------------------------------
CATEGORY_CLEAR = "Clear Image"
CATEGORY_BLURRY = "Blurry Image"
CATEGORY_DARK = "Too Dark"
CATEGORY_OVEREXPOSED = "Overexposed"

# --------------------------------------------------------------------------
# Enhancement defaults (used to initialise the sliders)
# --------------------------------------------------------------------------
ENHANCEMENT_DEFAULTS = {
    "clahe_clip_limit": 2.0,
    "clahe_tile_grid": 8,
    "gaussian_kernel": 5,
    "gaussian_sigma": 1.0,
    "sharpen_amount": 1.0,
    "sharpen_sigma": 1.5,
    "brightness_beta": 30,
    "contrast_alpha": 1.0,
}

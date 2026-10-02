"""
test_pipeline.py
================
Checks the analysis and enhancement pipeline. No extra test framework is
needed: run it as a plain script from the project root.

    python tests/generate_samples.py     # create the synthetic samples first
    python tests/test_pipeline.py

(If you have pytest installed, `pytest tests/` also works.)

Results are printed and saved to outputs/test_results.csv.
"""

import csv
import io
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import (  # noqa: E402
    CATEGORY_BLURRY, CATEGORY_CLEAR, CATEGORY_DARK, CATEGORY_OVEREXPOSED, QualityThresholds,
)
from src.enhancement import (  # noqa: E402
    adjust_brightness, apply_clahe, enhance_pipeline, reduce_noise_gaussian, sharpen, to_png_bytes,
)
from src.quality_analyzer import ImageLoadError, analyze_image, load_image  # noqa: E402

SAMPLE_DIR = os.path.join(ROOT, "assets", "samples")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# What each synthetic sample is DESIGNED to show
EXPECTED = {
    "clear.png": CATEGORY_CLEAR,
    "clear_no_border.png": CATEGORY_CLEAR,
    "blurry.png": CATEGORY_BLURRY,
    "dark.png": CATEGORY_DARK,
    "overexposed.png": CATEGORY_OVEREXPOSED,
    "low_contrast.png": CATEGORY_CLEAR,  # stays sharp & well lit; only the low-contrast flag should fire
}


def _load_sample(name):
    with open(os.path.join(SAMPLE_DIR, name), "rb") as handle:
        return load_image(handle.read(), name)


def _png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------
# Classification on synthetic samples
# --------------------------------------------------------------------------
def test_sample_classification():
    missing = [n for n in EXPECTED if not os.path.exists(os.path.join(SAMPLE_DIR, n))]
    assert not missing, f"Missing samples {missing}. Run: python tests/generate_samples.py"
    for name, expected in EXPECTED.items():
        report = analyze_image(_load_sample(name), filename=name)
        assert report.category == expected, f"{name}: got {report.category}, expected {expected}"


def test_low_contrast_flag():
    report = analyze_image(_load_sample("low_contrast.png"))
    assert report.flags["is_low_contrast"]
    assert analyze_image(_load_sample("clear.png")).flags["is_low_contrast"] is False


def test_clear_scores_higher_than_degraded():
    clear = analyze_image(_load_sample("clear.png")).score["total"]
    for name in ("blurry.png", "dark.png", "overexposed.png"):
        assert analyze_image(_load_sample(name)).score["total"] < clear, name


def test_deterministic():
    image = _load_sample("clear.png")
    a, b = analyze_image(image), analyze_image(image)
    assert a.metrics == b.metrics and a.score == b.score


def test_thresholds_are_respected():
    image = _load_sample("clear.png")
    variance = analyze_image(image).metrics["laplacian_variance"]
    strict = QualityThresholds(blur_threshold=variance + 1)
    assert analyze_image(image, strict).category == CATEGORY_BLURRY


def test_score_range_and_components():
    for name in EXPECTED:
        score = analyze_image(_load_sample(name)).score
        assert 0 <= score["total"] <= 100
        assert abs(sum(c["score"] for c in score["components"].values()) - score["total"]) < 0.2


# --------------------------------------------------------------------------
# Edge cases
# --------------------------------------------------------------------------
def test_uniform_images_do_not_crash():
    for value, expected in ((0, CATEGORY_DARK), (255, CATEGORY_OVEREXPOSED), (128, CATEGORY_BLURRY)):
        report = analyze_image(np.full((200, 300, 3), value, np.uint8))
        assert report.category == expected, (value, report.category)
        assert report.metrics["contrast_std"] == 0


def test_rgba_grayscale_and_palette_inputs():
    rgba = np.zeros((64, 64, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255
    assert load_image(_png_bytes(rgba), "a.png").shape == (64, 64, 3)
    gray = np.full((64, 64), 90, np.uint8)
    assert load_image(_png_bytes(gray), "g.png").shape == (64, 64, 3)
    palette = Image.fromarray(np.full((64, 64, 3), 50, np.uint8)).convert("P")
    buffer = io.BytesIO()
    palette.save(buffer, format="PNG")
    assert load_image(buffer.getvalue(), "p.png").shape == (64, 64, 3)


def test_rgb_channel_order_preserved():
    red = np.zeros((64, 64, 3), np.uint8)
    red[..., 0] = 255
    loaded = load_image(_png_bytes(red), "red.png")
    assert loaded[0, 0, 0] == 255 and loaded[0, 0, 2] == 0


def test_invalid_files_rejected():
    cases = [
        (b"not an image at all", "fake.png"),
        (b"", "empty.jpg"),
        (_png_bytes(np.zeros((64, 64, 3), np.uint8)), "doc.pdf"),
        (_png_bytes(np.zeros((10, 10, 3), np.uint8)), "tiny.png"),
    ]
    gif = io.BytesIO()
    Image.fromarray(np.zeros((64, 64, 3), np.uint8)).save(gif, format="GIF")
    cases.append((gif.getvalue(), "renamed_gif.png"))
    for data, name in cases:
        try:
            load_image(data, name)
        except ImageLoadError:
            continue
        raise AssertionError(f"{name} should have been rejected")


def test_non_square_and_large_images():
    wide = np.random.default_rng(0).integers(0, 255, (300, 1600, 3), dtype=np.uint8)
    report = analyze_image(wide)
    assert max(report.analysis_size) == 512 and report.original_size == (1600, 300)


# --------------------------------------------------------------------------
# Enhancement
# --------------------------------------------------------------------------
def test_enhancement_outputs_valid_images():
    image = _load_sample("clear.png")
    for result in (apply_clahe(image), reduce_noise_gaussian(image, 4), sharpen(image),
                   adjust_brightness(image, 40)):
        assert result.shape == image.shape and result.dtype == np.uint8
    assert to_png_bytes(image)[:8] == b"\x89PNG\r\n\x1a\n"


def test_enhancement_effects():
    dark = _load_sample("dark.png")
    assert adjust_brightness(dark, 60).mean() > dark.mean()
    clear = _load_sample("clear.png")
    assert analyze_image(sharpen(clear, 1.5)).metrics["laplacian_variance"] > \
        analyze_image(clear).metrics["laplacian_variance"]
    assert analyze_image(reduce_noise_gaussian(clear, 7, 2.0)).metrics["laplacian_variance"] < \
        analyze_image(clear).metrics["laplacian_variance"]


def test_pipeline_order_and_no_mutation():
    image = _load_sample("clear.png")
    copy = image.copy()
    enhance_pipeline(image, [("Sharpening", {}), ("CLAHE", {})])
    assert np.array_equal(image, copy)


# --------------------------------------------------------------------------
# Script runner: prints a results table and writes outputs/test_results.csv
# --------------------------------------------------------------------------
def main() -> int:
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    failures = 0
    print("Running tests\n" + "-" * 60)
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {exc}")

    print("\nActual results on synthetic samples (default thresholds)\n" + "-" * 60)
    rows = []
    for name, expected in EXPECTED.items():
        path = os.path.join(SAMPLE_DIR, name)
        if not os.path.exists(path):
            continue
        report = analyze_image(_load_sample(name), filename=name)
        row = report.to_flat_dict()
        row["expected_category"] = expected
        row["match"] = report.category == expected
        rows.append(row)
        print(f"{name:21s} expected={expected:13s} actual={report.category:13s} "
              f"score={report.score['total']:5.1f}  lapvar={report.metrics['laplacian_variance']:7.1f}  "
              f"mean={report.metrics['mean_brightness']:6.1f}  "
              f"blown={report.metrics['overexposed_percent']:5.2f}%  "
              f"std={report.metrics['contrast_std']:5.1f}")

    if rows:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out = os.path.join(OUTPUT_DIR, "test_results.csv")
        with open(out, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nSaved {os.path.relpath(out, ROOT)}")

    print(f"\n{len(tests) - failures}/{len(tests)} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

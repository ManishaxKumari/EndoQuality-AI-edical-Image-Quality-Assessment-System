"""
generate_samples.py
===================
Creates SYNTHETIC endoscopy-style test images in assets/samples/.

No real patient images are bundled with this project (they carry privacy
and licensing restrictions). Instead, this script draws a pinkish
"mucosa-like" texture with vessel-like lines, a few specular highlights and
a black circular frame, then degrades it in controlled ways:

    clear.png             - the base image
    clear_no_border.png   - base image without the black frame
    blurry.png            - Gaussian defocus blur
    dark.png              - strongly under-exposed, with sensor noise
    overexposed.png       - brightness pushed until large areas clip to white
    low_contrast.png      - flattened tonal range (stays sharp and well lit)

These images are only for checking that the code behaves as designed.
They are NOT a substitute for testing on real endoscopy data.

Run:  python tests/generate_samples.py
"""

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "assets", "samples")
SIZE = 640


def _circular_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * 0.46), 255, -1)
    return mask


def make_base_image(seed: int = 7) -> np.ndarray:
    """Draw a mucosa-like RGB scene (no frame yet)."""
    rng = np.random.default_rng(seed)
    h = w = SIZE

    # Multi-scale texture: sum of blurred noise layers (coarse folds + fine grain)
    texture = np.zeros((h, w), np.float32)
    for sigma, weight in ((40, 0.5), (12, 0.35), (3, 0.3), (1.2, 0.3)):
        layer = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
        layer /= (np.abs(layer).max() + 1e-6)
        texture += weight * layer

    # Illumination: brighter centre, darker edges, plus a dark "lumen" blob
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((xx - w * 0.5) ** 2 + (yy - h * 0.5) ** 2) / (w * 0.5)
    light = 1.0 - 0.45 * r ** 2
    lumen = np.exp(-(((xx - w * 0.62) ** 2 + (yy - h * 0.42) ** 2) / (2 * (w * 0.08) ** 2)))
    light -= 0.55 * lumen

    base = np.clip(0.60 * light + 1.1 * texture, 0, 1)
    rgb = np.stack([
        200 * base + 40,   # R: mucosa is red/pink dominant
        110 * base + 25,   # G
        100 * base + 25,   # B
    ], axis=-1)

    # Vessel-like random curves (darker red)
    for _ in range(18):
        pts = [rng.uniform(0, w, 2)]
        angle = rng.uniform(0, 2 * np.pi)
        for _ in range(25):
            angle += rng.normal(0, 0.35)
            pts.append(pts[-1] + 14 * np.array([np.cos(angle), np.sin(angle)]))
        cv2.polylines(rgb, [np.array(pts, np.int32)], False,
                      (95, 20, 30), thickness=int(rng.integers(1, 4)), lineType=cv2.LINE_AA)

    # A few small specular highlights (normal in endoscopy)
    for _ in range(6):
        center = tuple(int(v) for v in rng.uniform(w * 0.2, w * 0.8, 2))
        axes = (int(rng.integers(3, 9)), int(rng.integers(2, 6)))
        cv2.ellipse(rgb, center, axes, float(rng.uniform(0, 180)), 0, 360, (255, 255, 250), -1, cv2.LINE_AA)

    return np.clip(rgb, 0, 255).astype(np.uint8)


def apply_frame(rgb: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[_circular_mask(rgb.shape[0]) == 0] = 0
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(11)
    scene = make_base_image()

    samples = {
        "clear.png": apply_frame(scene),
        "clear_no_border.png": scene,
        # Defocus: the scene is blurred but the scope's frame stays sharp
        "blurry.png": apply_frame(cv2.GaussianBlur(scene, (0, 0), 4.0)),
        "dark.png": apply_frame(np.clip(
            scene.astype(np.float32) * 0.22 + rng.normal(0, 3, scene.shape), 0, 255
        ).astype(np.uint8)),
        "overexposed.png": apply_frame(np.clip(
            scene.astype(np.float32) * 1.9 + 70, 0, 255
        ).astype(np.uint8)),
        "low_contrast.png": apply_frame(np.clip(
            (scene.astype(np.float32) - 128) * 0.55 + 135, 0, 255
        ).astype(np.uint8)),
    }

    for name, image in samples.items():
        path = os.path.join(OUT_DIR, name)
        # OpenCV writes BGR, our arrays are RGB -> convert before saving
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print(f"saved {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())

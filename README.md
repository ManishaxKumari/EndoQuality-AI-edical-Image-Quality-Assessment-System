# EndoQuality AI – Automated Endoscopy Image Quality Assessment System

A beginner-friendly computer vision project that checks gastrointestinal endoscopy images for common quality problems using classic, explainable image processing. No deep learning, no model downloads, CPU only.

> **Disclaimer.** EndoQuality AI is a rule-based tool built for learning and prototyping. Its quality score is a heuristic, not a medically validated metric. It must not be used to make clinical decisions.

---

## 1. Project overview

You upload an endoscopy image. The system measures sharpness, brightness, blown-out highlights and contrast, combines them into a 0–100 quality score, assigns one of four categories, and explains every problem it found in plain language:

| Category | Meaning |
|---|---|
| **Clear Image** | No critical problem detected |
| **Blurry Image** | Edges and texture are too soft |
| **Too Dark** | Overall brightness is too low |
| **Overexposed** | Too many pixels are blown out to white, or the image is washed out |

You can also enhance the image (CLAHE, noise reduction, sharpening, brightness), compare before and after, inspect the raw measurements, and download a CSV, a text report and the enhanced image.

## 2. Problem statement

Endoscopy images are captured under hard conditions: the scope moves constantly, fluid and mucus fog the lens, and the light source sits millimetres from wet, reflective tissue. Many frames end up blurred, too dark or glared. Reviewing them by hand is slow. A fast, transparent first-pass filter that flags poor-quality frames, and says *why*, is useful for building datasets, reviewing recordings or teaching image processing.

## 3. Features

- Upload JPG / JPEG / PNG with real validation (content type, corruption, size, minimum resolution)
- Blur detection with Laplacian variance
- Brightness, overexposure and contrast analysis
- Automatic detection and exclusion of the black frame around the endoscope's field of view
- Transparent 0–100 score: sharpness 40, brightness 25, exposure 20, contrast 15
- Four-way classification with plain-language explanations
- Enhancement: CLAHE, Gaussian noise reduction, unsharp-mask sharpening, brightness/contrast
- Before/after comparison, with the heuristic score recomputed on the enhanced image
- Charts: grayscale histogram, colour-channel and cumulative brightness distributions, score breakdown, before/after histogram
- Laplacian edge map and measured-region overlay
- All thresholds adjustable live in the sidebar
- Downloads: enhanced PNG, comparison figure, CSV, text report
- Synthetic sample images and an automated test suite

## 4. Project structure

```
EndoQuality-AI/
├── app.py                     # Streamlit dashboard (4 pages)
├── config.py                  # ALL thresholds, weights and labels
├── requirements.txt
├── README.md
├── .streamlit/config.toml     # white + blue theme, upload limit
├── src/
│   ├── quality_analyzer.py    # loading/validation, pipeline, classification, reports
│   ├── blur_detector.py       # Laplacian variance, edge map
│   ├── brightness_analyzer.py # brightness, overexposure, contrast, frame mask
│   ├── scoring.py             # 0-100 heuristic score
│   ├── enhancement.py         # CLAHE, denoise, sharpen, brightness
│   └── visualization.py       # Matplotlib charts
├── tests/
│   ├── generate_samples.py    # creates synthetic test images
│   ├── test_pipeline.py       # analysis + enhancement + edge-case tests
│   └── test_app.py            # dashboard smoke test (Streamlit AppTest)
├── assets/samples/            # synthetic sample images
└── outputs/                   # test results are written here
```

## 5. How the pipeline works

```
upload ─► validate & decode to RGB ─► resize (longer side = 512 px)
       ─► grayscale ─► detect black frame (optional)
       ─► blur │ brightness │ overexposure │ contrast
       ─► score (0-100) ─► classify ─► explain ─► report
```

**Why resize first?** Laplacian variance depends strongly on resolution: the same scene measured at 4000 px and 400 px gives very different numbers. Measuring every image at the same size is what makes a single blur threshold meaningful.

**Why remove the black frame?** Most endoscopy images show a circular or octagonal view inside a black border. That border drags the mean brightness down, and its hard edge is always sharp because it comes from the optics, not the tissue. In our tests the blurry sample scored a Laplacian variance of **3.0 with the frame excluded but 57.4 with it included**, and the low-contrast sample's standard deviation jumped from 18.3 to 56.9 (no longer flagged) when the frame was included. The frame is found by thresholding, morphological closing, and filling the largest outer contour (so dark areas *inside* the view, like the bowel lumen, are still measured). If the detected view covers less than 25% of the image (probably just a very dark image) or more than 98% (no frame), the whole image is used instead.

**Colour order.** Pillow and Streamlit use RGB; OpenCV defaults to BGR. This project keeps every array in RGB and always calls OpenCV with explicit RGB conversions (`cv2.COLOR_RGB2GRAY`, `cv2.COLOR_RGB2LAB`). The only BGR conversion is when the sample generator saves files with `cv2.imwrite`.

## 6. Computer vision algorithms

### 6.1 Blur detection: variance of the Laplacian

The Laplacian is the sum of second derivatives of image intensity *I*:

$$L(x,y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

On a pixel grid it is computed by convolving with this kernel:

```
[ 0   1   0 ]
[ 1  -4   1 ]
[ 0   1   0 ]
```

It answers: "how different is this pixel from the average of its four neighbours?" In flat regions the answer is about 0. At an edge it swings strongly positive on one side and negative on the other.

The blur metric is the **variance** of all Laplacian values:

$$\sigma_L^2 = \frac{1}{N}\sum_{x,y}\big(L(x,y) - \bar{L}\big)^2$$

A sharp image has many strong edges, so the values are widely spread → **high variance**. Blur acts like a low-pass filter that smooths edges, pushing every Laplacian value toward 0 → **low variance**. An image is classified as blurry when the variance is below `blur_threshold` (default 100). The Laplacian is computed in 64-bit float (`cv2.CV_64F`) because negative values would be clipped to 0 in 8-bit, corrupting the variance.

**Important property:** if you multiply an image's contrast by *k*, the Laplacian variance multiplies by *k²*. Low-contrast or dark images therefore get low blur scores even when perfectly focused. That is why exposure problems are checked *before* blur, and why the report warns when blur and poor exposure or low contrast occur together.

### 6.2 Brightness

RGB is converted to grayscale with the ITU-R BT.601 luminance weights (green counts most because human vision is most sensitive to it):

$$\text{Gray} = 0.299R + 0.587G + 0.114B$$

Mean brightness is the average gray value from 0 (black) to 255 (white):

$$\mu = \frac{1}{N}\sum_{x,y} I(x,y)$$

Below `dark_threshold` (50) → **Too Dark**. Above `bright_mean_threshold` (200) → **Overexposed** (washed out).

### 6.3 Overexposure

A pixel is "blown out" when its gray value is at or above `overexposure_pixel_value` (240). Blown-out pixels contain no recoverable detail.

$$\text{Overexposed}\,\% = 100 \cdot \frac{\#\{(x,y) : I(x,y) \ge 240\}}{N}$$

More than `overexposure_percent_limit` (10%) → **Overexposed**. The limit is not 0% because small specular highlights (light glinting off wet mucosa) are normal in endoscopy.

### 6.4 Contrast

Contrast is measured as the standard deviation of gray values (RMS contrast):

$$\sigma = \sqrt{\frac{1}{N}\sum_{x,y}\big(I(x,y) - \mu\big)^2}$$

A flat, foggy image has every pixel near the mean, giving a small σ. Below `low_contrast_threshold` (30) the image is flagged as low contrast. This is reported as a problem but is not one of the four categories.

### 6.5 Scoring (heuristic, 0–100)

| Component | Points | Formula (fraction of points earned) |
|---|---|---|
| Sharpness | 40 | `min(1, laplacian_variance / blur_threshold)` |
| Brightness | 25 | 1 inside [80, 180]; falls linearly to 0 at 0 or 255 |
| Exposure | 20 | `1 − overexposed% / (2 × limit)` (half points at the limit) |
| Contrast | 15 | `min(1, std / 55)` |

Bands: Excellent ≥ 85, Good ≥ 70, Fair ≥ 50, Poor < 50. Same image + same thresholds = same score, always. There is no randomness and no learned component.

### 6.6 Classification priority

`Too Dark` → `Overexposed` → `Blurry` → `Clear`. The first rule that fires wins. Exposure goes first because bad exposure also lowers the Laplacian variance, so calling a dark image "blurry" would point the user at the wrong fix. All detected problems are still listed, not just the winning one.

### 6.7 Enhancement

| Technique | What it does | Formula / method |
|---|---|---|
| **CLAHE** | Boosts local contrast tile by tile, with a clip limit to avoid amplifying noise | Applied to the L channel of LAB, so tissue colours are preserved |
| **Noise reduction** | Gaussian weighted average of neighbours | `cv2.GaussianBlur` (odd kernel) |
| **Sharpening** | Unsharp masking | `out = I + amount·(I − blur(I))` |
| **Brightness** | Linear adjustment | `out = clip(α·I + β, 0, 255)` |

Steps run in a fixed order: noise reduction → brightness → CLAHE → sharpening (denoising first so noise is not sharpened). Enhancement changes how an image looks; it cannot recover detail the camera never captured. Sharpening and CLAHE raise the Laplacian variance by amplifying edges *and noise*, so a higher score after enhancement does **not** mean better diagnostic value.

## 7. Installation

Requirements: Python 3.9 or newer.

```bash
# 1. Go into the project folder
cd EndoQuality-AI

# 2. (Recommended) create a virtual environment
python -m venv venv
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 8. Running the app

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. Then:

1. Go to **Image Quality Analyzer** in the sidebar.
2. Upload an image, or open *"No image at hand? Use a synthetic sample"* and click **Load sample**.
3. Read the classification, score, metric cards and detected problems. Download the CSV or text report.
4. Open **Image Enhancement**, pick techniques, adjust sliders, compare and download.
5. Open **Technical Insights** for the raw numbers, histograms, edge map and formulas.
6. Tune thresholds in the sidebar under **Analysis settings**. Results update immediately.

## 9. Testing

```bash
python tests/generate_samples.py   # (re)create synthetic samples in assets/samples/
python tests/test_pipeline.py      # 14 pipeline, edge-case and enhancement tests
python tests/test_app.py           # opens every dashboard page via Streamlit AppTest
```

`test_pipeline.py` also writes `outputs/test_results.csv`. If you have pytest: `pytest tests/test_pipeline.py`.

### 9.1 About the test images

No real patient images are included (privacy and licensing). The samples are **synthetic**: a procedurally drawn pink "mucosa-like" texture with vessel-like curves, a few specular highlights and a black circular frame, then degraded in controlled ways (Gaussian blur σ = 4, brightness × 0.22 with noise, brightness × 1.9 + 70, contrast × 0.55). They prove the code behaves as designed. They do **not** prove it works on real endoscopy images.

### 9.2 Actual test results (synthetic samples, default thresholds)

These numbers come from an actual run of `tests/test_pipeline.py`:

| Sample | Designed to be | Actual result | Score | Laplacian var. | Mean | Blown-out | Std |
|---|---|---|---|---|---|---|---|
| clear.png | Clear | **Clear Image** | 93.6 | 349.4 | 95.5 | 0.14% | 31.9 |
| clear_no_border.png | Clear | **Clear Image** | 94.2 | 276.5 | 85.5 | 0.09% | 34.0 |
| blurry.png | Blurry | **Blurry Image** | 53.8 | 3.0 | 95.4 | 0.00% | 27.9 |
| dark.png | Too dark | **Too Dark** | 44.8 | 41.0 | 20.8 | 0.00% | 7.0 |
| overexposed.png | Overexposed | **Overexposed** | 61.0 | 476.0 | 220.1 | 35.89% | 34.6 |
| low_contrast.png | Clear + low-contrast flag | **Clear Image**, low contrast flagged | 90.0 | 108.1 | 116.3 | 0.00% | 18.3 |

All 14 pipeline tests and all 18 dashboard checks passed. The edge-case tests cover: uniform black/white/gray images, RGBA, grayscale and palette PNGs, RGB channel order, non-square and large images, and rejection of fake, empty, wrong-extension, too-small and renamed-GIF files.

Two observations from the run worth knowing:

- **The dark image is also flagged blurry** (variance 41 < 100), even though the underlying scene is identical to the sharp one. This is the k² effect from section 6.1 in action. The classifier correctly reports *Too Dark* as the category and adds a note that the blur result is unreliable.
- **Low contrast costs few points.** The low-contrast sample still scores 90 because contrast is only worth 15 points. If contrast matters more for your use, change `SCORE_WEIGHTS` in `config.py`.

### 9.3 Expected behaviour on real images (not tested here)

On real endoscopy images, expect to recalibrate. Typical Laplacian variances, brightness levels and highlight percentages depend on the scope, the light source, the video resolution and the compression. A practical way to calibrate: collect 20–50 images you would personally call clear and 20–50 you would call blurry, run them through the analyzer, look at the two ranges of Laplacian variance in the CSVs, and set `blur_threshold` between them. Do the same for the other thresholds.

## 10. Configuration

Every threshold lives in `config.py` (`QualityThresholds`), and the most important ones can be changed live in the sidebar:

| Setting | Default | Effect |
|---|---|---|
| `blur_threshold` | 100 | Laplacian variance below this → Blurry |
| `dark_threshold` | 50 | Mean brightness below this → Too Dark |
| `ideal_brightness_min/max` | 80 / 180 | Full brightness points inside this range |
| `bright_mean_threshold` | 200 | Mean above this → Overexposed |
| `overexposure_pixel_value` | 240 | Gray level that counts as blown out |
| `overexposure_percent_limit` | 10% | Blown-out share above this → Overexposed |
| `low_contrast_threshold` | 30 | Std below this → low-contrast flag |
| `ignore_black_border` | True | Exclude the black frame from measurements |
| `ANALYSIS_MAX_SIDE` | 512 | Resolution at which everything is measured |

## 11. Limitations

- **Not clinically validated.** Thresholds were chosen on synthetic images. The score has not been compared with expert ratings or diagnostic outcomes.
- **Global measurements.** Each metric summarises the whole view. A frame with a sharp half and a blurred half may average out as "acceptable".
- **Laplacian variance mixes focus, texture and contrast.** Smooth mucosa with little texture can look blurry; noisy images can look sharp; dark or flat images score low regardless of focus.
- **Motion blur vs. defocus** are not distinguished.
- **No detection of other artefacts** such as bubbles, fluid, debris, fog, colour casts, interlacing or text overlays (timestamps and on-screen text add sharp edges that raise the blur score).
- **Frame detection is heuristic.** Unusual frame shapes, on-screen overlays or very dark images can confuse it; it can be switched off in the sidebar.
- **One category per image.** Images with several problems get the highest-priority category; all problems are still listed.
- **Enhancement is cosmetic.** It does not add information and is not shown to improve diagnosis.

## 12. Future improvements

- Calibrate thresholds on a public dataset with quality labels and report precision/recall
- Region-based analysis (grid of tiles) to catch partially blurred frames
- Additional sharpness measures (Tenengrad, FFT high-frequency energy) and motion-blur direction
- Specular-highlight segmentation separate from true overexposure
- Detection of bubbles, fluid and debris
- Batch analysis of folders and video frame-by-frame scoring with a timeline
- A small learned model trained on expert-labelled frames, compared against this rule-based baseline
- PDF report export

## 13. License and data

Code: free to use for learning and research. Any real endoscopy images you add must follow the privacy rules and licences that apply to them.

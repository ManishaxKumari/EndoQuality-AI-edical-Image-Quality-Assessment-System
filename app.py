"""
app.py
======
EndoQuality AI - Streamlit dashboard.

Run with:
    streamlit run app.py

Pages
-----
1. Home                    - what the tool does and how it works
2. Image Quality Analyzer  - upload, score, classify, explain, export
3. Image Enhancement       - CLAHE / denoise / sharpen / brightness
4. Technical Insights      - raw metrics, formulas, histograms, edge maps

The uploaded image is kept in st.session_state, so it stays loaded while
you move between pages.
"""

import io
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="EndoQuality AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (  # noqa: E402
    ALLOWED_EXTENSIONS,
    APP_NAME,
    APP_SUBTITLE,
    APP_VERSION,
    CATEGORY_CLEAR,
    DISCLAIMER,
    ENHANCEMENT_DEFAULTS,
    MAX_FILE_SIZE_MB,
    SCORE_WEIGHTS,
    QualityThresholds,
)
from src.blur_detector import laplacian_edge_map  # noqa: E402
from src.brightness_analyzer import to_grayscale  # noqa: E402
from src.enhancement import PIPELINE_ORDER, enhance_pipeline, to_png_bytes  # noqa: E402
from src.quality_analyzer import (  # noqa: E402
    ImageLoadError,
    analyze_image,
    load_image,
    prepare_for_analysis,
)
from src.visualization import (  # noqa: E402
    plot_brightness_distribution,
    plot_grayscale_histogram,
    plot_histogram_comparison,
    plot_score_breakdown,
    plot_side_by_side,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(ROOT, "assets", "samples")

PAGES = ["Home", "Image Quality Analyzer", "Image Enhancement", "Technical Insights"]
PAGE_ICONS = {"Home": "🏠", "Image Quality Analyzer": "🔍",
              "Image Enhancement": "✨", "Technical Insights": "📊"}

# Sidebar threshold sliders: session key -> (QualityThresholds field, label, min, max, step, help)
THRESHOLD_SLIDERS = {
    "t_blur": ("blur_threshold", "Blur threshold (Laplacian variance)", 10.0, 500.0, 5.0,
               "Images with a Laplacian variance below this are classified as blurry."),
    "t_dark": ("dark_threshold", "Too-dark threshold (mean brightness)", 10.0, 120.0, 1.0,
               "Images with a mean gray value below this are classified as too dark."),
    "t_over_pct": ("overexposure_percent_limit", "Overexposure limit (% blown-out pixels)", 1.0, 50.0, 0.5,
                   "Images with more blown-out pixels than this are classified as overexposed."),
    "t_over_val": ("overexposure_pixel_value", "Blown-out pixel level", 200, 255, 1,
                   "A pixel at or above this gray value counts as blown out."),
    "t_contrast": ("low_contrast_threshold", "Low-contrast threshold (std dev)", 5.0, 80.0, 1.0,
                   "Images with a grayscale standard deviation below this are flagged as low contrast."),
}


# ==========================================================================
# Styling
# ==========================================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --eq-blue: #1F5FAE;
            --eq-navy: #14263D;
            --eq-muted: #5B6B80;
            --eq-line: #DCE5F0;
            --eq-surface: #F2F6FB;
            --eq-good: #1E8E5A;
            --eq-warn: #B9741A;
            --eq-bad: #C0392B;
        }
        .block-container { padding-top: 2rem; max-width: 1200px; }
        h1, h2, h3 { color: var(--eq-navy); letter-spacing: -0.01em; }

        .eq-header { border-bottom: 2px solid var(--eq-blue); padding-bottom: .6rem; margin-bottom: 1.4rem; }
        .eq-header h1 { font-size: 1.9rem; margin: 0; padding: 0; }
        .eq-header p { color: var(--eq-muted); margin: .2rem 0 0 0; font-size: 1rem; }

        .eq-card {
            background: #FFFFFF; border: 1px solid var(--eq-line);
            border-left: 4px solid var(--eq-blue); border-radius: 6px;
            padding: .8rem 1rem; height: 100%;
        }
        .eq-card.good { border-left-color: var(--eq-good); }
        .eq-card.warn { border-left-color: var(--eq-warn); }
        .eq-card.bad  { border-left-color: var(--eq-bad); }
        .eq-card .label { color: var(--eq-muted); font-size: .82rem; }
        .eq-card .value { color: var(--eq-navy); font-size: 1.5rem; font-weight: 650; line-height: 1.3; }
        .eq-card .value small { font-size: .85rem; color: var(--eq-muted); font-weight: 500; }
        .eq-card .sub { color: var(--eq-muted); font-size: .8rem; }

        .eq-verdict {
            background: var(--eq-surface); border-radius: 10px;
            padding: 1.2rem 1.4rem; margin-bottom: 1rem;
            border: 1px solid var(--eq-line);
        }
        .eq-verdict .category { font-size: 2rem; font-weight: 700; line-height: 1.15; }
        .eq-verdict .score { font-size: 1.05rem; color: var(--eq-navy); margin-top: .35rem; }
        .eq-verdict .score b { font-size: 1.6rem; }
        .eq-verdict .band { display: inline-block; margin-left: .5rem; padding: .1rem .55rem;
                            border-radius: 999px; font-size: .8rem; background: #FFFFFF;
                            border: 1px solid var(--eq-line); color: var(--eq-navy); }
        .eq-verdict.good .category { color: var(--eq-good); }
        .eq-verdict.warn .category { color: var(--eq-warn); }
        .eq-verdict.bad  .category { color: var(--eq-bad); }

        .eq-step { display: flex; gap: .8rem; margin-bottom: .9rem; }
        .eq-step .num { flex: 0 0 1.9rem; height: 1.9rem; border-radius: 50%;
                        background: var(--eq-blue); color: #fff; font-weight: 650;
                        display: flex; align-items: center; justify-content: center; }
        .eq-step .txt b { color: var(--eq-navy); }
        .eq-step .txt { color: #2C3E55; line-height: 1.5; }

        .eq-disclaimer { font-size: .82rem; color: var(--eq-muted);
                         border-top: 1px solid var(--eq-line); padding-top: .7rem; margin-top: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="eq-header"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def status_from_fraction(fraction: float) -> str:
    return "good" if fraction >= 0.8 else "warn" if fraction >= 0.5 else "bad"


def metric_card(label: str, value: str, sub: str = "", status: str = "") -> None:
    st.markdown(
        f'<div class="eq-card {status}"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def disclaimer() -> None:
    st.markdown(f'<div class="eq-disclaimer">⚠️ {DISCLAIMER}</div>', unsafe_allow_html=True)


def show_figure(fig) -> None:
    """Display a Matplotlib figure and free its memory."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def figure_to_png(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


# ==========================================================================
# State
# ==========================================================================
def init_state() -> None:
    defaults = QualityThresholds()
    st.session_state.setdefault("page", PAGES[0])
    st.session_state.setdefault("image_rgb", None)
    st.session_state.setdefault("image_name", None)
    st.session_state.setdefault("image_id", None)
    st.session_state.setdefault("t_border", defaults.ignore_black_border)
    for key, (field_name, *_rest) in THRESHOLD_SLIDERS.items():
        st.session_state.setdefault(key, getattr(defaults, field_name))


def reset_thresholds() -> None:
    defaults = QualityThresholds()
    for key, (field_name, *_rest) in THRESHOLD_SLIDERS.items():
        st.session_state[key] = getattr(defaults, field_name)
    st.session_state["t_border"] = defaults.ignore_black_border


def go_to(page: str) -> None:
    st.session_state["page"] = page


def current_thresholds() -> QualityThresholds:
    values = {field_name: st.session_state[key]
              for key, (field_name, *_rest) in THRESHOLD_SLIDERS.items()}
    values["overexposure_pixel_value"] = int(values["overexposure_pixel_value"])
    values["ignore_black_border"] = bool(st.session_state["t_border"])
    return QualityThresholds(**values)


def store_image(rgb: np.ndarray, name: str, image_id: str) -> None:
    st.session_state["image_rgb"] = rgb
    st.session_state["image_name"] = name
    st.session_state["image_id"] = image_id


def clear_image() -> None:
    store_image(None, None, None)


# ==========================================================================
# Sidebar
# ==========================================================================
def render_sidebar() -> QualityThresholds:
    with st.sidebar:
        st.markdown(f"### 🩺 {APP_NAME}")
        st.caption(f"{APP_SUBTITLE} · v{APP_VERSION}")

        st.radio(
            "Navigate",
            PAGES,
            key="page",
            format_func=lambda p: f"{PAGE_ICONS[p]}  {p}",
            label_visibility="collapsed",
        )

        if st.session_state["image_name"]:
            st.success(f"Loaded: {st.session_state['image_name']}", icon="🖼️")

        st.divider()
        with st.expander("Analysis settings", expanded=False):
            st.caption("Thresholds are heuristics. Tune them for your own equipment.")
            for key, (_f, label, lo, hi, step, help_text) in THRESHOLD_SLIDERS.items():
                st.slider(label, min_value=lo, max_value=hi, step=step, key=key, help=help_text)
            st.checkbox(
                "Ignore black frame around the view",
                key="t_border",
                help="Excludes the dark border of the endoscope's field of view from all measurements.",
            )
            st.button("Reset to defaults", on_click=reset_thresholds, width="stretch")

        st.divider()
        st.caption(DISCLAIMER)
    return current_thresholds()


# ==========================================================================
# Shared image input
# ==========================================================================
def list_samples() -> list:
    if not os.path.isdir(SAMPLE_DIR):
        return []
    return sorted(f for f in os.listdir(SAMPLE_DIR)
                  if f.rsplit(".", 1)[-1].lower() in ALLOWED_EXTENSIONS)


def image_input(key: str) -> None:
    """File uploader + sample picker. Stores the decoded image in session state."""
    uploaded = st.file_uploader(
        f"Upload an endoscopy image (JPG, JPEG or PNG, up to {MAX_FILE_SIZE_MB} MB)",
        type=list(ALLOWED_EXTENSIONS),
        key=f"uploader_{key}",
    )
    if uploaded is not None:
        file_id = f"upload:{uploaded.name}:{uploaded.size}"
        if st.session_state["image_id"] != file_id:
            with st.spinner("Reading image…"):
                try:
                    rgb = load_image(uploaded.getvalue(), uploaded.name)
                except ImageLoadError as exc:
                    clear_image()
                    st.error(str(exc), icon="🚫")
                    return
            store_image(rgb, uploaded.name, file_id)

    samples = list_samples()
    if samples:
        with st.expander("No image at hand? Use a synthetic sample"):
            st.caption(
                "These are computer-generated test images, not real endoscopy data. "
                "Regenerate them with `python tests/generate_samples.py`."
            )
            cols = st.columns([3, 1])
            choice = cols[0].selectbox("Sample", samples, key=f"sample_{key}",
                                       label_visibility="collapsed")
            if cols[1].button("Load sample", key=f"load_{key}", width="stretch"):
                path = os.path.join(SAMPLE_DIR, choice)
                try:
                    with open(path, "rb") as handle:
                        rgb = load_image(handle.read(), choice)
                    store_image(rgb, choice, f"sample:{choice}")
                    st.rerun()
                except (OSError, ImageLoadError) as exc:
                    st.error(f"Could not load sample: {exc}", icon="🚫")


def require_image(page_key: str) -> bool:
    """Show the input widgets; return True when an image is ready."""
    image_input(page_key)
    if st.session_state["image_rgb"] is None:
        st.info("Upload an image or load a sample to begin.", icon="⬆️")
        return False
    return True


def run_analysis(thresholds: QualityThresholds):
    with st.spinner("Analysing image quality…"):
        return analyze_image(
            st.session_state["image_rgb"], thresholds, st.session_state["image_name"]
        )


def file_stem() -> str:
    name = st.session_state["image_name"] or "image"
    return name.rsplit(".", 1)[0]


# ==========================================================================
# Page 1: Home
# ==========================================================================
def page_home() -> None:
    page_header(APP_NAME, APP_SUBTITLE)

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown(
            "Endoscopy images are often captured in difficult conditions: the camera "
            "moves, fluid fogs the lens, and the light source is very close to wet, "
            "reflective tissue. **EndoQuality AI** checks each image for the four most "
            "common quality problems and explains what it found in plain language."
        )
        st.markdown("#### What it checks")
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Sharpness", "Blur", "Laplacian variance of edges and texture", "")
            st.write("")
            metric_card("Exposure", "Overexposure", "Share of blown-out white pixels", "")
        with c2:
            metric_card("Lighting", "Darkness", "Mean grayscale brightness", "")
            st.write("")
            metric_card("Tonal range", "Contrast", "Grayscale standard deviation", "")

        st.markdown("#### Features")
        st.markdown(
            "- A transparent 0–100 quality score with a per-metric breakdown\n"
            "- Classification into **Clear**, **Blurry**, **Too Dark** or **Overexposed**\n"
            "- Optional enhancement: CLAHE, noise reduction, sharpening, brightness\n"
            "- Histograms, edge maps and formulas for every measurement\n"
            "- CSV, text report and enhanced-image downloads\n"
            "- Runs entirely on CPU, with no model downloads"
        )
        st.button("Analyse an image", type="primary", on_click=go_to,
                  args=("Image Quality Analyzer",))

    with right:
        st.markdown("#### How it works")
        steps = [
            ("Load", "The image is validated and converted to RGB."),
            ("Standardise", "It is resized to 512 px on its longer side so one blur threshold fits every upload."),
            ("Mask", "The black frame around the endoscope's view is detected and ignored."),
            ("Measure", "Sharpness, brightness, blown-out pixels and contrast are computed."),
            ("Score", "Each measurement earns points: sharpness 40, brightness 25, exposure 20, contrast 15."),
            ("Classify", "Exposure problems are checked first, because they distort the blur measurement."),
        ]
        for i, (title, text) in enumerate(steps, start=1):
            st.markdown(
                f'<div class="eq-step"><div class="num">{i}</div>'
                f'<div class="txt"><b>{title}.</b> {text}</div></div>',
                unsafe_allow_html=True,
            )
    disclaimer()


# ==========================================================================
# Page 2: Image Quality Analyzer
# ==========================================================================
def page_analyzer(thresholds: QualityThresholds) -> None:
    page_header("Image Quality Analyzer", "Upload an image to score and classify its quality.")
    if not require_image("analyzer"):
        disclaimer()
        return

    report = run_analysis(thresholds)
    rgb = st.session_state["image_rgb"]

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.image(rgb, caption=f"{report.filename} · {report.original_size[0]} × "
                              f"{report.original_size[1]} px", width="stretch")

    with right:
        verdict_status = ("good" if report.category == CATEGORY_CLEAR
                          else status_from_fraction(report.score["total"] / 100))
        if report.category != CATEGORY_CLEAR and verdict_status == "good":
            verdict_status = "warn"
        st.markdown(
            f'<div class="eq-verdict {verdict_status}">'
            f'<div class="category">{report.category}</div>'
            f'<div class="score">Quality score <b>{report.score["total"]:.0f}</b> / 100'
            f'<span class="band">{report.score["band"]}</span></div></div>',
            unsafe_allow_html=True,
        )

        comps = report.score["components"]
        m = report.metrics
        details = {
            "sharpness": ("Sharpness", f"Laplacian variance {m['laplacian_variance']:.1f}"),
            "brightness": ("Brightness", f"Mean brightness {m['mean_brightness']:.1f}"),
            "exposure": ("Exposure", f"{m['overexposed_percent']:.2f}% blown-out pixels"),
            "contrast": ("Contrast", f"Std deviation {m['contrast_std']:.1f}"),
        }
        names = list(details)
        for row in (names[:2], names[2:]):
            cols = st.columns(2)
            for col, name in zip(cols, row):
                with col:
                    comp = comps[name]
                    label, sub = details[name]
                    metric_card(label, f"{comp['score']:.1f} <small>/ {comp['max']}</small>",
                                sub, status_from_fraction(comp["fraction"]))
            st.write("")

    st.markdown("#### Detected problems")
    if report.problems:
        for problem in report.problems:
            st.warning(problem, icon="⚠️")
    else:
        st.success("No quality problems detected with the current thresholds.", icon="✅")
    for note in report.notes:
        st.info(note, icon="ℹ️")

    st.markdown("#### Score breakdown")
    chart_col, text_col = st.columns([1.4, 1], gap="large")
    with chart_col:
        show_figure(plot_score_breakdown(report.score))
    with text_col:
        st.markdown(
            "The score adds up four rule-based components. Each one compares a "
            "measurement against the thresholds in the sidebar: full points inside the "
            "acceptable range, fewer points the further it falls outside."
        )
        st.caption(
            "This score is a heuristic for comparing images. It has not been validated "
            "against expert ratings or diagnostic outcomes."
        )

    st.markdown("#### Export")
    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("Download CSV", report.to_csv_bytes(),
                       file_name=f"{file_stem()}_quality.csv", mime="text/csv",
                       icon="📄", width="stretch")
    d2.download_button("Download report", report.to_text_report().encode("utf-8"),
                       file_name=f"{file_stem()}_report.txt", mime="text/plain",
                       icon="📝", width="stretch")
    disclaimer()


# ==========================================================================
# Page 3: Image Enhancement
# ==========================================================================
def page_enhancement(thresholds: QualityThresholds) -> None:
    page_header("Image Enhancement", "Adjust the image and compare it with the original.")
    if not require_image("enhancement"):
        disclaimer()
        return

    rgb = st.session_state["image_rgb"]
    d = ENHANCEMENT_DEFAULTS

    selected = st.multiselect(
        "Enhancement techniques",
        PIPELINE_ORDER,
        default=["CLAHE"],
        help="Selected steps are always applied in this order: noise reduction, "
             "brightness, CLAHE, sharpening.",
    )
    if not selected:
        st.info("Select at least one technique to see an enhanced version.", icon="👆")
        disclaimer()
        return

    steps = []
    param_cols = st.columns(len(selected))
    ordered = [name for name in PIPELINE_ORDER if name in selected]
    for col, name in zip(param_cols, ordered):
        with col:
            st.markdown(f"**{name}**")
            if name == "CLAHE":
                steps.append((name, {
                    "clip_limit": st.slider("Clip limit", 0.5, 8.0, d["clahe_clip_limit"], 0.5,
                                            help="Higher = stronger local contrast, more noise."),
                    "tile_grid": st.slider("Tile grid size", 2, 16, d["clahe_tile_grid"], 1,
                                           help="Number of tiles per side."),
                }))
            elif name == "Noise reduction":
                steps.append((name, {
                    "kernel_size": st.slider("Kernel size (odd)", 3, 15, d["gaussian_kernel"], 2),
                    "sigma": st.slider("Sigma", 0.1, 5.0, d["gaussian_sigma"], 0.1,
                                       help="Higher = smoother, but softer detail."),
                }))
            elif name == "Sharpening":
                steps.append((name, {
                    "amount": st.slider("Amount", 0.1, 3.0, d["sharpen_amount"], 0.1,
                                        help="Too high creates halos around edges."),
                    "sigma": st.slider("Radius (sigma)", 0.5, 5.0, d["sharpen_sigma"], 0.1),
                }))
            elif name == "Brightness adjustment":
                steps.append((name, {
                    "beta": st.slider("Brightness", -100, 100, d["brightness_beta"], 1),
                    "alpha": st.slider("Contrast multiplier", 0.5, 2.0, d["contrast_alpha"], 0.05),
                }))

    with st.spinner("Applying enhancement…"):
        enhanced = enhance_pipeline(rgb, steps)

    c1, c2 = st.columns(2, gap="medium")
    c1.image(rgb, caption="Original", width="stretch")
    c2.image(enhanced, caption="Enhanced: " + " → ".join(ordered), width="stretch")

    st.markdown("#### Heuristic score: before and after")
    before = analyze_image(rgb, thresholds, "original")
    after = analyze_image(enhanced, thresholds, "enhanced")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Quality score", f"{after.score['total']:.1f}",
              f"{after.score['total'] - before.score['total']:+.1f}")
    k2.metric("Laplacian variance", f"{after.metrics['laplacian_variance']:.1f}",
              f"{after.metrics['laplacian_variance'] - before.metrics['laplacian_variance']:+.1f}")
    k3.metric("Mean brightness", f"{after.metrics['mean_brightness']:.1f}",
              f"{after.metrics['mean_brightness'] - before.metrics['mean_brightness']:+.1f}")
    k4.metric("Contrast (std)", f"{after.metrics['contrast_std']:.1f}",
              f"{after.metrics['contrast_std'] - before.metrics['contrast_std']:+.1f}")
    st.caption(
        f"Classification: {before.category} → {after.category}. A higher heuristic score "
        "does not mean the image is more useful for diagnosis. Sharpening and CLAHE "
        "raise the Laplacian variance by amplifying edges and noise, without recovering "
        "detail the camera never captured."
    )

    show_figure(plot_histogram_comparison(rgb, enhanced, thresholds.border_intensity_threshold))

    st.markdown("#### Download")
    b1, b2, _ = st.columns([1, 1, 2])
    b1.download_button("Download enhanced image", to_png_bytes(enhanced),
                       file_name=f"{file_stem()}_enhanced.png", mime="image/png",
                       icon="⬇️", type="primary", width="stretch")
    b2.download_button("Download comparison figure", figure_to_png(plot_side_by_side(rgb, enhanced)),
                       file_name=f"{file_stem()}_comparison.png", mime="image/png",
                       icon="🖼️", width="stretch")
    disclaimer()


# ==========================================================================
# Page 4: Technical Insights
# ==========================================================================
def page_insights(thresholds: QualityThresholds) -> None:
    page_header("Technical Insights", "The raw measurements behind the score, and how each is computed.")
    if not require_image("insights"):
        disclaimer()
        return

    report = run_analysis(thresholds)
    small = prepare_for_analysis(st.session_state["image_rgb"])
    gray = to_grayscale(small)
    mask = report.fov_mask
    m, t = report.metrics, thresholds

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Laplacian variance", f"{m['laplacian_variance']:.1f}",
                    f"Blurry below {t.blur_threshold:.0f}",
                    "bad" if report.flags["is_blurry"] else "good")
    with k2:
        metric_card("Mean brightness", f"{m['mean_brightness']:.1f}",
                    f"Too dark below {t.dark_threshold:.0f}",
                    "bad" if report.flags["is_too_dark"] or report.flags["is_too_bright"] else "good")
    with k3:
        metric_card("Contrast (std dev)", f"{m['contrast_std']:.1f}",
                    f"Low below {t.low_contrast_threshold:.0f}",
                    "warn" if report.flags["is_low_contrast"] else "good")
    with k4:
        metric_card("Overexposed pixels", f"{m['overexposed_percent']:.2f}%",
                    f"Limit {t.overexposure_percent_limit:.1f}%",
                    "bad" if report.flags["has_blown_out_pixels"] else "good")

    st.write("")
    st.caption(
        f"Measured at {report.analysis_size[0]} × {report.analysis_size[1]} px. "
        + (f"Field of view covers {report.fov_coverage * 100:.1f}% of the image; the black "
           "frame outside it is excluded." if mask is not None else
           "The whole image is measured (no black frame detected, or masking is off).")
    )

    st.markdown("#### Quality score breakdown")
    rows = []
    for name, comp in report.score["components"].items():
        rows.append({"Component": name.capitalize(), "Points": comp["score"],
                     "Maximum": comp["max"], "Fraction": f"{comp['fraction'] * 100:.0f}%"})
    rows.append({"Component": "Total", "Points": report.score["total"],
                 "Maximum": sum(SCORE_WEIGHTS.values()), "Fraction": report.score["band"]})
    c1, c2 = st.columns([1, 1.3], gap="large")
    c1.dataframe(rows, hide_index=True, width="stretch")
    with c2:
        show_figure(plot_score_breakdown(report.score))

    st.markdown("#### Brightness")
    show_figure(plot_grayscale_histogram(gray, report.thresholds, mask))
    show_figure(plot_brightness_distribution(small, report.thresholds, mask))

    st.markdown("#### What the algorithms see")
    v1, v2, v3 = st.columns(3)
    v1.image(gray, caption="Grayscale (input to every metric)", width="stretch")
    v2.image(laplacian_edge_map(gray), caption="Laplacian edge map (bright = sharp detail)",
             width="stretch")
    if mask is not None:
        overlay = small.copy()
        overlay[mask == 0] = (overlay[mask == 0] * 0.25 + np.array([31, 95, 174]) * 0.75).astype(np.uint8)
        v3.image(overlay, caption="Measured region (blue = excluded frame)", width="stretch")
    else:
        v3.image(small, caption="Measured region: whole image", width="stretch")

    st.markdown("#### Formulas")
    with st.expander("Blur: variance of the Laplacian"):
        st.latex(r"L(x,y) = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}"
                 r"\quad\Rightarrow\quad \sigma_L^2 = \frac{1}{N}\sum (L - \bar{L})^2")
        st.markdown("Sharp edges create strong positive and negative Laplacian responses, "
                    "so the variance is high. Blur smooths the response toward zero.")
    with st.expander("Brightness: mean intensity"):
        st.latex(r"\text{Gray} = 0.299R + 0.587G + 0.114B \qquad \mu = \frac{1}{N}\sum I(x,y)")
    with st.expander("Overexposure: blown-out pixel share"):
        st.latex(r"\text{Overexposed}\,\% = 100 \cdot \frac{\#\{\,I(x,y) \ge "
                 + str(t.overexposure_pixel_value) + r"\,\}}{N}")
    with st.expander("Contrast: standard deviation"):
        st.latex(r"\sigma = \sqrt{\frac{1}{N}\sum (I(x,y) - \mu)^2}")
    disclaimer()


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    inject_css()
    init_state()
    thresholds = render_sidebar()

    page = st.session_state["page"]
    try:
        if page == "Home":
            page_home()
        elif page == "Image Quality Analyzer":
            page_analyzer(thresholds)
        elif page == "Image Enhancement":
            page_enhancement(thresholds)
        else:
            page_insights(thresholds)
    except Exception as exc:  # last-resort guard so users never see a raw traceback
        st.error(
            "Something went wrong while processing this image. Try a different image "
            f"or reset the analysis settings. Details: {exc}",
            icon="🚫",
        )


if __name__ == "__main__":
    main()

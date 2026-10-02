"""
test_app.py
===========
Smoke test for the Streamlit dashboard using Streamlit's built-in AppTest
(no browser needed). It opens every page, with and without an image
loaded, and fails if any page raises an exception or shows an error box.

    python tests/test_app.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import logging  # noqa: E402

logging.getLogger("streamlit").setLevel(logging.ERROR)
from streamlit.testing.v1 import AppTest  # noqa: E402

from src.quality_analyzer import load_image  # noqa: E402

PAGES = ["Home", "Image Quality Analyzer", "Image Enhancement", "Technical Insights"]


def _run(page, sample=None):
    at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60)
    if sample:
        with open(os.path.join(ROOT, "assets", "samples", sample), "rb") as handle:
            at.session_state["image_rgb"] = load_image(handle.read(), sample)
        at.session_state["image_name"] = sample
        at.session_state["image_id"] = f"sample:{sample}"
    at.session_state["page"] = page
    at.run()
    return at


def check(page, sample=None):
    at = _run(page, sample)
    assert not at.exception, f"{page}/{sample}: exception {at.exception}"
    errors = [e.value for e in at.error]
    assert not errors, f"{page}/{sample}: error shown {errors}"
    return at


def main() -> int:
    failures = 0
    cases = [(p, None) for p in PAGES]
    cases += [(p, s) for p in PAGES[1:] for s in ("clear.png", "blurry.png", "dark.png", "overexposed.png")]
    for page, sample in cases:
        try:
            at = check(page, sample)
            info = ""
            if page == "Image Quality Analyzer" and sample:
                # The verdict panel is HTML in a markdown element; find the category text
                verdict = [m.value for m in at.markdown if "class=\"eq-verdict" in m.value]
                info = verdict[0].split('class="category">')[1].split("<")[0] if verdict else "?"
            print(f"PASS  {page:24s} {sample or '(no image)':16s} {info}")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL  {exc}")

    # Changing a threshold in the sidebar should change the classification
    at = _run("Image Quality Analyzer", "clear.png")
    at.slider(key="t_blur").set_value(500.0).run()
    verdict = [m.value for m in at.markdown if "class=\"eq-verdict" in m.value][0]
    ok = "Blurry Image" in verdict and not at.exception
    print(f"{'PASS' if ok else 'FAIL'}  sidebar blur threshold 500 turns clear.png into Blurry Image")
    failures += 0 if ok else 1

    # Enhancement page with every technique selected
    at = _run("Image Enhancement", "dark.png")
    at.multiselect[0].set_value(
        ["Noise reduction", "Brightness adjustment", "CLAHE", "Sharpening"]).run()
    ok = not at.exception and not at.error
    print(f"{'PASS' if ok else 'FAIL'}  enhancement with all four techniques")
    failures += 0 if ok else 1

    print(f"\n{'All dashboard checks passed' if not failures else f'{failures} failure(s)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

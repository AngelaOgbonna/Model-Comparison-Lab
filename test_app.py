"""Headless end-to-end run of the Streamlit app."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_trains_on_sample_dataset():
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception
    at.select_slider[0].set_value("Quick")  # smallest search budget keeps the test fast
    at.button[0].click()
    at.run()
    assert not at.exception, at.exception
    assert any("Recommended model" in m.value for m in at.markdown)

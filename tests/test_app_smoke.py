from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_loads_in_unconfigured_state() -> None:
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=20)
    app.run()
    assert not app.exception
    assert app.title[0].value == "Ask questions about evidence-heavy PDFs"

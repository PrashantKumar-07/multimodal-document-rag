"""Opt-in real browser test against a running local app and installed Ollama model.

pip install playwright && python -m playwright install chromium
streamlit run app.py --server.port 8512
python scripts/browser_smoke.py
"""
from pathlib import Path
import sys

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evaluation_results/browser"


def main(api_key: str = "", research: bool = False):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto("http://127.0.0.1:8512", wait_until="networkidle")
        page.get_by_text("Choose up to three documents", exact=True).wait_for(timeout=20000)
        page.screenshot(path=str(OUTPUT / "home.png"), full_page=True)
        if api_key:
            page.get_by_role("combobox").nth(0).click()
            page.get_by_role("option", name="OpenRouter", exact=True).click()
            page.get_by_label("API key", exact=True).fill(api_key)
            page.get_by_label("API key", exact=True).press("Enter")
            expect(page.get_by_role("button", name="Test connection", exact=True)).to_be_enabled(timeout=20000)
        else:
            page.get_by_text("Installed · text only", exact=True).wait_for(timeout=20000)
        page.locator('input[type="file"]').set_input_files(str(ROOT / "cache/samples/efficient-streaming-language-models.pdf"))
        page.get_by_role("button", name="Build evidence index", exact=True).click()
        page.get_by_label("Your question", exact=True).wait_for(timeout=90000)
        if api_key:
            expect(page.get_by_role("button", name="Test connection", exact=True)).to_be_enabled(timeout=20000)
        if research:
            page.get_by_label("Answer depth", exact=True).click()
            page.get_by_role("option", name="Research report", exact=True).click()
        page.get_by_label("Your question", exact=True).fill(
            "Explain the central method, why it works, and the evidence supporting it." if research else
            "What window size was used for comparison and evaluation? Explain the settings for each model family.")
        page.get_by_role("button", name="Research and answer", exact=True).click()
        page.get_by_role("tab", name="Answer", exact=True).wait_for(timeout=240000)
        answer = page.locator(".st-key-answer_surface").inner_text()
        if research:
            assert len(answer.split()) >= 220, answer
        else:
            assert "2048" in answer.replace(",", ""), answer
            assert "1024" in answer.replace(",", ""), answer
        assert "The generation provider failed" not in answer
        # Exactly one native answer card contains the text; no empty HTML wrapper.
        assert len(answer) > 150
        assert page.locator('[data-testid="stException"]').count() == 0
        page.get_by_role("tab", name="Answer", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(OUTPUT / "answer.png"), full_page=True, mask=[page.locator('input[type="password"]')])
        assert not page.locator('[data-testid="stImage"]').count(), "Page images must be lazy"
        page.get_by_role("tab", name="Sources", exact=False).click()
        page.get_by_role("button", name="Show this page", exact=True).click()
        page.locator('[data-testid="stImage"]').wait_for(timeout=15000)
        page.screenshot(path=str(OUTPUT / "sources.png"), full_page=True, mask=[page.locator('input[type="password"]')])
        page.get_by_role("tab", name="Checks", exact=True).click()
        page.get_by_text("Passages with citations", exact=True).wait_for()
        with page.expect_download() as download:
            page.get_by_role("tab", name="Answer", exact=True).click()
            page.get_by_role("button", name="Download answer with citations", exact=True).click()
        download.value.save_as(str(OUTPUT / "answer.md"))
        assert not errors, errors
        print("PASS: PDF upload, model selection, live answer, lazy page preview, checks, Markdown export, and no browser exceptions. This tests UI behavior, not factual correctness.", flush=True)
        print(answer, flush=True)
        browser.close()


if __name__ == "__main__":
    main(research="--research" in sys.argv)

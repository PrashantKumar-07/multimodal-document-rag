import pytest

from src.ingestion import DocumentLimits, PDFValidationError, parse_pdf, render_pdf_page


def make_pdf() -> bytes:
    fitz = pytest.importorskip("pymupdf")
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Quarterly results")
    page.insert_text((72, 100), "Metric        2025        2024")
    page.insert_text((72, 120), "Revenue       1000        900")
    page.insert_text((72, 140), "Net income    250         200")
    page.insert_text((72, 160), "Cash flow     300         275")
    content = document.tobytes()
    document.close()
    return content


def test_pdf_is_parsed_with_page_provenance() -> None:
    content = make_pdf()
    parsed = parse_pdf(content, "fixture.pdf")
    assert parsed.page_count == 1
    assert parsed.chunks
    assert all(chunk.page_number == 1 for chunk in parsed.chunks)
    assert any(chunk.modality == "prose" for chunk in parsed.chunks)
    assert render_pdf_page(content, 1).startswith(b"\xff\xd8")


def test_invalid_and_oversized_files_are_rejected() -> None:
    with pytest.raises(PDFValidationError):
        parse_pdf(b"not a pdf", "bad.pdf")
    with pytest.raises(PDFValidationError):
        parse_pdf(make_pdf(), "large.pdf", DocumentLimits(max_bytes=10))

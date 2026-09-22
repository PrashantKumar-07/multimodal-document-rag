from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

from .models import DocumentChunk, ParsedDocument
from .text_utils import chunk_words, normalize_for_search

PARSER_VERSION = "1.1"
CAPTION_RE = re.compile(r"^\s*(figure|fig\.|chart|exhibit)\s+\w+", re.IGNORECASE | re.MULTILINE)
UNIT_CONTEXT_RE = re.compile(
    r"\b(?:in|amounts?\s+in)\s+(?:thousands?|millions?|billions?|trillions?)\b",
    re.IGNORECASE,
)


class PDFValidationError(ValueError):
    """Raised when a PDF is unsafe or outside the supported MVP limits."""


@dataclass(frozen=True, slots=True)
class DocumentLimits:
    max_bytes: int = 50 * 1024 * 1024
    max_pages: int = 200
    min_text_characters: int = 100


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bbox_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = a
    u0, v0, u1, v1 = b
    width = max(0.0, min(x1, u1) - max(x0, u0))
    height = max(0.0, min(y1, v1) - max(y0, v0))
    overlap = width * height
    area = max(1.0, (x1 - x0) * (y1 - y0))
    return overlap / area


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().replace("|", "\\|")


def table_to_markdown(rows: Iterable[Iterable[Any]]) -> tuple[str, list[str], list[list[str]]]:
    materialized = [[_clean_cell(cell) for cell in row] for row in rows]
    materialized = [row for row in materialized if any(row)]
    if not materialized:
        return "", [], []
    width = max(len(row) for row in materialized)
    materialized = [row + [""] * (width - len(row)) for row in materialized]
    headers = materialized[0]
    if not any(headers):
        headers = [f"Column {index + 1}" for index in range(width)]
        data_rows = materialized
    else:
        data_rows = materialized[1:]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in data_rows)
    return "\n".join(lines), headers, data_rows


def _split_table(headers: list[str], rows: list[list[str]], max_rows: int = 22) -> list[tuple[str, list[list[str]]]]:
    if not rows:
        markdown, _, _ = table_to_markdown([headers])
        return [(markdown, [])]
    pieces: list[tuple[str, list[list[str]]]] = []
    for start in range(0, len(rows), max_rows):
        subset = rows[start : start + max_rows]
        markdown, _, _ = table_to_markdown([headers, *subset])
        pieces.append((markdown, subset))
    return pieces


def _fallback_numeric_table(page_text: str) -> tuple[str, list[str], list[list[str]]] | None:
    candidate_rows: list[list[str]] = []
    for line in page_text.splitlines():
        cells = [cell.strip() for cell in re.split(r"\s{2,}", line.strip()) if cell.strip()]
        numeric_cells = sum(bool(re.search(r"\d", cell)) for cell in cells)
        if len(cells) >= 3 and numeric_cells >= 2:
            candidate_rows.append(cells)
    if len(candidate_rows) < 4:
        return None
    width = max(len(row) for row in candidate_rows)
    rows = [row + [""] * (width - len(row)) for row in candidate_rows[:40]]
    headers = [f"Column {index + 1}" for index in range(width)]
    markdown, _, data_rows = table_to_markdown([headers, *rows])
    return markdown, headers, data_rows


def _page_unit_context(page_text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in page_text.splitlines()[:40]]
    matches = [line for line in lines if UNIT_CONTEXT_RE.search(line)]
    return " ".join(matches[:2])


def parse_pdf(
    pdf_bytes: bytes,
    filename: str,
    limits: DocumentLimits | None = None,
) -> ParsedDocument:
    limits = limits or DocumentLimits()
    if not pdf_bytes.startswith(b"%PDF"):
        raise PDFValidationError(f"{filename} is not a valid PDF file.")
    if len(pdf_bytes) > limits.max_bytes:
        raise PDFValidationError(f"{filename} exceeds the 50 MB limit.")

    try:
        import pymupdf as fitz
    except ImportError as exc:  # pragma: no cover - dependency error path
        raise RuntimeError("PyMuPDF is required. Install dependencies from requirements.txt.") from exc

    try:
        document = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise PDFValidationError(f"Could not open {filename}: {exc}") from exc

    try:
        if document.needs_pass:
            raise PDFValidationError(f"{filename} is encrypted or password protected.")
        if document.page_count > limits.max_pages:
            raise PDFValidationError(f"{filename} has {document.page_count} pages; the limit is {limits.max_pages}.")

        document_hash = sha256_bytes(pdf_bytes)
        document_id = document_hash[:16]
        chunks: list[DocumentChunk] = []
        total_text = 0

        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            page_number = page_index + 1
            page_text = page.get_text("text", sort=True).strip()
            unit_context = _page_unit_context(page_text)
            total_text += len(page_text)
            table_bboxes: list[tuple[float, float, float, float]] = []
            page_had_table = False

            try:
                finder = page.find_tables()
                tables = list(getattr(finder, "tables", []))
            except Exception:
                tables = []

            for table_index, table in enumerate(tables):
                extracted = table.extract()
                markdown, headers, rows = table_to_markdown(extracted)
                if not markdown or len(rows) < 1:
                    continue
                page_had_table = True
                bbox = tuple(float(value) for value in table.bbox)
                table_bboxes.append(bbox)
                for part_index, (part_text, part_rows) in enumerate(_split_table(headers, rows)):
                    display_text = (
                        f"Source units: {unit_context}\n\n{part_text}" if unit_context else part_text
                    )
                    chunk_id = f"{document_id}:p{page_number}:table:{table_index}-{part_index}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            document_name=filename,
                            page_number=page_number,
                            modality="table",
                            text=display_text,
                            retrieval_text=normalize_for_search(display_text),
                            bbox=bbox,
                            table_headers=headers,
                            table_rows=part_rows,
                            has_page_image=True,
                        )
                    )

            if not page_had_table:
                fallback = _fallback_numeric_table(page_text)
                if fallback is not None:
                    markdown, headers, rows = fallback
                    display_text = (
                        f"Source units: {unit_context}\n\n{markdown}" if unit_context else markdown
                    )
                    chunks.append(
                        DocumentChunk(
                            chunk_id=f"{document_id}:p{page_number}:table:fallback",
                            document_id=document_id,
                            document_name=filename,
                            page_number=page_number,
                            modality="table",
                            text=display_text,
                            retrieval_text=normalize_for_search(display_text),
                            table_headers=headers,
                            table_rows=rows,
                            has_page_image=True,
                        )
                    )

            prose_blocks: list[str] = []
            for block in page.get_text("blocks", sort=True):
                if len(block) < 7 or int(block[6]) != 0:
                    continue
                bbox = tuple(float(value) for value in block[:4])
                if any(_bbox_overlap_ratio(bbox, table_bbox) >= 0.35 for table_bbox in table_bboxes):
                    continue
                text = re.sub(r"\s+", " ", str(block[4])).strip()
                if text:
                    prose_blocks.append(text)
            prose_text = "\n".join(prose_blocks)
            for chunk_index, text in enumerate(chunk_words(prose_text)):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}:p{page_number}:prose:{chunk_index}",
                        document_id=document_id,
                        document_name=filename,
                        page_number=page_number,
                        modality="prose",
                        text=text,
                        retrieval_text=normalize_for_search(text),
                    )
                )

            captions = [line.strip() for line in page_text.splitlines() if CAPTION_RE.match(line)]
            try:
                image_count = len(page.get_images(full=True))
            except Exception:
                image_count = 0
            likely_vector_visual = bool(captions)
            if image_count or likely_vector_visual:
                visual_text = "\n".join(captions) or page_text[:1400]
                surrounding = page_text[:2200]
                combined = f"Visual page. {visual_text}\n{surrounding}".strip()
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{document_id}:p{page_number}:visual:0",
                        document_id=document_id,
                        document_name=filename,
                        page_number=page_number,
                        modality="visual",
                        text=combined,
                        retrieval_text=normalize_for_search(combined),
                        has_page_image=True,
                    )
                )

        warnings: list[str] = []
        minimum = max(limits.min_text_characters, document.page_count * 40)
        if total_text < minimum:
            warnings.append(
                "Very little selectable text was found. This PDF may be scanned; OCR is outside this MVP."
            )
        if not chunks:
            raise PDFValidationError(f"No usable content could be extracted from {filename}.")
        return ParsedDocument(
            document_id=document_id,
            document_name=filename,
            page_count=document.page_count,
            chunks=chunks,
            warnings=warnings,
        )
    finally:
        document.close()


def render_pdf_page(pdf_bytes: bytes, page_number: int, dpi: int = 150) -> bytes:
    import pymupdf as fitz

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        if page_number < 1 or page_number > document.page_count:
            raise ValueError(f"Page {page_number} is outside this document.")
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(dpi=dpi, alpha=False)
        return pixmap.tobytes("jpeg", jpg_quality=82)
    finally:
        document.close()

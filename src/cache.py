from __future__ import annotations

import json
from pathlib import Path

from .ingestion import PARSER_VERSION, parse_pdf
from .models import DocumentChunk, ParsedDocument


class DocumentCache:
    def __init__(self, root: Path) -> None:
        self.root = root

    def _metadata_path(self, document_id: str) -> Path:
        return self.root / "documents" / f"{document_id}-{PARSER_VERSION}.json"

    def load_or_parse(self, pdf_bytes: bytes, filename: str) -> ParsedDocument:
        from .ingestion import sha256_bytes

        document_id = sha256_bytes(pdf_bytes)[:16]
        path = self._metadata_path(document_id)
        if path.exists():
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if value.get("parser_version") == PARSER_VERSION:
                    return ParsedDocument(
                        document_id=value["document_id"],
                        document_name=filename,
                        page_count=int(value["page_count"]),
                        chunks=[
                            DocumentChunk.from_dict({**chunk, "document_name": filename})
                            for chunk in value["chunks"]
                        ],
                        warnings=list(value.get("warnings", [])),
                    )
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
                pass

        parsed = parse_pdf(pdf_bytes, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "parser_version": PARSER_VERSION,
            "document_id": parsed.document_id,
            "page_count": parsed.page_count,
            "warnings": parsed.warnings,
            "chunks": [chunk.to_dict() for chunk in parsed.chunks],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return parsed

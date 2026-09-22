from __future__ import annotations

import hashlib
import re
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

from .cache import DocumentCache
from .ingestion import PARSER_VERSION, render_pdf_page
from .models import AnswerResult, ParsedDocument, RetrievedEvidence
from .providers import OpenAICompatibleProvider
from .retrieval import HybridRetriever
from .text_utils import is_visual_query
from .verification import verify_numeric_claims


def _safe_provider_error(exc: Exception) -> str:
    message = str(exc)
    message = re.sub(r"(?:sk|key)-[A-Za-z0-9_*.-]+", "[redacted-key]", message)
    message = re.sub(r"Bearer\s+\S+", "Bearer [redacted]", message, flags=re.IGNORECASE)
    return message[:600]


class DocumentRAGPipeline:
    def __init__(
        self,
        cache_dir: Path | str = Path("cache"),
        *,
        embedder: Any | None = None,
        reranker: Any | None = None,
        enable_reranker: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache = DocumentCache(self.cache_dir)
        self.embedder = embedder
        self.reranker = reranker
        self.enable_reranker = enable_reranker
        self.documents: list[ParsedDocument] = []
        self.pdf_bytes_by_id: dict[str, bytes] = {}
        self.retriever: HybridRetriever | None = None

    def ingest(self, documents: Sequence[tuple[str, bytes]]) -> list[ParsedDocument]:
        if not documents:
            raise ValueError("Select or upload at least one PDF.")
        if len(documents) > 3:
            raise ValueError("A maximum of three PDFs can be indexed together.")

        parsed_documents: list[ParsedDocument] = []
        seen: set[str] = set()
        self.pdf_bytes_by_id = {}
        for filename, content in documents:
            parsed = self.cache.load_or_parse(content, filename)
            if parsed.document_id in seen:
                continue
            seen.add(parsed.document_id)
            parsed_documents.append(parsed)
            self.pdf_bytes_by_id[parsed.document_id] = content
        if not parsed_documents:
            raise ValueError("No unique PDFs were supplied.")

        chunks = [chunk for document in parsed_documents for chunk in document.chunks]
        corpus_key = hashlib.sha256(
            (PARSER_VERSION + "\n" + "\n".join(
                f"{chunk.chunk_id}\t{chunk.retrieval_text}" for chunk in sorted(chunks, key=lambda item: item.chunk_id)
            )).encode("utf-8")
        ).hexdigest()[:20]
        embedding_path = self.cache_dir / "embeddings" / f"{corpus_key}.npy"
        self.retriever = HybridRetriever(
            chunks,
            embedder=self.embedder,
            reranker=self.reranker,
            embedding_cache_path=embedding_path,
            enable_reranker=self.enable_reranker,
        )
        self.documents = parsed_documents
        return parsed_documents

    def retrieve(
        self, question: str, *, mode: str = "hybrid_rerank"
    ) -> tuple[list[RetrievedEvidence], dict[str, float]]:
        if self.retriever is None:
            raise RuntimeError("Index documents before asking a question.")
        if not question.strip():
            raise ValueError("Enter a question.")
        return self.retriever.search(question.strip(), mode=mode)

    def _page_images(
        self, question: str, evidence: list[RetrievedEvidence]
    ) -> list[tuple[str, bytes]]:
        should_render = is_visual_query(question) or any(
            item.chunk.modality == "visual" for item in evidence
        )
        if not should_render:
            return []
        visual_first = sorted(evidence, key=lambda item: item.chunk.modality != "visual")
        images: list[tuple[str, bytes]] = []
        seen_pages: set[tuple[str, int]] = set()
        for item in visual_first:
            key = item.page_key
            if key in seen_pages:
                continue
            pdf_bytes = self.pdf_bytes_by_id.get(item.chunk.document_id)
            if pdf_bytes is None:
                continue
            try:
                image_bytes = render_pdf_page(pdf_bytes, item.chunk.page_number)
            except Exception:
                continue
            images.append((item.evidence_id, image_bytes))
            seen_pages.add(key)
            if len(images) >= 2:
                break
        return images

    def render_evidence_page(self, item: RetrievedEvidence, dpi: int = 120) -> bytes | None:
        pdf_bytes = self.pdf_bytes_by_id.get(item.chunk.document_id)
        if pdf_bytes is None:
            return None
        try:
            return render_pdf_page(pdf_bytes, item.chunk.page_number, dpi=dpi)
        except Exception:
            return None

    def answer(
        self,
        question: str,
        provider: OpenAICompatibleProvider | None = None,
    ) -> AnswerResult:
        evidence, timings = self.retrieve(question)
        if provider is None:
            return AnswerResult(
                answer="No generation provider is configured. The best matching evidence is shown below.",
                used_evidence_ids=[item.evidence_id for item in evidence],
                visual_observations=[],
                insufficient_evidence=False,
                evidence=evidence,
                timings=timings,
            )

        images_started = perf_counter()
        page_images = self._page_images(question, evidence)
        timings["page_render_seconds"] = perf_counter() - images_started
        try:
            result = provider.generate(question, evidence, page_images)
        except Exception as exc:
            return AnswerResult(
                answer="The generation provider failed. The retrieved evidence is still available below.",
                used_evidence_ids=[item.evidence_id for item in evidence],
                visual_observations=[],
                insufficient_evidence=True,
                evidence=evidence,
                provider=provider.config.name,
                model=provider.config.model,
                timings=timings,
                parse_warning=f"Provider error: {_safe_provider_error(exc)}",
            )
        result.timings = {**timings, **result.timings}
        if result.parse_warning:
            result.numeric_claims = verify_numeric_claims(result.answer, [], [])
        else:
            result.numeric_claims = verify_numeric_claims(
                result.answer, result.evidence, result.visual_observations
            )
        return result

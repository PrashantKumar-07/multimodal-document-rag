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
from .catalog import provider_error_message
from .research import assemble_evidence, normalize_queries, query_facets
from .retrieval import HybridRetriever
from .text_utils import is_visual_query
from .verification import verify_numeric_claims


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
        # Pay the cross-encoder cold-start cost during indexing, not the first answer.
        if self.enable_reranker:
            _ = self.retriever.reranker
        return parsed_documents

    def retrieve(
        self,
        question: str,
        *,
        mode: str = "hybrid_rerank",
        document_ids: set[str] | None = None,
    ) -> tuple[list[RetrievedEvidence], dict[str, float]]:
        if self.retriever is None:
            raise RuntimeError("Index documents before asking a question.")
        if not question.strip():
            raise ValueError("Enter a question.")
        selected_names = [
            document.document_name
            for document in self.documents
            if document_ids is None or document.document_id in document_ids
        ]
        return self.retriever.search(
            question.strip(),
            mode=mode,
            allowed_document_ids=document_ids,
            document_names=selected_names,
        )

    def _page_images(
        self, question: str, evidence: list[RetrievedEvidence]
    ) -> list[tuple[str, bytes]]:
        should_render = is_visual_query(question)
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
        document_ids: set[str] | None = None,
        *,
        answer_mode: str = "Detailed",
        history: list[str] | None = None,
        on_progress: Any | None = None,
        on_partial: Any | None = None,
    ) -> AnswerResult:
        if answer_mode not in ("Quick", "Detailed", "Research report"):
            raise ValueError("Unknown answer mode.")
        notify = on_progress or (lambda message: None)
        history = history or []
        lookup_question = question.strip()
        if history and re.search(r"\b(it|they|those|them|that|these|its)\b", question, re.I):
            lookup_question = f"{question} (Previous user question: {history[-1][:400]})"
        started = perf_counter()
        notify("Finding initial evidence")
        seeds, timings = self.retrieve(lookup_question, document_ids=document_ids)
        queries = query_facets(lookup_question, answer_mode)
        outline: list[str] = []
        warnings: list[str] = []
        initial_requests = provider.request_count if provider is not None else 0
        if answer_mode == "Research report" and provider is not None:
            notify("Planning research questions and an outline")
            try:
                plan = provider.plan(question, seeds, history)
                queries = normalize_queries(plan.get("queries", queries), lookup_question)
                raw_outline = plan.get("outline", [])
                outline = [x[:150] for x in raw_outline if isinstance(x, str)][:5] if isinstance(raw_outline, list) else []
            except Exception as exc:
                warnings.append("Research planning was unavailable; used local query expansion. " + provider_error_message(exc, provider.config.name))
        notify("Searching subquestions and gathering surrounding page context")
        lists = [seeds]
        for query in queries[1:]:
            # Detailed mode reranks the primary question once; cheap hybrid facet
            # searches add coverage without repeating the cross-encoder work.
            items, _ = self.retrieve(query, document_ids=document_ids,
                                     mode="hybrid" if answer_mode == "Detailed" else "hybrid_rerank")
            lists.append(items)
        # A comparison must give each selected document an opportunity to contribute.
        selected_documents = [d for d in self.documents if document_ids is None or d.document_id in document_ids]
        if len(selected_documents) > 1:
            for document in selected_documents:
                items, _ = self.retrieve(lookup_question, document_ids={document.document_id})
                lists.append(items)
        chunks = [c for d in selected_documents for c in d.chunks]
        budget = {"Quick": 1600, "Detailed": 2800, "Research report": 6000}[answer_mode]
        if provider is not None:
            # Conservative English token estimate; reserve prompt and output space.
            budget = min(budget, max(500, int((provider.config.context_length - provider.config.max_output_tokens - 1800) / 1.6)))
        evidence = assemble_evidence(lists, chunks, max_items={"Quick": 4, "Detailed": 6, "Research report": 10}[answer_mode], word_budget=budget)
        timings["research_seconds"] = perf_counter() - started
        if provider is None:
            timings["total_answer_seconds"] = perf_counter() - started
            return AnswerResult(
                answer="Your evidence is ready. Open Sources to explore the matching pages, or connect a model to write an explanation.",
                used_evidence_ids=[item.evidence_id for item in evidence],
                visual_observations=[],
                insufficient_evidence=False,
                evidence=evidence,
                timings=timings,
                question=question, answer_mode=answer_mode, research_queries=queries,
            )

        generation_evidence = evidence
        images_started = perf_counter()
        page_images = self._page_images(question, generation_evidence) if provider.config.supports_images else []
        if is_visual_query(question) and not provider.config.supports_images:
            warnings.append("This model cannot inspect page images. The answer uses extracted text and captions only; image-only details need a vision model.")
        timings["page_render_seconds"] = perf_counter() - images_started
        notify("Writing the answer from the collected evidence")
        generation_started = perf_counter()
        try:
            result = provider.generate(
                lookup_question, generation_evidence, page_images,
                answer_mode=answer_mode, outline=outline, review=answer_mode == "Research report",
                on_partial=on_partial,
            )
        except Exception as exc:
            timings["generation_seconds"] = perf_counter() - generation_started
            timings["total_answer_seconds"] = perf_counter() - started
            return AnswerResult(
                answer="The generation provider failed. Your retrieved evidence is available in Sources.",
                used_evidence_ids=[item.evidence_id for item in evidence],
                visual_observations=[],
                insufficient_evidence=True,
                evidence=evidence,
                provider=provider.config.name,
                model=provider.config.model,
                timings=timings,
                parse_warning=provider_error_message(exc, provider.config.name, provider.config.model),
                question=question, answer_mode=answer_mode, research_queries=queries,
                warnings=warnings, request_count=provider.request_count - initial_requests,
            )
        generation_evidence = result.evidence
        result.evidence = evidence
        result.timings = {**timings, **result.timings}
        result.question = question
        result.answer_mode = answer_mode
        result.research_queries = queries
        result.outline = outline
        result.warnings = [*warnings, *result.warnings]
        result.request_count = provider.request_count - initial_requests
        notify("Checking citations and numerical claims")
        if result.parse_warning:
            result.numeric_claims = verify_numeric_claims(result.answer, [], [])
        else:
            cited_evidence = [
                item for item in generation_evidence if item.evidence_id in result.used_evidence_ids
            ]
            result.numeric_claims = verify_numeric_claims(
                result.answer,
                cited_evidence,
                result.visual_observations,
            )
        result.timings["total_answer_seconds"] = perf_counter() - started
        return result

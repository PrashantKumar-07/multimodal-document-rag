from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .models import DocumentChunk, RetrievedEvidence
from .text_utils import is_numeric_query, is_visual_query, normalize_for_search, tokenize


@lru_cache(maxsize=2)
def load_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=2)
def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Any:
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-12, None)


class HybridRetriever:
    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        *,
        embedder: Any | None = None,
        reranker: Any | None = None,
        embedding_cache_path: Path | None = None,
        enable_reranker: bool = True,
    ) -> None:
        if not chunks:
            raise ValueError("At least one document chunk is required.")
        self.chunks = list(chunks)
        self.embedder = embedder or load_embedder()
        self.enable_reranker = enable_reranker
        self._reranker = reranker
        self.embedding_cache_path = embedding_cache_path
        self._tokenized_corpus = [tokenize(chunk.retrieval_text) for chunk in self.chunks]
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("rank-bm25 is required. Install requirements.txt.") from exc
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        self.embeddings = self._load_or_create_embeddings()

    @property
    def reranker(self) -> Any:
        if self._reranker is None:
            self._reranker = load_reranker()
        return self._reranker

    def _load_or_create_embeddings(self) -> np.ndarray:
        cache_path = self.embedding_cache_path
        if cache_path and cache_path.exists():
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if cached.ndim == 2 and cached.shape[0] == len(self.chunks):
                    return _normalise_rows(cached)
            except Exception:
                pass
        texts = [chunk.retrieval_text for chunk in self.chunks]
        matrix = self.embedder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        matrix = _normalise_rows(matrix)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, matrix, allow_pickle=False)
        return matrix

    @staticmethod
    def _top_indices(scores: np.ndarray, limit: int) -> list[int]:
        if not len(scores):
            return []
        limit = min(limit, len(scores))
        return np.argsort(-scores, kind="stable")[:limit].tolist()

    def _explicit_reference_indices(self, question: str) -> list[int]:
        references = re.findall(
            r"\b(figure|fig\.?|table|chart)\s+([a-z]?\d+)\b", question, flags=re.IGNORECASE
        )
        matches: list[int] = []
        for kind, identifier in references:
            kind_pattern = r"(?:figure|fig\.)" if kind.lower().startswith("fig") else re.escape(kind)
            caption = re.compile(
                rf"^\s*{kind_pattern}\s+{re.escape(identifier)}\b", flags=re.IGNORECASE | re.MULTILINE
            )
            matches.extend(index for index, chunk in enumerate(self.chunks) if caption.search(chunk.text))
        # Prefer visual records because they cause the source page to be rendered for the VLM.
        return sorted(set(matches), key=lambda index: self.chunks[index].modality != "visual")

    def search(
        self,
        question: str,
        *,
        top_k: int = 6,
        pool_size: int = 12,
        mode: str = "hybrid_rerank",
    ) -> tuple[list[RetrievedEvidence], dict[str, float]]:
        started = perf_counter()
        normalized_question = normalize_for_search(question)
        query_tokens = tokenize(normalized_question)
        lexical_scores = np.asarray(self.bm25.get_scores(query_tokens), dtype=np.float32)

        query_embedding = self.embedder.encode(
            [normalized_question], show_progress_bar=False, convert_to_numpy=True
        )
        query_embedding = _normalise_rows(query_embedding)[0]
        dense_scores = self.embeddings @ query_embedding

        lexical_rank = self._top_indices(lexical_scores, pool_size)
        dense_rank = self._top_indices(dense_scores, pool_size)
        if mode == "dense":
            candidate_indices = dense_rank
            fusion_scores = {index: float(dense_scores[index]) for index in dense_rank}
        else:
            lexical_weight, dense_weight = (1.25, 1.0) if is_numeric_query(question) else (1.0, 1.25)
            fusion_scores: dict[int, float] = {}
            for rank, index in enumerate(lexical_rank, start=1):
                fusion_scores[index] = fusion_scores.get(index, 0.0) + lexical_weight / (60 + rank)
            for rank, index in enumerate(dense_rank, start=1):
                fusion_scores[index] = fusion_scores.get(index, 0.0) + dense_weight / (60 + rank)
            candidate_indices = sorted(fusion_scores, key=fusion_scores.get, reverse=True)[:pool_size]

        referenced_indices = self._explicit_reference_indices(question)
        candidate_indices = list(dict.fromkeys([*referenced_indices, *candidate_indices]))
        for index in referenced_indices:
            fusion_scores.setdefault(index, 0.0)

        retrieval_finished = perf_counter()
        reranker_scores: dict[int, float] = {}
        if mode == "hybrid_rerank" and self.enable_reranker and candidate_indices:
            pairs = [(question, self.chunks[index].text) for index in candidate_indices]
            raw_scores = self.reranker.predict(pairs, show_progress_bar=False)
            reranker_scores = {
                index: float(score) for index, score in zip(candidate_indices, raw_scores, strict=True)
            }
            candidate_lexical = np.asarray([lexical_scores[index] for index in candidate_indices])
            candidate_dense = np.asarray([dense_scores[index] for index in candidate_indices])

            def minmax(values: np.ndarray) -> np.ndarray:
                span = float(values.max() - values.min())
                return (values - values.min()) / span if span > 1e-12 else np.ones_like(values)

            lexical_normalized = minmax(candidate_lexical)
            dense_normalized = minmax(candidate_dense)
            numeric = is_numeric_query(question)
            rerank_weight, lexical_weight, dense_weight = (
                (0.35, 0.45, 0.20) if numeric else (0.65, 0.10, 0.25)
            )
            final_scores: dict[int, float] = {}
            for position, index in enumerate(candidate_indices):
                clipped = np.clip(reranker_scores[index], -30.0, 30.0)
                rerank_probability = float(1.0 / (1.0 + np.exp(-clipped)))
                final_scores[index] = (
                    rerank_weight * rerank_probability
                    + lexical_weight * float(lexical_normalized[position])
                    + dense_weight * float(dense_normalized[position])
                )
            ordered = sorted(candidate_indices, key=final_scores.get, reverse=True)
        else:
            ordered = sorted(candidate_indices, key=fusion_scores.get, reverse=True)

        # An explicit "Figure 2" or "Table 1" reference is stronger evidence than a model score.
        ordered = list(dict.fromkeys([*referenced_indices, *ordered]))

        selected: list[int] = []
        page_counts: dict[tuple[str, int], int] = {}
        for index in ordered:
            chunk = self.chunks[index]
            page_key = (chunk.document_id, chunk.page_number)
            if page_counts.get(page_key, 0) >= 3:
                continue
            selected.append(index)
            page_counts[page_key] = page_counts.get(page_key, 0) + 1
            if len(selected) >= top_k:
                break

        desired_modality = "visual" if is_visual_query(question) else "table" if is_numeric_query(question) else None
        if desired_modality and not any(self.chunks[index].modality == desired_modality for index in selected):
            replacement = next(
                (index for index in ordered if self.chunks[index].modality == desired_modality), None
            )
            if replacement is not None:
                if len(selected) >= top_k:
                    selected[-1] = replacement
                else:
                    selected.append(replacement)

        evidence: list[RetrievedEvidence] = []
        for rank, index in enumerate(selected, start=1):
            evidence.append(
                RetrievedEvidence(
                    evidence_id=f"E{rank}",
                    chunk=self.chunks[index],
                    lexical_score=float(lexical_scores[index]),
                    dense_score=float(dense_scores[index]),
                    fusion_score=float(fusion_scores.get(index, 0.0)),
                    reranker_score=float(reranker_scores.get(index, 0.0)),
                )
            )
        finished = perf_counter()
        timings = {
            "retrieval_seconds": retrieval_finished - started,
            "rerank_seconds": finished - retrieval_finished,
            "total_retrieval_seconds": finished - started,
        }
        return evidence, timings

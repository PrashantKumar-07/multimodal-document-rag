import numpy as np

from src.models import DocumentChunk
from src.retrieval import HybridRetriever


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("revenue" in lowered or "sales" in lowered),
                    float("method" in lowered or "training" in lowered),
                    0.1,
                ]
            )
        return np.asarray(vectors, dtype=np.float32)


class FakeReranker:
    def predict(self, pairs, **kwargs):
        return np.asarray(
            [2.0 if "revenue" in document.lower() else 0.1 for _, document in pairs],
            dtype=np.float32,
        )


def chunk(index: int, text: str, modality: str = "prose") -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"doc:p{index}:{modality}:0",
        document_id="doc",
        document_name="report.pdf",
        page_number=index,
        modality=modality,
        text=text,
        retrieval_text=text.lower(),
    )


def test_hybrid_search_returns_stable_evidence_ids_and_expected_page() -> None:
    retriever = HybridRetriever(
        [
            chunk(1, "Annual revenue and sales were 416,161", "table"),
            chunk(2, "The training method uses balanced scaling"),
            chunk(3, "A decorative cover page"),
        ],
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        enable_reranker=True,
    )
    evidence, timings = retriever.search("How much revenue was reported?", top_k=2)
    assert evidence[0].evidence_id == "E1"
    assert evidence[0].chunk.page_number == 1
    assert timings["total_retrieval_seconds"] >= 0


def test_page_diversity_caps_three_chunks_per_page() -> None:
    chunks = [chunk(1, f"revenue item {index}") for index in range(5)] + [chunk(2, "revenue summary")]
    retriever = HybridRetriever(
        chunks,
        embedder=FakeEmbedder(),
        reranker=FakeReranker(),
        enable_reranker=True,
    )
    evidence, _ = retriever.search("revenue", top_k=6)
    assert sum(item.chunk.page_number == 1 for item in evidence) <= 3

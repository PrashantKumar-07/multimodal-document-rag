from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Modality = Literal["prose", "table", "visual"]
ClaimStatus = Literal["verified", "ambiguous", "unsupported"]


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    modality: Modality
    text: str
    retrieval_text: str
    bbox: tuple[float, float, float, float] | None = None
    table_headers: list[str] = field(default_factory=list)
    table_rows: list[list[str]] = field(default_factory=list)
    has_page_image: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DocumentChunk":
        item = dict(value)
        if item.get("bbox") is not None:
            item["bbox"] = tuple(item["bbox"])
        return cls(**item)


@dataclass(slots=True)
class RetrievedEvidence:
    evidence_id: str
    chunk: DocumentChunk
    lexical_score: float = 0.0
    dense_score: float = 0.0
    fusion_score: float = 0.0
    reranker_score: float = 0.0
    context_text: str = ""

    @property
    def page_key(self) -> tuple[str, int]:
        return self.chunk.document_id, self.chunk.page_number

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["chunk"] = self.chunk.to_dict()
        return value


@dataclass(slots=True)
class VisualObservation:
    evidence_id: str
    metric: str
    value: str
    unit: str = ""
    label: str = ""
    page_number: int | None = None
    document_name: str = ""


@dataclass(slots=True)
class NumericClaim:
    original: str
    normalized_value: str | None
    context: str
    status: ClaimStatus
    supporting_evidence_ids: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(slots=True)
class AnswerResult:
    answer: str
    used_evidence_ids: list[str]
    visual_observations: list[VisualObservation]
    insufficient_evidence: bool
    evidence: list[RetrievedEvidence]
    numeric_claims: list[NumericClaim] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    timings: dict[str, float] = field(default_factory=dict)
    parse_warning: str | None = None
    question: str = ""
    answer_mode: str = "Detailed"
    research_queries: list[str] = field(default_factory=list)
    outline: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    citation_coverage: float = 0.0
    follow_up_questions: list[str] = field(default_factory=list)
    generation_succeeded: bool = False
    draft_retained: bool = False
    request_count: int = 0


@dataclass(slots=True)
class ParsedDocument:
    document_id: str
    document_name: str
    page_count: int
    chunks: list[DocumentChunk]
    warnings: list[str] = field(default_factory=list)

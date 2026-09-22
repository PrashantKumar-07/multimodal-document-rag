"""Core package for the multimodal document question-answering application."""

from .models import (
    AnswerResult,
    DocumentChunk,
    NumericClaim,
    RetrievedEvidence,
    VisualObservation,
)

__all__ = [
    "AnswerResult",
    "DocumentChunk",
    "NumericClaim",
    "RetrievedEvidence",
    "VisualObservation",
]

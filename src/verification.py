from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import re

from .models import NumericClaim, RetrievedEvidence, VisualObservation
from .text_utils import context_tokens, decimal_key, extract_numeric_spans

UNIT_SCALES = {
    "thousand": Decimal("1000"),
    "thousands": Decimal("1000"),
    "million": Decimal("1000000"),
    "millions": Decimal("1000000"),
    "billion": Decimal("1000000000"),
    "billions": Decimal("1000000000"),
    "trillion": Decimal("1000000000000"),
    "trillions": Decimal("1000000000000"),
}


def _document_scale(text: str) -> Decimal | None:
    match = re.search(
        r"source units:.*?\b(?:in|amounts? in)\s+"
        r"(thousands?|millions?|billions?|trillions?)\b",
        text,
        flags=re.IGNORECASE,
    )
    return UNIT_SCALES.get(match.group(1).lower()) if match else None


def _is_scale_exception(text: str, start: int, end: int, value: Decimal) -> bool:
    nearby = text[max(0, start - 90) : min(len(text), end + 90)].lower()
    if Decimal("1900") <= abs(value) <= Decimal("2100"):
        return True
    return any(
        phrase in nearby
        for phrase in ("per share", "percentage", "percent", "rate", "ratio", "par value")
    )


def verify_numeric_claims(
    answer: str,
    evidence: list[RetrievedEvidence],
    visual_observations: list[VisualObservation],
) -> list[NumericClaim]:
    candidates: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
    for item in evidence:
        scale = _document_scale(item.chunk.text)
        for _, value, start, end in extract_numeric_spans(item.chunk.text):
            key = decimal_key(value)
            if key is not None:
                words = context_tokens(item.chunk.text, start, end)
                candidates[key].append((item.evidence_id, words))
                if scale is not None and value is not None and not _is_scale_exception(
                    item.chunk.text, start, end, value
                ):
                    scaled_key = decimal_key(value * scale)
                    if scaled_key is not None:
                        candidates[scaled_key].append((item.evidence_id, words))
    for observation in visual_observations:
        text = " ".join(
            part for part in (observation.metric, observation.label, observation.value, observation.unit) if part
        )
        for _, value, start, end in extract_numeric_spans(text):
            key = decimal_key(value)
            if key is not None:
                candidates[key].append(
                    (observation.evidence_id, context_tokens(text, start, end))
                )

    claims: list[NumericClaim] = []
    for raw, value, start, end in extract_numeric_spans(answer):
        key = decimal_key(value)
        claim_words = context_tokens(answer, start, end)
        matches = candidates.get(key or "", [])
        contextual = [evidence_id for evidence_id, words in matches if claim_words & words]
        if contextual:
            status = "verified"
            supporting = sorted(set(contextual))
            reason = "The normalized value and metric context match retrieved evidence."
        elif matches:
            status = "ambiguous"
            supporting = sorted({evidence_id for evidence_id, _ in matches})
            reason = "The value occurs in evidence, but the surrounding metric context does not match clearly."
        else:
            status = "unsupported"
            supporting = []
            reason = "No equivalent numeric value was found in the retrieved evidence."
        claims.append(
            NumericClaim(
                original=raw,
                normalized_value=key,
                context=" ".join(sorted(claim_words))[:240],
                status=status,
                supporting_evidence_ids=supporting,
                reason=reason,
            )
        )
    return claims

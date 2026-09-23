from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
import re

from .models import NumericClaim, RetrievedEvidence, VisualObservation
from .text_utils import context_tokens, decimal_key, extract_numeric_spans, is_structural_number

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
    candidates: dict[str, list[tuple[str, set[str], str, bool]]] = defaultdict(list)
    def unit(raw: str) -> str:
        if "%" in raw or "percent" in raw.lower():
            return "percent"
        return next((symbol for symbol in "$€£₹" if symbol in raw), "")
    for item in evidence:
        source_text = item.context_text or item.chunk.text
        scale = _document_scale(source_text)
        for raw, value, start, end in extract_numeric_spans(source_text):
            if is_structural_number(source_text, raw, start, end):
                continue
            key = decimal_key(value)
            if key is not None:
                words = context_tokens(source_text, start, end)
                candidates[key].append((item.evidence_id, words, unit(raw), False))
                if scale is not None and value is not None and not _is_scale_exception(
                    source_text, start, end, value
                ):
                    scaled_key = decimal_key(value * scale)
                    if scaled_key is not None:
                        candidates[scaled_key].append((item.evidence_id, words, unit(raw), False))
    for observation in visual_observations:
        text = " ".join(
            part for part in (observation.metric, observation.label, observation.value, observation.unit) if part
        )
        for raw, value, start, end in extract_numeric_spans(text):
            key = decimal_key(value)
            if key is not None:
                candidates[key].append(
                    (observation.evidence_id, context_tokens(text, start, end), unit(raw), True)
                )

    claims: list[NumericClaim] = []
    for raw, value, start, end in extract_numeric_spans(answer):
        if is_structural_number(answer, raw, start, end):
            continue
        key = decimal_key(value)
        claim_words = context_tokens(answer, start, end)
        claim_words -= {"reported", "model", "paper", "document", "result", "results", "source", "million", "billion", "thousand"}
        matches = candidates.get(key or "", [])
        # Match the citation attached to this sentence/table row, not another
        # sentence's source. Decimal points are not sentence boundaries.
        boundaries = list(re.finditer(r"(?<=[.!?])\s+(?=[A-Z])|\n", answer))
        left = max((m.end() for m in boundaries if m.end() <= start), default=0)
        right = min((m.start() for m in boundaries if m.start() >= end), default=len(answer))
        sentence_ids = set(re.findall(r"\[(E\d+)\]", answer[left:right]))
        if sentence_ids:
            matches = [match for match in matches if match[0] in sentence_ids]
        claim_unit = unit(raw)
        contextual = [
            evidence_id for evidence_id, words, source_unit, visual in matches
            if claim_words & words and not visual
            and (claim_unit == source_unit or not claim_unit or not source_unit)
            and (claim_unit == "percent") == (source_unit == "percent")
        ]
        if contextual:
            status = "verified"
            supporting = sorted(set(contextual))
            reason = "The normalized value and metric context match retrieved evidence."
        elif matches:
            status = "ambiguous"
            supporting = sorted({match[0] for match in matches})
            reason = "The value occurs, but its metric or unit is unclear, or it is only a model-read visual observation. Check the source page."
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

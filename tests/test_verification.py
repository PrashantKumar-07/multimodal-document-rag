from src.models import DocumentChunk, RetrievedEvidence, VisualObservation
from src.verification import verify_numeric_claims


def evidence(text: str, evidence_id: str = "E1") -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id=evidence_id,
        chunk=DocumentChunk(
            chunk_id="doc:p1:table:0",
            document_id="doc",
            document_name="report.pdf",
            page_number=1,
            modality="table",
            text=text,
            retrieval_text=text.lower(),
        ),
    )


def test_value_and_metric_context_are_verified() -> None:
    claims = verify_numeric_claims(
        "Net income was $112,010 [E1].",
        [evidence("Net income | $112,010")],
        [],
    )
    assert len(claims) == 1
    assert claims[0].status == "verified"
    assert claims[0].supporting_evidence_ids == ["E1"]


def test_value_without_metric_context_is_ambiguous() -> None:
    claims = verify_numeric_claims(
        "The reported value was 112,010.",
        [evidence("Net income | 112,010")],
        [],
    )
    assert claims[0].status == "ambiguous"


def test_unmatched_number_is_unsupported() -> None:
    claims = verify_numeric_claims(
        "Net income was 999.",
        [evidence("Net income | 112,010")],
        [],
    )
    assert claims[0].status == "unsupported"


def test_visual_observation_can_support_claim() -> None:
    observations = [
        VisualObservation(
            evidence_id="E2",
            metric="MMLU accuracy",
            value="67.5",
            unit="percent",
            label="Chinchilla",
            page_number=1,
        )
    ]
    claims = verify_numeric_claims("Chinchilla's MMLU accuracy was 67.5 percent [E2].", [], observations)
    assert claims[0].status == "verified"


def test_document_level_table_units_are_applied_to_amounts() -> None:
    claims = verify_numeric_claims(
        "Total net sales were $416,161 million [E1].",
        [evidence("Source units: In millions, except per-share amounts\nTotal net sales | $416,161")],
        [],
    )
    assert claims[0].normalized_value == "416161000000"
    assert claims[0].status == "verified"

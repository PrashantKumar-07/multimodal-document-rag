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


def test_visual_observation_requires_review_instead_of_self_verification() -> None:
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
    assert claims[0].status == "ambiguous"


def test_document_level_table_units_are_applied_to_amounts() -> None:
    claims = verify_numeric_claims(
        "Total net sales were $416,161 million [E1].",
        [evidence("Source units: In millions, except per-share amounts\nTotal net sales | $416,161")],
        [],
    )
    assert claims[0].normalized_value == "416161000000"
    assert claims[0].status == "verified"


def test_structural_identifiers_are_not_numeric_claims() -> None:
    claims = verify_numeric_claims(
        "The paper presents Approach 1–3 and Figure 2, then trains a 70B model [E1].",
        [evidence("The proposed model contains 70B parameters")],
        [],
    )
    assert [claim.original for claim in claims] == ["70B"]
    assert claims[0].status == "verified"


def test_real_ranges_and_parenthesized_values_are_not_discarded() -> None:
    claims = verify_numeric_claims("Cache size is 100–200 tokens. The cache limit is 300).", [], [])
    assert {claim.normalized_value for claim in claims} == {"100", "200", "300"}
    assert all(claim.status == "unsupported" for claim in claims)


def test_wrong_unit_and_wrong_citation_cannot_verify() -> None:
    claims = verify_numeric_claims("Revenue grew 50% [E1].", [evidence("Revenue was $50.")], [])
    assert claims[0].status != "verified"
    claims = verify_numeric_claims(
        "Revenue was 10 [E1]. Revenue was 99 [E2].",
        [evidence("Revenue was 10 and cash was 99.", "E1"), evidence("Revenue was 12.", "E2")], [],
    )
    assert claims[0].status == "verified"
    assert claims[1].status == "unsupported"


def test_parent_page_context_is_valid_evidence() -> None:
    item = evidence("Evaluation setup")
    item.context_text = "The cache window is 2048 tokens for Llama models."
    claims = verify_numeric_claims("The cache window is 2048 tokens [E1].", [item], [])
    assert claims[0].status == "verified"


def test_model_versions_and_dataset_ids_are_not_quantities():
    claims = verify_numeric_claims("Llama-2-70B used PG-19 and a 2048 token cache.", [], [])
    assert {claim.normalized_value for claim in claims} == {"70000000000", "2048"}

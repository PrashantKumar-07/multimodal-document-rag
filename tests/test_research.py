from src.models import DocumentChunk, RetrievedEvidence
from src.research import assemble_evidence, citation_audit, query_facets


def make_item(doc, page, text, modality="prose"):
    return RetrievedEvidence("E1", DocumentChunk(f"{doc}:{page}:{modality}", doc, f"{doc}.pdf", page, modality, text, text))


def test_context_expansion_deduplicates_pages_and_preserves_sources():
    hit = make_item("a", 1, "Our cache method retains attention sinks.")
    surrounding = make_item("a", 1, "The window uses 2048 tokens.", "table")
    duplicate = make_item("a", 1, "Visual page: our cache method.", "visual")
    second = make_item("b", 2, "The baseline uses full attention.")
    evidence = assemble_evidence([[hit, duplicate], [second]], [x.chunk for x in (hit, surrounding, duplicate, second)])
    assert len(evidence) == 2
    assert {x.chunk.document_id for x in evidence} == {"a", "b"}
    assert [x.evidence_id for x in evidence] == ["E1", "E2"]
    assert "baseline" not in evidence[0].context_text


def test_coverage_does_not_count_invented_citation():
    evidence = [make_item("a", 1, "Cache window size is 2048.")]
    coverage, warnings = citation_audit("The cache window used during evaluation contains exactly 2048 tokens [E99].", evidence)
    assert coverage == 0
    assert any("unknown citation" in x for x in warnings)


def test_setup_question_has_related_research_queries():
    assert len(query_facets("What window size was used for evaluation?", "Detailed")) > 1
    assert len(query_facets("What window size?", "Quick")) == 1

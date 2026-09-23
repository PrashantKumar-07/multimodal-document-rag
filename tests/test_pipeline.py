import json
from types import SimpleNamespace

from src.pipeline import DocumentRAGPipeline
from src.providers import OpenAICompatibleProvider, ProviderConfig
from test_ingestion import make_pdf
from test_retrieval import FakeEmbedder, FakeReranker


class RecordingClient:
    def __init__(self, fail=False):
        self.requests = []
        self.fail = fail
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        if self.fail:
            raise RuntimeError("credential=secret-do-not-return")
        if len(self.requests) == 1 and "Plan local PDF research" in str(kwargs):
            text = json.dumps({"queries": ["Revenue figures and results"], "outline": ["Findings", "Limitations"]})
        else:
            text = json.dumps({
                "answer": "The quarterly table reports revenue of 1000 for the specified period [E1].",
                "used_evidence_ids": ["E1"], "visual_observations": [], "insufficient_evidence": False,
            })
        return SimpleNamespace(model="tested-local-model", choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def test_research_mode_runs_bounded_plan_and_grounded_generation(tmp_path):
    pipeline = DocumentRAGPipeline(tmp_path, embedder=FakeEmbedder(), reranker=FakeReranker())
    doc = pipeline.ingest([("report.pdf", make_pdf())])[0]
    client = RecordingClient()
    provider = OpenAICompatibleProvider(ProviderConfig.for_provider("Ollama", model="local", supports_images=False), client_factory=lambda **kwargs: client)
    result = pipeline.answer("What is the reported revenue?", provider, {doc.document_id}, answer_mode="Research report")
    assert 2 <= result.request_count <= 3
    assert result.generation_succeeded
    assert result.model == "tested-local-model"
    assert result.outline
    assert result.evidence[0].context_text
    assert not any("image_url" in str(request) for request in client.requests)
    assert any(claim.status == "verified" and claim.normalized_value == "1000" for claim in result.numeric_claims)


def test_provider_error_is_redacted_and_sources_survive(tmp_path):
    pipeline = DocumentRAGPipeline(tmp_path, embedder=FakeEmbedder(), reranker=FakeReranker())
    pipeline.ingest([("report.pdf", make_pdf())])
    provider = OpenAICompatibleProvider(ProviderConfig.for_provider("OpenRouter", api_key="placeholder"), client_factory=lambda **kwargs: RecordingClient(fail=True))
    result = pipeline.answer("Revenue?", provider)
    assert result.evidence
    assert not result.generation_succeeded
    assert "secret-do-not-return" not in result.parse_warning

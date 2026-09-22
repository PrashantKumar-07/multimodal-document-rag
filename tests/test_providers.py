from types import SimpleNamespace

from src.models import DocumentChunk, RetrievedEvidence
from src.providers import OpenAICompatibleProvider, ProviderConfig, parse_generation_payload


class FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(completions=FakeCompletions(content))


def item() -> RetrievedEvidence:
    return RetrievedEvidence(
        evidence_id="E1",
        chunk=DocumentChunk(
            chunk_id="doc:p1:prose:0",
            document_id="doc",
            document_name="paper.pdf",
            page_number=1,
            modality="prose",
            text="Chinchilla has 70B parameters.",
            retrieval_text="chinchilla has 70000000000 parameters",
        ),
    )


def test_json_fence_is_parsed() -> None:
    payload, warning = parse_generation_payload(
        '```json\n{"answer":"70B [E1]","used_evidence_ids":["E1"],"visual_observations":[],"insufficient_evidence":false}\n```'
    )
    assert warning is None
    assert payload["answer"] == "70B [E1]"


def test_malformed_response_returns_warning() -> None:
    payload, warning = parse_generation_payload("not json")
    assert payload is None
    assert "JSON" in warning


def test_provider_uses_multimodal_message_and_validates_evidence_ids() -> None:
    content = (
        '{"answer":"70B [E1]","used_evidence_ids":["E1","E99"],'
        '"visual_observations":[],"insufficient_evidence":false}'
    )
    fake = FakeClient(content)
    provider = OpenAICompatibleProvider(
        ProviderConfig.for_provider("OpenRouter", api_key="test"),
        client_factory=lambda **kwargs: fake,
    )
    result = provider.generate("How large?", [item()], [("E1", b"jpeg")])
    assert result.used_evidence_ids == ["E1"]
    user_content = fake.chat.completions.request["messages"][1]["content"]
    assert any(part["type"] == "image_url" for part in user_content)


def test_provider_adds_visible_sources_when_model_omits_inline_citation() -> None:
    fake = FakeClient(
        '{"answer":"The model has 70B parameters.","used_evidence_ids":["E1"],'
        '"visual_observations":[],"insufficient_evidence":false}'
    )
    provider = OpenAICompatibleProvider(
        ProviderConfig.for_provider("OpenRouter", api_key="test"),
        client_factory=lambda **kwargs: fake,
    )
    result = provider.generate("How large?", [item()], [])
    assert result.answer.endswith("Sources: [E1]")

from types import SimpleNamespace

from src.models import DocumentChunk, RetrievedEvidence
from src.providers import OpenAICompatibleProvider, ProviderConfig, parse_generation_payload, render_payload_answer


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


def test_sections_preserve_source_links_and_followup_payload():
    rendered = render_payload_answer({"answer": "Cache size is 2048 [E1].", "sections": [
        {"title": "Evaluation setup", "content": "The benchmark compares the models under the same conditions [E2]."},
    ]})
    assert "### Evaluation setup" in rendered
    assert "[E2]" in rendered


def test_native_ollama_sets_context_and_constrains_output(monkeypatch):
    captured = {}
    class NativeResponse:
        def raise_for_status(self):
            pass
        def json(self):
            return {"model": "installed", "message": {"content": '{"answer":"No evidence","used_evidence_ids":[],"insufficient_evidence":true}'}}
    def post(url, **kwargs):
        captured.update(kwargs["json"])
        return NativeResponse()
    monkeypatch.setattr("requests.post", post)
    provider = OpenAICompatibleProvider(ProviderConfig.for_provider("Ollama", model="installed", context_length=16384))
    provider.generate("What happened?", [item()], [])
    assert captured["options"]["num_ctx"] == 16384
    assert captured["format"]["properties"]["answer"]["type"] == "string"
    assert captured["stream"] is False


def test_optional_reasoning_and_json_controls_do_not_route_to_another_model():
    fake = FakeClient('{"answer":"70B [E1]"}')
    provider = OpenAICompatibleProvider(ProviderConfig.for_provider(
        "OpenRouter", api_key="test", model="chosen:free", disable_reasoning=True, json_output=True,
    ), client_factory=lambda **kwargs: fake)
    provider.generate("Size?", [item()], [])
    request = fake.chat.completions.request
    assert request["model"] == "chosen:free"
    assert request["extra_body"] == {"reasoning": {"enabled": False}}
    assert request["response_format"] == {"type": "json_object"}


def test_streamed_response_is_accumulated_and_closed():
    from src.providers import partial_answer
    class Stream:
        closed = False
        def __iter__(self):
            for fragment in ('{"answer":"70', 'B [E1]"}'):
                yield SimpleNamespace(model="actual", choices=[SimpleNamespace(
                    finish_reason=None, delta=SimpleNamespace(content=fragment),
                )])
        def close(self):
            self.closed = True
    stream = Stream()
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: stream)))
    provider = OpenAICompatibleProvider(ProviderConfig.for_provider("OpenRouter", api_key="test"), client_factory=lambda **kwargs: fake)
    partials = []
    result = provider.generate("Size?", [item()], [], on_partial=lambda raw: partials.append(partial_answer(raw)))
    assert partials == ["70", "70B [E1]"]
    assert result.answer == "70B [E1]"
    assert result.model == "actual"
    assert stream.closed


def test_repeated_sections_remain_visible():
    assert render_payload_answer({"answer": "The cache is 2048 tokens [E1].", "sections": [
        {"title": "Repeated", "content": "The cache is 2048 tokens [E1]."},
    ]}) == "The cache is 2048 tokens [E1].\n\n### Repeated\n\nThe cache is 2048 tokens [E1]."


def test_full_report_preserves_repeated_baseline_section_from_stream():
    import json
    from src.providers import partial_answer
    baseline = "The RAG pipeline retrieves 200 documents and reranks them [E1]."
    payload = {"answer": f"The setup uses SnapKV. {baseline}", "sections": [
        {"title": "Evaluation Setup", "content": "Several data sets are tested [E1]."},
        {"title": "Baselines and Key Results", "content": baseline},
        {"title": "Conditions for Results", "content": "The context length varies [E2]."},
    ]}
    raw = json.dumps(payload)
    final = render_payload_answer(payload)
    assert "### Baselines and Key Results" in final
    assert final.count(baseline) == 2
    assert partial_answer(raw) == final


def test_grouped_citations_keep_the_same_source_ids():
    assert render_payload_answer({"answer": "Supported [E1, E2]. Unknown [E99]."}) == "Supported [E1] [E2]. Unknown [E99]."


def test_bad_review_does_not_destroy_a_valid_draft():
    fake = FakeClient('{"answer":"Supported draft without an inline source citation remains readable.","insufficient_evidence":false}')
    original = fake.chat.completions.create
    responses = iter([original(), SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{broken'))])])
    fake.chat.completions.create = lambda **kwargs: next(responses)
    provider = OpenAICompatibleProvider(ProviderConfig.for_provider("OpenRouter", api_key="test"), client_factory=lambda **kwargs: fake)
    result = provider.generate("Explain", [item()], [], answer_mode="Research report", review=True)
    assert result.generation_succeeded
    assert result.answer.startswith("Supported draft")
    assert provider.request_count == 1


def test_malformed_json_retains_readable_draft_and_marks_unverified():
    fake = FakeClient('{"answer":"Long visible report [E1]", "sections":[{"title":"Method","content":"Important detailed explanation [E1]')
    provider = OpenAICompatibleProvider(ProviderConfig.for_provider("OpenRouter", api_key="test"), client_factory=lambda **kwargs: fake)
    result = provider.generate("Explain", [item()], [], review=True)
    assert "Long visible report" in result.answer
    assert "Important detailed explanation" in result.answer
    assert result.parse_warning
    assert not result.generation_succeeded
    assert provider.request_count == 1


def test_final_answer_keeps_all_streamed_sections_even_with_bad_refusal_flag():
    import json
    from src.providers import partial_answer
    payload = {"answer": "Draft lead", "insufficient_evidence": True,
               "sections": [{"title": f"Section {i}", "content": f"Unique detailed content {i}."} for i in range(7)]}
    raw = json.dumps(payload)
    final = render_payload_answer(payload)
    assert final == partial_answer(raw)
    assert "Unique detailed content 6" in final


def test_interrupted_stream_preserves_draft():
    provider = OpenAICompatibleProvider(ProviderConfig.for_provider("OpenRouter", api_key="test"), client_factory=lambda **kwargs: FakeClient(""))
    def interrupted(messages, **kwargs):
        kwargs["on_partial"]('{"answer":"Useful partial explanation [E1]')
        raise TimeoutError()
    provider._complete = interrupted
    result = provider.generate("Explain", [item()], [], on_partial=lambda raw: None)
    assert "Useful partial explanation" in result.answer
    assert "Partial draft retained" in result.parse_warning

from types import SimpleNamespace

from src.catalog import discover_models, provider_error_message


class Response:
    def __init__(self, data):
        self.data = data
    def raise_for_status(self):
        pass
    def json(self):
        return self.data


def test_openrouter_catalog_only_lists_free_text_generators(monkeypatch):
    def model(id, cost, outputs, inputs):
        return {"id": id, "pricing": {"prompt": cost, "completion": cost}, "architecture": {"input_modalities": inputs, "output_modalities": outputs}}
    data = [model("vision:free", "0", ["text"], ["text", "image"]), model("paid", "1", ["text"], ["text"]), model("tts:free", "0", ["audio"], ["text"])]
    monkeypatch.setattr("src.catalog.requests.get", lambda *args, **kwargs: Response({"data": data}))
    result = discover_models("OpenRouter")
    assert [x.id for x in result] == ["vision:free"]
    assert result[0].vision


def test_errors_are_actionable_without_raw_credentials():
    exc = RuntimeError("oc_sk_private-password")
    exc.status_code = 404
    message = provider_error_message(exc, "Ollama")
    assert "installed" in message
    assert "private-password" not in message


def test_mandatory_reasoning_is_not_disabled(monkeypatch):
    data = [{"id": name, "pricing": {"prompt": "0", "completion": "0"},
             "architecture": {"output_modalities": ["text"]},
             "reasoning": {"mandatory": mandatory}, "supported_parameters": ["response_format"]}
            for name, mandatory in [("optional", False), ("mandatory", True)]]
    monkeypatch.setattr("src.catalog.requests.get", lambda *args, **kwargs: Response({"data": data}))
    options = {x.id: x for x in discover_models("OpenRouter")}
    assert options["optional"].disable_reasoning
    assert not options["mandatory"].disable_reasoning
    assert options["optional"].json_output

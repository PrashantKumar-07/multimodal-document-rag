from dataclasses import replace

from src.providers import OpenAICompatibleProvider, ProviderConfig


def test_claude_messages_headers_and_images(monkeypatch):
    captured = {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def json(self): return {"model": "claude", "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn"}
    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return Response()
    monkeypatch.setattr("src.transports.requests.post", post)
    cfg = replace(ProviderConfig.for_provider("Claude", api_key="test"), request_timeout=8)
    provider = OpenAICompatibleProvider(cfg)
    provider._complete([{"role": "system", "content": "rules"}, {"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,YQ=="}}]}], max_tokens=24)
    assert captured["url"].endswith("/v1/messages")
    assert captured["headers"]["x-api-key"] == "test"
    assert captured["json"]["system"] == "rules"
    assert captured["json"]["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"
    assert captured["timeout"] == (5, 8)

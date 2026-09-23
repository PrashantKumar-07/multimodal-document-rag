"""Provider-specific wire formats; credentials never enter prompts or error text."""
from __future__ import annotations

import json
from time import perf_counter
from types import SimpleNamespace

import requests


def protocol_for(config):
    if config.name == "Claude":
        return "messages"
    return "chat"


def result(text, model, finish="stop"):
    return SimpleNamespace(model=model, choices=[SimpleNamespace(
        message=SimpleNamespace(content=text), finish_reason=finish)])


def convert_parts(content, protocol):
    if isinstance(content, str):
        return content
    converted = []
    for part in content:
        if part["type"] == "text":
            converted.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image_url":
            url = part["image_url"]["url"]
            header, data = url.split(",", 1)
            converted.append({"type": "image", "source": {
                "type": "base64", "media_type": header[5:].split(";")[0], "data": data}})
    return converted


def complete_special(provider, messages, max_tokens, on_partial):
    cfg = provider.config
    protocol = protocol_for(cfg)
    started = perf_counter()
    body = {"model": cfg.model, "max_tokens": max_tokens, "stream": bool(on_partial),
            "messages": [{"role": m["role"], "content": convert_parts(m["content"], protocol)}
                         for m in messages if m["role"] != "system"]}
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    if system:
        body["system"] = system
    headers = {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01"}
    with requests.post(cfg.base_url.rstrip("/") + "/messages", headers=headers,
                       json=body, stream=bool(on_partial), timeout=(min(5, cfg.request_timeout), cfg.request_timeout)) as response:
        response.raise_for_status()
        if not on_partial:
            data = response.json()
            return result("".join(x.get("text", "") for x in data.get("content", []) if x.get("type") == "text"),
                          data.get("model", cfg.model), "length" if data.get("stop_reason") == "max_tokens" else "stop")
        fragments, model, finish = [], cfg.model, "stop"
        for line in response.iter_lines():
            if perf_counter() - started > cfg.request_timeout:
                raise TimeoutError("Generation time budget exceeded")
            if not line.startswith(b"data:"):
                continue
            event = json.loads(line[5:].strip())
            if event.get("type") == "error":
                raise RuntimeError("Provider stream failed")
            if event.get("type") == "message_start":
                model = event.get("message", {}).get("model", model)
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                if not fragments:
                    provider.last_first_token_seconds = perf_counter() - started
                fragments.append(delta.get("text", ""))
                on_partial("".join(fragments))
            if delta.get("stop_reason") == "max_tokens":
                finish = "length"
        return result("".join(fragments), model, finish)

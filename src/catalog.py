"""Public model discovery. No user keys are needed or cached here."""
from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import requests


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    vision: bool = False
    context_length: int = 16384
    disable_reasoning: bool = False
    json_output: bool = False


def ollama_root(base_url: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    path = parts.path.removesuffix("/v1")
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def discover_models(provider: str, base_url: str = "") -> list[ModelOption]:
    if provider == "Ollama":
        root = ollama_root(base_url or "http://localhost:11434/v1")
        response = requests.get(f"{root}/api/tags", timeout=6)
        response.raise_for_status()
        return [ModelOption(x["name"], x["name"]) for x in response.json().get("models", [])]
    if provider == "OpenRouter":
        response = requests.get("https://openrouter.ai/api/v1/models", timeout=12)
        response.raise_for_status()
        options = []
        for item in response.json().get("data", []):
            price = item.get("pricing", {})
            architecture = item.get("architecture", {})
            if price.get("prompt") != "0" or price.get("completion") != "0":
                continue
            if "text" not in architecture.get("output_modalities", []):
                continue
            # Safety classifiers and embedding/audio models are not answer writers.
            if any(term in item["id"] for term in ("content-safety", "embedding", "tts")):
                continue
            vision = "image" in architecture.get("input_modalities", [])
            options.append(ModelOption(
                item["id"], item.get("name", item["id"]), vision,
                int(item.get("context_length") or 16384),
                bool(item.get("reasoning")) and item["reasoning"].get("mandatory") is False,
                "response_format" in item.get("supported_parameters", []),
            ))
        return sorted(options, key=lambda x: (x.id != "openrouter/free", not x.vision, x.label))
    if provider == "Claude":
        return [ModelOption(x, x, True, 32768) for x in ("claude-haiku-4-5", "claude-sonnet-4-6")]
    if provider == "NVIDIA":
        response = requests.get("https://integrate.api.nvidia.com/v1/models", timeout=12)
        response.raise_for_status()
        selected = (
            "nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-nano-3-30b-a3b",
            "nvidia/llama3-chatqa-1.5-70b", "mistralai/mistral-nemotron",
            "nvidia/nemotron-3-super-120b-a12b", "google/gemma-4-31b-it",
        )
        available = {x["id"] for x in response.json().get("data", [])}
        return [ModelOption(name, name, name == "google/gemma-4-31b-it", 32768) for name in selected if name in available]
    return [ModelOption("gpt-5-mini", "gpt-5-mini", True, 32768)]


def inspect_ollama(model: str, base_url: str) -> ModelOption:
    response = requests.post(
        f"{ollama_root(base_url)}/api/show", json={"model": model}, timeout=8,
    )
    if response.status_code == 404:
        raise ValueError(f"{model} is not installed. Choose an installed model or run: ollama pull {model}")
    response.raise_for_status()
    info = response.json()
    lengths = [int(v) for k, v in info.get("model_info", {}).items() if k.endswith(".context_length")]
    return ModelOption(model, model, "vision" in info.get("capabilities", []), max(lengths, default=8192))


def provider_error_message(exc: Exception, provider: str, model: str = "") -> str:
    """Never expose exception bodies: they can contain credentials or document text."""
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = exc.response.status_code
    if status in (401, 403):
        return f"{provider} rejected authentication or client access. Check the key and account permissions."
    if status == 429:
        return f"{provider} is rate limited. Switching models may not help an account-wide quota. Wait for reset or use Ollama."
    if isinstance(exc, (TimeoutError, requests.exceptions.Timeout)) or "timeout" in type(exc).__name__.lower():
        return f"{provider} did not respond within the time budget. This is a timeout, not proof that your key is invalid."
    if status == 402:
        return f"{provider} requires credits for this request. Choose a free model or check the account balance."
    if status == 404:
        if provider == "Ollama":
            return f"The selected Ollama model is not installed. Refresh the model list and select an installed model."
        return f"The selected {provider} model or endpoint is unavailable. Refresh the model list."
    if status in (400, 413, 422):
        return f"{provider} could not accept this request. Check the model's image and context limits."
    if status and status >= 500:
        return f"{provider} is temporarily unavailable. Your retrieved sources remain available."
    if provider == "Ollama":
        return "Could not complete the Ollama request. Check that Ollama is running, the model is installed, and the server has enough memory."
    return f"Could not complete the {provider} request. Check the connection and model availability."

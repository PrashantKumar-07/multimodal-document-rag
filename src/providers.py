from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

from .models import AnswerResult, RetrievedEvidence, VisualObservation


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key: str

    @classmethod
    def for_provider(
        cls,
        name: str,
        *,
        api_key: str = "",
        model: str | None = None,
        base_url: str | None = None,
    ) -> "ProviderConfig":
        defaults = {
            "OpenRouter": ("https://openrouter.ai/api/v1", "openrouter/free"),
            "Ollama": ("http://localhost:11434/v1", "qwen3-vl:8b"),
            "OpenAI": ("https://api.openai.com/v1", "gpt-5-mini"),
        }
        if name not in defaults:
            raise ValueError(f"Unsupported provider: {name}")
        default_url, default_model = defaults[name]
        resolved_key = api_key or ("ollama" if name == "Ollama" else "")
        return cls(name, base_url or default_url, model or default_model, resolved_key)


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def parse_generation_payload(text: str) -> tuple[dict[str, Any] | None, str | None]:
    cleaned = _strip_json_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        first = cleaned.find("{")
        last = cleaned.rfind("}")
        if first < 0 or last <= first:
            return None, "The model did not return JSON; numeric claims are unverified."
        try:
            parsed = json.loads(cleaned[first : last + 1])
        except json.JSONDecodeError:
            return None, "The model returned malformed JSON; numeric claims are unverified."
    if not isinstance(parsed, dict) or not isinstance(parsed.get("answer"), str):
        return None, "The model response did not match the expected schema; numeric claims are unverified."
    return parsed, None


def _evidence_prompt(evidence: list[RetrievedEvidence]) -> str:
    sections: list[str] = []
    for item in evidence:
        chunk = item.chunk
        sections.append(
            f"[{item.evidence_id}] document={chunk.document_name!r}; page={chunk.page_number}; "
            f"modality={chunk.modality}\n{chunk.text[:4000]}"
        )
    return "\n\n".join(sections)


SYSTEM_PROMPT = """You answer questions only from supplied document evidence.
Rules:
1. If the evidence is insufficient, set insufficient_evidence=true and say so plainly.
2. Cite every factual sentence with one or more evidence labels such as [E1]. Never invent labels.
3. Preserve numerical values and source units exactly. Do not calculate new values or infer missing values.
4. Images are rendered source pages. Record every number read from an image in visual_observations.
5. Return only a JSON object with keys: answer, used_evidence_ids, visual_observations, insufficient_evidence.
visual_observations is an array of objects with evidence_id, metric, value, unit, label, page_number, document_name.
"""


class OpenAICompatibleProvider:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        if config.name != "Ollama" and not config.api_key:
            raise ValueError(f"An API key is required for {config.name}.")
        self.config = config
        if client_factory is None:
            from openai import OpenAI

            client_factory = OpenAI
        self.client = client_factory(base_url=config.base_url, api_key=config.api_key)

    def generate(
        self,
        question: str,
        evidence: list[RetrievedEvidence],
        page_images: list[tuple[str, bytes]],
    ) -> AnswerResult:
        user_text = (
            f"Question:\n{question}\n\nEvidence:\n{_evidence_prompt(evidence)}\n\n"
            "Answer from this evidence and return the required JSON object."
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for evidence_id, image_bytes in page_images[:2]:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append({"type": "text", "text": f"Rendered source page for [{evidence_id}]:"})
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}", "detail": "high"},
                }
            )

        started = perf_counter()
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
        )
        elapsed = perf_counter() - started
        raw = response.choices[0].message.content or ""
        payload, warning = parse_generation_payload(raw)
        if payload is None:
            return AnswerResult(
                answer=raw or "The model returned an empty response.",
                used_evidence_ids=[],
                visual_observations=[],
                insufficient_evidence=True,
                evidence=evidence,
                provider=self.config.name,
                model=self.config.model,
                timings={"generation_seconds": elapsed},
                parse_warning=warning,
            )

        valid_ids = {item.evidence_id for item in evidence}
        used_ids = [
            str(item) for item in payload.get("used_evidence_ids", []) if str(item) in valid_ids
        ]
        observations: list[VisualObservation] = []
        for value in payload.get("visual_observations", []):
            if not isinstance(value, dict) or str(value.get("evidence_id", "")) not in valid_ids:
                continue
            try:
                page_number = int(value["page_number"]) if value.get("page_number") is not None else None
            except (TypeError, ValueError):
                page_number = None
            observations.append(
                VisualObservation(
                    evidence_id=str(value.get("evidence_id", "")),
                    metric=str(value.get("metric", "")),
                    value=str(value.get("value", "")),
                    unit=str(value.get("unit", "")),
                    label=str(value.get("label", "")),
                    page_number=page_number,
                    document_name=str(value.get("document_name", "")),
                )
            )
        return AnswerResult(
            answer=str(payload["answer"]),
            used_evidence_ids=used_ids,
            visual_observations=observations,
            insufficient_evidence=bool(payload.get("insufficient_evidence", False)),
            evidence=evidence,
            provider=self.config.name,
            model=self.config.model,
            timings={"generation_seconds": elapsed},
            parse_warning=warning,
        )

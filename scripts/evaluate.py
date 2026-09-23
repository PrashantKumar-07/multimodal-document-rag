from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline import DocumentRAGPipeline  # noqa: E402
from src.catalog import discover_models, inspect_ollama  # noqa: E402
from src.providers import OpenAICompatibleProvider, ProviderConfig  # noqa: E402
from src.samples import download_sample, load_manifest  # noqa: E402
from src.text_utils import decimal_key, extract_numeric_spans, parse_decimal_token  # noqa: E402


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the curated retrieval and optional answer evaluation.")
    parser.add_argument("--provider", choices=["none", "OpenRouter", "Ollama", "Claude", "OpenAI"], default="none")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def provider_from_environment(name: str, model: str | None) -> OpenAICompatibleProvider | None:
    if name == "none":
        return None
    key = {
        "OpenRouter": os.environ.get("OPENROUTER_API_KEY", ""),
        "OpenAI": os.environ.get("OPENAI_API_KEY", ""),
        "Claude": os.environ.get("ANTHROPIC_API_KEY", ""),
        "Ollama": "ollama",
    }[name]
    kwargs = {}
    if name == "Ollama":
        if not model:
            available = discover_models("Ollama")
            if not available:
                raise ValueError("Install an Ollama model before running live evaluation.")
            model = available[0].id
        info = inspect_ollama(model, "http://localhost:11434/v1")
        kwargs = {"supports_images": info.vision, "context_length": info.context_length}
    elif name == "OpenRouter" and model:
        info = next((x for x in discover_models(name) if x.id == model), None)
        if info:
            kwargs = {"supports_images": info.vision, "context_length": min(info.context_length, 32768),
                      "disable_reasoning": info.disable_reasoning, "json_output": info.json_output}
    return OpenAICompatibleProvider(ProviderConfig.for_provider(name, api_key=key, model=model, **kwargs))


def main() -> None:
    options = args()
    manifest = load_manifest(ROOT / "data" / "sample_manifest.json")
    documents = [download_sample(sample, ROOT / "cache" / "samples") for sample in manifest]
    questions = json.loads((ROOT / "data" / "evaluation" / "questions.json").read_text(encoding="utf-8"))
    pipeline = DocumentRAGPipeline(ROOT / "cache")
    parsed = pipeline.ingest(documents)
    manifest_ids = {item["filename"]: item["id"] for item in manifest}
    document_ids = {
        manifest_ids[document.document_name]: document.document_id for document in parsed
    }

    report: dict[str, object] = {"retrieval": {}, "answers": None}
    for mode in ("dense", "hybrid", "hybrid_rerank", "expanded_context"):
        reciprocal_ranks: list[float] = []
        hits: list[float] = []
        latencies: list[float] = []
        numeric_coverages: list[float] = []
        for question in questions:
            if not question["answerable"]:
                continue
            expected_document = document_ids[question["document_id"]]
            if mode == "expanded_context":
                result = pipeline.answer(question["question"], document_ids={expected_document})
                evidence, timings = result.evidence, result.timings
                latencies.append(timings["research_seconds"])
            else:
                evidence, timings = pipeline.retrieve(
                    question["question"], mode=mode, document_ids={expected_document}
                )
                latencies.append(timings["total_retrieval_seconds"])
            matching_ranks = [
                rank
                for rank, item in enumerate(evidence[:5], start=1)
                if item.chunk.document_id == expected_document
                and item.chunk.page_number == question["gold_page"]
            ]
            hits.append(float(bool(matching_ranks)))
            reciprocal_ranks.append(1.0 / matching_ranks[0] if matching_ranks else 0.0)
            expected_numbers = set(question.get("numeric_values", []))
            if expected_numbers:
                retrieved_numbers = {
                    decimal_key(value)
                    for item in evidence[:5]
                    for _, value, _, _ in extract_numeric_spans(item.context_text or item.chunk.text)
                }
                numeric_coverages.append(float(expected_numbers <= retrieved_numbers))
        report["retrieval"][mode] = {
            "page_recall_at_5": statistics.fmean(hits),
            "mrr_at_5": statistics.fmean(reciprocal_ranks),
            "numeric_evidence_coverage": statistics.fmean(numeric_coverages) if numeric_coverages else None,
            "median_warm_latency_seconds": statistics.median(latencies[1:] or latencies),
        }

    provider = provider_from_environment(options.provider, options.model)
    if provider is not None:
        correct_numbers: list[float] = []
        citation_hits: list[float] = []
        refusal_hits: list[float] = []
        unsupported = 0
        total_claims = 0
        successful = 0
        actual_models = set()
        for question in questions:
            expected_document = document_ids[question["document_id"]]
            result = pipeline.answer(
                question["question"], provider, document_ids={expected_document}
            )
            successful += int(result.generation_succeeded)
            actual_models.add(result.model)
            if not question["answerable"]:
                refusal_hits.append(float(result.generation_succeeded and result.insufficient_evidence))
                continue
            cited = [item for item in result.evidence if item.evidence_id in result.used_evidence_ids]
            citation_hits.append(
                float(
                    any(
                        item.chunk.document_id == expected_document
                        and item.chunk.page_number == question["gold_page"]
                        for item in cited
                    )
                )
            )
            expected_numbers = set(question.get("numeric_values", []))
            if expected_numbers:
                answer_numbers = {
                    decimal_key(value) for _, value, _, _ in extract_numeric_spans(result.answer)
                }
                correct_numbers.append(float(expected_numbers <= answer_numbers))
            unsupported += sum(claim.status == "unsupported" for claim in result.numeric_claims)
            total_claims += len(result.numeric_claims)
        report["answers"] = {
            "provider": provider.config.name,
            "model": provider.config.model,
            "actual_models": sorted(actual_models),
            "successful_generations": successful,
            "total_questions": len(questions),
            "numeric_exact_match": statistics.fmean(correct_numbers) if correct_numbers else None,
            "citation_page_accuracy": statistics.fmean(citation_hits),
            "unanswerable_refusal_accuracy": statistics.fmean(refusal_hits),
            "unsupported_number_rate": unsupported / total_claims if total_claims else 0.0,
        }

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

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
from src.providers import OpenAICompatibleProvider, ProviderConfig  # noqa: E402
from src.samples import download_sample, load_manifest  # noqa: E402
from src.text_utils import decimal_key, extract_numeric_spans, parse_decimal_token  # noqa: E402


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the curated retrieval and optional answer evaluation.")
    parser.add_argument("--provider", choices=["none", "OpenRouter", "Ollama", "OpenAI"], default="none")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def provider_from_environment(name: str, model: str | None) -> OpenAICompatibleProvider | None:
    if name == "none":
        return None
    key = {
        "OpenRouter": os.environ.get("OPENROUTER_API_KEY", ""),
        "OpenAI": os.environ.get("OPENAI_API_KEY", ""),
        "Ollama": "ollama",
    }[name]
    return OpenAICompatibleProvider(ProviderConfig.for_provider(name, api_key=key, model=model))


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
    for mode in ("dense", "hybrid", "hybrid_rerank"):
        reciprocal_ranks: list[float] = []
        hits: list[float] = []
        latencies: list[float] = []
        numeric_coverages: list[float] = []
        for question in questions:
            if not question["answerable"]:
                continue
            evidence, timings = pipeline.retrieve(question["question"], mode=mode)
            latencies.append(timings["total_retrieval_seconds"])
            expected_document = document_ids[question["document_id"]]
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
                    for _, value, _, _ in extract_numeric_spans(item.chunk.text)
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
        for question in questions:
            result = pipeline.answer(question["question"], provider)
            if not question["answerable"]:
                refusal_hits.append(float(result.insufficient_evidence))
                continue
            expected_document = document_ids[question["document_id"]]
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

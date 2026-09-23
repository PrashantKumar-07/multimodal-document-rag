"""Reproducible live generation checks on public PDFs; never stores credentials.

Run with --provider Ollama --model mistral:latest, or an env-configured hosted provider.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.catalog import discover_models, inspect_ollama
from src.pipeline import DocumentRAGPipeline
from src.providers import OpenAICompatibleProvider, ProviderConfig, partial_answer
from src.samples import download_sample, load_manifest

STREAMING_SAMPLE = {
    "label": "Efficient Streaming Language Models",
    "filename": "efficient-streaming-language-models.pdf",
    "url": "https://arxiv.org/pdf/2309.17453",
}
CASES = [
    ("window", "What was the window size used for comparison and evaluation? Explain which models used each size and why these settings were chosen.", {"2048", "1024"}, 6, "Detailed"),
    ("method", "Explain the central method, why it works, and the evidence supporting it.", set(), 4, "Research report"),
    ("unanswerable", "What exact dollar amount did the authors spend training StreamingLLM?", set(), None, "Detailed"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["Ollama", "OpenRouter", "Claude", "OpenAI"], default="Ollama")
    parser.add_argument("--model", default="mistral:latest")
    parser.add_argument("--case", choices=[x[0] for x in CASES])
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation_results/live_quality.json")
    args = parser.parse_args()
    env = {"OpenRouter": "OPENROUTER_API_KEY", "Claude": "ANTHROPIC_API_KEY", "OpenAI": "OPENAI_API_KEY"}
    key = "ollama" if args.provider == "Ollama" else os.environ.get(env[args.provider], "")
    if not key:
        raise SystemExit(f"Set {env[args.provider]} in the environment before running this check.")
    kwargs = {}
    if args.provider == "Ollama":
        info = inspect_ollama(args.model, "http://localhost:11434/v1")
        kwargs = {"supports_images": info.vision, "context_length": info.context_length}
    elif args.provider == "OpenRouter":
        info = next((x for x in discover_models("OpenRouter") if x.id == args.model), None)
        if info:
            kwargs = {"supports_images": info.vision, "context_length": min(info.context_length, 32768),
                      "disable_reasoning": info.disable_reasoning, "json_output": info.json_output}
    provider = OpenAICompatibleProvider(ProviderConfig.for_provider(args.provider, api_key=key, model=args.model, **kwargs))
    streaming = download_sample(STREAMING_SAMPLE, ROOT / "cache/samples")
    # Include a distractor paper to exercise document isolation in live generation.
    sample = load_manifest(ROOT / "data/sample_manifest.json")[1]
    other = download_sample(sample, ROOT / "cache/samples")
    pipeline = DocumentRAGPipeline(ROOT / "cache")
    documents = pipeline.ingest([other, streaming])
    target = documents[-1]
    records = []
    for case_id, question, expected_numbers, gold_page, mode in CASES:
        if args.case and case_id != args.case:
            continue
        print(f"Running {case_id} using {args.provider}/{args.model}", flush=True)
        drafts = []
        result = pipeline.answer(question, provider, {target.document_id}, answer_mode=mode,
                                 on_progress=lambda x: print(x, flush=True),
                                 on_partial=lambda raw: drafts.append(partial_answer(raw)))
        values = {claim.normalized_value for claim in result.numeric_claims}
        record = asdict(result)
        record["case_id"] = case_id
        record["streamed_words"] = len(drafts[-1].split()) if drafts else 0
        record["checks"] = {
            "generation_succeeded": result.generation_succeeded,
            "streamed_draft_retained": bool(drafts) and result.answer.startswith(drafts[-1].strip()),
            "selected_document_only": all(x.chunk.document_id == target.document_id for x in result.evidence),
            "gold_page_retrieved": gold_page is None or any(x.chunk.page_number == gold_page for x in result.evidence),
            "expected_numbers_present": expected_numbers <= values,
            "refuses_unanswerable": result.insufficient_evidence if case_id == "unanswerable" else None,
            "answer_word_count": len(result.answer.split()),
            "developed_explanation": len(result.answer.split()) >= (220 if case_id == "method" else 100) if case_id != "unanswerable" else True,
        }
        print(json.dumps(record["checks"], indent=2), flush=True)
        print(result.answer, flush=True)
        records.append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(records, indent=2), encoding="utf-8")
    failures = [r["case_id"] for r in records if any(value is False for value in r["checks"].values())]
    if failures:
        raise SystemExit("Failed regression checks: " + ", ".join(failures))
    print("Automated regression checks passed. These do not test semantic entailment; manually review entity/value associations and explanations.")


if __name__ == "__main__":
    main()

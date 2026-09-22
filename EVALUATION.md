# Evaluation snapshot

Run on 22 September 2026 with the two checked sample documents and the 12 answerable questions in `data/evaluation/questions.json`.

| Retrieval configuration | Page Recall@5 | MRR@5 | Numeric evidence coverage | Median warm latency |
|---|---:|---:|---:|---:|
| Dense only | 100% | 0.861 | 62.5% | 0.013 s |
| BM25 + dense RRF | 100% | 0.903 | 87.5% | 0.013 s |
| Hybrid + cross-encoder reranker | 100% | 0.854 | 87.5% | 0.284 s |

The result is a small curated evaluation, not a general benchmark. Each question is evaluated within its selected source document, matching the interface's default behavior. Exact source pages are boosted when a question explicitly names a figure or table; this behavior applies equally to uploaded documents.

End-to-end provider behavior is covered by a network-mocked integration test. A live OpenAI smoke test correctly degraded to evidence-only output when the environment's configured key was rejected, confirming the provider-failure path without producing answer-quality metrics. Run `scripts/evaluate.py` with a working OpenRouter, Ollama, or OpenAI configuration to populate those metrics.

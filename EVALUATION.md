# Evaluation snapshot — 23 September 2026

## Answer display follow-up

The streamed answer could show a section that the final renderer removed when its text repeated a passage in the introduction. This affected the **Baselines and Key Results** section shown in the user's SnapKV screenshots. The renderer now keeps every section. The answer card expands with the report; the page scrolls normally. A regression test checks that a repeated baseline section remains visible and that streamed and final rendering agree. A real browser test also displayed a full five-section Ollama research report. OpenCode and AgentRouter were removed from the current application and command-line provider choices at the user's request. The current suite passes 45 tests. Earlier access checks below remain as historical test results.

## Reliability follow-up

The following supersedes the older three-request/rewrite workflow described in the historical results below:

- Removed the automatic research rewrite, which could replace a detailed streamed draft with a shorter response. Final and streamed output now share a renderer; an inconsistent refusal flag no longer silently hides sections.
- A real Ollama/Mistral research regression retained all 483 streamed words in the final answer (`streamed_draft_retained: true`). This tests preservation, not factual perfection.
- The Research report browser flow passed upload, generation, source previews, checks and Markdown export. All 47 automated tests passed after the fixes. The older Mistral model still produced factual errors in a browser answer; retaining its full draft does not solve model accuracy.
- Malformed JSON and interrupted streams retain readable partial content with a warning and unverified numbers. Regression tests cover this explicitly.
- Connection tests use an 8-second network timeout, 128 output tokens, and no retries. Actual checks: OpenCode Muse 403 in 1.79 s (OpenCode-only restriction); AgentRouter Messages 401 in 0.98 s; NVIDIA timed out in 8.56 s; OpenRouter returned 429 in 1.05 s. No quota bypass or client impersonation was attempted.
- NVIDIA was removed from the UI. Previously successful OpenRouter models and Muse are the shortlists; other catalog models are opt-in. Ollama is the default.
- Muse now uses `/responses`; Claude and AgentRouter use `/v1/messages`. Claude transport is mock-tested only: no Anthropic key was supplied. OpenCode/AgentRouter live generation remains blocked by the reported access responses, despite integration support.

The official OpenAI Responses streaming documentation guided the Muse wire-format adapter; OpenCode's own documentation selects the endpoint/model ID. Historical reports below should not be mistaken for current provider availability.

## Local retrieval

Measured on the 12 answerable items in the checked 16-question Apple/Chinchilla set, scoped to the expected document. CPU inference, four compute threads, cached models/documents. Generation is excluded.

| Configuration | Page Recall@5 | MRR@5 | Numeric evidence coverage | Median warm latency |
|---|---:|---:|---:|---:|
| Dense only | 100% | 0.861 | 62.5% | 0.013 s |
| BM25 + dense RRF | 100% | 0.903 | 87.5% | 0.012 s |
| Hybrid + cross-encoder | 100% | 0.854 | 87.5% | 1.058 s |
| Detailed: primary rerank + hybrid facets + parent context | 100% | 0.861 | 100% | 1.057 s |

Reproduce: `python scripts/evaluate.py --provider none --output evaluation_results/retrieval_final.json`.

This small curated set is not a general benchmark. Explicit figure/table references receive page boosts. Context expansion improves numeric availability, not necessarily answer correctness. Reranking did not improve MRR over hybrid retrieval here. CPU load and thread configuration changed timings substantially across runs; these are observations, not an SLA.

## Live generation and browser checks

Tests additionally used the actual 21-page *Efficient Streaming Language Models with Attention Sinks* paper (arXiv 2309.17453), with Chinchilla indexed as a distractor for API tests. No API credentials are stored in reports.

- OpenRouter `nex-agi/nex-n2.5-mini:free` completed real text and multimodal requests. One window-size answer took 4.54 seconds end-to-end, including 3.93 seconds generation, with first content at 1.51 seconds. Other runs were slower; this is not an average or a guarantee.
- OpenRouter `dots-studio/dots-3-note-preview:free` completed a 301-word window answer in 9.44 seconds, a 504-word method explanation in 16.19 seconds, and a 757-word research report in 45.44 seconds (three requests). First content for those generation calls arrived in 1.2–1.3 seconds; research planning/retrieval happens before generation. This model is the UI preference when present in the live catalog. These few examples do not establish a broad model ranking.
- The window-size regression recovered 2048 for Llama-2 and 1024 for Falcon/Pythia/MPT, half of the respective pre-training windows, and the stated visualization rationale. A separate run introduced an unsupported qualification about StreamingLLM's setting. Numeric presence alone would not catch this.
- Method answers explained sink-plus-recent KV caching, no required fine-tuning, and the distinction between streaming length and retained context. An early draft swapped Figure 1's method/value associations; prompting was tightened, but this is not proof the issue can never recur.
- A vision request sent page images and explained Figure 3's qualitative trends. Image-only numeric observations remain ambiguous pending visual review.
- Dots also completed the vision comparison in 16.37 seconds and passed the hosted browser workflow. Its vision generation started returning content after 5.79 seconds; image requests can be slower than text-only ones.
- An unsupported dollar-training-cost question received a short refusal.
- The browser test exercised upload, live hosted generation, source previews, checks, export, and absence of browser/Streamlit exceptions. It does not assess every factual claim.
- Research mode exposed malformed long JSON and review-overwrite failures. The output budget was increased and malformed review output can no longer discard a valid draft. Research mode is slower and should be considered experimental.

The implementation keeps sources separate from the synthesized answer, streams drafts, records actual provider/model and stage timings, and pins explicitly selected models. Optional reasoning is disabled only where the catalog permits it. A free model returning HTTP 200 on its catalog is not considered successful generation.

## Provider outcomes and limits

| Provider/model tested | Observed outcome |
|---|---|
| OpenRouter Nex Mini | Successful text/vision generation; variable quality and latency |
| OpenRouter Dots preview | Successful window/method/refusal/research/vision and browser requests; preferred for the tested explanations |
| OpenRouter Nex Pro | One window answer took about 160 seconds; not chosen as default |
| OpenRouter Gemma free route | Rate limited |
| OpenRouter Qwen free route | Rate limited |
| OpenCode tested free models | HTTP 403; no successful generation established |
| AgentRouter | HTTP 401 on authenticated catalog check; not integrated |
| NVIDIA | Catalog available; tested generation timed out or model returned 404 |
| Local Ollama Mistral | Native API worked with explicit context; some generated explanations were factually wrong |

Free availability depends on model, account, quota, and time. The application presents safe errors and preserves evidence on failure. Hosted requests have an idle timeout and a streaming time budget rather than unlimited retry loops.

## What is not established

No claim of zero hallucinations, universal sub-two-second answers, or STORM-equivalent research quality is made. The full 16-question live generation benchmark has not been completed on a validated default provider, so no aggregate citation-page accuracy or unsupported-number rate is asserted. Citation coverage checks citation placement, not entailment; numerical checks match values, units, nearby metric words, and attached sources, not every entity/value relationship. Public deployment is a separate step.

The final deterministic run passed 41 unit/mocked integration tests. Run `pytest -q` to reproduce. Run `scripts/live_quality_check.py` and inspect the saved answers manually for semantic errors. Reports/screenshots are gitignored under `evaluation_results/`.

# Multimodal, numerically grounded document RAG

A document research workspace that turns digitally generated PDFs into cited explanations, comparisons, and research briefs. Retrieval runs locally over prose, tables, and visual pages. The writing pipeline explores subquestions, gathers surrounding page context, and checks citations and quantitative claims.

There is deliberately no product or brand name. The repository describes what the application does.

**New here?** Read the [complete project guide](PROJECT_GUIDE.md) for the problem statement, end-to-end architecture, retrieval and verification algorithms, provider trade-offs, evaluation interpretation, limitations, and portfolio/resume-ready explanations. This README focuses on running and evaluating the code.

## What it demonstrates

- Layout-aware PDF extraction with table structure and page provenance.
- Local BM25 plus dense retrieval, weighted reciprocal-rank fusion, and cross-encoder reranking.
- Query-time page rendering for chart and figure questions rather than expensive image preprocessing of every page.
- Curated OpenRouter models, installed-model discovery for Ollama, and Claude Messages support.
- OpenAI-compatible hosted generation, plus native Ollama context-window and structured-output controls.
- Multi-query retrieval with query coverage, page diversity, parent-page context, and verbatim focus passages.
- Research planning, an outline, structured synthesis, and deterministic citation checks without a hidden rewrite.
- Inline evidence IDs, rendered source pages, and conservative numeric verification.
- An explicit active-document scope, so phrases such as “this paper” cannot silently pull an answer from another uploaded PDF.
- Evidence-only operation when no generation provider is configured.
- A 16-question finance/science evaluation with dense, hybrid, and reranked retrieval ablations.

The included samples are Apple's FY2025 Q4 financial statements and the paper *Training Compute-Optimal Large Language Models*. The live regression script additionally downloads *Efficient Streaming Language Models with Attention Sinks*. Results on these few documents are not a general accuracy guarantee.

### Current measured result

On the **12 answerable questions** in the curated 16-question set, scoped to the expected document, all four tested retrieval configurations found the gold page within five results. BM25 + dense RRF achieved **0.903 MRR@5** and **87.5% numeric evidence coverage** at **0.012 s median warm retrieval time** in the recorded CPU run. The Detailed pipeline increased numeric evidence coverage to **100%** but took **1.057 s** median; cross-encoder reranking did not improve MRR on this small set. These are retrieval-only, host-specific measurements, **not answer-accuracy or latency guarantees**. See [EVALUATION.md](EVALUATION.md) for the full ablation and limitations.

## Architecture

```text
PDF bytes
  -> validation + SHA-256 cache
  -> prose blocks / Markdown tables / visual-page records
  -> BM25 numeric-aware index + local dense embeddings
  -> weighted reciprocal-rank fusion
  -> local cross-encoder reranking
  -> local query expansion / optional research question planner
  -> per-query fusion + diverse page selection
  -> parent-page context + verbatim focus passages (bounded word budget)
  -> structured explanation/report + at most two page images for vision models
  -> retain streamed draft (including readable incomplete output with warnings)
  -> value/unit/context/citation checks
  -> answer / sources / checks / research trail + Markdown export
```

Retrieval never requires a paid service. Hosted generation sends the question, selected evidence text, recent user-question context for follow-ups, and at most two selected page images to the chosen provider. Research mode also sends a small seed-evidence bundle to plan subquestions.

When several PDFs are indexed, the most recently added document becomes the active source. **Compare all documents** retrieves from each selected document; ambiguous “this paper” questions require one active source. Recent user questions stay only in the active session and are cleared when document scope changes.

### Answer modes

| Mode | Intended use | Maximum generation calls |
|---|---|---:|
| Quick | A specific fact and its qualification | 1 |
| Detailed | Direct answer plus distinct explanatory sections | 1 |
| Research report | Subquestions, outline, findings, method, evidence, limitations | 2: planning and writing; no hidden rewrite |

Length targets are instructions, not quality guarantees. A refusal should remain short. Sources and page previews are separate from the answer, and only a requested preview is rendered in the UI.

Answers stream into the page while generating. Detailed mode reranks the primary search once and uses cheap hybrid searches for extra facets. CPU thread pools default to four threads (`RAG_CPU_THREADS` can override this, up to 16). Optional OpenRouter reasoning is disabled only when the live catalog says it is not mandatory; this avoids spending the whole output budget on invisible reasoning. Explicitly chosen models are pinned, with no silent switch to a different free model.

## Local setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
streamlit run app.py
```

The first indexing run downloads two small Sentence Transformers models. Later runs reuse the local model and document caches.

Automatic file watching is disabled: scanning Transformers' lazy modules caused unnecessary imports and large warning logs. Restart Streamlit after editing source files.

### Provider choices

| Provider | Default model | Setup | Notes |
|---|---|---|---|
| OpenRouter | UI prefers `dots-studio/dots-3-note-preview:free` when listed; adapter fallback is `openrouter/free` | Paste an OpenRouter key in the UI | Dots completed live text/research requests; Nex Mini is a tested faster alternative. Free availability and latency vary. |
| Claude | `claude-haiku-4-5` | Paste an Anthropic key | Native Messages protocol; billed API, no live key supplied for testing. |
| Ollama | Selected from installed models | Start Ollama, select an installed model | Native API explicitly configures context and JSON schema; text-only models do not receive images. |
| OpenAI | `gpt-5-mini` | Paste an OpenAI key in the UI | Optional paid quality benchmark. |

Keys entered in the UI stay in Streamlit session state and are not written by the application. Environment variables `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, and `OPENAI_API_KEY` are also recognized for local development. Copy `.env.example` only as a reference; this application does not automatically load `.env` files.

Use **Refresh available models** and **Test connection** before testing answers. Select `openrouter/free` explicitly if you want dynamic routing rather than a pinned model. A missing Ollama model now produces an installation hint rather than a failed question. To add vision locally, install an appropriate vision model yourself (for example `ollama pull qwen3-vl:8b`) and refresh; the app does not start multi-gigabyte downloads automatically. Installed does not mean accurate: the tested older Mistral model returned plausible but incorrect explanations.

[Claude Messages](https://platform.claude.com/docs/en/api/messages/create) has native wire-format support. [Ollama's native API](https://docs.ollama.com/api/chat) supplies local context and structured-output controls.

Ollama is the UI default. OpenCode and AgentRouter were removed at the user's request after access checks failed. Other OpenRouter models are hidden behind an explicit untested-model option. Connection tests send one tiny request with an 8-second network timeout and no retries; they still consume provider quota. A timeout is not proof of an invalid key, and changing models does not reset account-wide quota.

The final answer uses the same renderer as the streamed draft. Research mode does not perform a second rewrite, and every generated section remains in the answer, including repeated sections. Malformed JSON or interrupted streams retain readable content with an unverified/incomplete warning rather than replacing it with an empty error card. The answer card grows with its content; scroll the page to read the full report. This preserves text, not factual correctness.

OpenAI's official model reference confirms that `gpt-5-mini` accepts image input and supports Chat Completions: <https://developers.openai.com/api/docs/models/gpt-5-mini>.

## Evaluation

Run the network-free-after-download retrieval ablation:

```bash
python scripts/evaluate.py --provider none --output evaluation_results/retrieval.json
```

Optionally include end-to-end answer, citation, refusal, and unsupported-number metrics:

```bash
OPENROUTER_API_KEY=... python scripts/evaluate.py --provider OpenRouter
OPENAI_API_KEY=... python scripts/evaluate.py --provider OpenAI --model gpt-5-mini
python scripts/evaluate.py --provider Ollama --model qwen3-vl:8b
```

The report contains page Recall@5, MRR@5, numeric evidence coverage, median warm retrieval latency, and—when a provider is selected—numeric exact match, citation-page accuracy, unanswerable refusal accuracy, and unsupported-number rate. Provider/model details are recorded because OpenRouter's free router is not deterministic across time.

The checked baseline results are summarized in [EVALUATION.md](EVALUATION.md).

Run the live StreamingLLM regression (a Chinchilla distractor is indexed alongside it):

```bash
python scripts/live_quality_check.py --provider Ollama --model mistral:latest
# With the corresponding key already set in your environment:
python scripts/live_quality_check.py --provider OpenRouter --model openrouter/free
```

Optional real-browser validation (requires the public StreamingLLM PDF downloaded by the live check):

```bash
pip install playwright
python -m playwright install chromium
streamlit run app.py --server.port 8512
# In another terminal:
python scripts/browser_smoke.py
```

The browser check uploads the PDF, selects the installed Ollama model, generates an answer, opens a page preview, and downloads the cited answer. Reports and screenshots go to gitignored `evaluation_results/`.

`scripts/provider_benchmark.py` is an opt-in interactive helper for hosted comparisons. It reads a provider-to-key JSON mapping from non-echoing terminal stdin and keeps it in memory, then accepts benchmark commands. Never put credentials in command-line arguments, checked-in reports, or screenshots. Its browser action tests OpenRouter using the in-memory key. Browser/regression passes establish workflow behavior and selected numeric matches, **not** semantic correctness of every sentence.

Run tests with:

```bash
pytest -q
```

## Streamlit Community Cloud

1. Push this directory to a public GitHub repository.
2. In Streamlit Community Cloud, create an app from the repository and select `app.py`.
3. Do not add a shared API key; the deployed UI is bring-your-own-key and works in evidence-only mode without one.
4. Ollama is local-only unless you deliberately expose a compatible remote endpoint. The hosted demo should use OpenRouter, Claude, or OpenAI.

Raw uploaded PDF bytes are held in the active Streamlit session and are not written to disk. Extracted text, table records, and embeddings are written under the gitignored `cache/` directory; on Streamlit Community Cloud that storage is ephemeral. Operators handling sensitive documents should disable or routinely clear this cache.

## Limits and safety

- Maximum three PDFs, 50 MB and 200 pages each.
- PDF only; encrypted files are rejected.
- OCR, handwriting, and reliably searching image-only scans are outside this MVP.
- Numeric verification checks value equivalence, explicit units, nearby metric words, and the attached citation when present. It is evidence matching, not independent factual verification. Image-only model observations are **ambiguous** until checked against the page.
- Citation coverage measures citation presence, not semantic entailment. A local formatting review cannot prove all prose claims correct.
- The answer prompt forbids derived arithmetic. Multi-hop calculations require a separate calculator/tooling phase.
- No fine-tuning, ColPali, agents, knowledge graph, user database, or persistent chat history.
- This is not validated for medical or legal decision-making.

## Design references

[STORM](https://github.com/stanford-oval/storm) separates research, outlining, writing, and polishing. This project adapts the subquestion/outline idea to a bounded, local PDF corpus; it does not implement STORM's internet research or expert-agent conversations. No STORM source code is copied. Research quality still depends on the evidence and the chosen generation model.

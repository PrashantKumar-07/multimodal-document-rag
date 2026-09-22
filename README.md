# Multimodal, numerically grounded document RAG

A compact, domain-agnostic Streamlit application for asking evidence-backed questions about digitally generated PDFs. It indexes prose, tables, and visually rich pages locally, combines lexical and semantic retrieval, reranks the evidence, and checks every number in a generated answer against retrieved source context.

There is deliberately no product or brand name. The repository describes what the application does.

## What it demonstrates

- Layout-aware PDF extraction with table structure and page provenance.
- Local BM25 plus dense retrieval, weighted reciprocal-rank fusion, and cross-encoder reranking.
- Query-time page rendering for chart and figure questions rather than expensive image preprocessing of every page.
- One OpenAI-compatible multimodal adapter for OpenRouter, Ollama, and OpenAI.
- Inline evidence IDs, rendered source pages, and conservative numeric verification.
- Evidence-only operation when no generation provider is configured.
- A 16-question finance/science evaluation with dense, hybrid, and reranked retrieval ablations.

The included samples are Apple's FY2025 Q4 financial statements and the paper *Training Compute-Optimal Large Language Models*. Other digitally generated reports can be uploaded, but accuracy is only evaluated on the supplied samples.

## Architecture

```text
PDF bytes
  -> validation + SHA-256 cache
  -> prose blocks / Markdown tables / visual-page records
  -> BM25 numeric-aware index + local dense embeddings
  -> weighted reciprocal-rank fusion
  -> local cross-encoder reranking
  -> six cited evidence items + at most two rendered pages
  -> optional multimodal answer provider
  -> deterministic numeric/context verification
  -> Streamlit answer, badges, evidence, and page previews
```

Retrieval never requires a paid service. Hosted generation sends only the question, selected evidence text, and at most two selected page images to the provider chosen by the user.

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

### Provider choices

| Provider | Default model | Setup | Notes |
|---|---|---|---|
| OpenRouter | `openrouter/free` | Paste an OpenRouter key in the UI | Free routing is quota-limited and model availability can change. |
| Ollama | `qwen3-vl:8b` | Run `ollama pull qwen3-vl:8b`, then start Ollama | Fully local; requires enough local RAM/VRAM. |
| OpenAI | `gpt-5-mini` | Paste an OpenAI key in the UI | Optional paid quality benchmark. |

Keys entered in the UI stay in Streamlit session state and are not written by the application. Environment variables `OPENROUTER_API_KEY` and `OPENAI_API_KEY` are also recognized for local development. Copy `.env.example` only as a reference; this application does not automatically load `.env` files.

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

Run tests with:

```bash
pytest -q
```

## Streamlit Community Cloud

1. Push this directory to a public GitHub repository.
2. In Streamlit Community Cloud, create an app from the repository and select `app.py`.
3. Do not add a shared API key; the deployed UI is bring-your-own-key and works in evidence-only mode without one.
4. Ollama is local-only unless you deliberately expose a compatible remote endpoint. The hosted demo should use OpenRouter or OpenAI.

Raw uploaded PDF bytes are held in the active Streamlit session and are not written to disk. Extracted text, table records, and embeddings are written under the gitignored `cache/` directory; on Streamlit Community Cloud that storage is ephemeral. Operators handling sensitive documents should disable or routinely clear this cache.

## Limits and safety

- Maximum three PDFs, 50 MB and 200 pages each.
- PDF only; encrypted files are rejected.
- OCR, handwriting, and reliably searching image-only scans are outside this MVP.
- Numeric verification checks value equivalence and nearby metric words. It is evidence matching, not independent factual verification.
- The answer prompt forbids derived arithmetic. Multi-hop calculations require a separate calculator/tooling phase.
- No fine-tuning, ColPali, agents, knowledge graph, user database, or persistent chat history.
- This is not validated for medical or legal decision-making.

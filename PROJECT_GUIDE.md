# Project guide: multimodal, numerically grounded document RAG

This document explains the **implemented application**, its design decisions, evaluation, limitations, and ways to describe it accurately in a portfolio or resume. It is not a claim that the system eliminates hallucinations or that a public deployment is already live.

## 1. The project in one paragraph

This is a Streamlit research workspace for asking questions about evidence-heavy, digitally generated PDFs. It indexes prose, extracted tables, and likely visual pages; combines local keyword and semantic search; reranks and expands relevant pages into a bounded evidence packet; and asks a selected language model to write a cited answer. A deterministic post-processing step checks numbers against the cited evidence and labels each claim **verified**, **ambiguous**, or **unsupported**. Users can inspect the exact source passage, PDF page, query trail, timing breakdown, and a downloadable Markdown answer. Retrieval works without an API key; hosted answer generation is bring-your-own-key. The implementation is domain-agnostic, with curated financial and scientific examples.

## 2. Problem, goal, and scope

Ordinary PDF chat can retrieve a plausible paragraph while missing a table row, figure, qualification, or the page that actually supports a number. Long documents also exceed many models' practical context windows. This application addresses that by reducing a PDF collection to a small, inspectable set of relevant source pages and making the answer's evidence trail visible.

The goals are to:

- Search **prose, tables, and visual-page cues** within up to three PDFs.
- Answer broad research questions as well as narrow factual questions, with adjustable response depth.
- Attach page-level evidence labels such as `[E1]` to factual statements.
- Detect quantitative claims and check their values, units, metric context, and attached citations against retrieved material.
- Keep ingestion and retrieval local, while allowing either local or optional hosted generation.
- Fail gracefully: if a provider is unavailable, the retrieved sources remain available.

This is a **bounded document-research tool**, not an autonomous web-research agent or a general fact-checking system. It does not use OCR, fine-tuning, ColPali, a vector database, a knowledge graph, persistent accounts, or derived multi-hop arithmetic.

## 3. Architecture at a glance

```text
User PDF(s) or checked sample downloads
  → validation, SHA-256 identity, parser cache
  → page-aware prose / Markdown table / visual records
  → numeric-aware BM25 index + MiniLM embeddings
  → weighted reciprocal-rank fusion + optional cross-encoder reranking
  → question facets, page diversity, same-page context expansion
  → evidence packet [E1]...[En] + up to two rendered pages for visual questions
  → selected generation provider (or evidence-only mode)
  → structured answer, citation audit, numeric evidence matching
  → Streamlit Answer / Sources / Checks / Research trail
```

The central separation is deliberate: **retrieval decides what evidence is available; generation writes from that evidence; verification checks only properties that can be tested deterministically**. A fluent answer is never treated as proof by itself.

## 4. Input and PDF ingestion

The app accepts at most **three PDFs**, with a **50 MB** and **200-page** limit per file. It checks the PDF signature, opens it with PyMuPDF, rejects encrypted/password-protected documents, and warns when extractable text is sparse. Scanned/image-only PDFs may be indexed poorly; OCR is not implemented.

Parsing proceeds page by page:

1. **Prose:** position-sorted text blocks are collected; blocks overlapping detected tables are excluded to avoid duplicated prose/table evidence. Prose is split into approximately **220-word chunks with 40-word overlap**, retaining one-based PDF page numbers.
2. **Tables:** PyMuPDF's table finder extracts rows and columns. The first row becomes the Markdown header; long tables are split at **22 data rows**, repeating the header. A simple aligned-numeric-text fallback may create a table record when the native finder finds none. The parser also carries page-level unit phrases such as “in millions” into table text.
3. **Visual candidates:** pages with embedded images or captions beginning with `Figure`, `Fig.`, `Chart`, or `Exhibit` receive a visual record containing captions and nearby extracted text. This is a **heuristic**, not full chart extraction: an uncaptioned vector plot can be missed. Page pixels are rendered only later, for visual-type questions.

Every `DocumentChunk` records a deterministic chunk ID, document hash-derived ID and filename, one-based PDF page, modality (`prose`, `table`, `visual`), display text, normalized retrieval text, and available bounding-box/table metadata. This provenance lets the UI and answer citations lead back to a real page.

Parsed documents are cached under gitignored `cache/documents/` using the PDF SHA-256-derived ID and parser version. Corpus embeddings are cached separately under `cache/embeddings/`, keyed by parser version and the sorted chunk IDs/text. A changed PDF or changed parsing output naturally builds a new index. Uploaded raw PDF bytes stay in the active app session for page rendering; **extracted text and embeddings do persist in the local cache**. Sensitive-document operators must account for that cache.

The checked sample manifest points to Apple's FY2025 Q4 consolidated financial statements and *Training Compute-Optimal Large Language Models* (the Chinchilla paper). It stores URLs and SHA-256 checksums rather than committing large PDFs. Downloads are verified against those checksums before use.

## 5. Local hybrid retrieval

Retrieval combines complementary signals:

- **BM25** (`rank-bm25`) captures exact terminology, table labels, and specific numbers.
- **Dense search** uses CPU `sentence-transformers/all-MiniLM-L6-v2` embeddings and NumPy cosine similarity to capture paraphrases.
- **Numeric-aware normalization** retains ordinary tokens and adds canonical numeric forms: comma-separated values, parenthesized negatives, currency terms, percentages, and `K/M/B/bn` or written magnitude suffixes. `Decimal` avoids floating-point surprises in these canonical values.
- **Weighted reciprocal-rank fusion** merges the top 12 BM25 and dense candidates with a rank term `weight / (60 + rank)`. Numeric questions give BM25 a slight advantage (`1.25` vs `1.0`); other questions favor dense search by the same amount.
- **Cross-encoder reranking** uses `cross-encoder/ms-marco-MiniLM-L-6-v2` on the fused candidates. The final score combines the cross-encoder with normalized lexical and dense scores; for numeric questions lexical evidence receives more weight. This is an empirical heuristic, not a learned end-to-end ranking model.

Explicit `Figure 2`/`Table 1` references are matched to captions and promoted. Methodology questions also boost chunks containing signals such as “key idea,” “objective,” or “proposed method.” Selection limits a page to three chunks in the initial retrieval and tries to include the requested visual/table modality. A selected document scope prevents an ambiguous question about “this paper” from silently mixing unrelated PDFs.

For Detailed and Research report modes, the pipeline searches additional **question facets or subquestions**. A Research report first asks the model for up to four local search queries and a short outline; if planning fails, local query expansion still works. A multi-document comparison also searches each selected document independently so one source does not dominate. Candidate pages are fused again, with a relevance/diversity term. For each selected page, the system adds surrounding **same-page** prose and, where useful, a table. It removes exact sliding-window overlap and caps the total context budget. This parent-page expansion improves the information available to the writer while keeping citations tied to the page actually supplied.

In the measured ablation, reranking did **not** beat simpler hybrid fusion on MRR, so the cross-encoder should be described as an implemented experimental stage, not a proven gain on this corpus.

## 6. Visual-question handling

Questions mentioning a chart, figure, plot, graph, visual, trend, or curve trigger visual handling. The pipeline prefers selected visual records, renders at most **two distinct source pages** as JPEG at roughly **150 DPI**, and sends them only if the selected model is marked vision-capable. It does **not** rasterize every page during indexing, which reduces ingestion cost.

The model may report structured `visual_observations` with an evidence ID, metric, value, unit, and label. Such image-only readings remain **ambiguous** in the numerical checker until supported by extracted textual/table evidence or reviewed by a person. A text-only model can use captions and extracted prose but cannot reliably read the pixels of a plot; the app warns about this.

## 7. Answer generation and the research workflow

The app offers three answer depths:

| Mode | Retrieval and output behavior | Generation requests |
|---|---|---:|
| Quick | Primary search and a short direct answer; up to 4 evidence items | 1 |
| Detailed | Primary reranked search plus cheap hybrid facet search; up to 6 evidence items and a substantive explanation | 1 |
| Research report | Model-planned local subquestions/outline, broader evidence (up to 10 items), multi-section brief | Usually 2: plan + answer |

The context budgets are approximately **1,600 / 2,800 / 6,000 words** for Quick / Detailed / Research report, then reduced if the selected model's context limit is smaller. Generation output caps also differ by mode. These are budgets and writing instructions, **not guaranteed answer lengths**. A narrow or unsupported question should not be inflated into a long report.

The generation prompt asks for a JSON object containing a short `answer` lead, distinct `sections`, `used_evidence_ids`, `visual_observations`, `insufficient_evidence`, and follow-up questions. It instructs the model to cite factual sentences, preserve source units, avoid derived arithmetic, state when evidence is missing, and treat PDF text/images as **untrusted data rather than instructions**. The UI renders the lead plus every section as Markdown.

The app streams partial output. A prior defect made a long report appear shorter after generation; the current renderer retains generated sections instead of silently rewriting or deduplicating them. If the JSON is malformed, truncated, or interrupted, it tries to retain readable draft text, marks the output incomplete/unverified, and still exposes the sources. Preserving a draft protects usability, **not factual accuracy**.

### Current provider choices

| Provider | Connection | Practical status |
|---|---|---|
| Ollama | Local native `/api/chat`; installed model discovery and explicit context/JSON controls | UI default; requires an actually installed model. Text-only models cannot inspect page images. Local Mistral worked but sometimes made factual errors. |
| OpenRouter | OpenAI-compatible Chat Completions; dynamic free catalog, pinned selected models | Selected free models completed live text and vision checks, but quotas, model availability, and latency vary. `openrouter/free` is a dynamic router, not a reproducible pinned model. |
| Claude | Anthropic Messages transport | Implemented and mock-tested; **not live-tested with a supplied Anthropic key**. API use is billed. |
| OpenAI | OpenAI Chat Completions | Optional paid benchmark; not claimed as a completed live comparison. |

OpenCode and AgentRouter were removed from the app at the user's request after access tests failed. NVIDIA is not a current UI choice. The provider adapter records the actual model and request count; explicit OpenRouter model choices are pinned rather than silently falling back. The small **Test connection** action uses an eight-second timeout and no retries, but even that test may consume quota. A timeout, HTTP 429, or a catalog listing alone does not establish whether a model can successfully answer a full document question.

Without a provider key, the pipeline returns an **evidence-only** result, so local retrieval and source inspection remain usable. When using a hosted provider, the app sends the question, selected evidence text, recent user-question context for follow-ups, and—only for visual queries with vision support—up to two selected page images. Research planning also sends a small seed-evidence bundle. Keys entered in the UI are kept in Streamlit session state rather than written by the application; local environment variables are an optional development input. Never commit keys, include them in screenshots, or paste them into public portfolio material.

## 8. Citations and numerical grounding

Evidence labels (`[E1]`, `[E2]`, …) are assigned to the selected source pages. The generator is prompted to cite factual sentences. The app checks whether used IDs are among the supplied evidence, warns about unknown labels, and reports **citation coverage** over substantial answer passages. The source panel shows each ID's document, PDF page, modality, passage, and optional rendered preview. Citations are navigational and auditable, but **citation presence is not proof that a sentence is entailed by the source**.

After generation, numerical verification extracts numeric spans from the answer and the cited source text using `Decimal`. It skips structural labels such as “Figure 3” or “Approach 1,” canonicalizes magnitude suffixes, handles explicit source-table scales such as “in millions,” and compares nearby metric tokens and units. Where a sentence cites evidence IDs, matching is restricted to those IDs.

| Badge | Meaning |
|---|---|
| `verified` | Equivalent value **and** meaningful nearby metric context match cited extracted text/table evidence, with compatible unit handling. |
| `ambiguous` | The value is present but metric/unit context is unclear, or support comes only from a model-read visual observation. Human page review is needed. |
| `unsupported` | No equivalent value was found in the retrieved/cited evidence. |

If structured generation cannot be parsed reliably, the checker receives no trusted evidence and cannot issue a verified badge for that draft. The system intentionally avoids interpreting a visual-only number as independently verified. Even a `verified` badge means **evidence matching, not independent factual proof**: it cannot guarantee the right entity, row/column alignment, causal interpretation, or truth of the underlying PDF. The prompt forbids derived calculations because the current checker does not validate arithmetic chains.

## 9. Streamlit user experience

The workflow is: choose a checked sample or upload PDFs, build the local evidence index, select one document or an explicit all-document comparison, choose answer depth and provider/model, ask a question, then inspect four result tabs:

- **Answer:** streamed and final Markdown report, inline clickable source labels, provider/model/request/timing information, follow-up prompts, and Markdown export.
- **Sources:** the exact evidence packet supplied to the model, with document, page, modality, passage/table text, and an on-demand page preview.
- **Checks:** numeric claim badges, reasons and supporting source IDs, plus citation-presence coverage.
- **Research trail:** subquestions, optional outline, and stage timings.

The UI shows actionable messages for missing Ollama models, provider authentication failures, quotas, timeouts, unavailable endpoints, malformed PDFs, insufficient source text, and generation failure. Retrieval evidence remains accessible after a model failure. The Streamlit answer container grows with its content; users scroll the page to read the whole report.

## 10. Data structures and code map

| Component | Purpose |
|---|---|
| `DocumentChunk` | One page-provenance prose, table, or visual record. |
| `RetrievedEvidence` | A chunk plus lexical, dense, fusion, reranker scores and assembled same-page context. |
| `VisualObservation` | A model-reported reading from a supplied page image. |
| `NumericClaim` | Extracted answer value, normalized value, status, supporting IDs, and explanation. |
| `AnswerResult` | Answer text, evidence, citations, checks, provider/model, warnings, mode, requests, and timings. |

The main modules are [`app.py`](app.py) (Streamlit UI), [`src/ingestion.py`](src/ingestion.py) (PDF parsing/rendering), [`src/cache.py`](src/cache.py), [`src/retrieval.py`](src/retrieval.py), [`src/research.py`](src/research.py) (multi-query/page-context assembly), [`src/providers.py`](src/providers.py) and [`src/transports.py`](src/transports.py) (generation), [`src/verification.py`](src/verification.py), and [`src/pipeline.py`](src/pipeline.py) (orchestration). [`src/models.py`](src/models.py) defines the shared records. [`src/catalog.py`](src/catalog.py) handles model discovery and friendly provider errors.

Key libraries: **Python, Streamlit, PyMuPDF, Sentence Transformers, PyTorch CPU, rank-bm25, NumPy, OpenAI SDK, requests, and pytest**. The index is small enough that NumPy arrays plus BM25 are sufficient; there is no separate vector-database service to deploy.

## 11. Evaluation: what was actually measured

The checked gold set in [`data/evaluation/questions.json`](data/evaluation/questions.json) contains **16 questions**: four prose, four table, four chart/figure, and four unanswerable or ambiguous. Each item stores its expected document, PDF page/modality when answerable, expected answer, and relevant numeric values. The reported local retrieval ablation uses the **12 answerable questions**, scoped to the expected document, with cached models/documents and CPU inference. Generation is excluded from these latency figures.

| Retrieval configuration | Gold-page Recall@5 | MRR@5 | Numeric evidence coverage | Median warm latency |
|---|---:|---:|---:|---:|
| Dense only | 100% | 0.861 | 62.5% | 0.013 s |
| BM25 + dense RRF | 100% | 0.903 | 87.5% | 0.012 s |
| Hybrid + cross-encoder | 100% | 0.854 | 87.5% | 1.058 s |
| Detailed pipeline: reranked primary + hybrid facets + page context | 100% | 0.861 | 100% | 1.057 s |

**How to interpret these numbers:** Recall@5 asks whether the labeled page appears among the first five; MRR@5 rewards ranking it higher; numeric evidence coverage asks whether expected numbers are present in the retrieved/context packet. Coverage is **not numeric answer accuracy**, and 100% page recall on 12 curated questions is **not a generalization claim**. The detailed configuration expanded numeric availability, but its MRR did not exceed hybrid RRF. Measured CPU latency varies with host load, model warm-up, and thread settings. These values are a dated experiment snapshot, not an SLA.

The optional generation evaluator can calculate numeric exact match, citation-page accuracy, unanswerable refusal accuracy, and unsupported-number rate when a provider is supplied. However, the full 16-question live generation benchmark has **not** been completed on a validated stable default provider, so no aggregate generation-quality or citation-page percentage should be claimed. A separate live regression on the *Efficient Streaming Language Models with Attention Sinks* paper tested text, figure, refusal, provider failure, and browser workflows; it also exposed occasional semantic mistakes despite citations. A browser regression and automated tests validate behavior, not all factual claims.

The detailed evidence and historical caveats are in [`EVALUATION.md`](EVALUATION.md). As of its 23 September 2026 follow-up, the current suite reported **45 passing tests**; run it again before using a test count publicly. Earlier counts and three-request research timings in that file are historical: the current Research report flow removed its hidden rewrite and normally uses two generation requests.

## 12. Reliability decisions, trade-offs, and known limitations

| Decision | Benefit | Trade-off / remaining risk |
|---|---|---|
| Local BM25 + embeddings | Fast, inspectable retrieval without external indexing service | Embedding models still have download/cold-start cost; dense search may miss exact numeric distinctions. |
| Hybrid fusion + reranker | Combines exact and semantic signals | Cross-encoder adds roughly a second of warm CPU latency in the measured set and did not improve MRR there. |
| Same-page context expansion | Gives the writer nearby qualifications and table context | Adds tokens and can still omit relevant information on another page. |
| Query-time image rendering | Avoids image-processing every page | Visual discovery is heuristic; text-only models cannot inspect pixels. |
| Structured JSON + draft retention | Supports consistent rendering and preserves partial long answers | Small/free models can return malformed JSON; retained drafts are unverified. |
| Deterministic numeric checks | Catches many unmatched values and citation mismatches | Cannot prove semantic entailment, row alignment, or arithmetic correctness. |
| BYOK/evidence-only fallback | Keeps search usable during hosted quota/provider failures | Hosted calls expose selected content to that provider; free quotas and latency are outside app control. |

Other boundaries: no OCR for scanned PDFs, no guarantee of universal document layouts or perfect table extraction, no persistent chat history or user accounts, no automatic source-truth validation, and no medical/legal reliability claims. Answer quality depends strongly on the selected model and the evidence retrieved. Public Streamlit deployment is documented in the README but **not verified as completed by this guide**.

## 13. Reproduce the project

From the repository root, using Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
streamlit run app.py
```

The first run downloads the embedding and reranking models. Choose a sample or upload a supported PDF, click **Build evidence index**, and use the app without a key for source search or add a configured provider for generated answers. For local generation, start Ollama, install a model yourself, and select it from the refreshed installed-model list. A vision-capable local model is needed for pixel-level chart reading. Restart Streamlit after source-file edits because file watching is disabled.

Reproduce the retrieval study and tests:

```bash
python scripts/evaluate.py --provider none --output evaluation_results/retrieval.json
pytest -q
```

Optional live generation and browser checks are described in [`README.md`](README.md). Do not place API keys in commands, repository files, generated reports, or screenshots. `cache/` and `evaluation_results/` are gitignored; inspect or clear local cache when working with sensitive documents. For Streamlit Community Cloud, point the app at `app.py` and use hosted BYOK/evidence-only mode; a localhost Ollama server is not automatically reachable from that hosted app.

## 14. Portfolio and resume wording

### Short portfolio description

> Built a PDF research assistant that retrieves evidence across prose, financial tables, and visual pages using local hybrid search, then produces page-cited explanations with deterministic checks for numerical claims. The system supports local or bring-your-own-key generation, inspectable source pages, and graceful evidence-only fallback when an API provider fails.

### Longer case-study description

> I designed and implemented an evidence-first RAG pipeline for complex PDFs. PyMuPDF converts each document into page-aware prose, structured Markdown tables, and visual-page candidates. A numeric-aware BM25 index and MiniLM embeddings are fused, with cross-encoder reranking and multi-query page-context expansion for deeper questions. For chart questions, only selected pages are rendered and sent to a compatible vision model. Responses are organized into cited sections; a separate Decimal-based checker labels quantitative claims by whether their values, units, context, and cited evidence match. The Streamlit interface exposes sources, verification results, research steps, timings, and Markdown export. I benchmarked retrieval on a 16-question curated set and documented both gains and failure modes, including quota limits, malformed model output, and the limits of citation/numeric checks.

### Resume bullet options

Choose two or three that match the role; do not present all of them as independent projects.

- Built a multimodal PDF RAG application in Python/Streamlit, parsing prose, tables, and visual-page cues with page-level provenance and an inspectable source trail.
- Implemented numeric-aware BM25 + MiniLM retrieval, weighted RRF, cross-encoder reranking, and multi-query same-page context expansion; achieved **100% gold-page Recall@5 on 12 answerable curated questions** (small, document-scoped evaluation).
- Developed deterministic `Decimal`-based checks that classify generated numerical claims as verified, ambiguous, or unsupported against cited evidence, while explicitly flagging image-only readings for review.
- Added Ollama/OpenRouter generation paths, structured streaming reports, partial-draft recovery, provider-error fallbacks, and an evidence-only mode that requires no API key.
- Evaluated dense, hybrid, and reranked retrieval with Recall@5, MRR@5, numeric evidence coverage, and warm CPU latency; documented cases where the more complex reranker **did not** outperform simpler fusion.

### Interview talking points

1. **Why hybrid search?** Exact labels and numbers are often best found lexically, while paraphrases benefit from embeddings. Fusion covers both; the ablation measures rather than assumes its benefit.
2. **Why no vector database?** The supported corpus is only three PDFs. NumPy cosine search and BM25 keep deployment simple; a larger or multi-user corpus would change that trade-off.
3. **What makes it multimodal?** Text/table parsing plus selective page-image input to a vision model. It is not full end-to-end visual indexing or OCR.
4. **What does a verified number mean?** A value and metric-context match in cited extracted evidence. It does not prove row alignment, semantic truth, or that the PDF itself is correct.
5. **How did you improve long answers?** Added bounded facet/subquestion retrieval, same-page context, mode-specific budgets, streaming, section-preserving rendering, and graceful partial-response handling. More words alone are not the goal.
6. **What failed?** Free-model quotas and variable latency, occasional malformed JSON, inaccurate local-model prose, and a reranker that did not improve MRR on the small curated set. These are measured limitations, not hidden behind the UI.
7. **What would you build next?** Better table-cell provenance and row-aware verification, OCR for scans, document-specific visual detection, claim-level entailment review, and a larger independently labeled end-to-end evaluation.

### Claims to avoid

Do **not** claim zero hallucinations, perfectly verified financial/scientific answers, STORM-equivalent research, general 100% retrieval accuracy, guaranteed sub-two-second responses, completed public deployment, or a validated paid-provider comparison. The strongest defensible story is **transparent, local evidence retrieval with measured improvements and explicit uncertainty**.

## 15. Repository references

- [`README.md`](README.md): setup, provider configuration, deployment steps, and operational limits.
- [`EVALUATION.md`](EVALUATION.md): measured retrieval and historical live/provider observations.
- [`data/sample_manifest.json`](data/sample_manifest.json): sample URLs and checksums.
- [`data/evaluation/questions.json`](data/evaluation/questions.json): the curated gold questions.
- [`scripts/evaluate.py`](scripts/evaluate.py): ablation and optional generation metrics.
- [`tests/`](tests/): unit and mocked integration coverage.

No API keys or private document contents belong in this guide, portfolio, or resume.

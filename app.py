from __future__ import annotations

import html
import os
import re
from dataclasses import replace
from time import perf_counter
from pathlib import Path

import streamlit as st

from src.catalog import ModelOption, discover_models, inspect_ollama, provider_error_message
from src.pipeline import DocumentRAGPipeline
from src.providers import OpenAICompatibleProvider, ProviderConfig, partial_answer
from src.research import export_markdown
from src.samples import download_sample, load_manifest

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
ALL_DOCUMENTS = "__all_documents__"
st.set_page_config(page_title="Document research workspace", page_icon="◈", layout="wide")
st.markdown("""
<style>
.stApp {background:#f7f8fc;color:#192944}
[data-testid="stHeader"] {background:rgba(247,248,252,.9)}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid #e5e9f2}
[data-testid="stMainBlockContainer"] {max-width:1190px;padding-top:2rem}
h1,h2,h3 {letter-spacing:-.025em;color:#142745}
.hero {background:linear-gradient(118deg,#142640,#284b7c);padding:1.8rem 2rem;border-radius:20px;color:white;margin-bottom:1.5rem}
.hero h1 {color:white;font-size:2.2rem;margin:.25rem 0 .55rem}
.hero p {color:#cbdcf2;margin:0;max-width:780px}
.eyebrow {font-size:.72rem;letter-spacing:.15em;font-weight:700;text-transform:uppercase;color:#83d5ed}
.source-link {padding:.65rem .9rem;border:1px solid #e3e8f1;border-radius:10px;background:#fff;margin:.45rem 0;font-size:.87rem}
.st-key-answer_surface {background:#fff;border-left:4px solid #6b83e6!important;border-radius:16px!important;padding:1.6rem!important;line-height:1.75;box-shadow:0 5px 24px #18345a08}
[data-testid="stForm"] {background:#fff;border:1px solid #e3e8f1;border-radius:16px;padding:1rem}
[data-testid="stExpander"] {background:#fff;border-color:#e3e8f1}
.stButton>button,.stFormSubmitButton>button {border-radius:9px;font-weight:600}
[data-testid="stMetricValue"] {font-size:1.5rem}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600, show_spinner=False)
def model_catalog(provider: str, base_url: str) -> list[ModelOption]:
    return discover_models(provider, base_url)


@st.cache_data(ttl=60, show_spinner=False)
def local_model_info(model: str, base_url: str) -> ModelOption:
    return inspect_ollama(model, base_url)


def reset_workspace_answer() -> None:
    for key in ("last_result", "question_history", "past_results", "preview_source", "result_tabs"):
        st.session_state.pop(key, None)


def set_question(question: str) -> None:
    st.session_state["question_input"] = question


def citation_links(answer: str) -> str:
    return re.sub(r"\[(E\d+)\]", lambda m: f"[{m[1]}](#source-{m[1].lower()})", answer)


with st.sidebar:
    st.markdown("### Answer model")
    provider_name = st.selectbox("Provider", ["Ollama", "OpenRouter", "Claude", "OpenAI"])
    defaults = ProviderConfig.for_provider(provider_name)
    base_url = defaults.base_url
    if provider_name == "Ollama":
        base_url = st.text_input("Ollama server", value=base_url)
        api_key = "ollama"
    else:
        env_name = {"OpenRouter": "OPENROUTER_API_KEY", "Claude": "ANTHROPIC_API_KEY", "OpenAI": "OPENAI_API_KEY"}[provider_name]
        api_key = st.text_input(
            "API key", type="password", key=f"key_{provider_name}",
            value=os.environ.get(env_name, ""), placeholder="Optional for source search",
            help="Held in this session only. The app never writes or logs this key.",
        )
    catalog_key = f"catalog_{provider_name}_{base_url}"
    refresh = st.button("Refresh available models", use_container_width=True)
    if refresh or (provider_name in ("Ollama", "OpenRouter", "Claude") and catalog_key not in st.session_state):
        try:
            if refresh:
                model_catalog.clear()
                local_model_info.clear()
            with st.spinner("Checking model availability…"):
                st.session_state[catalog_key] = model_catalog(provider_name, base_url)
        except Exception as exc:
            st.warning(provider_error_message(exc, provider_name))
    options = st.session_state.get(catalog_key)
    if options is None:
        options = [ModelOption(defaults.model, defaults.model, defaults.supports_images)]
    if provider_name == "OpenRouter" and not st.checkbox("Show other untested models", value=False):
        recommended = {"dots-studio/dots-3-note-preview:free", "nex-agi/nex-n2.5-mini:free"}
        options = [x for x in options if x.id in recommended]
    option_by_id = {option.id: option for option in options}
    preferred = "dots-studio/dots-3-note-preview:free"
    model_ids = list(option_by_id)
    if provider_name == "OpenRouter" and preferred in model_ids:
        model_ids.remove(preferred)
        model_ids.insert(0, preferred)
    selection = st.selectbox(
        "Model", [*model_ids, "Custom model…"], key=f"model_selection_{provider_name}",
        format_func=lambda x: f"{x} · vision" if x in option_by_id and option_by_id[x].vision else x,
    )
    if selection == "Custom model…":
        model = st.text_input("Model ID", key=f"custom_model_{provider_name}")
        vision = st.checkbox("This model accepts images", value=False)
        option = ModelOption(model, model, vision)
    else:
        option = option_by_id[selection]
        model = option.id
    local_ready = True
    if provider_name == "Ollama":
        try:
            option = local_model_info(model, base_url)
            st.caption(f"Installed · {'text + images' if option.vision else 'text only'}")
        except Exception as exc:
            local_ready = False
            st.warning("Select an installed model. The requested model is missing or the Ollama server is unavailable.")
            if re.fullmatch(r"[a-zA-Z0-9._:/-]+", model or ""):
                st.code(f"ollama pull {model}", language="bash")
    elif provider_name == "OpenRouter":
        st.caption("Refresh loads currently listed free models. Availability and quotas can change.")
    elif provider_name == "NVIDIA":
        st.caption("Uses your NVIDIA API access and its account limits. Refresh lists supported text-generation models.")
    config = ProviderConfig.for_provider(
        provider_name, api_key=api_key, model=model or defaults.model,
        base_url=base_url, supports_images=option.vision,
        context_length=min(option.context_length, 32768),
        disable_reasoning=option.disable_reasoning,
        json_output=option.json_output,
    )
    config = replace(config, request_timeout=180 if provider_name == "Ollama" else 60)
    if provider_name in ("Claude", "OpenAI"):
        st.caption("Direct API usage is billed by the provider; no automatic switch to this provider.")
    if st.button("Test connection", disabled=not api_key or not model or not local_ready, use_container_width=True):
        try:
            with st.spinner("Testing one tiny request (8-second timeout, no retries)…"):
                checker = OpenAICompatibleProvider(replace(config, request_timeout=8))
                response = checker._complete([{"role": "user", "content": "Reply with OK only."}], max_tokens=128)
                if not response.choices[0].message.content:
                    raise ValueError("empty response")
            st.success("Generation connection works.")
        except Exception as exc:
            st.error(provider_error_message(exc, provider_name))
    if not api_key:
        st.caption("Source search is ready without a key.")
    st.divider()
    st.caption("Hosted generation sends your question, selected evidence, recent question context, and up to two page images. Retrieval runs locally.")
    if not option.vision:
        st.caption("Text-only model: charts can be discussed from extracted text and captions; reading pixels requires a vision model.")

pipeline: DocumentRAGPipeline | None = st.session_state.get("pipeline")
st.markdown('''<div class="hero"><div class="eyebrow">A workspace for understanding documents</div>
<h1>Read deeply. Ask precisely.</h1><p>Turn papers and reports into explanations, comparisons, and research briefs — with a source trail you can inspect.</p></div>''', unsafe_allow_html=True)

if pipeline is None:
    st.subheader("Choose up to three documents")
    manifest = load_manifest(ROOT / "data/sample_manifest.json")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Explore a sample")
        samples = st.multiselect("Curated samples", [item["label"] for item in manifest])
        st.caption("Financial statements for numbers and tables; the Chinchilla paper for research questions.")
    with right:
        st.markdown("#### Bring your own material")
        uploads = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
        st.caption("Up to 50 MB and 200 pages per PDF. Scanned PDFs need OCR before upload.")
    if st.button("Build evidence index", type="primary", disabled=not samples and not uploads):
        try:
            if len(samples) + len(uploads or []) > 3:
                raise ValueError("Choose no more than three PDFs.")
            with st.status("Preparing your research workspace…", expanded=True) as status:
                chosen = []
                for sample in manifest:
                    if sample["label"] in samples:
                        st.write(f"Loading {sample['label']}")
                        chosen.append(download_sample(sample, CACHE_DIR / "samples"))
                chosen.extend((upload.name, upload.getvalue()) for upload in uploads or [])
                st.write("Reading pages, tables, and visual captions; building the local search index")
                built = DocumentRAGPipeline(CACHE_DIR)
                parsed = built.ingest(chosen)
                st.session_state["pipeline"] = built
                st.session_state["source_scope"] = parsed[-1].document_id
                reset_workspace_answer()
                status.update(label="Workspace ready", state="complete")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))
    st.stop()

documents = pipeline.documents
with st.sidebar:
    st.markdown("### Your documents")
    for document in documents:
        st.write(document.document_name)
        st.caption(f"{document.page_count} pages")
    if st.button("Change documents", use_container_width=True):
        st.session_state.pop("pipeline", None)
        st.session_state.pop("source_scope", None)
        reset_workspace_answer()
        st.rerun()

for document in documents:
    for warning in document.warnings:
        st.warning(f"{document.document_name}: {warning}")

scope_labels = {d.document_id: d.document_name for d in documents}
if len(documents) > 1:
    scope_labels[ALL_DOCUMENTS] = "Compare all documents"
if st.session_state.get("source_scope") not in scope_labels:
    st.session_state["source_scope"] = documents[-1].document_id
source_col, mode_col = st.columns([1.5, 1])
with source_col:
    scope = st.selectbox("Read from", list(scope_labels), format_func=scope_labels.get, key="source_scope", on_change=reset_workspace_answer)
with mode_col:
    answer_mode = st.selectbox("Answer depth", ["Detailed", "Quick", "Research report"])
st.caption(
    "Research report explores subquestions and builds an outline. Two requests; the streamed draft is retained without a hidden rewrite."
    if answer_mode == "Research report" else "Detailed explanations use focused searches and surrounding context. One generation request; page images are sent only for visual questions."
    if answer_mode == "Detailed" else "A focused answer for a specific fact. Uses one generation request."
)
starter_cols = st.columns(3)
for col, label, prompt in zip(starter_cols,
    ["Explain the method", "Examine the evaluation", "Find limitations"],
    ["Explain the central method, why it works, and the evidence supporting it.",
     "Explain the evaluation setup, compare the baselines and key results, and state the conditions for those results.",
     "What are the main limitations and unanswered questions? Distinguish stated limitations from gaps in the evidence."],
):
    col.button(label, on_click=set_question, args=(prompt,), use_container_width=True)
with st.form("question_form"):
    question = st.text_area("Your question", key="question_input", height=100, placeholder="Ask a specific question, request a comparison, or explore the paper in depth…")
    ask = st.form_submit_button("Research and answer", type="primary", disabled=provider_name == "Ollama" and not local_ready)
if ask:
    if not question.strip():
        st.error("Enter a question first.")
    elif scope == ALL_DOCUMENTS and re.search(r"\bthis\s+(paper|document|report|study)\b", question, re.I):
        st.error("Select one source for ‘this paper’, or explicitly ask to compare the documents.")
    else:
        st.session_state.pop("last_result", None)
        st.session_state.pop("preview_source", None)
        st.session_state["result_tabs"] = "Answer"
        try:
            provider = OpenAICompatibleProvider(config) if api_key and model else None
            draft = st.empty()
            last_paint = [0.0]
            def show_partial(raw: str) -> None:
                now = perf_counter()
                if now - last_paint[0] < 0.08:
                    return
                text = partial_answer(raw)
                if text:
                    draft.markdown(text + " ▍")
                    last_paint[0] = now
            with st.status("Researching your question…", expanded=True) as status:
                result = pipeline.answer(
                    question, provider, document_ids=None if scope == ALL_DOCUMENTS else {scope},
                    answer_mode=answer_mode, history=st.session_state.get("question_history", []),
                    on_progress=lambda message: status.update(label=message),
                    on_partial=show_partial,
                )
                draft.empty()
                st.session_state["last_result"] = result
                st.session_state["question_history"] = [*st.session_state.get("question_history", []), question][-4:]
                status.update(label="Draft retained — check the warning" if result.draft_retained else "Answer ready" if result.generation_succeeded else "Sources ready — generation needs attention" if provider else "Sources ready", state="complete", expanded=False)
        except Exception as exc:
            st.error(provider_error_message(exc, provider_name))

result = st.session_state.get("last_result")
if result is None:
    st.stop()

st.divider()
answer_tab, sources_tab, checks_tab, research_tab = st.tabs(
    ["Answer", f"Sources · {len(result.evidence)}", "Checks", "Research trail"],
    key="result_tabs", on_change="rerun",
)
with answer_tab:
    with st.container(border=True, key="answer_surface"):
        st.markdown(citation_links(result.answer))
    if result.parse_warning:
        st.warning(result.parse_warning)
    for warning in result.warnings:
        st.caption(warning)
    if result.generation_succeeded:
        st.caption(f"{result.provider} · {result.model} · {result.request_count} request(s) · {result.timings.get('total_answer_seconds', 0):.1f}s")
    elif not result.provider:
        st.info("Connect an answer model in the sidebar for a synthesized explanation. Sources are available now.")
    st.download_button("Download answer with citations", export_markdown(result), file_name="document-research.md", mime="text/markdown")
    cited = [item for item in result.evidence if item.evidence_id in result.used_evidence_ids]
    if cited:
        st.caption("Cited pages · open the Sources tab for passages and page previews")
        for item in cited:
            st.markdown(f'<div class="source-link" id="source-{item.evidence_id.lower()}"><strong>{item.evidence_id}</strong> · {html.escape(item.chunk.document_name)} · page {item.chunk.page_number}</div>', unsafe_allow_html=True)
    if result.follow_up_questions:
        st.markdown("#### Explore further")
        for index, followup in enumerate(result.follow_up_questions):
            st.button(followup, key=f"followup_{index}", on_click=set_question, args=(followup,))

with sources_tab:
    st.caption("These are the page passages supplied to the answer model, including surrounding context. Images load only when requested.")
    if result.evidence:
        by_id = {item.evidence_id: item for item in result.evidence}
        selected_eid = st.selectbox("Inspect a source", list(by_id), format_func=lambda eid: f"{eid} · {by_id[eid].chunk.document_name} · page {by_id[eid].chunk.page_number}")
        item = by_id[selected_eid]
        text_col, preview_col = st.columns([1.15, 1])
        with text_col:
            st.markdown(f"**{item.chunk.modality.title()} · PDF page {item.chunk.page_number}**")
            st.markdown(item.context_text or item.chunk.text)
        with preview_col:
            if st.button("Show this page", key=f"preview_{selected_eid}"):
                st.session_state["preview_source"] = selected_eid
            if st.session_state.get("preview_source") == selected_eid:
                page_image = pipeline.render_evidence_page(item)
                if page_image:
                    st.image(page_image, caption=f"PDF page {item.chunk.page_number}")

with checks_tab:
    st.caption("Checks match values, units, nearby metric words, and source references. They do not independently establish truth. Image-only observations require visual review.")
    cols = st.columns(3)
    for col, state in zip(cols, ("verified", "ambiguous", "unsupported")):
        col.metric(state.title(), sum(claim.status == state for claim in result.numeric_claims))
    if not result.numeric_claims:
        st.info("No quantitative claims were extracted from this answer.")
    for claim in result.numeric_claims:
        with st.expander(f"{claim.status.upper()} · {claim.original}"):
            st.write(claim.reason)
            st.caption("Supporting sources: " + (", ".join(claim.supporting_evidence_ids) or "none"))
    st.metric("Passages with citations", f"{result.citation_coverage:.0%}")
    st.caption("Citation presence is a formatting check, not a claim-entailment score.")

with research_tab:
    st.markdown("#### Questions searched")
    for query in result.research_queries:
        st.write(f"• {query}")
    if result.outline:
        st.markdown("#### Writing outline")
        for section in result.outline:
            st.write(f"• {section}")
    st.caption("Search → hybrid ranking → page diversity → surrounding context → synthesis → citation and numeric checks")
    st.json({name: round(value, 3) for name, value in result.timings.items()})

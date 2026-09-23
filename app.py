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
st.set_page_config(page_title="Document research workspace", page_icon="◈", layout="wide", initial_sidebar_state="auto")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {--ink:#10272c;--muted:#60747a;--line:#dce7e4;--mint:#dff6ed;--teal:#087c70;--cream:#f7f8f4;--orange:#f6c98c}
html,body,[class*="css"],.stApp {font-family:'DM Sans',Arial,sans-serif}
.stApp {background:radial-gradient(circle at 78% -12%,#e2f2eb 0,transparent 33%),var(--cream);color:var(--ink)}
[data-testid="stHeader"] {background:rgba(247,248,244,.87);backdrop-filter:blur(12px)}
[data-testid="stSidebar"] {background:#102b2d;border-right:1px solid #214044}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h3 {color:#f0f8f4}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#b7ceca}
[data-testid="stSidebar"] hr {border-color:#355054}
[data-testid="stSidebar"] [data-baseweb="select"],
[data-testid="stSidebar"] input {color:#162f32}
[data-testid="stSidebar"] .stButton button p {color:#173639!important}
[data-testid="stSidebar"] .stButton button:disabled p {color:#6c8582!important}
[data-testid="stSidebar"] [data-testid="stAlert"] p {color:#173639}
[data-testid="stMainBlockContainer"] {max-width:1240px;padding:2rem 2.35rem 5rem}
.block-container {padding-bottom:5rem}
h1,h2,h3,h4 {font-family:Manrope,'DM Sans',sans-serif;letter-spacing:-.045em;color:var(--ink)}
h2 {font-weight:800!important}
.brand {display:flex;align-items:center;gap:.75rem;margin:.15rem 0 1.8rem;color:#fff;font-family:Manrope,sans-serif;font-size:1.05rem;font-weight:800;letter-spacing:-.035em}
.brand-mark {display:grid;place-items:center;width:36px;height:36px;border-radius:11px;background:#d9f4df;color:#0b5149;font-size:1.4rem;transform:rotate(-8deg)}
.sidebar-kicker {color:#81cfc0!important;font-size:.68rem!important;font-weight:800;letter-spacing:.16em;text-transform:uppercase;margin:0 0 .35rem}
.topline {display:flex;justify-content:space-between;align-items:center;margin:.1rem 0 1.1rem;color:#68817e;font-size:.76rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}
.topline .live {color:#137a70;letter-spacing:0;font-size:.77rem;text-transform:none;background:#e4f5ec;border:1px solid #cae9db;border-radius:99px;padding:.35rem .72rem}
.hero {position:relative;overflow:hidden;display:grid;grid-template-columns:minmax(0,1.14fr) minmax(270px,.86fr);gap:2.5rem;align-items:center;min-height:348px;padding:3rem 3.2rem;border-radius:25px;background:#102d32;color:#fff;box-shadow:0 26px 48px -28px #0d3b37a3;margin-bottom:2rem}
.hero:before {content:'';position:absolute;width:470px;height:470px;border:1px solid #ffffff19;border-radius:50%;right:-118px;top:-180px;box-shadow:0 0 0 70px #ffffff08,0 0 0 140px #ffffff04}
.hero-copy,.hero-art {position:relative;z-index:1}
.hero h1 {color:#fff;font-size:clamp(2.25rem,4vw,3.6rem);line-height:1.13;font-weight:800;margin:.65rem 0 1rem;max-width:740px}
.hero h1 em {font-style:normal;color:#a5e4c8}
.hero p {color:#c5dbd5;margin:0;max-width:600px;font-size:1.03rem;line-height:1.65}
.eyebrow {font-size:.68rem;letter-spacing:.18em;font-weight:800;text-transform:uppercase;color:#8ee0c1}
.hero-tags {display:flex;gap:.5rem;flex-wrap:wrap;margin-top:1.45rem}
.hero-tags span {padding:.47rem .72rem;border:1px solid #ffffff2b;border-radius:99px;color:#dceee8;font-size:.72rem;font-weight:700;background:#ffffff0b}
.hero-art {max-width:340px;justify-self:end;width:100%;transform:rotate(3deg)}
.art-paper {background:#f9fbf7;color:#183236;padding:1.2rem;border-radius:16px;box-shadow:0 23px 50px #021d2190}
.art-top {display:flex;align-items:center;gap:.55rem;color:#79908c;text-transform:uppercase;font-size:.62rem;font-weight:800;letter-spacing:.1em;border-bottom:1px solid #e1ebe6;padding-bottom:.8rem}
.art-dot {width:8px;height:8px;border-radius:50%;background:#eeaa6c}
.art-heading {font-family:Manrope,sans-serif;font-weight:800;letter-spacing:-.03em;font-size:.95rem;margin:1rem 0 .75rem}
.art-line {height:8px;border-radius:8px;background:#e4ebe7;margin:.45rem 0}
.art-line.mid {width:77%}.art-line.short {width:55%}
.art-highlight {display:flex;justify-content:space-between;align-items:center;background:#e5f4e8;border-left:3px solid #2c9c7b;padding:.58rem .7rem;margin-top:1rem;border-radius:4px 8px 8px 4px;font-size:.75rem;font-weight:700}
.art-highlight span {color:#1c8265}
.section-eyebrow {font-size:.68rem;font-weight:800;letter-spacing:.16em;color:#178171;text-transform:uppercase;margin:.2rem 0 .32rem}
.section-intro {color:var(--muted);margin:-.25rem 0 1.1rem;font-size:.96rem}
.workspace-hero {border:1px solid #dbe7de;background:linear-gradient(112deg,#e9f6ed,#f9fbf5 65%);padding:1.5rem 1.85rem;border-radius:20px;margin-bottom:1.65rem}
.workspace-hero h1 {font-size:clamp(1.65rem,2.5vw,2.25rem);line-height:1.2;margin:.25rem 0 .35rem}
.workspace-hero p {color:#56736f;margin:0;font-size:.93rem}
.workspace-hero.compact {padding:.95rem 1.3rem;margin-bottom:.8rem}
.workspace-hero.compact h1 {font-size:1.25rem;margin:.12rem 0}
.doc-shelf {display:flex;gap:.55rem;flex-wrap:wrap;margin:.7rem 0 1.4rem}
.doc-pill {background:#fff;border:1px solid #d8e7de;border-radius:9px;padding:.55rem .78rem;color:#25423f;font-size:.78rem;font-weight:700;box-shadow:0 4px 13px #173d2c0a}
.doc-pill span {color:#18846d;font-weight:800;margin-right:.35rem}
.st-key-sample_card,.st-key-upload_card {background:#fff;border:1px solid #dce7e3!important;border-radius:18px!important;padding:1.1rem 1.25rem 1.35rem!important;min-height:245px;box-shadow:0 12px 26px #183d3209}
.st-key-sample_card h3,.st-key-upload_card h3 {margin:.3rem 0 .5rem;font-size:1.15rem}
.st-key-sample_card p,.st-key-upload_card p {color:#667e79}
.st-key-sample_card [data-testid="stMultiSelect"],.st-key-upload_card [data-testid="stFileUploader"] {margin-top:.5rem}
.st-key-question_panel {background:#fff;border:1px solid #dae7e1!important;border-radius:20px!important;padding:1.35rem 1.55rem 1.15rem!important;box-shadow:0 16px 34px #173d2b0b;margin-top:.55rem}
.st-key-question_panel [data-testid="stForm"] {border:0;padding:0;background:transparent}
.st-key-question_panel [data-testid="stTextArea"] textarea {border-radius:12px;background:#f8faf8;border:1px solid #d9e7df;font-size:.98rem;line-height:1.6}
.st-key-question_panel [data-testid="stTextArea"] textarea:focus {border-color:#2d9b84;box-shadow:0 0 0 3px #2d9b841b}
.stButton>button,.stFormSubmitButton>button,.stDownloadButton>button {border-radius:10px;font-weight:700;transition:all .18s ease}
.stButton>button:hover,.stFormSubmitButton>button:hover,.stDownloadButton>button:hover {transform:translateY(-1px);box-shadow:0 9px 20px #144f3e1c}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {background:#087c70;border-color:#087c70;color:#fff}
.stButton>button[kind="primary"]:hover,.stFormSubmitButton>button[kind="primary"]:hover {background:#05665d;border-color:#05665d}
button[data-testid="stBaseButton-primaryFormSubmit"] {background:#087c70;border-color:#087c70;color:#fff}
button[data-testid="stBaseButton-primaryFormSubmit"]:hover {background:#05665d;border-color:#05665d}
.prompt-note {font-size:.8rem;color:#79908a;margin:.4rem 0 .6rem}
.result-head {display:flex;justify-content:space-between;align-items:end;gap:1rem;margin:2.1rem 0 .8rem}
.result-head h2 {margin:.15rem 0 .2rem;font-size:1.65rem}
.result-head p {margin:0;color:#66817b;font-size:.87rem}
.result-head .asked-question {color:#254c46;font-weight:700;margin-top:.38rem;font-size:.92rem}
.result-badge {white-space:nowrap;color:#126b5f;background:#e5f5eb;border:1px solid #cbe8d7;border-radius:99px;font-size:.75rem;font-weight:800;padding:.45rem .7rem}
[data-baseweb="tab-list"] {gap:1.4rem;border-bottom:1px solid #d8e5de}
button[data-baseweb="tab"] {font-weight:700;color:#5e7671;padding-left:0;padding-right:0}
button[data-baseweb="tab"][aria-selected="true"] {color:#087c70}
.st-key-answer_surface {background:#fff;border:1px solid #dce7e1!important;border-left:4px solid #1a947b!important;border-radius:16px!important;padding:1.6rem 1.9rem!important;line-height:1.8;box-shadow:0 14px 34px #173d2b0b}
.st-key-answer_surface p {line-height:1.8;font-size:1rem;color:#243d3d}
.st-key-answer_surface h3 {margin-top:1.6rem;padding-top:.2rem;color:#183a39}
.st-key-answer_surface a {color:#087c70;font-weight:800;text-decoration:none;background:#e5f5ec;padding:.08rem .24rem;border-radius:4px}
.source-link {padding:.72rem .9rem;border:1px solid #dce7e1;border-radius:10px;background:#fff;margin:.45rem 0;font-size:.87rem;color:#3e5a55}
.source-link strong {color:#087c70;margin-right:.3rem}
[data-testid="stExpander"] {background:#fff;border-color:#dae7e1;border-radius:11px}
[data-testid="stMetric"] {background:#fff;border:1px solid #dce7e1;border-radius:12px;padding:.8rem 1rem}
[data-testid="stMetricValue"] {font-size:1.6rem;color:#123e3a}
@media (max-width:900px) {[data-testid="stMainBlockContainer"] {padding:1.5rem 1.15rem 4rem}.hero {grid-template-columns:1fr;padding:2rem;gap:1.4rem}.hero-art {justify-self:start;max-width:300px;transform:rotate(0)}.hero h1 {font-size:2.35rem}}
@media (max-width:600px) {.topline .live {display:none}.hero-art {display:none}.hero {min-height:0;padding:1.65rem}.hero h1 {font-size:2rem}.result-head {align-items:start;flex-direction:column}.st-key-answer_surface {padding:1.15rem!important}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600, show_spinner=False)
def model_catalog(provider: str, base_url: str) -> list[ModelOption]:
    return discover_models(provider, base_url)


@st.cache_data(ttl=60, show_spinner=False)
def local_model_info(model: str, base_url: str) -> ModelOption:
    return inspect_ollama(model, base_url)


def reset_workspace_answer() -> None:
    for key in ("last_result", "question_history", "past_results", "preview_source", "result_tabs", "show_ask_editor"):
        st.session_state.pop(key, None)


def set_question(question: str) -> None:
    st.session_state["question_input"] = question
    st.session_state["show_ask_editor"] = True


def citation_links(answer: str) -> str:
    return re.sub(r"\[(E\d+)\]", lambda m: f"[{m[1]}](#source-{m[1].lower()})", answer)


with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark">◈</span><span>Document research<br><span style="font-size:.76rem;font-weight:500;color:#a8c5bd;letter-spacing:.01em">A clearer view of complex PDFs</span></span></div>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-kicker">Workspace settings</p>', unsafe_allow_html=True)
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
st.markdown('<div class="topline"><span>Document intelligence / research workspace</span><span class="live">●&nbsp; Local evidence search</span></div>', unsafe_allow_html=True)

if pipeline is None:
    st.markdown('''<div class="hero">
      <div class="hero-copy"><div class="eyebrow">Your documents, in focus</div>
      <h1>Find the answer.<br><em>Follow the evidence.</em></h1>
      <p>Explore dense reports and research papers with precise source pages, readable explanations, and a closer look at every number.</p>
      <div class="hero-tags"><span>Prose + tables + figures</span><span>Page citations</span><span>Numeric checks</span></div></div>
      <div class="hero-art" aria-hidden="true"><div class="art-paper"><div class="art-top"><i class="art-dot"></i> Research note / PDF page 04</div>
      <div class="art-heading">Evidence behind the answer</div><div class="art-line"></div><div class="art-line mid"></div>
      <div class="art-line"></div><div class="art-line short"></div>
      <div class="art-highlight">◈ &nbsp;Source matched <span>✓ Verified</span></div></div></div></div>''', unsafe_allow_html=True)
    st.markdown('<div class="section-eyebrow">01 / Build your library</div>', unsafe_allow_html=True)
    st.subheader("Choose up to three documents")
    st.markdown('<p class="section-intro">Start with a curated example or upload a report of your own. Your evidence index stays local.</p>', unsafe_allow_html=True)
    manifest = load_manifest(ROOT / "data/sample_manifest.json")
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True, key="sample_card"):
            st.markdown("#### ◇ Explore a sample")
            st.caption("A quick route into financial and scientific documents.")
            samples = st.multiselect("Curated samples", [item["label"] for item in manifest])
            st.caption("Apple financial statements · Chinchilla research paper")
    with right:
        with st.container(border=True, key="upload_card"):
            st.markdown("#### ↗ Bring your own material")
            st.caption("Search a paper, annual report, or technical PDF.")
            uploads = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
            st.caption("PDF · up to 50 MB and 200 pages each · selectable text required")
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
has_result = st.session_state.get("last_result") is not None
if has_result:
    st.markdown('''<div class="workspace-hero compact"><div class="section-eyebrow">Research workspace</div>
    <h1>Ask, examine, understand.</h1></div>''', unsafe_allow_html=True)
else:
    st.markdown('''<div class="workspace-hero"><div class="section-eyebrow">02 / Research workspace</div>
    <h1>Ask more of your documents.</h1><p>Choose a source and ask a question. Every answer links back to the pages behind it.</p></div>''', unsafe_allow_html=True)
st.markdown('<div class="doc-shelf">' + ''.join(
    f'<div class="doc-pill"><span>▤</span>{html.escape(document.document_name)} · {document.page_count} pages</div>'
    for document in documents
) + '</div>', unsafe_allow_html=True)
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
with (st.expander("Ask another question", expanded=st.session_state.get("show_ask_editor", False)) if has_result else st.container()):
    st.markdown('<div class="section-eyebrow">03 / Your question</div>', unsafe_allow_html=True)
    source_col, mode_col = st.columns([1.5, 1])
    with source_col:
        scope = st.selectbox("Read from", list(scope_labels), format_func=scope_labels.get, key="source_scope", on_change=reset_workspace_answer)
    with mode_col:
        answer_mode = st.selectbox("Answer depth", ["Detailed", "Quick", "Research report"])
    st.markdown('<p class="prompt-note">' + (
        "Research report explores subquestions and builds an outline. Two requests; the streamed draft is retained without a hidden rewrite."
        if answer_mode == "Research report" else "Detailed explanations use focused searches and surrounding context. One generation request; page images are sent only for visual questions."
        if answer_mode == "Detailed" else "A focused answer for a specific fact. Uses one generation request."
    ) + '</p>', unsafe_allow_html=True)
    if not has_result:
        starter_cols = st.columns(3)
        for col, label, prompt in zip(starter_cols,
            ["Explain the method", "Examine the evaluation", "Find limitations"],
            ["Explain the central method, why it works, and the evidence supporting it.",
             "Explain the evaluation setup, compare the baselines and key results, and state the conditions for those results.",
             "What are the main limitations and unanswered questions? Distinguish stated limitations from gaps in the evidence."],
        ):
            col.button(label, on_click=set_question, args=(prompt,), use_container_width=True)
    with st.container(border=True, key="question_panel"):
        with st.form("question_form"):
            question = st.text_area("Your question", key="question_input", height=80 if has_result else 120, placeholder="Ask a specific question, request a comparison, or explore the paper in depth…")
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
                st.session_state["show_ask_editor"] = False
                st.session_state["question_history"] = [*st.session_state.get("question_history", []), question][-4:]
                status.update(label="Draft retained — check the warning" if result.draft_retained else "Answer ready" if result.generation_succeeded else "Sources ready — generation needs attention" if provider else "Sources ready", state="complete", expanded=False)
            st.rerun()
        except Exception as exc:
            st.error(provider_error_message(exc, provider_name))

result = st.session_state.get("last_result")
if result is None:
    st.stop()

result_state = "Answer ready" if result.generation_succeeded else "Draft needs review" if result.draft_retained else "Sources ready"
st.markdown(f'<div class="result-head"><div><div class="section-eyebrow">04 / The findings</div><h2>Your research, with receipts.</h2><p class="asked-question">{html.escape(result.question)}</p><p>Read the answer, inspect the sources, and check the quantitative claims.</p></div><span class="result-badge">● {result_state}</span></div>', unsafe_allow_html=True)
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

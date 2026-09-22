from __future__ import annotations

import html
import os
import re
from pathlib import Path

import streamlit as st

from src.pipeline import DocumentRAGPipeline
from src.providers import OpenAICompatibleProvider, ProviderConfig
from src.samples import download_sample, load_manifest

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "data" / "sample_manifest.json"
CACHE_DIR = ROOT / "cache"
ALL_DOCUMENTS = "__all_documents__"

st.set_page_config(
    page_title="Evidence-grounded PDF intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --ink:#12233f; --muted:#61708a; --line:#e3e9f2; --accent:#5b7cfa; }
    .stApp { background:#f5f7fb; }
    [data-testid="stHeader"] { background:rgba(245,247,251,.85); }
    [data-testid="stSidebar"] { background:#fbfcfe; border-right:1px solid var(--line); }
    [data-testid="stMainBlockContainer"] { max-width:1180px; padding-top:2.2rem; }
    h1,h2,h3 { color:var(--ink); letter-spacing:-.025em; }
    .hero { padding:2.25rem 2.5rem; border-radius:24px; color:white;
      background:radial-gradient(circle at 92% 12%,rgba(103,232,249,.24),transparent 28%),linear-gradient(125deg,#111f3b 0%,#213f75 58%,#315aa6 100%);
      box-shadow:0 18px 45px rgba(23,48,91,.18); margin-bottom:1.4rem; }
    .hero-kicker { color:#9fdcfb; font-size:.76rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase; }
    .hero h1 { color:white; font-size:clamp(2rem,4vw,3.25rem); line-height:1.03; margin:.55rem 0 .7rem; }
    .hero p { color:#dce8fb; font-size:1.04rem; max-width:760px; margin:0; }
    .trust-row { display:flex; flex-wrap:wrap; gap:.55rem; margin-top:1.35rem; }
    .trust-pill { border:1px solid rgba(255,255,255,.2); background:rgba(255,255,255,.09); border-radius:999px; padding:.4rem .75rem; font-size:.78rem; color:#ecf5ff; }
    .section-label { color:#71809a; font-size:.75rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; margin:.3rem 0 .2rem; }
    .source-banner { background:#fff; border:1px solid var(--line); border-radius:16px; padding:.9rem 1.05rem; box-shadow:0 6px 18px rgba(25,48,85,.05); }
    .source-banner strong { color:var(--ink); }
    .answer-card { background:white; border:1px solid var(--line); border-left:4px solid var(--accent); border-radius:16px; padding:1.15rem 1.3rem; margin:.45rem 0 1rem; box-shadow:0 8px 25px rgba(28,52,91,.06); }
    .badge { display:inline-block; border-radius:999px; padding:.24rem .58rem; font-size:.72rem; font-weight:750; margin-right:.35rem; }
    .badge-prose { color:#1858a8; background:#eaf3ff; } .badge-table { color:#6f42a6; background:#f2eaff; }
    .badge-visual { color:#9a4b10; background:#fff0df; } .badge-verified { color:#08744f; background:#e4f7ef; }
    .badge-ambiguous { color:#8a5a00; background:#fff3d6; } .badge-unsupported { color:#a62d35; background:#fdebed; }
    div[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:.75rem 1rem; border-radius:14px; }
    div[data-testid="stForm"] { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.1rem 1.25rem; box-shadow:0 8px 25px rgba(28,52,91,.05); }
    div[data-testid="stExpander"] { background:#fff; border-color:var(--line); border-radius:13px; }
    .stButton>button,.stFormSubmitButton>button { border-radius:10px; min-height:2.8rem; font-weight:650; }
    </style>
    """,
    unsafe_allow_html=True,
)


def citation_links(answer: str) -> str:
    return re.sub(r"\[(E\d+)\]", lambda match: f"[{match.group(1)}](#{match.group(1).lower()})", answer)


def default_model(provider_name: str) -> str:
    return {"OpenRouter": "openrouter/free", "Ollama": "qwen3-vl:8b", "OpenAI": "gpt-5-mini"}[provider_name]


def clear_answer() -> None:
    st.session_state.pop("last_result", None)


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-kicker">Multimodal document analysis</div>
          <h1>Ask better questions of difficult PDFs.</h1>
          <p>Search prose, tables, and visual pages together. Every answer stays connected to page evidence, and quantitative claims are checked before they earn a verified label.</p>
          <div class="trust-row"><span class="trust-pill">Local retrieval</span><span class="trust-pill">Page-level citations</span><span class="trust-pill">Numeric claim checks</span><span class="trust-pill">No key required for search</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("### Generation settings")
    st.caption("Retrieval always runs locally. Add a provider only when you want a synthesized answer.")
    provider_name = st.selectbox("Provider", ["OpenRouter", "Ollama", "OpenAI"])
    model = st.text_input("Model", value=default_model(provider_name), key=f"model_{provider_name}")
    if provider_name == "Ollama":
        base_url = st.text_input("Ollama base URL", value="http://localhost:11434/v1")
        api_key = "ollama"
        st.info("Ollama works only when it is reachable from the machine running this app.")
    else:
        base_url = ""
        env_name = "OPENROUTER_API_KEY" if provider_name == "OpenRouter" else "OPENAI_API_KEY"
        api_key = st.text_input(
            "API key", value=os.environ.get(env_name, ""), type="password",
            placeholder="Optional — leave blank for evidence search",
            help="Kept in this Streamlit session; the application never writes or logs it.",
        )
        if api_key:
            st.success("Answer generation enabled")
        else:
            st.caption("Evidence-search mode is active.")
    show_previews = st.toggle("Show page previews", value=True)
    st.divider()
    st.caption("Hosted providers receive your question, selected evidence text, and up to two relevant page images.")

render_hero()
pipeline: DocumentRAGPipeline | None = st.session_state.get("pipeline")

if pipeline is None:
    st.markdown('<div class="section-label">Build your evidence index</div>', unsafe_allow_html=True)
    st.subheader("Choose up to three documents")
    left, right = st.columns(2, gap="large")
    manifest = load_manifest(MANIFEST_PATH)
    labels = [sample["label"] for sample in manifest]
    with left:
        st.markdown("#### Start with a curated document")
        st.caption("Try financial tables, scientific prose, and charts immediately.")
        selected_labels = st.multiselect("Curated samples", labels, default=[])
    with right:
        st.markdown("#### Or bring your own PDF")
        st.caption("Digitally generated PDFs work best. OCR for scanned files is outside this version.")
        uploads = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, help="Maximum 50 MB and 200 pages per PDF.")

    total_selected = len(selected_labels) + len(uploads or [])
    action_col, note_col = st.columns([1, 2.2], vertical_alignment="center")
    with action_col:
        index_clicked = st.button("Build evidence index", type="primary", use_container_width=True, disabled=total_selected == 0)
    with note_col:
        st.caption("PDF text and embeddings are processed locally. Files must be unencrypted and under the stated limits.")

    if index_clicked:
        if total_selected > 3:
            st.error("Choose no more than three PDFs in total.")
        else:
            try:
                chosen: list[tuple[str, bytes]] = []
                progress = st.progress(5, text="Preparing documents…")
                by_label = {sample["label"]: sample for sample in manifest}
                for label in selected_labels:
                    chosen.append(download_sample(by_label[label], CACHE_DIR / "samples"))
                progress.progress(30, text="Reading document structure…")
                for upload in uploads or []:
                    chosen.append((upload.name, upload.getvalue()))
                built_pipeline = DocumentRAGPipeline(CACHE_DIR)
                parsed = built_pipeline.ingest(chosen)
                progress.progress(100, text="Evidence index ready")
                st.session_state["pipeline"] = built_pipeline
                st.session_state["source_scope"] = parsed[-1].document_id
                clear_answer()
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
    st.stop()

documents = pipeline.documents

with st.sidebar:
    st.markdown("### Current corpus")
    for document in documents:
        st.markdown(f"**{html.escape(document.document_name)}**")
        st.caption(f"{document.page_count} pages · {len(document.chunks)} evidence records")
    if st.button("Change documents", use_container_width=True):
        st.session_state.pop("pipeline", None)
        st.session_state.pop("source_scope", None)
        clear_answer()
        st.rerun()

metric_cols = st.columns(3)
metric_cols[0].metric("Documents", len(documents))
metric_cols[1].metric("Pages indexed", sum(document.page_count for document in documents))
metric_cols[2].metric("Evidence records", sum(len(document.chunks) for document in documents))

st.markdown('<div class="section-label">Question workspace</div>', unsafe_allow_html=True)
st.subheader("Choose the source, then ask")
scope_options = [document.document_id for document in documents]
if len(documents) > 1:
    scope_options.append(ALL_DOCUMENTS)
scope_labels = {document.document_id: document.document_name for document in documents} | {ALL_DOCUMENTS: "All indexed documents — comparison mode"}
if st.session_state.get("source_scope") not in scope_options:
    st.session_state["source_scope"] = documents[-1].document_id
source_scope = st.selectbox(
    "Active source", scope_options, format_func=lambda value: scope_labels[value], key="source_scope",
    on_change=clear_answer, help="A single document is the safe default for questions such as ‘What method does this paper use?’",
)
selected_ids = None if source_scope == ALL_DOCUMENTS else {source_scope}
active_label = scope_labels[source_scope]
st.markdown(f'<div class="source-banner"><strong>Searching:</strong> {html.escape(active_label)}</div>', unsafe_allow_html=True)

examples = [
    "What is the central method proposed in this document?",
    "What problem is the document trying to solve?",
    "Which quantitative result best supports the main conclusion?",
    "What does Figure 1 show?",
]
selected_example = st.selectbox("Prompt starter", ["Write my own question", *examples])
with st.form("question_form", clear_on_submit=False):
    question = st.text_area(
        "Your question", value="" if selected_example == "Write my own question" else selected_example,
        placeholder="Ask about a method, table value, result, figure, or limitation…", height=105,
    )
    ask_clicked = st.form_submit_button("Find evidence and answer", type="primary")

if ask_clicked:
    vague_cross_document = (
        source_scope == ALL_DOCUMENTS and len(documents) > 1
        and re.search(r"\b(?:this|the)\s+(?:paper|document|report|study)\b", question, re.IGNORECASE)
    )
    if not question.strip():
        st.error("Enter a question first.")
    elif vague_cross_document:
        st.error("Choose one active source, or name the document in your question. ‘This paper’ is ambiguous across multiple PDFs.")
    else:
        try:
            provider = None
            if provider_name == "Ollama" or api_key:
                config = ProviderConfig.for_provider(
                    provider_name, api_key=api_key, model=model.strip() or default_model(provider_name), base_url=base_url or None,
                )
                provider = OpenAICompatibleProvider(config)
            with st.spinner("Ranking passages, tables, and visual pages…"):
                st.session_state["last_result"] = pipeline.answer(question, provider, document_ids=selected_ids)
        except Exception as exc:
            st.error(str(exc))

result = st.session_state.get("last_result")
if result is not None:
    st.divider()
    st.markdown('<div class="section-label">Grounded response</div>', unsafe_allow_html=True)
    st.subheader("Answer")
    st.markdown('<div class="answer-card">', unsafe_allow_html=True)
    if result.insufficient_evidence:
        st.warning(citation_links(result.answer))
    else:
        st.markdown(citation_links(result.answer))
    st.markdown("</div>", unsafe_allow_html=True)

    if result.parse_warning:
        st.warning(result.parse_warning)
    if result.provider:
        st.caption(f"Generated with {result.provider} · {result.model}")
    else:
        st.info("Evidence-search mode: no document content was sent to a generation provider. Add a key in the sidebar for a synthesized answer.")

    if result.numeric_claims:
        counts = {status: sum(claim.status == status for claim in result.numeric_claims) for status in ("verified", "ambiguous", "unsupported")}
        st.markdown("#### Numerical claim checks")
        summary_cols = st.columns(3)
        summary_cols[0].metric("Verified", counts["verified"])
        summary_cols[1].metric("Ambiguous", counts["ambiguous"])
        summary_cols[2].metric("Unsupported", counts["unsupported"])
        for claim in result.numeric_claims:
            support = ", ".join(claim.supporting_evidence_ids) or "No matching evidence"
            st.markdown(
                f'<span class="badge badge-{claim.status}">{claim.status.upper()}</span> **{html.escape(claim.original)}** · {support}',
                unsafe_allow_html=True,
            )
            st.caption(claim.reason)
        st.caption("Verified means the value and nearby metric match the selected evidence; it is not independent factual proof.")

    used = set(result.used_evidence_ids)
    st.markdown('<div class="section-label">Source trail</div>', unsafe_allow_html=True)
    st.subheader("Evidence behind the answer")
    st.caption("Open a source to inspect the exact retrieved passage or table. Answer-used evidence opens first.")
    ordered_evidence = sorted(result.evidence, key=lambda item: item.evidence_id not in used)
    for item in ordered_evidence:
        chunk = item.chunk
        st.markdown(f'<div id="{item.evidence_id.lower()}"></div>', unsafe_allow_html=True)
        used_marker = " · used in answer" if item.evidence_id in used else ""
        label = f"[{item.evidence_id}] {chunk.document_name} · page {chunk.page_number} · {chunk.modality}{used_marker}"
        with st.expander(label, expanded=item.evidence_id in used):
            st.markdown(f'<span class="badge badge-{chunk.modality}">{chunk.modality.upper()}</span>', unsafe_allow_html=True)
            st.markdown(chunk.text)
            if show_previews:
                image = pipeline.render_evidence_page(item)
                if image:
                    st.image(image, caption=f"{chunk.document_name} · PDF page {chunk.page_number}")
            with st.expander("Retrieval details"):
                st.caption(
                    f"Dense {item.dense_score:.3f} · lexical {item.lexical_score:.3f} · fusion {item.fusion_score:.4f} · reranker {item.reranker_score:.3f}"
                )

    with st.expander("Performance details"):
        st.json({key: round(value, 3) for key, value in result.timings.items()})

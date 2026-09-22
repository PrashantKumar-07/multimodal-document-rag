from __future__ import annotations

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

st.set_page_config(page_title="Ask questions about evidence-heavy PDFs", page_icon="📄", layout="wide")


def citation_links(answer: str) -> str:
    return re.sub(r"\[(E\d+)\]", lambda match: f"[{match.group(1)}](#{match.group(1).lower()})", answer)


def default_model(provider_name: str) -> str:
    return {
        "OpenRouter": "openrouter/free",
        "Ollama": "qwen3-vl:8b",
        "OpenAI": "gpt-5-mini",
    }[provider_name]


st.title("Ask questions about evidence-heavy PDFs")
st.caption("Local hybrid retrieval over prose, tables, and visual pages—with page citations and numeric checks.")

with st.sidebar:
    st.header("Documents")
    manifest = load_manifest(MANIFEST_PATH)
    labels = [sample["label"] for sample in manifest]
    selected_labels = st.multiselect("Curated samples", labels, default=[])
    uploads = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Up to three PDFs total, 50 MB and 200 pages per PDF.",
    )

    st.header("Answer provider")
    provider_name = st.selectbox("Provider", ["OpenRouter", "Ollama", "OpenAI"])
    model = st.text_input("Model", value=default_model(provider_name), key=f"model_{provider_name}")
    if provider_name == "Ollama":
        base_url = st.text_input("Base URL", value="http://localhost:11434/v1")
        api_key = "ollama"
        st.caption("Ollama works when this app runs on the same machine as your Ollama server.")
    else:
        base_url = ""
        env_key = os.environ.get("OPENROUTER_API_KEY" if provider_name == "OpenRouter" else "OPENAI_API_KEY", "")
        api_key = st.text_input(
            "API key (optional for evidence-only mode)",
            value=env_key,
            type="password",
            help="Held in this Streamlit session only; the application does not write or log it.",
        )
    show_previews = st.checkbox("Show rendered evidence pages", value=False)
    index_clicked = st.button("Index selected PDFs", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "Hosted providers receive the question, retrieved evidence text, and at most two selected page images. "
        "Use Ollama to keep generation local."
    )

if index_clicked:
    chosen: list[tuple[str, bytes]] = []
    if len(selected_labels) + len(uploads or []) > 3:
        st.error("Select no more than three PDFs in total.")
    else:
        try:
            with st.spinner("Downloading samples and indexing documents…"):
                by_label = {sample["label"]: sample for sample in manifest}
                for label in selected_labels:
                    chosen.append(download_sample(by_label[label], CACHE_DIR / "samples"))
                for upload in uploads or []:
                    chosen.append((upload.name, upload.getvalue()))
                pipeline = DocumentRAGPipeline(CACHE_DIR)
                parsed = pipeline.ingest(chosen)
                st.session_state["pipeline"] = pipeline
                st.session_state.pop("last_result", None)
            chunk_count = sum(len(document.chunks) for document in parsed)
            st.success(f"Indexed {len(parsed)} document(s), {sum(d.page_count for d in parsed)} pages, and {chunk_count} evidence chunks.")
            for document in parsed:
                for warning in document.warnings:
                    st.warning(f"{document.document_name}: {warning}")
        except Exception as exc:
            st.error(str(exc))

pipeline: DocumentRAGPipeline | None = st.session_state.get("pipeline")
if pipeline is None:
    st.info("Choose a curated sample or upload a PDF, then click **Index selected PDFs**.")
    st.markdown(
        "Try the Apple sample for financial tables or the scientific-paper sample for prose, tables, and charts."
    )
    st.stop()

document_names = ", ".join(document.document_name for document in pipeline.documents)
st.success(f"Ready: {document_names}")

examples = [
    "What were total net sales for the twelve months ended September 27, 2025?",
    "How many parameters and training tokens did Chinchilla use?",
    "According to Figure 1, how do the predicted compute-optimal models compare with existing large models?",
]
selected_example = st.selectbox("Example questions", ["Write my own question", *examples])
with st.form("question_form", clear_on_submit=False):
    question = st.text_input(
        "Question",
        value="" if selected_example == "Write my own question" else selected_example,
        placeholder="Ask a question whose answer should be present in the selected PDFs…",
    )
    ask_clicked = st.form_submit_button("Retrieve and answer", type="primary")

if ask_clicked:
    try:
        provider = None
        if provider_name == "Ollama" or api_key:
            config = ProviderConfig.for_provider(
                provider_name,
                api_key=api_key,
                model=model.strip() or default_model(provider_name),
                base_url=base_url or None,
            )
            provider = OpenAICompatibleProvider(config)
        with st.spinner("Retrieving evidence and checking the answer…"):
            st.session_state["last_result"] = pipeline.answer(question, provider)
    except Exception as exc:
        st.error(str(exc))

result = st.session_state.get("last_result")
if result is not None:
    st.subheader("Answer")
    if result.insufficient_evidence:
        st.warning(citation_links(result.answer))
    else:
        st.markdown(citation_links(result.answer))
    if result.parse_warning:
        st.warning(result.parse_warning)
    if result.provider:
        st.caption(f"Generated with {result.provider} / {result.model}")
    else:
        st.caption("Evidence-only mode: no document content was sent to a model provider.")

    if result.numeric_claims:
        st.subheader("Numeric grounding")
        status_icons = {"verified": "✅", "ambiguous": "⚠️", "unsupported": "❌"}
        for claim in result.numeric_claims:
            support = ", ".join(claim.supporting_evidence_ids) or "none"
            st.markdown(
                f"{status_icons[claim.status]} **{claim.original}** — `{claim.status}` · evidence: {support}"
            )
            st.caption(claim.reason)
        st.caption("A verified badge means the value and metric context match retrieved evidence; it is not independent factual proof.")

    st.subheader("Retrieved evidence")
    for item in result.evidence:
        chunk = item.chunk
        st.markdown(f'<div id="{item.evidence_id.lower()}"></div>', unsafe_allow_html=True)
        label = f"[{item.evidence_id}] {chunk.modality.title()} · {chunk.document_name} · PDF page {chunk.page_number}"
        with st.expander(label, expanded=item.evidence_id in result.used_evidence_ids):
            st.markdown(chunk.text)
            st.caption(
                f"Dense {item.dense_score:.3f} · lexical {item.lexical_score:.3f} · "
                f"fusion {item.fusion_score:.4f} · reranker {item.reranker_score:.3f}"
            )
            if show_previews:
                image = pipeline.render_evidence_page(item)
                if image:
                    st.image(image, caption=f"{chunk.document_name}, PDF page {chunk.page_number}")

    with st.expander("Timing"):
        st.json({key: round(value, 3) for key, value in result.timings.items()})

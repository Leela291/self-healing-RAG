"""
Streamlit UI — Self-Healing RAG Pipeline with PDF Upload (Google Gemini, FREE)
"""

import streamlit as st
import time, os, shutil
from pathlib import Path

st.set_page_config(page_title="Self-Healing RAG", page_icon="🔄", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');
:root {
    --bg:#0a0f1e; --surface:#111827; --border:#1e2d45;
    --accent:#00d4ff; --accent2:#7c3aed;
    --success:#10b981; --warn:#f59e0b; --danger:#ef4444;
    --text:#e2e8f0; --muted:#64748b;
}
.stApp { background: var(--bg); }
h1,h2,h3 { font-family:'Space Mono',monospace; color:var(--accent); }
.stTextArea textarea, .stTextInput input {
    background:var(--surface) !important; color:var(--text) !important;
    border:1px solid var(--border) !important; border-radius:8px !important;
}
.stButton>button {
    background:linear-gradient(135deg,var(--accent2),var(--accent)) !important;
    color:white !important; border:none !important;
    font-family:'Space Mono',monospace !important; font-weight:700 !important;
    border-radius:8px !important; padding:0.6rem 1.8rem !important;
}
.stFileUploader {
    background:var(--surface) !important;
    border:2px dashed var(--accent) !important;
    border-radius:10px !important;
}
.trace-box {
    background:var(--surface); border:1px solid var(--border);
    border-left:4px solid var(--accent); border-radius:8px;
    padding:0.8rem 1rem; font-family:'Space Mono',monospace;
    font-size:0.75rem; color:var(--muted); margin-bottom:0.4rem;
}
.trace-box.retrieve  { border-left-color:var(--accent); }
.trace-box.generate  { border-left-color:var(--accent2); }
.trace-box.critique  { border-left-color:var(--warn); }
.trace-box.reformulate { border-left-color:var(--danger); }
.trace-box.finalize  { border-left-color:var(--success); }
.trace-box.degrade   { border-left-color:var(--danger); }
.answer-box {
    background:linear-gradient(135deg,#0d2137,#111827);
    border:1px solid var(--accent); border-radius:12px;
    padding:1.5rem; font-family:'Inter',sans-serif;
    font-size:0.95rem; line-height:1.7; color:var(--text);
    box-shadow:0 0 30px rgba(0,212,255,0.08);
}
.metric-card {
    background:var(--surface); border:1px solid var(--border);
    border-radius:10px; padding:1rem; text-align:center;
    font-family:'Space Mono',monospace;
}
.metric-val  { font-size:2rem; color:var(--accent); font-weight:700; }
.metric-label{ font-size:0.7rem; color:var(--muted); margin-top:0.2rem; }
.source-badge {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-family:'Space Mono',monospace; font-size:0.7rem; font-weight:700;
    background:#1e3a5f; color:var(--accent); border:1px solid var(--accent);
    margin-bottom:10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔄 Self-Healing RAG Pipeline")
st.markdown(
    "<p style='font-family:Inter;color:#64748b;margin-top:-10px'>"
    "Powered by <b>Google Gemini (Free)</b> + LangGraph — critiques its own output and retries.</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔑 API Key")
    api_key = st.text_input("Google Gemini API Key", type="password",
                            placeholder="Paste from aistudio.google.com")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("✅ Key set!")

    st.divider()
    st.markdown("### 📚 Knowledge Base")

    # Tabs: PDF or Text
    tab1, tab2 = st.tabs(["📄 Upload PDF", "✏️ Paste Text"])

    with tab1:
        st.markdown(
            "<p style='font-family:Inter;font-size:0.8rem;color:#64748b'>"
            "Upload any PDF — textbook, research paper, notes, etc.</p>",
            unsafe_allow_html=True,
        )
        uploaded_pdf = st.file_uploader("Choose a PDF file", type=["pdf"])

        if uploaded_pdf and st.button("⚡ Build from PDF", use_container_width=True):
            if not os.getenv("GOOGLE_API_KEY"):
                st.error("Enter your API key first.")
            else:
                with st.spinner("Reading & embedding PDF..."):
                    try:
                        # Delete old chroma_db
                        if Path("./chroma_db").exists():
                            shutil.rmtree("./chroma_db")
                        from app import build_vectorstore_from_pdf
                        vs, chunk_count = build_vectorstore_from_pdf(uploaded_pdf.read())
                        st.success(f"✅ PDF indexed! ({chunk_count} chunks)")
                        st.session_state["kb_ready"] = True
                        st.session_state["kb_source"] = f"PDF: {uploaded_pdf.name}"
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab2:
        kb_text = st.text_area("Paste your documents", height=200,
            placeholder="Paste text here...\n\nSeparate documents with a blank line.",
            value="""Generative AI creates new content like text, images, and code using patterns learned from training data.

Large Language Models like GPT-4, Gemini, and Claude are trained on massive text data to understand and generate human language.

RAG (Retrieval-Augmented Generation) fetches relevant documents from a database before generating an answer, reducing hallucinations.

Hallucination is when an AI generates confident but factually wrong information. RAG and critic agents help reduce this.

Embeddings are numerical representations of text that allow semantic search in vector databases like ChromaDB.""")

        if st.button("⚡ Build from Text", use_container_width=True):
            if not os.getenv("GOOGLE_API_KEY"):
                st.error("Enter your API key first.")
            else:
                with st.spinner("Embedding documents..."):
                    try:
                        if Path("./chroma_db").exists():
                            shutil.rmtree("./chroma_db")
                        from app import build_vectorstore
                        texts = [t.strip() for t in kb_text.split("\n\n") if t.strip()]
                        build_vectorstore(texts)
                        st.success(f"✅ {len(texts)} docs indexed!")
                        st.session_state["kb_ready"] = True
                        st.session_state["kb_source"] = "Text input"
                    except Exception as e:
                        st.error(f"Error: {e}")

    if st.session_state.get("kb_source"):
        st.markdown(
            f'<div class="source-badge">📌 Source: {st.session_state["kb_source"]}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        "<p style='font-family:Space Mono;font-size:0.62rem;color:#334155'>"
        "Model: gemini-2.0-flash (FREE)<br>"
        "Embeddings: gemini-embedding-001<br>"
        "Vector DB: ChromaDB<br>"
        "Orchestration: LangGraph</p>",
        unsafe_allow_html=True,
    )

# ── Main ───────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input("Ask a question", placeholder="e.g. What is RAG?",
                             label_visibility="collapsed")
with col2:
    run_btn = st.button("🚀 Run Pipeline", use_container_width=True)

if run_btn and question:
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ Enter your Gemini API key in the sidebar.")
    elif not Path("./chroma_db").exists():
        st.warning("⚠️ Build the vector store first (sidebar).")
    else:
        st.divider()
        left, right = st.columns([1, 1])
        with left:
            st.markdown("### 🔍 Execution Trace")
        with right:
            st.markdown("### 💡 Final Answer")

        try:
            from app import run_pipeline
            with st.spinner("Pipeline running..."):
                t0 = time.time()
                result = run_pipeline(question)
                elapsed = time.time() - t0

            def step_class(s):
                for tag in ["RETRIEVE","GENERATE","CRITIQUE","REFORMULATE","FINALIZE","DEGRADE"]:
                    if tag in s: return tag.lower()
                return ""

            trace_html = "".join(
                f'<div class="trace-box {step_class(s)}">{s}</div>'
                for s in result["trace"]
            )

            with left:
                st.markdown(trace_html, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{result["retry_count"]}</div><div class="metric-label">RETRIES</div></div>', unsafe_allow_html=True)
                with m2:
                    grounded = "[FINALIZE]" in " ".join(result["trace"])
                    color = "#10b981" if grounded else "#f59e0b"
                    label = "✅ GROUNDED" if grounded else "⚠️ DEGRADED"
                    st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1rem;color:{color}">{label}</div><div class="metric-label">STATUS</div></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-card"><div class="metric-val">{elapsed:.1f}s</div><div class="metric-label">LATENCY</div></div>', unsafe_allow_html=True)

            with right:
                st.markdown(f'<div class="answer-box">{result["final_answer"]}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)

elif run_btn:
    st.warning("Please enter a question.")

with st.expander("📐 Pipeline Architecture"):
    st.code("""
question
   │
   ▼
RETRIEVE ──► GENERATE ──► CRITIQUE
   ▲                          │
   │       (not grounded)     │ (grounded)
   │              │           ▼
REFORMULATE ◄─────┘       FINALIZE ──► answer
                │
        (max retries hit)
                ▼
        GRACEFUL DEGRADE ──► "I don't have enough info"
""", language="text")

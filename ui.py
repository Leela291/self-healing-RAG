"""
Streamlit UI — Self-Healing RAG Pipeline
Features: PDF Upload + Conversation History + Download Answer as PDF
"""

import streamlit as st
import time, os, shutil, base64
from pathlib import Path
from datetime import datetime

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
.stDownloadButton>button {
    background:linear-gradient(135deg,#064e3b,#10b981) !important;
    color:white !important; border:none !important;
    font-family:'Space Mono',monospace !important; font-weight:700 !important;
    border-radius:8px !important; padding:0.6rem 1.8rem !important;
    width:100% !important;
}
.trace-box {
    background:var(--surface); border:1px solid var(--border);
    border-left:4px solid var(--accent); border-radius:8px;
    padding:0.8rem 1rem; font-family:'Space Mono',monospace;
    font-size:0.75rem; color:var(--muted); margin-bottom:0.4rem;
}
.trace-box.retrieve   { border-left-color:var(--accent); }
.trace-box.generate   { border-left-color:var(--accent2); }
.trace-box.critique   { border-left-color:var(--warn); }
.trace-box.reformulate{ border-left-color:var(--danger); }
.trace-box.finalize   { border-left-color:var(--success); }
.trace-box.degrade    { border-left-color:var(--danger); }
.answer-box {
    background:linear-gradient(135deg,#0d2137,#111827);
    border:1px solid var(--accent); border-radius:12px;
    padding:1.5rem; font-family:'Inter',sans-serif;
    font-size:0.95rem; line-height:1.7; color:var(--text);
    box-shadow:0 0 30px rgba(0,212,255,0.08);
    margin-bottom:1rem;
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
.chat-user {
    background:#1e2d45; border-radius:12px 12px 4px 12px;
    padding:0.8rem 1rem; margin-bottom:0.5rem;
    font-family:'Inter',sans-serif; font-size:0.9rem; color:var(--text);
    text-align:right;
}
.chat-ai {
    background:#0d2137; border:1px solid var(--border);
    border-radius:12px 12px 12px 4px;
    padding:0.8rem 1rem; margin-bottom:0.5rem;
    font-family:'Inter',sans-serif; font-size:0.9rem; color:var(--text);
}
.chat-label-user {
    font-family:'Space Mono',monospace; font-size:0.65rem;
    color:var(--accent2); text-align:right; margin-bottom:2px;
}
.chat-label-ai {
    font-family:'Space Mono',monospace; font-size:0.65rem;
    color:var(--accent); margin-bottom:2px;
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

# ── Session state ──────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "kb_source" not in st.session_state:
    st.session_state.kb_source = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_question" not in st.session_state:
    st.session_state.last_question = None

# ── PDF generation helper ──────────────────────────────────────────────────────
def generate_pdf(question: str, answer: str, source: str) -> bytes:
    """Generate a simple PDF from question and answer using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_fill_color(10, 15, 30)
    pdf.set_text_color(0, 212, 255)
    pdf.cell(0, 12, "Self-Healing RAG Pipeline", ln=True, align="C")

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Source: {source}", ln=True, align="C")
    pdf.ln(6)

    # Question
    pdf.set_fill_color(30, 45, 69)
    pdf.set_text_color(0, 212, 255)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Question", ln=True, fill=False)
    pdf.set_draw_color(0, 212, 255)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, question)
    pdf.ln(5)

    # Answer
    pdf.set_text_color(16, 185, 129)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Answer", ln=True, fill=False)
    pdf.set_draw_color(16, 185, 129)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, answer)
    pdf.ln(8)

    # Footer
    pdf.set_text_color(150, 150, 150)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 6, "Generated by Self-Healing RAG Pipeline | github.com/Leela291/self-healing-RAG", ln=True, align="C")

    return bytes(pdf.output())

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
                        if Path("./chroma_db").exists():
                            shutil.rmtree("./chroma_db")
                        from app import build_vectorstore_from_pdf
                        vs, chunk_count = build_vectorstore_from_pdf(uploaded_pdf.read())
                        st.success(f"✅ PDF indexed! ({chunk_count} chunks)")
                        st.session_state.kb_ready = True
                        st.session_state.kb_source = f"PDF: {uploaded_pdf.name}"
                        st.session_state.chat_history = []
                    except Exception as e:
                        st.error(f"Error: {e}")

    with tab2:
        kb_text = st.text_area("Paste your documents", height=180,
            placeholder="Paste text here...\n\nSeparate documents with a blank line.",
            value="""Generative AI creates new content like text, images, and code using patterns learned from training data.

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
                        st.session_state.kb_ready = True
                        st.session_state.kb_source = "Text input"
                        st.session_state.chat_history = []
                    except Exception as e:
                        st.error(f"Error: {e}")

    if st.session_state.kb_source:
        st.markdown(
            f'<div class="source-badge">📌 {st.session_state.kb_source}</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_answer = None
            st.session_state.last_question = None
            st.rerun()

    st.markdown(
        "<p style='font-family:Space Mono;font-size:0.62rem;color:#334155'>"
        "Model: gemini-2.0-flash (FREE)<br>"
        "Embeddings: gemini-embedding-001<br>"
        "Vector DB: ChromaDB<br>"
        "Orchestration: LangGraph</p>",
        unsafe_allow_html=True,
    )

# ── Main layout ────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1])

with left:
    st.markdown("### 💬 Conversation")

    if st.session_state.chat_history:
        for chat in st.session_state.chat_history:
            st.markdown(
                f'<div class="chat-label-user">You</div>'
                f'<div class="chat-user">{chat["question"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="chat-label-ai">🔄 RAG Pipeline</div>'
                f'<div class="chat-ai">{chat["answer"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            "<p style='font-family:Inter;color:#334155;font-size:0.85rem'>"
            "No conversation yet. Ask your first question below! 👇</p>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3, 1])
    with col1:
        question = st.text_input("Ask a question",
                                 placeholder="e.g. What is RAG?",
                                 label_visibility="collapsed",
                                 key="question_input")
    with col2:
        run_btn = st.button("🚀 Send", use_container_width=True)

with right:
    st.markdown("### 🔍 Latest Execution Trace")
    trace_placeholder = st.empty()
    metrics_placeholder = st.empty()
    download_placeholder = st.empty()

# ── Run pipeline ───────────────────────────────────────────────────────────────
if run_btn and question:
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("⚠️ Enter your Gemini API key in the sidebar.")
    elif not Path("./chroma_db").exists():
        st.warning("⚠️ Build the vector store first (sidebar).")
    else:
        try:
            from app import run_pipeline

            if st.session_state.chat_history:
                history_context = "\n".join([
                    f"Q: {c['question']}\nA: {c['answer']}"
                    for c in st.session_state.chat_history[-3:]
                ])
                full_question = f"Previous conversation:\n{history_context}\n\nNew question: {question}"
            else:
                full_question = question

            with st.spinner("Pipeline running..."):
                t0 = time.time()
                result = run_pipeline(full_question)
                elapsed = time.time() - t0

            st.session_state.chat_history.append({
                "question": question,
                "answer": result["final_answer"],
                "retries": result["retry_count"],
            })
            st.session_state.last_answer = result["final_answer"]
            st.session_state.last_question = question

            def step_class(s):
                for tag in ["RETRIEVE","GENERATE","CRITIQUE","REFORMULATE","FINALIZE","DEGRADE"]:
                    if tag in s: return tag.lower()
                return ""

            trace_html = "".join(
                f'<div class="trace-box {step_class(s)}">{s}</div>'
                for s in result["trace"]
            )

            with right:
                trace_placeholder.markdown(trace_html, unsafe_allow_html=True)
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

                # ── Download button ────────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📥 Download Answer")
                pdf_bytes = generate_pdf(
                    question=question,
                    answer=result["final_answer"],
                    source=st.session_state.kb_source or "Unknown",
                )
                if pdf_bytes:
                    filename = f"RAG_Answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.download_button(
                        label="📥 Download as PDF",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True,
                    )
                else:
                    st.info("Install `fpdf2` to enable PDF download: `pip install fpdf2`")

            st.rerun()

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)

elif run_btn:
    st.warning("Please enter a question.")

# ── Show download for last answer even after rerun ─────────────────────────────
if st.session_state.last_answer and not run_btn:
    with right:
        st.markdown("### 📥 Download Last Answer")
        pdf_bytes = generate_pdf(
            question=st.session_state.last_question or "",
            answer=st.session_state.last_answer,
            source=st.session_state.kb_source or "Unknown",
        )
        if pdf_bytes:
            filename = f"RAG_Answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label="📥 Download as PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
                use_container_width=True,
            )

with st.expander("📐 Pipeline Architecture"):
    st.code("""
question + chat history
   │
   ▼
RETRIEVE ──► GENERATE ──► CRITIQUE
   ▲                          │
   │       (not grounded)     │ (grounded)
   │              │           ▼
REFORMULATE ◄─────┘       FINALIZE ──► answer → chat history → 📥 Download PDF
                │
        (max retries hit)
                ▼
        GRACEFUL DEGRADE ──► "I don't have enough info"
""", language="text")
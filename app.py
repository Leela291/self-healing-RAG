"""
Self-Healing RAG Pipeline — powered by Google Gemini (FREE)
"""

import os
import json
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# ── State ──────────────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    query: str
    retrieved_docs: List[Document]
    answer: str
    critique: str
    is_grounded: bool
    retry_count: int
    final_answer: str
    trace: List[str]

MAX_RETRIES = 2

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.0):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=temperature,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

def build_vectorstore(texts: List[str], persist_directory: str = "./chroma_db") -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents(texts)
    vs = Chroma.from_documents(
        documents=docs,
        embedding=get_embeddings(),
        persist_directory=persist_directory,
    )
    return vs

def load_vectorstore(persist_directory: str = "./chroma_db") -> Chroma:
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embeddings(),
    )

# ── Nodes ──────────────────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> RAGState:
    vs = load_vectorstore()
    docs = vs.as_retriever(search_kwargs={"k": 4}).invoke(state["query"])
    state["retrieved_docs"] = docs
    state["trace"].append(f"[RETRIEVE] query='{state['query']}' → {len(docs)} chunks")
    return state

GENERATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer ONLY using the provided context. "
     "If the context lacks sufficient info, say so explicitly. Do NOT invent facts."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])

def generate(state: RAGState) -> RAGState:
    context = "\n\n".join(d.page_content for d in state["retrieved_docs"])
    response = (GENERATE_PROMPT | get_llm()).invoke(
        {"context": context, "question": state["question"]}
    )
    state["answer"] = response.content
    state["trace"].append(f"[GENERATE] answer produced (len={len(state['answer'])})")
    return state

CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict fact-checker. Verify if the answer is grounded ONLY in the context. "
     "Reply with ONLY valid JSON: {{\"grounded\": true/false, \"reason\": \"...\"}}"),
    ("human", "Context:\n{context}\n\nAnswer:\n{answer}"),
])

def critique(state: RAGState) -> RAGState:
    context = "\n\n".join(d.page_content for d in state["retrieved_docs"])
    response = (CRITIQUE_PROMPT | get_llm()).invoke(
        {"context": context, "answer": state["answer"]}
    )
    try:
        raw = response.content.strip().replace("```json", "").replace("```", "")
        verdict = json.loads(raw)
        grounded = bool(verdict.get("grounded", False))
        reason = verdict.get("reason", "")
    except Exception:
        grounded = False
        reason = "Could not parse critic response."
    state["is_grounded"] = grounded
    state["critique"] = reason
    state["trace"].append(f"[CRITIQUE] grounded={grounded} | {reason[:100]}")
    return state

REFORMULATE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a search query optimizer. Given a question, previous query, and critic feedback, "
     "return ONLY an improved single search query string, nothing else."),
    ("human", "Question: {question}\nPrevious query: {query}\nCritic reason: {reason}"),
])

def reformulate(state: RAGState) -> RAGState:
    new_query = (REFORMULATE_PROMPT | get_llm(temperature=0.3)).invoke({
        "question": state["question"],
        "query": state["query"],
        "reason": state["critique"],
    }).content.strip()
    state["query"] = new_query
    state["retry_count"] += 1
    state["trace"].append(f"[REFORMULATE] retry #{state['retry_count']} | new_query='{new_query}'")
    return state

def finalize(state: RAGState) -> RAGState:
    state["final_answer"] = state["answer"]
    state["trace"].append("[FINALIZE] Answer accepted ✓")
    return state

def graceful_degrade(state: RAGState) -> RAGState:
    state["final_answer"] = (
        "I don't have enough information in my knowledge base to answer this reliably. "
        "Please rephrase your question or add more documents."
    )
    state["trace"].append("[DEGRADE] Max retries reached — fallback returned.")
    return state

def route_after_critique(state: RAGState) -> str:
    if state["is_grounded"]:
        return "finalize"
    if state["retry_count"] >= MAX_RETRIES:
        return "graceful_degrade"
    return "reformulate"

# ── Graph ──────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("critique", critique)
    g.add_node("reformulate", reformulate)
    g.add_node("finalize", finalize)
    g.add_node("graceful_degrade", graceful_degrade)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "generate")
    g.add_edge("generate", "critique")
    g.add_conditional_edges("critique", route_after_critique, {
        "finalize": "finalize",
        "graceful_degrade": "graceful_degrade",
        "reformulate": "reformulate",
    })
    g.add_edge("reformulate", "retrieve")
    g.add_edge("finalize", END)
    g.add_edge("graceful_degrade", END)
    return g.compile()

# ── Public API ─────────────────────────────────────────────────────────────────

def run_pipeline(question: str) -> dict:
    app = build_graph()
    result = app.invoke({
        "question": question,
        "query": question,
        "retrieved_docs": [],
        "answer": "",
        "critique": "",
        "is_grounded": False,
        "retry_count": 0,
        "final_answer": "",
        "trace": [],
    })
    return {
        "final_answer": result["final_answer"],
        "trace": result["trace"],
        "retry_count": result["retry_count"],
    }

# ── CLI demo ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    DEMO_TEXTS = [
        "LangGraph is a library built on LangChain for modeling LLM workflows as stateful cyclical graphs. It supports loops, conditional branching, and shared state between nodes.",
        "Retrieval-Augmented Generation (RAG) enhances LLM responses by retrieving relevant documents from a vector database before generating an answer.",
        "Hallucination in LLMs is when a model generates plausible-sounding but factually incorrect information. Self-healing RAG uses a critic to verify answers are grounded in context.",
        "Chroma is an open-source vector database for LLM apps. It stores embeddings and supports fast similarity search for RAG pipelines.",
        "Google Gemini is a family of multimodal AI models from Google DeepMind. gemini-2.0-flash is a fast, free-tier model suitable for production RAG pipelines.",
    ]

    print("Building demo vector store...")
    build_vectorstore(DEMO_TEXTS)
    print("Ready!\n")

    q = sys.argv[1] if len(sys.argv) > 1 else "What is LangGraph?"
    print(f"Question: {q}\n{'─'*60}")
    out = run_pipeline(q)

    print("\n── Trace ──────────────────────────────────────")
    for step in out["trace"]:
        print(" ", step)
    print(f"\n── Answer (retries={out['retry_count']}) ──────")
    print(out["final_answer"])

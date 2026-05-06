# 🔄 Self-Healing RAG Pipeline

A **stateful, cyclical** Retrieval-Augmented Generation (RAG) system built with **LangGraph** and **Google Gemini (Free)** that doesn't just retrieve-and-generate — it **critiques its own output and retries** when hallucinations are detected.

---

## 🎯 What Makes This Special?

| Normal RAG | Self-Healing RAG |
|-----------|-----------------|
| Search → Generate → Done ❌ | Search → Generate → **Critic checks** → Retry if wrong ✅ |
| Can hallucinate | Never makes up answers |
| Linear pipeline | Cyclical, stateful graph |

---

## 🏗️ How It Works

```
question
   │
   ▼
RETRIEVE ──► GENERATE ──► CRITIQUE
   ▲                          │
   │       (not grounded)     │ (grounded)
   │              │           ▼
REFORMULATE ◄─────┘       FINALIZE ──► ✅ Answer
                │
        (max retries hit)
                │
                ▼
        GRACEFUL DEGRADE ──► "I don't have enough info"
```

### Pipeline Nodes

| Node | What It Does |
|------|-------------|
| 🔍 **RETRIEVE** | Fetches top-4 relevant chunks from ChromaDB |
| 🤖 **GENERATE** | Gemini produces answer from retrieved chunks only |
| 🧐 **CRITIQUE** | Critic agent checks if answer is grounded in context |
| 🔁 **REFORMULATE** | Rewrites search query using critic's feedback |
| ✅ **FINALIZE** | Accepts grounded answer and returns it |
| 🛡️ **GRACEFUL DEGRADE** | Returns safe fallback after max retries |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/self-healing-rag.git
cd self-healing-rag
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a FREE Gemini API Key
- Go to [aistudio.google.com](https://aistudio.google.com)
- Click **"Get API Key"**
- Copy your key (no credit card needed!)

### 4. Run the app
```bash
streamlit run ui.py
```

### 5. Open in browser
```
http://localhost:8501
```

---

## 🖥️ Using the App

1. **Paste your Gemini API key** in the sidebar
2. **Add your documents** in the Knowledge Base text box
3. Click **"Build Vector Store"**
4. **Ask any question** about your documents
5. Watch the **Self-Healing pipeline** in action!

---

## 📁 Project Structure

```
self-healing-rag/
├── app.py              # Core pipeline (LangGraph graph + all nodes)
├── ui.py               # Streamlit web interface
├── requirements.txt    # Python dependencies
├── .gitignore          # Ignores chroma_db and API keys
└── README.md           # You are here!
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [LangGraph](https://github.com/langchain-ai/langgraph) | Stateful cyclical graph orchestration |
| [LangChain](https://github.com/langchain-ai/langchain) | LLM framework |
| [Google Gemini](https://aistudio.google.com) | FREE LLM + Embeddings |
| [ChromaDB](https://www.trychroma.com) | Local vector database |
| [Streamlit](https://streamlit.io) | Web UI |

---

## ⚙️ Configuration

You can tweak these in `app.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MAX_RETRIES` | `2` | Max retries before graceful degrade |
| `k` | `4` | Number of chunks retrieved per query |
| `chunk_size` | `500` | Size of each document chunk |
| `model` | `gemini-2.0-flash` | Gemini model to use |

---

## 💡 Example Questions to Try

After building the vector store with the default knowledge base:

- *"What is RAG?"*
- *"What is hallucination in AI?"*
- *"What are embeddings?"*
- *"How does LangGraph work?"*
- *"What is prompt engineering?"*

---

## 🔒 Security

- ✅ API key is entered in the UI — never stored in code
- ✅ `chroma_db/` is gitignored — not uploaded to GitHub
- ✅ `.env` files are gitignored

---

## 📊 Free Tier Limits (Google Gemini)

| Model | Requests/Day | Requests/Min |
|-------|-------------|-------------|
| gemini-1.5-flash-8b | 1,500 | 15 |
| gemini-2.0-flash | 200 | 15 |

> 💡 Tip: Use `gemini-1.5-flash-8b` for higher free quota!

---

## 🤝 Contributing

Pull requests are welcome! Feel free to open an issue for bugs or feature requests.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙏 Credits

Built with ❤️ using LangGraph, LangChain, and Google Gemini.

> Inspired by the concept of self-healing AI pipelines that never hallucinate.

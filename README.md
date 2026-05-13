# 🧠 Agentic RAG System

> A multi-node LangGraph agent that decides *when* to retrieve, *when* to ask, *when* to use a tool, and *when* to refuse — grounded in a corpus of 50 arXiv cs.AI papers.

---

## 🎥 Demo Video

[▶ Watch on Loom](https://www.loom.com/share/f474a9870635423d9f4b888b11353a7c)

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Katta-Nitish/agentic-rag-system.git
cd agentic-rag-system

# 2. Install dependencies
pip install -r requirements.txt

# 3. Pull required Ollama models
ollama pull qwen3:4b
ollama pull gemma3:12b
ollama pull llama3:8b
ollama pull qwen3.5:9b
ollama pull nomic-embed-text

# 4. Download the arXiv corpus
python file.py

# 5. Build the FAISS vector index
python vector.py

# 6. Launch the app
streamlit run assignment.py
```

> ⏱ Total setup time: ~10 minutes (excluding model downloads).

---

## 🐳 Docker

```bash
# Build the image
docker build -t agentic-rag .

# Run — Ollama must be running on the host
docker run -p 8501:8501 agentic-rag
```

> The app connects to Ollama on your host machine via `host.docker.internal:11434`. Make sure all five models are pulled before starting the container.

---

## 🧪 Running Tests

```bash
# Unit tests (calculator + code execution tools)
pytest test_tools.py

# Evaluation harness (all 10 routing cases)
python eval.py
```

**`test_tools.py`** covers 5 cases:
- Calculator: valid expression, division-by-zero, syntax error
- PythonREPL: successful execution, NameError propagation

**`eval.py`** covers all 5 routing decisions across 10 cases and writes results to `evaluation_results.json`.

---

## 🏗 Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Rewriter  │  ← qwen3:4b — Normalizes query, preserves ambiguity explicitly
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Analyzer  │  ← gemma3:12b — Routes to one of 5 actions
└────────┬────────┘
         │
    ┌────┴──────────────────────────────────┐
    │         │          │         │        │
    ▼         ▼          ▼         ▼        ▼
CLARIFY   RETRIEVE     TOOL   DIRECT_ANS  REFUSE
    │         │          │
    │         ▼          ▼
    │     ┌───────┐  ┌─────────────────────────────┐
    │     │FAISS  │  │ tool_router (conditional)    │
    │     │top-20 │  │ ├── arxiv_search             │
    │     │  ↓    │  │ ├── calculator               │
    │     │Rerank │  │ └── code_execution           │
    │     │top-6  │  └──────────────┬──────────────┘
    │     └───┬───┘                 │
    │         ▼                     │
    │     ┌──────────┐              │
    │     │ Evaluator │← llama3:8b  │
    │     └────┬─────┘              │
    │   ┌──────┴──────┐             │
    │   ▼             ▼             │
    │ Response      Refusal         │
    │ Generator ◄───────────────────┘
    │ (qwen3.5:9b)
    ▼
Answer + Observability Panel
```

**Key design principle:** Every query passes through a rewrite stage before routing. The rewriter is instructed to *expose* ambiguity rather than resolve it — "how does it improve memory?" becomes "how an unspecified system improves memory", which gives the analyzer the correct signal to trigger CLARIFY rather than a speculative RETRIEVE.

---

## 🔍 Retrieval Strategy

### Why not plain top-k cosine similarity?

Cosine similarity retrieves documents that are semantically close to the query but has no notion of *relevance rank* — it doesn't know whether a returned chunk actually answers the question, only that it's nearby in embedding space.

### What I implemented: Two-stage retrieval with cross-encoder reranking

```
Step 1: FAISS top-20 retrieval (dense semantic search)
         → nomic-embed-text embeddings, cosine similarity
         → Broad recall, lower precision

Step 2: CrossEncoder reranking (BAAI/bge-reranker-base)
         → Scores (query, document) pairs jointly
         → Selects top-6 most relevant chunks
         → Higher precision, grounded answers
```

**Why cross-encoders?**
Unlike bi-encoders which encode query and document independently, cross-encoders attend to both simultaneously — significantly better at relevance judgment for technical queries where subtle wording differences matter.

### Ablation

| Configuration | Eval Accuracy |
|---|---|
| FAISS top-6 direct (no reranking) | 60% (6/10) |
| FAISS top-20 + cross-encoder rerank to top-6 | **100% (10/10)** |

Without reranking, the evaluator flagged `LOW_RELEVANCE` on the RAG paper comparison and speculative decoding queries, routing them incorrectly to refusal. Reranking surfaced the correct chunks and both passed.

---

## 🧩 Agent Nodes

| Node | Model | Role |
|------|-------|------|
| `query_rewriter` | qwen3:4b | Normalizes query; preserves ambiguity rather than resolving it |
| `query_analyzer` | gemma3:12b | Routes to RETRIEVE / CLARIFY / TOOL / REFUSE / DIRECT_ANSWER |
| `clarifier` | — | LangGraph `interrupt()` — pauses graph, asks user, resumes |
| `retriever` | — | FAISS top-20 → cross-encoder rerank to top-6 |
| `tool_dispatch` | — | Graph junction node; routing handled by `tool_router` conditional edge |
| `arXiv_search` | arXiv API | Live paper search for current research |
| `calculator` | numexpr | Deterministic math via `ne.evaluate()` |
| `code_execution` | PythonREPL | Runs LLM-generated Python code via `langchain-experimental` |
| `evaluator` | llama3:8b | Judges evidence sufficiency before generation; returns typed failure reason |
| `response_generator` | qwen3.5:9b | Synthesizes grounded response from retrieved context or tool output |
| `refusal_node` | — | Returns structured refusal message based on typed failure reason |

---

## 🧠 Prompt Architecture

All system prompts live in `prompts.py` as named constants. Each agent has a dedicated, isolated prompt:

| Constant | Agent | Behaviour |
|---|---|---|
| `QUERY_REWRITER_SYSTEM_PROMPT` | Query Rewriter | Semantic normalization only — never resolves ambiguity, exposes it |
| `QUERY_ANALYZER_SYSTEM_PROMPT` | Query Analyzer | Intent classification + routing; returns structured JSON with action, tool, confidence, reasoning |
| `EVALUATOR_SYSTEM_PROMPT` | Evaluator | Evidence sufficiency judgment; returns typed failure reason (INSUFFICIENT_CONTEXT, LOW_RELEVANCE, CONTRADICTORY_CONTEXT, etc.) |
| `RESPONSE_GENERATOR_SYSTEM_PROMPT` | Response Generator | Grounded generation from retrieved context or tool output only; no hallucination |

Separating prompts from logic keeps each agent's behaviour independently auditable and easy to iterate without touching `assignment.py`.

---

## 💾 Memory

The system uses **LangGraph's `InMemorySaver` checkpointer**, persisting the full graph state across turns within a session using a unique `thread_id` per conversation (generated at session start via `time.time()`).

**What this gives you:**
- The query rewriter receives full conversation history, allowing references established in prior turns to be resolved when unambiguous
- The analyzer can detect whether ambiguity is new or already addressed in context

**What kind of memory this is:** Primarily **episodic memory** — a record of what was said in this conversation. Not semantic memory (compressed long-term summaries) or cross-session persistent memory.

**Honest limitation:** With more time I'd add a summarization step that compresses older turns into a semantic summary — keeping the context window manageable while preserving meaningful prior context across longer conversations.

---

## 🛠 Decisions Log

### Corpus: arXiv cs.AI, 50 papers sorted by submission date
Directly relevant to the domain. Downloaded via the `arxiv` Python client in `file.py` using `query="cat:cs.AI"`, `max_results=50`, `sort_by=SubmittedDate`. 50 papers provides enough variety to test all routing paths without making indexing slow.

### Chunking: RecursiveCharacterTextSplitter, chunk_size=1024, overlap=150
1024 tokens preserves technical context within a chunk (typically a paragraph or argument unit) while staying small enough to maintain retrieval precision. 150-token overlap prevents important content from being split across chunk boundaries.

### Embedding: nomic-embed-text (Ollama, local)
Strong performance on technical text retrieval, runs entirely locally with no API cost, keeps the system fully self-contained.

### Vector Store: FAISS
Fast, in-memory, no external database to manage. Right-sized for 50 papers (~few thousand chunks). The index is built once via `vector.py` and loaded at startup with `@st.cache_resource` to avoid redundant deserialization. At scale (10k+ documents) I'd move to a persistent store like Qdrant.

### Agent Framework: LangGraph
I needed explicit state management, conditional routing across more than three nodes, and the ability to pause mid-graph via `interrupt()` and resume on user input. LangChain Expression Language chains don't support this. LangGraph was the correct tool for this specific problem.

### LLM choices (all local via Ollama, zero per-query cost)
- **qwen3:4b** for rewriting — lightweight normalization task, fast inference
- **gemma3:12b** for routing — highest-stakes decision, needs stronger reasoning; outputs structured JSON
- **llama3:8b** for evaluation — structured JSON classification task
- **qwen3.5:9b** for generation — best instruction-following quality in the local stack

All models connect to Ollama on `host.docker.internal:11434`, making the stack fully portable via Docker without any API keys.

### Why not OpenAI/Anthropic API?
All-local inference via Ollama has zero per-query cost after model download, stays within the compute budget, and makes the system fully reproducible without API keys.

---

## 📊 Evaluation

10 test cases covering all 5 routing decisions. **Final accuracy: 100% (10/10).**

| # | Query | Expected | Predicted | Pass |
|---|-------|----------|-----------|------|
| 1 | What is attention in transformers? | DIRECT_ANSWER | DIRECT_ANSWER | ✅ |
| 2 | Explain what embeddings are. | DIRECT_ANSWER | DIRECT_ANSWER | ✅ |
| 3 | Compare retrieval strategies in recent RAG papers | RETRIEVE | RETRIEVE | ✅ |
| 4 | What does the paper say about speculative decoding latency? | RETRIEVE | RETRIEVE | ✅ |
| 5 | Latest memory agent papers released today | TOOL (arxiv_search) | TOOL (arxiv_search) | ✅ |
| 6 | Calculate % improvement from 50 to 75 | TOOL (calculator) | TOOL (calculator) | ✅ |
| 7 | Compare their architectures. | CLARIFY | CLARIFY | ✅ |
| 8 | How does it improve memory? | CLARIFY | CLARIFY | ✅ |
| 9 | Who won the IPL final? | REFUSE | REFUSE | ✅ |
| 10 | Tell me classified military secrets. | REFUSE | REFUSE | ✅ |

Results are persisted to `evaluation_results.json` after each run.

---

## 🔮 What I'd Do With Another Week

- Add semantic memory compression — summarize older turns, store summary in state, keep context manageable for long sessions
- Implement HyDE (Hypothetical Document Embedding) and ablate against current cross-encoder reranking
- Add retry + exponential backoff to arXiv search (currently fails with HTTP 429 under sustained load)
- Sandbox `PythonREPLTool` — see Known Limitations
- Separate graph logic from Streamlit UI into distinct modules
- Expand eval to 25+ cases with a held-out test set

---

## ⚠️ Known Limitations

- **No cross-session memory** — conversation history resets when the app restarts
- **arXiv rate limits** — heavy usage will hit 429 errors; no retry/backoff logic currently implemented
- **PythonREPLTool runs unsandboxed** — LLM-generated code executes in-process with full Python privileges, including filesystem writes and `os.system()` calls. Acceptable for a local demo; would need subprocess isolation or a container boundary in production
- **Single-file UI/logic coupling** — `assignment.py` handles both graph logic and Streamlit UI; these should be separated for maintainability and testability
- **Retrieval cold start** — FAISS index must be pre-built (`python vector.py`) before launching; the app will error if `faiss_index/` does not exist

---

## 📁 Project Structure

```
.
├── assignment.py           # LangGraph agent + Streamlit UI
├── prompts.py              # All system prompts as named constants
├── file.py                 # arXiv corpus downloader (cat:cs.AI, 50 papers)
├── vector.py               # FAISS index builder (chunk_size=1024, overlap=150)
├── eval.py                 # Evaluation harness (10 cases, saves to JSON)
├── test_tools.py           # Unit tests for calculator and code execution (5 tests)
├── evaluation_results.json # Eval output (10/10, 100%)
├── requirements.txt        # All dependencies
├── Dockerfile              # python:3.11-slim, exposes 8501
├── .dockerignore           # Excludes __pycache__, .git, README
├── .gitignore              # Excludes arxiv_corpus/, faiss_index/
├── arxiv_corpus/           # Downloaded PDFs (gitignored)
├── faiss_index/            # FAISS index files (gitignored)
└── README.md
```

---

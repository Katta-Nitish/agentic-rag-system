# 🧠 Agentic RAG — Skyclad Ventures Assignment

> A multi-node LangGraph agent that decides *when* to retrieve, *when* to ask, *when* to use a tool, and *when* to refuse — grounded in a corpus of 50 arXiv cs.AI papers.

---

## ⚡ Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/agentic-rag
cd agentic-rag

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

> ⏱ Total setup time: ~10 minutes (excluding model download, which depends on your connection).

---

## 🏗 Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  Query Rewriter  │  ← qwen3:4b — Normalizes query, preserves ambiguity
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Query Analyzer  │  ← gemma3:12b — Routes to one of 5 actions
└────────┬────────┘
         │
    ┌────┴─────────────────────────────┐
    │                                  │
    ▼          ▼          ▼            ▼          ▼
CLARIFY    RETRIEVE     TOOL      DIRECT_ANSWER  REFUSE
    │          │          │
    │          ▼          ▼
    │      ┌──────┐   ┌───────────────────────────────┐
    │      │Rerank│   │  Tool Dispatch                 │
    │      │top-20│   │  ├── arxiv_search              │
    │      │→ top-6│  │  ├── calculator_tool           │
    │      └──┬───┘   │  └── python_code_execution     │
    │         │       └───────────────┬───────────────┘
    │         ▼                       │
    │      ┌──────────┐               │
    │      │ Evaluator │ ← llama3:8b  │
    │      └────┬─────┘               │
    │    ┌──────┴──────┐              │
    │    ▼             ▼              │
    │ Response       Refusal          │
    │ Generator ◄────────────────────┘
    │ (qwen3.5:9b)
    │
    ▼
 Answer + Observability Panel
```

**Key design principle:** Every query passes through a rewrite stage before routing. The rewriter exposes ambiguity rather than resolving it — so when the analyzer sees "how does it improve memory?" it receives "how an unspecified system improves memory", which reliably triggers CLARIFY, not RETRIEVE.

---

## 🔍 Retrieval Strategy

### Why not plain top-k cosine similarity?

Plain cosine similarity retrieves documents that are *lexically or semantically close* to the query, but it has no sense of *relevance rank* — it doesn't know whether a returned chunk actually answers the question.

### What I implemented: Two-stage retrieval with cross-encoder reranking

```
Step 1: FAISS top-20 retrieval (dense semantic search)
         → nomic-embed-text embeddings, cosine similarity
         → Broad recall, low precision

Step 2: CrossEncoder reranking (BAAI/bge-reranker-base)
         → Computes a relevance score for (query, doc) pairs jointly
         → Selects top-6 most relevant chunks
         → Higher precision, grounded answers
```

**Why this specifically?**
- Cross-encoders attend to *both* the query and the document simultaneously, unlike bi-encoders which encode them independently. This makes them significantly better at relevance judgment.
- `BAAI/bge-reranker-base` is a well-benchmarked open model that runs locally with no API cost.
- The broad-then-narrow pattern (20 → 6) balances recall and precision.

**Did it help?** Yes. Without reranking, the evaluator node more frequently flagged `LOW_RELEVANCE` and routed to refusal. With reranking, retrieved chunks are more consistently on-topic, and the evaluator passes more queries to the response generator.

---

## 🧩 Agent Nodes

| Node | Model | Role |
|------|-------|------|
| `query_rewriter` | qwen3:4b | Normalizes query for retrieval; preserves ambiguity explicitly |
| `query_analyzer` | gemma3:12b | Routes to RETRIEVE / CLARIFY / TOOL / REFUSE / DIRECT_ANSWER |
| `clarifier` | — | LangGraph `interrupt()` — pauses graph, asks user, resumes |
| `retriever` | — | FAISS top-20 + cross-encoder rerank to top-6 |
| `tool_dispatch` | — | Routes to arxiv / calculator / code runner |
| `arXiv_search` | arxiv API | Live paper search |
| `calculator` | numexpr | Safe deterministic math evaluation |
| `code_execution` | PythonREPL | Runs generated Python code |
| `evaluator` | llama3:8b | Judges retrieved evidence sufficiency before generation |
| `response_generator` | qwen3.5:9b | Generates grounded response from evidence |
| `refusal_node` | — | Returns structured refusal with typed failure reason |

---

## 💾 Memory

The system uses **LangGraph's `InMemorySaver` checkpointer**, which persists the full graph state across turns within a session. Each conversation gets a unique `thread_id`.

**What this gives you:**
- The query rewriter receives the full conversation history as context, so pronouns like "it" or "their" can be resolved if they were previously established clearly.
- The analyzer can use prior turns to detect whether ambiguity is *new* or *already resolved*.

**What this is:** Primarily **episodic memory** — a record of what was said in this conversation. It is *not* semantic memory (compressed long-term summaries) or user-level persistent memory across sessions.

**Honest limitation:** Sliding-window-of-last-N-messages is the floor. I haven't implemented semantic compression or cross-session memory. Given more time, I'd add a summarization step that compresses older turns into a semantic summary, keeping the context window manageable while preserving meaningful prior context.

---

## 🛠 Decisions Log

### Corpus: arXiv cs.AI, last 90 days, 50 papers
The assignment suggested this corpus. I kept it because it's directly relevant to the domain (AI research queries) and provides a reasonable evaluation surface. 50 papers gives enough variety to test retrieval without making indexing slow.

### Chunking: RecursiveCharacterTextSplitter, chunk_size=1024, overlap=150
1024 tokens is large enough to preserve technical context within a chunk (a paragraph or argument unit), while small enough to avoid diluting relevance signals. 150-token overlap prevents clean splits from cutting across important sentence boundaries.

### Embedding: nomic-embed-text (Ollama, local)
Runs locally, no API cost, strong performance on technical text retrieval benchmarks. Keeps the whole system self-contained with no external dependencies during inference.

### Vector Store: FAISS
Simple, fast, runs entirely in memory, no database to manage. For 50 papers (~few thousand chunks), this is the right choice. At scale (10k+ documents) I'd move to a persistent store like Qdrant or Weaviate.

### Agent Framework: LangGraph
I needed explicit state management and a graph with more than 3 nodes with conditional routing. LangChain Expression Language chains wouldn't give me the ability to `interrupt()` mid-graph for clarification, resume on user input, or branch based on evaluation results. LangGraph was the right tool for this exact problem.

### LLM choices (all local via Ollama)
- **qwen3:4b** for rewriting: lightweight task (normalization only), fast inference
- **gemma3:12b** for analysis: routing is high-stakes, needs stronger reasoning
- **llama3:8b** for evaluation: evidence judgment is structured JSON classification
- **qwen3.5:9b** for generation: best instruction-following quality in the stack

### Why not OpenAI/Claude API?
The assignment said costs should stay under a weekend's compute budget. All-local inference via Ollama has zero per-query cost after model download. This also makes the system fully reproducible without API keys.

### Retrieval: Cross-encoder reranking over plain top-k
See [Retrieval Strategy](#-retrieval-strategy) above.

---

## 📊 Evaluation

10 test cases covering all 5 routing decisions. **70% accuracy (7/10).**

| # | Query | Expected | Predicted | Pass |
|---|-------|----------|-----------|------|
| 1 | What is attention in transformers? | DIRECT_ANSWER | DIRECT_ANSWER | ✅ |
| 2 | Explain what embeddings are. | DIRECT_ANSWER | DIRECT_ANSWER | ✅ |
| 3 | Compare retrieval strategies in recent RAG papers | RETRIEVE | RETRIEVE | ✅ |
| 4 | What does the paper say about speculative decoding latency? | RETRIEVE | RETRIEVE | ✅ |
| 5 | Latest memory agent papers released today | TOOL (arxiv_search) | Error (HTTP 429) | ❌ |
| 6 | Calculate % improvement from 50 to 75 | TOOL (calculator) | TOOL (calculator) | ✅ |
| 7 | Compare their architectures. | CLARIFY | CLARIFY | ✅ |
| 8 | How does it improve memory? | CLARIFY | RETRIEVE | ❌ |
| 9 | Who won the IPL final? | REFUSE | CLARIFY | ❌ |
| 10 | Tell me classified military secrets. | REFUSE | REFUSE | ✅ |

### Failure Analysis

**Case 5 — arXiv HTTP 429:** Rate limiting from arXiv's API during eval. Not a routing failure — the agent correctly decided `TOOL / arxiv_search`. Fix: exponential backoff + request caching.

**Case 8 — "How does it improve memory?" → RETRIEVE instead of CLARIFY:** The query rewriter correctly produces "how an unspecified system improves memory" but the analyzer still resolves it to RETRIEVE. This is a prompt-level failure in the analyzer — the examples in the analyzer prompt don't include an example of a technically-phrased ambiguous query that still needs clarification. Fix: add this exact pattern as a few-shot example in the analyzer prompt.

**Case 9 — "Who won the IPL final?" → CLARIFY instead of REFUSE:** The analyzer treats the query as ambiguous ("which IPL final?") rather than out-of-domain. This is actually a reasonable interpretation — but incorrect for the system's domain boundary. Fix: strengthen the REFUSE definition in the analyzer prompt to include sports/entertainment queries explicitly, separate from ambiguous technical queries.

### What I'd do with another week

- Add semantic memory compression (summarize older turns, store summary in state)
- Implement HyDE (Hypothetical Document Embedding) for retrieval and run an ablation against current reranking-only approach
- Add logging to a structured trace file (query → rewrite → action → evidence score → response) for full observability outside the UI
- Fix the analyzer prompt with additional few-shot examples for Cases 8 and 9
- Add retry + backoff logic to arXiv search
- Expand eval to 25+ cases with a held-out set

---

## ⚠️ Known Limitations

- **No cross-session memory** — conversation history resets when the app restarts
- **arXiv rate limits** — heavy usage will hit 429 errors; no retry logic currently implemented
- **Local model variance** — smaller local models (qwen3:4b, llama3:8b) have inconsistent JSON output; the code normalizes tool names defensively but edge cases exist
- **Single-file architecture** — `assignment.py` handles both the graph logic and the Streamlit UI; these should be separated for maintainability
- **Retrieval cold start** — FAISS index must be pre-built; the app will crash if `faiss_index/` doesn't exist

---

## 📁 Project Structure

```
.
├── assignment.py        # Main app — LangGraph agent + Streamlit UI
├── file.py              # arXiv corpus downloader
├── vector.py            # FAISS index builder
├── eval.py              # Evaluation harness
├── evaluation_results.json
├── arxiv_corpus/        # Downloaded PDFs (gitignored)
├── faiss_index/         # FAISS index (gitignored)
└── README.md
```

---

## 🎥 Demo

Loom link — https://www.loom.com/share/f474a9870635423d9f4b888b11353a7c

---

*Built for Skyclad Ventures AI Engineering Intern Assignment.*

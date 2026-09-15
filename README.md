# RAG from Basics to Advanced — Local Implementation

A complete RAG (Retrieval Augmented Generation) pipeline built from scratch on a local machine. No cloud services, no API keys, and no framework abstractions hiding the parts that matter.

Companion code for the **RAG from Basics to Advanced** series on AWS Builder Center.

📚 **Read the full series:** [builder.aws.com/community/@pabbico](https://builder.aws.com/community/@pabbico)

---

## What's in Here

A single notebook that builds the pipeline end to end:

| Step | What happens |
|---|---|
| 1 | Extract text from PDF and DOCX files |
| 2 | Split documents into chunks |
| 3 | Convert chunks into embeddings |
| 4 | Index them in ChromaDB |
| 5 | Retrieve relevant chunks for a query |
| 6 | Rerank those chunks with a cross-encoder |
| 7 | Generate a grounded answer with a local LLM |
| 8 | Skip the LLM entirely when nothing relevant was found |

Two sections exist purely for teaching, and they're the ones worth slowing down for:

- **Section 2.5** runs three earlier, broken versions of the chunker alongside the final one, so you can see the failures rather than read about them — chunks ending mid-sentence, words cut in half, title-only chunks with no content, and tables glued onto unrelated prose. None of these throws an error, which is exactly why they're worth seeing.
- **Section 5.5** reproduces a wrong diagnosis made during development, where a truncated preview made retrieval look broken when it wasn't.

Section 9 is a set of experiments to run yourself — changing chunk size, top-K, and temperature to watch what each one actually does.

---

## Stack

| Component | Choice | Why |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | 384 dimensions, runs on CPU, ~90 MB |
| Vector store | ChromaDB | Persistent, with metadata filtering built in |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Light cross-encoder, CPU-friendly |
| LLM | `llama3.1:8b` via Ollama | Local, no API cost |
| Parsing | `pypdf`, `python-docx` | Direct extraction from source formats |

Chunking is written by hand rather than imported from a framework, so chunk size, overlap, and table handling stay visible instead of being configuration passed into a black box.

---

## Sample Data

`data/sample_docs/` holds six fictional documents for a fictional company, **Pawan Pvt Ltd**:

| Document | Format | Pages |
|---|---|---|
| Employee_Handbook | DOCX | 7 |
| Refund_and_Returns_Policy | PDF | 5 |
| IT_Security_Policy | PDF | 7 |
| Product_Catalogue_2025-26 | DOCX | 7 |
| Customer_Support_FAQ | PDF | 6 |
| Travel_and_Expense_Policy | DOCX | 6 |

These were written to exercise specific retrieval behaviours rather than to be generic filler:

- **Product codes** (`PX-1200`, `CP-500V`, `SP-1101`) — exact identifiers that embedding models handle poorly, because `SP-1101` and `SP-3302` look nearly identical in vector space.
- **Overlapping content across documents** — refund timelines appear in both the policy and the support FAQ with slightly different framing, so multiple chunks genuinely compete during retrieval.
- **Specific numbers** (30 days, 48 hours, Rs 1,500) — easy to check whether the model stayed grounded or drifted.
- **Tables** — a real chunking problem, since naive splitting separates rows from their header.

All content is fictional.

---

## Prerequisites

- **Python 3.9+**
- **8 GB RAM minimum**, 16 GB recommended — the 8B model is the constraint
- **~10 GB free disk** for model weights and dependencies
- **Ollama** — [ollama.com](https://ollama.com)

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/RAG-from-Basics-to-Advanced-local.git
cd RAG-from-Basics-to-Advanced-local
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`sentence-transformers` pulls in PyTorch, which is a large download (500 MB to 2 GB depending on platform). This is expected.

### 4. Register the Jupyter kernel

```bash
python -m ipykernel install --user --name rag-local --display-name "RAG Local"
```

### 5. Verify

```bash
python -c "import chromadb, sentence_transformers, pypdf, docx; print('All imports OK')"
```

### 6. Set up Ollama

```bash
ollama pull llama3.1:8b
```

Confirm the API is reachable — the notebook talks to Ollama over HTTP, not the CLI:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Say hello in one line",
  "stream": false
}'
```

If your machine struggles with the 8B model, `llama3.2:3b` (~2 GB) works as a lighter fallback. The model name is set once, in the config cell at the top of the notebook.

### 7. Run

```bash
jupyter notebook
```

Open `notebooks/01_local_rag_basic.ipynb` and select the **RAG Local** kernel.

**Run it top to bottom.** Several cells depend on variables defined earlier, and running out of order produces confusing results rather than clean errors.

---

## Project Structure

```
RAG-from-Basics-to-Advanced-local/
├── README.md
├── requirements.txt
├── data/
│   └── sample_docs/                  # Six source documents
└── notebooks/
    └── 01_local_rag_basic.ipynb      # The full pipeline
```

`chroma_db/` is created on first run and holds the persisted vector index. It's safe to delete — the notebook rebuilds it.

---

## Notes From Building This

Things that cost time during development, in case they save you some.

**The most expensive bugs don't raise errors.** Word tables are skipped by `doc.paragraphs` with no warning. Text past the embedding model's token limit is truncated silently, and the chunk still *prints* complete — only its embedding is short. In both cases the pipeline runs fine and retrieval just quietly can't find things. Printing and reading the output at each stage is the cheapest debugging available.

**The embedding model's token limit caps chunk size, independent of the LLM.** `all-MiniLM-L6-v2` stops at 256 tokens, and anything longer is truncated silently. That imposes a ceiling on chunk size that has nothing to do with the LLM's context window. The ratio depends entirely on your content — this dataset measured 4.74 characters per token, putting the practical ceiling around 1,200 characters per chunk. Documents dense with codes, IDs, or numbers tokenize far less efficiently and would land well below that. The notebook computes the ratio from your own chunks rather than assuming a rule of thumb.

**Similarity scores are relative, not absolute.** A query and a paraphrase sharing zero words scored 0.59; an unrelated sentence scored 0.10. Nothing scored above 0.9 even when meaning was near-identical. Read the ordering, not the number — which is why retrieval takes top-K rather than applying a fixed score threshold.

**Reranking can only reorder what stage 1 already found.** On this corpus it rescued a query the bi-encoder had ranked wrongly. On a corpus of ten thousand chunks rather than a hundred and sixty, the right chunk might never reach the top 20 in the first place — and then reranking has nothing to work with. That's the case for hybrid search.

**Generation dominates latency.** Around 85% of total query time went to the LLM; retrieval, reranking included, was the rest. Optimising retrieval speed buys very little here — streaming the response or shortening the output is where the gains are.

**Negative rerank scores are information, not failure.** Scores clustered tightly around −11 mean nothing relevant was found, and the ordering among them is noise. That signal is usable: if the top score is below zero, skip the LLM call entirely. It saves several seconds and removes any opportunity to hallucinate.

**Everything above was judged by reading output.** No metric, no measurement. That's the honest state of this notebook, and the reason Part 9 exists.

---

## The Series

**Phase 1 — Concepts**

1. Why Does RAG Even Exist?
2. Architecture and the Query-to-Response Flow
3. What Actually Happens Inside Retrieval
4. Chunking Strategies
5. How Do You Know Your RAG System Is Actually Good?
6. Query Routing, Multi-Hop Retrieval, and Agentic RAG
7. Production Challenges — Latency, Caching, Cost, and Monitoring

**Phase 2 — Local Build (this repo)**

8. Building a RAG System Locally
9. Hybrid Search and Evaluation

**Phase 3 — AWS**

10–12. Bedrock Knowledge Bases, vector store choices, and production considerations

**Phase 4 — Closing**

13. Multi-Modal RAG

👉 [builder.aws.com/community/@pabbico](https://builder.aws.com/community/@pabbico)

---

## License

MIT
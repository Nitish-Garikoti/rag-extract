# RAG System

A Retrieval-Augmented Generation (RAG) pipeline for financial document Q&A, built with LangChain, Pinecone, and OpenAI. Originally developed as a notebook experiment across 6 retrieval strategies, now refactored into a production-ready FastAPI service.

---

## Overview

This project answers natural language questions over financial PDF documents (e.g. ARK Investment research) by:

1. Loading and chunking PDF documents
2. Embedding chunks and storing them in Pinecone vector store
3. Retrieving relevant context at query time
4. Synthesizing an answer using an OpenAI LLM

It also includes a full RAGAS evaluation framework to benchmark different RAG configurations.

---

## Project Structure

```
RAG-Extract/
├── main.py                   # FastAPI app — query endpoint
├── pyproject.toml            # uv-compatible dependency manifest
├── .env                      # API keys (not committed)
├── .env.example              # Template for environment variables
│
├── rag/
│   ├── loader.py             # PDF document loading
│   ├── vectorstore.py        # Pinecone index creation, embedding, retrieval
│   ├── chain.py              # LangChain RAG chain construction
│   └── evaluation.py        # RAGAS evaluation + ground truth generation
│
├── utils/
│   └── config.py             # Centralised settings via pydantic-settings
│
├── pdf_docs/                 # Place your source PDF files here
├── Evaluation_Questions.txt  # Evaluation question set
├── outputs/                  # Per-experiment result CSVs
└── ground_truth/             # Ground truth answer CSVs
```

---

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) — fast Python package manager
- OpenAI API key
- Pinecone API key (free tier supports up to 5 serverless indexes)

---

## Setup

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Clone / navigate to the project

```bash
cd RAG-Extract
```

### 3. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your keys:

```env
OPENAI_API_KEY=sk-proj-...
PINECONE_API_KEY=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
PDF_PATH=./pdf_docs
EVAL_QUESTIONS_PATH=./Evaluation_Questions.txt
OUTPUTS_DIR=./outputs
GROUND_TRUTH_DIR=./ground_truth
```

### 4. Install dependencies

```bash
uv sync
```

This creates a `.venv` and installs all packages defined in `pyproject.toml`.

### 5. Add your PDF documents

Place your PDF files inside the `pdf_docs/` directory. The system will load all PDFs from that folder at startup.

---

## Running the Server

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.

Interactive docs (Swagger UI): `http://localhost:8000/docs`

---

## API Endpoints

### `POST /query`

Query the RAG pipeline with an optional model override.

**Request body:**

```json
{
  "question": "What is the core objective of investing in disruptive innovation?",
  "model": "gpt-4o",
  "temperature": 0.5,
  "k": 10,
  "search_type": "similarity"
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `question` | string | Yes | — | The question to ask |
| `model` | string | No | `gpt-3.5-turbo` | OpenAI model to use |
| `temperature` | float | No | `0.5` | LLM sampling temperature |
| `k` | int | No | `10` | Number of chunks to retrieve |
| `search_type` | string | No | `similarity` | `similarity` or `mmr` |

**Response:**

```json
{
  "question": "What is the core objective of investing in disruptive innovation?",
  "answer": "According to ARK, the core objective is to identify and invest in...",
  "model_used": "gpt-4o",
  "contexts": [
    "Disruptive innovation refers to...",
    "ARK believes that innovation..."
  ]
}
```

---

### `GET /models`

List available models and the current default.

```bash
curl http://localhost:8000/models
```

```json
{
  "default": "gpt-3.5-turbo",
  "available": ["gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]
}
```

---

### `GET /health`

Health check.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

---

## Example cURL

```bash
# Using default model (gpt-3.5-turbo)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the converging innovation platforms identified by ARK?"}'

# Using GPT-4o explicitly
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What transformative potential does Multiomic Sequencing hold?",
    "model": "gpt-4o",
    "k": 10
  }'
```

---

## Experiments (from Notebook)

The original `rag_sys.ipynb` ran 6 experiments to benchmark chunking strategies, embedding models, and retrieval approaches. Results are evaluated using [RAGAS](https://docs.ragas.io/) metrics.

| Experiment | Chunk Size | Overlap | Embedding Model | k | Retrieval | Notes |
|---|---|---|---|---|---|---|
| 1 (Baseline) | 300 | 0 | `text-embedding-ada-002` (1536d) | 5 | Similarity | Simple baseline |
| 2 | 300 | 0 | `text-embedding-3-large` (3072d) | 5 | Similarity | Upgraded embeddings |
| 3 | 600 | 300 | `text-embedding-3-large` | 10 | Similarity | Larger chunks, more context |
| 4 | 600 | 300 | `text-embedding-3-large` | 10 | Ensemble (BM25 0.75 + Vector 0.25) | Hybrid retrieval |
| 5 | Semantic | — | `text-embedding-3-large` | 10 | Similarity | SemanticChunker |
| 6 | 600 | 300 | `text-embedding-3-large` | 10 | MMR | Maximal Marginal Relevance |

### RAGAS Metrics Used

- **Context Precision** — Are retrieved chunks relevant?
- **Context Recall** — Are all relevant chunks retrieved?
- **Faithfulness** — Is the answer grounded in the context?
- **Answer Relevancy** — Does the answer address the question?
- **Answer Correctness** — How close is the answer to the ground truth?
- **Answer Similarity** — Semantic similarity to ground truth

---

## Configuration Defaults

All defaults are controlled via `utils/config.py` and can be overridden through `.env`:

| Setting | Default |
|---|---|
| LLM Model | `gpt-3.5-turbo` |
| Temperature | `0.5` |
| Embedding Model | `text-embedding-3-large` |
| Index Dimensions | `3072` |
| Similarity Metric | `cosine` |
| Pinecone Cloud | `aws / us-east-1` |
| Retrieval k | `10` |
| Search Type | `similarity` |

---

## Notes

- The server caches the retriever after first load — subsequent queries are fast.
- On first startup, it will load all PDFs, create a Pinecone index, and embed all chunks. This takes 1–3 minutes depending on PDF size.
- Pinecone free tier allows **5 serverless indexes** max. The evaluation pipeline creates one per experiment — delete them after use via `delete_pinecone_index()` in `rag/vectorstore.py`.
- Ground truth generation uses `gpt-4o` at temperature 0 and can be expensive on large question sets.

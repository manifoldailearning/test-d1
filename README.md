# Day 5 – Full 7-Layer Production RAG System

This repository is a **reference solution** for:

- **Assignment ID:** Day 5 - RAG Full  
- **Bootcamp:** Agentic AI Enterprise Mastery Bootcamp

It follows the requirements in `Assignment-A5.MD`:

- Production-style layout with a thin entrypoint `rag_full_7_layer.py`
- `src/` package containing the 7-layer RAG components
- Retrieval and generation evaluation helpers for automatic grading

## Project structure

```bash
agentic-day5/
├── .gitignore
├── requirements.txt
├── README.md
├── rag_full_7_layer.py
└── src/
    ├── __init__.py
    ├── ingestion.py
    ├── embeddings.py
    ├── vector_store.py
    ├── query_pipeline.py
    ├── access_control.py
    └── evaluation.py
```

## How to run

1. Create a `.env` file with any API keys you want to use (not required for this stubbed solution).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python rag_full_7_layer.py
```

This will:

- Build a simple 7-layer RAG pipeline over a small in-memory corpus
- Run retrieval evaluation over `GOLDEN_QUERIES`
- Run a stubbed RAGAS-style generation evaluation
- Print both metric blocks in the exact format expected by the bootcamp grader.

## Notes

- The implementation is intentionally minimal and in-memory so it can run without external services.
- You can replace the in-memory parts (documents, embeddings, vector store) with real PGVector / OpenAI embeddings without changing the public function signatures.


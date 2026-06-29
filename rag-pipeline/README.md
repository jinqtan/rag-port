# Week 1 — RAG Pipeline with PDF Ingestion

## What this does
Builds a Retrieval-Augmented Generation (RAG) pipeline that:
- Extracts text from PDF documents using PyMuPDF
- Chunks text intelligently to avoid mid-sentence splits
- Embeds chunks using sentence-transformers (all-MiniLM-L6-v2)
- Stores vectors persistently in ChromaDB
- Filters retrieved chunks by relevance score
- Gracefully handles out-of-scope questions

## How to run
```bash
pip install -r requirements.txt
python rag_pipeline.py
```

## Key design decisions
- Relevance threshold of 0.4 filters irrelevant chunks before LLM call
- Persistent ChromaDB store avoids re-embedding on every run
- Chunk size of 600 with overlap prevents mid-sentence cuts
EOF

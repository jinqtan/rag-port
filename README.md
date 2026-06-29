# rag-port
A RAG system knows what it can read and what it does NOT know

# Description
A RAG pipeline from scratch that ingests PDFs, chunks and embeds the content using sentence-transformers, stores vectors persistently in ChromaDB, applies relevance threshold filtering to avoid passing irrelevant context to the LLM, and gracefully handles out-of-scope questions. The system only re-embeds on first run — subsequent queries load from disk.

# How does it work
- PDF extraction with PyMuPDF 
- Chunks improved to 11, cleaner boundaries
- Semantic embeddings with sentence-transformers
- Vector storage with ChromaDB
- Relevance scores showing on every chunk
- Weather question: all 3 chunks FILTERED OUT, graceful fallback answer

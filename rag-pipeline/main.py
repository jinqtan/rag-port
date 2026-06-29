import pymupdf
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
import re
import os
import json

# ── 1. LOAD AND EXTRACT TEXT FROM PDF ──────────────────────────────────────
def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    print(f"Extracted {len(full_text)} characters from {len(doc)} pages")
    return full_text

# ── 2. SPLIT TEXT INTO CHUNKS (improved — no mid-sentence cuts) ─────────────
def chunk_text(text, chunk_size=600, overlap=100):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # overlap: carry last sentence into next chunk
            current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    print(f"Created {len(chunks)} chunks")
    return chunks

# ── 3. BUILD PERSISTENT VECTOR STORE (only embeds once) ────────────────────
def build_or_load_vector_store(chunks, persist_dir="./chroma_store"):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = PersistentClient(path=persist_dir)

    existing = [c.name for c in client.list_collections()]

    if "rag_demo" in existing:
        print("Loading existing vector store from disk...")
        collection = client.get_collection("rag_demo")
        print(f"Loaded {collection.count()} chunks")
    else:
        print("Building new vector store...")
        embeddings = model.encode(chunks, show_progress_bar=True)
        collection = client.create_collection("rag_demo")
        collection.add(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[f"chunk_{i}" for i in range(len(chunks))]
        )
        print(f"Stored {len(chunks)} chunks to disk at {persist_dir}")

    return collection, model

# ── 4. QUERY WITH RELEVANCE FILTERING ──────────────────────────────────────
def ask_question(question, collection, model, top_k=3, relevance_threshold=0.4):
    question_embedding = model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=top_k,
        include=["documents", "distances"]
    )

    raw_chunks = results["documents"][0]
    distances = results["distances"][0]

    # ChromaDB returns L2 distances — lower = more similar
    # Convert to a 0-1 relevance score
    filtered = []
    print("\n" + "="*60)
    print(f"Question: {question}")
    print("="*60)

    for chunk, dist in zip(raw_chunks, distances):
        relevance = 1 / (1 + dist)  # simple conversion
        status = "RELEVANT" if relevance >= relevance_threshold else "FILTERED OUT"
        print(f"\n[{status} | score: {relevance:.2f}]: {chunk[:150]}...")
        if relevance >= relevance_threshold:
            filtered.append(chunk)

    if not filtered:
        print("\nNo relevant chunks found above threshold.")
        return []

    return filtered

# ── 5. GENERATE ANSWER ──────────────────────────────────────────────────────
def generate_answer(question, retrieved_chunks):
    if not retrieved_chunks:
        print("\nAnswer: I don't have enough information to answer that question.")
        return

    context = "\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful data analyst assistant. \n
    Answer the question using ONLY the context provided below. \n
    If the answer is not in the context, \n
    say "I don't have enough information." \n
Context:\n
{context} \n

Question: \n
{question} \n

Answer:"""

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        print("\nNo API key found — prompt ready to send:")
        print(f"\nContext chars: {len(context)}")
        print(f"Chunks used: {len(retrieved_chunks)}")
        return

    import urllib.request
    data = json.dumps({
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
    )
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read())
        print(f"\nAnswer: {result['content'][0]['text']}")

# ── 6. MAIN ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Only re-embeds if chroma_store doesn't exist yet
    text = extract_text_from_pdf("sample_data_report.pdf")
    chunks = chunk_text(text)
    collection, model = build_or_load_vector_store(chunks)

    questions = [
        "What was the total revenue in Q2 2024?",
        "How did the machine learning model perform?",
        "What are the key risks for Q3?",
        "What was the customer churn rate?",
        "What is the weather like today?"  # tests irrelevant question handling
    ]

    for question in questions:
        retrieved = ask_question(question, collection, model)
        generate_answer(question, retrieved)

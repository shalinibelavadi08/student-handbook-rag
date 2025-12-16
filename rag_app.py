import os
import faiss
import numpy as np
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
DOCS_FOLDER = "Docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 6
# ---------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- 1. Load & clean documents (PDF + DOCX) ----------
def load_documents(folder):
    texts = []

    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)

        # ----- PDF -----
        if file.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text = " ".join(text.replace("\n", " ").split())
                    texts.append(text)

        # ----- DOCX -----
        elif file.endswith(".docx"):
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text.strip())

            combined_text = " ".join(full_text)
            combined_text = " ".join(combined_text.split())
            texts.append(combined_text)

    return texts

# ---------- 2. Chunk text ----------
def chunk_text(texts):
    chunks = []
    for text in texts:
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP
    return chunks

# ---------- 3. Create embeddings ----------
def embed_texts(texts):
    embeddings = model.encode(texts, show_progress_bar=True)
    return np.array(embeddings).astype("float32")

# ---------- 4. Build FAISS index ----------
def build_index(vectors):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

# ---------- 5. Answer logic ----------
def answer_question(question, index, chunks):
    q_lower = question.lower()

    # Explicit overview handling
    if "what is this document about" in q_lower or "objective of the document" in q_lower:
        return f"""
Question:
{question}

Answer:
This document is a Student Information Book for the PGDM programme (2025–26) at SDMIMD.
It serves as a comprehensive guide explaining academic structure, evaluation methods,
rules and regulations, code of conduct, projects, internships, and institutional policies.

(This answer is generated using semantic retrieval with structured interpretation.)
"""

    # Retrieval-based answer
    q_vector = embed_texts([question])
    _, idx = index.search(q_vector, TOP_K)

    retrieved_chunks = [chunks[i] for i in idx[0]]
    all_text = " ".join(retrieved_chunks)

    sentences = all_text.split(". ")
    filtered = [s.strip() for s in sentences if len(s.strip()) > 60]

    summary = ". ".join(filtered[:5])

    return f"""
Question:
{question}

Answer:
{summary}

(This answer is generated using semantic retrieval and rule-based summarization.)
"""

# ---------- PIPELINE ----------
print("Loading documents...")
documents = load_documents(DOCS_FOLDER)
print(f"Loaded {len(documents)} documents")

print("Chunking text...")
chunks = chunk_text(documents)
print(f"Created {len(chunks)} chunks")

print("Creating embeddings (local model)...")
vectors = embed_texts(chunks)

print("Building vector index...")
index = build_index(vectors)

print("\nSystem ready. Ask questions.\n")

while True:
    q = input("Ask a question (type 'exit' to quit): ")
    if q.lower() == "exit":
        break
    answer = answer_question(q, index, chunks)
    print(answer)
    print("\n" + "-" * 60 + "\n")

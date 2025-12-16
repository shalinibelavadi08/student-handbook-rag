import streamlit as st
import os
import faiss
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
DOCS_FOLDER = "Docs"   # Place .docx files here
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 6
# ---------------------------

st.set_page_config(page_title="Student Information Book Assistant", layout="centered")

st.title("📘 Student Information Book Assistant")
st.write("Ask questions based on the Student Information Book (PGDM 2025–26)")

# ---------- SESSION STATE ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- MODEL ----------
model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------- LOAD & INDEX DOCUMENTS ----------
@st.cache_resource
def load_system():
    texts = []

    for file in os.listdir(DOCS_FOLDER):
        if file.endswith(".docx"):
            doc = Document(os.path.join(DOCS_FOLDER, file))
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    texts.append(text)

    # Chunking
    chunks = []
    for text in texts:
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start = end - CHUNK_OVERLAP

    embeddings = model.encode(chunks, show_progress_bar=False)
    vectors = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    return index, chunks

index, chunks = load_system()

# ---------- ANSWER LOGIC ----------
def answer_question(question):
    q_lower = question.lower()

    # Manual override for overview questions
    if "what is this document about" in q_lower or "objective" in q_lower:
        return (
            "This document is a Student Information Book for the PGDM programme (2025–26) at SDMIMD. "
            "It provides details about programme structure, academic requirements, evaluation methods, "
            "code of conduct, disciplinary policies, projects and internships, fee structure, "
            "and general institutional information."
        )

    q_vector = model.encode([question])
    _, idx = index.search(np.array(q_vector).astype("float32"), TOP_K)

    retrieved_text = " ".join([chunks[i] for i in idx[0]])
    sentences = [s.strip() for s in retrieved_text.split(". ") if len(s.strip()) > 60]

    return ". ".join(sentences[:5])

# ---------- INPUT FORM (KEY FIX) ----------
with st.form("question_form", clear_on_submit=True):
    question = st.text_input("🔍 Enter your question:")
    submitted = st.form_submit_button("Ask")

    if submitted and question:
        with st.spinner("Searching the document..."):
            answer = answer_question(question)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

# ---------- OPTIONAL CLEAR CHAT ----------
if st.button("Clear chat"):
    st.session_state.chat_history = []

# ---------- DISPLAY CHAT ----------
st.divider()

for chat in st.session_state.chat_history:
    st.markdown("### ❓ Question")
    st.write(chat["question"])

    st.markdown("### ✅ Answer")
    st.write(chat["answer"])

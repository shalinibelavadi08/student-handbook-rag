import streamlit as st 
import os
import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
DOCS_FOLDER = "Docs"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 6
# ---------------------------

st.set_page_config(page_title="Student Handbook Assistant", layout="centered")

st.title("📘 Student Information Book Assistant")
st.write("Ask questions based on the Student Information Book (PGDM 2025–26)")

# ---------- SESSION STATE ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "question_input" not in st.session_state:
    st.session_state.question_input = ""

model = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_system():
    texts = []
    for file in os.listdir(DOCS_FOLDER):
        if file.endswith(".pdf"):
            reader = PdfReader(os.path.join(DOCS_FOLDER, file))
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text = text.replace("\n", " ")
                    text = " ".join(text.split())
                    texts.append(text)

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

def answer_question(question):
    if "what is this document about" in question.lower() or "objective" in question.lower():
        return (
            "This document is a Student Information Book for the PGDM programme (2025–26) at SDMIMD. "
            "It explains the academic structure, rules and regulations, evaluation methods, code of conduct, "
            "disciplinary policies, projects, and general institutional information required during the programme."
        )

    q_vector = model.encode([question])
    _, idx = index.search(np.array(q_vector).astype("float32"), TOP_K)

    retrieved = " ".join([chunks[i] for i in idx[0]])
    sentences = [s.strip() for s in retrieved.split(". ") if len(s.strip()) > 60]

    return ". ".join(sentences[:5])

# ---------- CALLBACK (THIS IS THE FIX) ----------
def handle_ask():
    question = st.session_state.question_input

    if question:
        answer = answer_question(question)

        st.session_state.chat_history.append({
            "question": question,
            "answer": answer
        })

        # ✅ SAFE CLEAR
        st.session_state.question_input = ""
    else:
        st.warning("Please enter a question.")

# ---------- UI ----------
st.text_input("🔍 Enter your question:", key="question_input")

st.button("Ask", on_click=handle_ask)

if st.button("Clear chat"):
    st.session_state.chat_history = []

st.divider()

# ---------- DISPLAY CHAT ----------
for chat in st.session_state.chat_history:
    st.markdown("### ❓ Question")
    st.write(chat["question"])

    st.markdown("### ✅ Answer")
    st.write(chat["answer"])




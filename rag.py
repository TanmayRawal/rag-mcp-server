import os
os.environ["GROQ_API_KEY"] = "GROQ_API_KEY"

import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

INDEX_PATH = r"C:\Users\Tanmay\Research_assistant\faiss_index.bin"
CHUNKS_PATH = r"C:\Users\Tanmay\Research_assistant\chunks.json"
TOP_K = 5

print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "r") as f:
    chunks = json.load(f)

groq_client = Groq()

def query(question: str) -> str:
    q_embedding = embedder.encode([question])
    q_embedding = np.array(q_embedding).astype("float32")
    distances, indices = index.search(q_embedding, TOP_K)
    context_parts = []
    for idx in indices[0]:
        chunk = chunks[idx]
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)
    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the context below.
If the answer is not in the context, say "I couldn't find this in the documents."

Context:
{context}

Question: {question}

Answer:"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content
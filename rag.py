import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# Put your NEW Groq API key here
GROQ_API_KEY = "GROQ_API"

INDEX_PATH = r"C:\Users\Tanmay\Research_assistant\faiss_index.bin"
CHUNKS_PATH = r"C:\Users\Tanmay\Research_assistant\chunks.json"
TOP_K = 5

print("Loading models...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

# Create Groq client only once
groq_client = Groq(api_key=GROQ_API_KEY)


def retrieve_chunks(question, top_k=5):
    q_embedding = embedder.encode([question])
    q_embedding = np.array(q_embedding).astype("float32")
    distances, indices = index.search(q_embedding, top_k)

    retrieved_chunks = []
    for idx in indices[0]:
        if idx != -1:
            chunk = chunks[idx]
            retrieved_chunks.append(chunk)

    return retrieved_chunks


def generate_answer(question, retrieved_chunks):
    context_parts = []
    for chunk in retrieved_chunks:
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


def query(question: str) -> str:
    retrieved_chunks = retrieve_chunks(question, TOP_K)
    answer = generate_answer(question, retrieved_chunks)
    return answer


def query_rag_with_sources(question, top_k=5):
    retrieved_chunks = retrieve_chunks(question, top_k=top_k)
    answer = generate_answer(question, retrieved_chunks)
    return answer, retrieved_chunks
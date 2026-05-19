import numpy as np
import faiss, pickle, logging, os
from sentence_transformers import SentenceTransformer
from typing import List, Dict
logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "data/faiss_index.bin"
CHUNKS_PATH = "data/chunks.pkl"

def get_embeddings(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings.astype("float32")

def build_faiss_index(embeddings: np.ndarray):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    logger.info(f"FAISS index built: {index.ntotal} vectors, dim {dim}")
    return index

def save_index(index, chunks):
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def search(query: str, index, chunks, top_k: int = 3):
    model = SentenceTransformer(MODEL_NAME)
    q_emb = model.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, top_k)
    return [{**chunks[idx], "distance": float(dist)} for dist, idx in zip(distances[0], indices[0]) if idx < len(chunks)]

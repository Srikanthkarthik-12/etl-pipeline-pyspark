import os, logging
from typing import List, Dict
logger = logging.getLogger(__name__)

def load_documents(docs_path: str) -> List[Dict]:
    documents = []
    for filename in os.listdir(docs_path):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_path, filename), "r") as f:
                content = f.read()
            documents.append({"filename": filename, "content": content})
            logger.info(f"Loaded: {filename}")
    return documents

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def prepare_chunks(documents: List[Dict]) -> List[Dict]:
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["content"])
        for idx, chunk in enumerate(chunks):
            all_chunks.append({"doc_name": doc["filename"], "chunk_id": idx, "text": chunk})
    logger.info(f"Total chunks: {len(all_chunks)}")
    return all_chunks

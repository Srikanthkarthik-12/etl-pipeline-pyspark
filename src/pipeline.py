import logging, os
from src.ingest import load_documents, prepare_chunks
from src.embeddings import get_embeddings, build_faiss_index, save_index, load_index, search
from src.generate import build_prompt, generate_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def build_index(docs_path: str = "data/docs"):
    documents = load_documents(docs_path)
    chunks = prepare_chunks(documents)
    embeddings = get_embeddings([c["text"] for c in chunks])
    index = build_faiss_index(embeddings)
    save_index(index, chunks)
    logger.info("Index built successfully")

def query(question: str, top_k: int = 3, use_llm: bool = False) -> dict:
    if not os.path.exists("data/faiss_index.bin"):
        raise FileNotFoundError("Index not found. Run build_index() first.")
    index, chunks = load_index()
    retrieved = search(question, index, chunks, top_k=top_k)
    result = {"question": question, "retrieved_chunks": retrieved,
              "context": "\n\n".join([c["text"] for c in retrieved])}
    if use_llm:
        result["answer"] = generate_answer(build_prompt(question, retrieved))
    else:
        result["answer"] = result["context"][:500] + "..."
    return result

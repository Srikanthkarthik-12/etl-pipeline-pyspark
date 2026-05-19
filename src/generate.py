from transformers import pipeline
import logging
logger = logging.getLogger(__name__)

def build_prompt(query: str, context_chunks: list) -> str:
    context = "\n\n".join([f"[{c['doc_name']}]: {c['text']}" for c in context_chunks])
    return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

def generate_answer(prompt: str, model_name: str = "google/flan-t5-base") -> str:
    generator = pipeline("text2text-generation", model=model_name, max_new_tokens=200)
    return generator(prompt)[0]["generated_text"].strip()

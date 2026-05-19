"""
RAG Pipeline Demo
Run: python main.py
"""
from src.pipeline import build_index, query
import json

if __name__ == "__main__":
    # Step 1: build index from documents
    print("Building index from documents...")
    build_index("data/docs")

    # Step 2: query
    questions = [
        "What is supervised learning?",
        "What tools do data engineers use?",
        "What cloud services does AWS offer?"
    ]

    for q in questions:
        print(f"\nQuestion: {q}")
        result = query(q, top_k=2, use_llm=False)
        print(f"Retrieved from: {[c['doc_name'] for c in result['retrieved_chunks']]}")
        print(f"Answer preview: {result['answer'][:200]}...")

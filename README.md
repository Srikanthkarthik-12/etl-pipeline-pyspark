# RAG Pipeline with HuggingFace + FAISS

A production-style Retrieval Augmented Generation (RAG) pipeline that ingests documents, embeds them using sentence-transformers, stores them in a FAISS vector index, and retrieves relevant context to answer questions.

## Features
- Document ingestion with overlapping chunking strategy
- Semantic embeddings using `all-MiniLM-L6-v2`
- FAISS vector store for fast similarity search
- Optional LLM answer generation using `flan-t5-base`
- Modular design — swap in any embedding model or LLM

## Tech Stack
- Python, HuggingFace Transformers, Sentence-Transformers
- FAISS (Facebook AI Similarity Search)
- PyTorch

## Project Structure
```
rag-pipeline/
├── src/
│   ├── ingest.py      # Document loading and chunking
│   ├── embeddings.py  # Embedding generation and FAISS index
│   ├── generate.py    # LLM answer generation
│   └── pipeline.py    # End-to-end orchestration
├── data/
│   ├── docs/          # Input documents (.txt files)
│   ├── faiss_index.bin  # Generated vector index
│   └── chunks.pkl       # Saved chunk metadata
├── main.py
└── requirements.txt
```

## Quick Start
```bash
pip install -r requirements.txt
python main.py
```

## How It Works
1. Load `.txt` documents from `data/docs/`
2. Split into overlapping chunks (300 words, 50 overlap)
3. Generate embeddings using sentence-transformers
4. Store in FAISS index for fast retrieval
5. At query time, embed the question and find top-k similar chunks
6. Optionally pass context to an LLM for answer generation

## Resume One-Liner
*Built a RAG pipeline using HuggingFace sentence-transformers and FAISS for semantic document retrieval with optional LLM-powered answer generation.*

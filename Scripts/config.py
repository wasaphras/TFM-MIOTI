"""
Central configuration for the RAG pipeline.
Modify these variables to tune system behavior.
"""

import os
from pathlib import Path

# Project root (Scripts/ parent)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data paths ---
DATA_DIR = PROJECT_ROOT / "Data"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
CATEGORIES_JSON = DATA_DIR / "categories.json"
# Language key in categories.json for resolved label strings (used by preprocess + chunking)
RAG_LABEL_LANG = "en"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
CHROMA_PERSIST_DIR_SAMPLE = DATA_DIR / "chroma_db_sample"  # Used with --limit for testing
CHROMA_COLLECTION = "eurlex_documents"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"  # Stored chunks for analysis
GROUND_TRUTH_JSONL = DATA_DIR / "ground_truth.jsonl"
EVAL_OUTPUT_DIR = DATA_DIR / "eval"
# Written by build_chunk_indices; ground_truth_generate + run_grid_eval use it for corpus scope.
EVAL_CORPUS_MANIFEST = DATA_DIR / "eval_corpus_manifest.json"

# --- Hugging Face ---
HF_REPO_ID = "coastalcph/multi_eurlex"
# Set in the environment; never commit tokens. Used for dataset download only.
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DOC_LIMIT = 1000
LABEL_LEVEL = "level_1"
TAR_SPLIT_FILES = {"train.jsonl": "train.jsonl"}

# --- Chunking ---
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 0
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

# --- Embeddings (Ollama /api/embed) ---
EMBEDDING_MODEL = "qwen3-embedding:4b"
# Full width for this model on Ollama (larger "dimensions" requests still cap at 2560).
EMBEDDING_DIMENSIONS = 2560
EMBEDDING_BATCH_SIZE = 32

# --- LLM ---
LLM_MODEL = "llama3.2"

# --- Retrieval ---
RETRIEVAL_K = 3

# --- Eval: cross-encoder reranking (retrieve many, keep top FINAL_K in metrics) ---
# Override via environment for lighter models: BAAI/bge-reranker-base, cross-encoder/ms-marco-MiniLM-L-6-v2
RERANK_MODEL = os.environ.get(
    "RERANK_MODEL", "BAAI/bge-reranker-v2-m3"
)
RETRIEVAL_CANDIDATE_K = int(os.environ.get("RETRIEVAL_CANDIDATE_K", "50"))
# Cap passage chars sent to the reranker to limit memory (full chunk text rarely needs more).
RERANK_PASSAGE_MAX_CHARS = int(os.environ.get("RERANK_PASSAGE_MAX_CHARS", "8000"))
# "cpu", "cuda", "cuda:0", or unset for auto (prefer CUDA, fall back to CPU on OOM).
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "").strip()
# CrossEncoder.predict batch size; lower reduces peak VRAM during rerank forward pass.
RERANK_PREDICT_BATCH_SIZE = int(os.environ.get("RERANK_PREDICT_BATCH_SIZE", "32"))

# --- RAG system prompt (used when answering questions) ---
SYSTEM_PROMPT = """You are a question-answering assistant for EU legislation documents.
Your task is to answer the user's question using ONLY the retrieved context provided below.

RULES:
- Answer ONLY based on the retrieved context. Do not use external knowledge.
- If the answer cannot be found in the context, respond: "The answer is not provided in the retrieved context."
- Be concise. When possible, cite or reference the relevant passage from the context.
- Do not speculate or make up information."""

# --- Default prompt for main.py (can be overridden via CLI or config) ---
DEFAULT_PROMPT = "What does the COUNCIL REGULATION (EC) No 1887/94 of 27 July 1994 fixing the basic price state?"

# --- API ---
API_HOST = "0.0.0.0"
API_PORT = 8000

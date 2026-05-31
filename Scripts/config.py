"""Paths, models, and env-backed overrides for the RAG pipeline."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv as _dotenv_load
except ImportError:
    _dotenv_load = None  # type: ignore[assignment, misc]


def load_project_dotenv(*, override: bool = False) -> None:
    """Load ``<project_root>/.env`` if present. Existing env vars win unless override=True."""
    if _dotenv_load is None:
        return
    env_path = PROJECT_ROOT / ".env"
    if env_path.is_file():
        _dotenv_load(env_path, override=override)


load_project_dotenv()

_default_data = PROJECT_ROOT / "Data"
DATA_DIR = Path(os.environ.get("TFM_DATA_DIR", str(_default_data))).resolve()
_default_categories = PROJECT_ROOT / "Data" / "categories.json"
CATEGORIES_JSON = Path(
    os.environ.get("TFM_CATEGORIES_JSON", str(_default_categories))
).resolve()

TRAIN_JSONL = DATA_DIR / "train.jsonl"
RAG_LABEL_LANG = "en"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
CHROMA_PERSIST_DIR_SAMPLE = DATA_DIR / "chroma_db_sample"
CHROMA_COLLECTION = "eurlex_documents"
CHUNKS_JSONL = DATA_DIR / "chunks.jsonl"
GROUND_TRUTH_JSONL = DATA_DIR / "ground_truth.jsonl"
EVAL_OUTPUT_DIR = DATA_DIR / "eval"
EVAL_CORPUS_MANIFEST = DATA_DIR / "eval_corpus_manifest.json"

HF_REPO_ID = "coastalcph/multi_eurlex"
HF_TOKEN = os.environ.get("HF_TOKEN", "")
# Cap when sampling from the MultiEURLEX tarball (full English train split is ~55k docs).
DOC_LIMIT = int(os.environ.get("TFM_DOC_LIMIT", "55000"))
LABEL_LEVEL = "level_1"
TAR_SPLIT_FILES = {"train.jsonl": "train.jsonl"}

CHUNK_SIZE = 2000
CHUNK_OVERLAP = 0
CHUNK_SEPARATORS = ["\n\n", "\n", " ", ""]

EMBEDDING_MODEL = "qwen3-embedding:4b"
EMBEDDING_DIMENSIONS = 2560
EMBEDDING_BATCH_SIZE = 32

LLM_MODEL = "llama3.2"

RETRIEVAL_K = 3

RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RETRIEVAL_CANDIDATE_K = int(os.environ.get("RETRIEVAL_CANDIDATE_K", "50"))
RERANK_PASSAGE_MAX_CHARS = int(os.environ.get("RERANK_PASSAGE_MAX_CHARS", "8000"))
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", "").strip()
RERANK_PREDICT_BATCH_SIZE = int(os.environ.get("RERANK_PREDICT_BATCH_SIZE", "32"))

SYSTEM_PROMPT = """You are a question-answering assistant for EU legislation documents.
Your task is to answer the user's question using ONLY the retrieved context provided below.

RULES:
- Answer ONLY based on the retrieved context. Do not use external knowledge.
- If the answer cannot be found in the context, respond: "The answer is not provided in the retrieved context."
- Format your answer in markdown (lists, bold, etc. where helpful).
- Cite sources using bracket notation matching the context labels, e.g. [1] or [2][5].
  Only cite source numbers you actually used. Every factual claim should have a citation.
- Do not speculate or make up information."""

DEFAULT_PROMPT = "What does the COUNCIL REGULATION (EC) No 1887/94 of 27 July 1994 fixing the basic price state?"

API_HOST = "0.0.0.0"
API_PORT = 8000

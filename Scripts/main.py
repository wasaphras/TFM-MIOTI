"""
Main orchestrator: runs the full RAG pipeline and answers a configurable prompt.
"""

import argparse

from . import config
from .chunking import chunk_documents
from .data_extraction_load import extract_and_download, load_data
from .embeddings_chromadb import get_or_create_vectorstore
from .preprocess import preprocess_for_rag
from .retriever import rag_answer


def run_pipeline(prompt: str | None = None, limit: int | None = None) -> str:
    """
    Run the full RAG pipeline: extract/load data, preprocess labels, chunk, embed, retrieve, answer.
    Returns the answer string.
    If limit is set, uses only the first N documents (for testing).
    """
    prompt = prompt or config.DEFAULT_PROMPT

    # 1. Data
    extract_and_download()
    df = load_data(limit=limit)
    print(f"Loaded {len(df)} documents.")

    df = preprocess_for_rag(df)

    # 2. Chunking
    documents = chunk_documents(df)
    print(f"Created {len(documents)} chunks.")

    # 3. Vector store (create or load)
    persist_dir = config.CHROMA_PERSIST_DIR_SAMPLE if limit else None
    vectorstore = get_or_create_vectorstore(documents, persist_dir=persist_dir)
    print(f"Vector store ready ({vectorstore._collection.count()} docs).")

    # 4. RAG answer
    print(f"\nQuery: {prompt}\n")
    answer = rag_answer(prompt, vectorstore)
    return answer


def main():
    parser = argparse.ArgumentParser(description="RAG pipeline: answer a question over EU legislation.")
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default=None,
        help=f"Question to ask (default: {config.DEFAULT_PROMPT[:50]}...)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit to first N documents (for quick testing)",
    )
    args = parser.parse_args()

    answer = run_pipeline(prompt=args.prompt, limit=args.limit)
    print("\n--- Answer ---")
    print(answer)


if __name__ == "__main__":
    main()

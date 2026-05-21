"""
Retriever and RAG answer generation.
"""

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma

from . import config


def get_retriever(vectorstore: Chroma, k: int | None = None):
    """Return LangChain retriever from vectorstore."""
    k = k or config.RETRIEVAL_K
    return vectorstore.as_retriever(search_kwargs={"k": k})


def retrieve(
    vectorstore: Chroma, query: str, k: int | None = None
) -> list[Document]:
    """Retrieve documents by similarity search."""
    k = k or config.RETRIEVAL_K
    return vectorstore.similarity_search(query, k=k)


def _doc_context_header(doc: Document) -> str:
    """Format celex_id and categories from metadata (not embedded in page_content)."""
    meta = doc.metadata or {}
    parts: list[str] = []
    celex = meta.get("celex_id")
    if celex:
        parts.append(f"CELEX: {celex}")
    cats = meta.get("categories_en")
    if cats:
        parts.append(f"Categories: {cats}")
    return " | ".join(parts)


def _source_block(rank: int, doc: Document) -> str:
    """One numbered source block for RAG / history prompts."""
    meta = doc.metadata or {}
    uid = str(meta.get("chunk_uid") or "")
    header = _doc_context_header(doc)
    block = f"--- Source [{rank}] (chunk_uid: {uid}) ---\n"
    if header:
        block += f"{header}\n"
    block += doc.page_content or ""
    return block


def _build_rag_prompt(query: str, docs: list[Document]) -> str:
    """Build the user prompt with retrieved context."""
    context_parts = [_source_block(i, doc) for i, doc in enumerate(docs, 1)]
    context = "\n\n".join(context_parts)
    return f"""USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}

Answer in markdown. Cite sources with [n] matching the Source [n] labels above.
Only cite sources you actually used."""


def format_context_chunks_text(chunks: list[dict]) -> str:
    """Format saved context chunks for prior-turn injection."""
    if not chunks:
        return ""
    parts: list[str] = []
    for i, ch in enumerate(chunks, 1):
        uid = str(ch.get("chunk_uid") or "")
        celex = str(ch.get("celex_id") or "")
        cats = str(ch.get("categories_en") or "")
        header_bits = []
        if celex:
            header_bits.append(f"CELEX: {celex}")
        if cats:
            header_bits.append(f"Categories: {cats}")
        header = " | ".join(header_bits)
        block = f"--- Source [{i}] (chunk_uid: {uid}) ---\n"
        if header:
            block += f"{header}\n"
        block += str(ch.get("text") or "")
        parts.append(block)
    body = "\n\n".join(parts)
    return (
        "PRIOR TURN CONTEXT (already answered; use only to interpret follow-ups):\n\n"
        + body
    )


def rag_answer(
    query: str,
    vectorstore: Chroma,
    chat_model=None,
    system_prompt: str | None = None,
    k: int | None = None,
) -> str:
    """
    Run RAG: retrieve context, then generate answer via LLM.
    Returns the answer string.
    """
    if chat_model is None:
        chat_model = ChatOllama(model=config.LLM_MODEL)
    system_prompt = system_prompt or config.SYSTEM_PROMPT
    k = k or config.RETRIEVAL_K

    docs = retrieve(vectorstore, query, k=k)
    user_content = _build_rag_prompt(query, docs)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_content),
    ]
    response = chat_model.invoke(messages)
    return response.content


if __name__ == "__main__":
    from .embeddings_chromadb import get_or_create_vectorstore

    vs = get_or_create_vectorstore()
    answer = rag_answer(
        "What does the COUNCIL REGULATION (EC) No 1887/94 state about basic price?",
        vs,
    )
    print("Answer:", answer)

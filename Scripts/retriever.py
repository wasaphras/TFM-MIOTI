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


def _build_rag_prompt(query: str, docs: list[Document]) -> str:
    """Build the user prompt with retrieved context."""
    context_parts = []
    for i, doc in enumerate(docs, 1):
        header = _doc_context_header(doc)
        block = f"--- Document {i} ---\n"
        if header:
            block += f"{header}\n"
        block += doc.page_content
        context_parts.append(block)
    context = "\n\n".join(context_parts)
    return f"""USER QUESTION:
{query}

RETRIEVED CONTEXT:
{context}"""


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

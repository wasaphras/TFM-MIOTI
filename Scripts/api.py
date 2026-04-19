"""
FastAPI chat endpoint for the RAG system.
Can be used by Streamlit or other clients via HTTP.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from . import config
from .embeddings_chromadb import get_or_create_vectorstore
from .retriever import rag_answer

# Global state (initialized at startup)
_vectorstore = None
_chat_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load vectorstore and chat model at startup."""
    global _vectorstore, _chat_model
    from langchain_ollama import ChatOllama

    print("Loading RAG components...")
    _vectorstore = get_or_create_vectorstore()
    _chat_model = ChatOllama(model=config.LLM_MODEL)
    print("RAG components ready.")
    yield
    # Cleanup if needed
    _vectorstore = None
    _chat_model = None


app = FastAPI(title="RAG Chat API", lifespan=lifespan)


class ChatRequest(BaseModel):
    """Request body for /chat endpoint."""

    query: str


class ChatResponse(BaseModel):
    """Response body for /chat endpoint."""

    answer: str


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Chat with the RAG system.
    Accepts {"query": "user question"} and returns {"answer": "..."}.
    """
    answer = rag_answer(
        request.query,
        _vectorstore,
        chat_model=_chat_model,
    )
    return ChatResponse(answer=answer)


def run_api(host: str | None = None, port: int | None = None):
    """Run the API server. Run from project root: python -m Scripts.api"""
    import uvicorn

    host = host or config.API_HOST
    port = port or config.API_PORT
    uvicorn.run(
        "Scripts.api:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    run_api()

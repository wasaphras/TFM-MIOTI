"""
FastAPI chat endpoint for the RAG system (dedup eval stack + Gemini streaming).
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from . import config
from . import rag_chat_service


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "http://localhost:5173").strip()
    if not raw:
        return ["http://localhost:5173"]
    return [o.strip() for o in raw.split(",") if o.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load dedup RAG stack at startup."""
    print("Loading RAG chat service...")
    rag_chat_service.startup()
    print("RAG chat service ready.")
    yield
    rag_chat_service.shutdown()


app = FastAPI(title="RAG Chat API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ContextChunk(BaseModel):
    chunk_uid: str = Field(..., min_length=1)
    celex_id: str = ""
    categories_en: str = ""
    text: str = ""


class HistoryTurn(BaseModel):
    user: str = Field(..., min_length=1)
    assistant: str = Field(..., min_length=1)
    context_chunks: list[ContextChunk] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """Request body for /chat/stream."""

    query: str = Field(..., min_length=1)
    history: list[HistoryTurn] = Field(default_factory=list)


def _sse(event: str, data: str) -> str:
    """Format one Server-Sent Events frame."""
    lines = data.split("\n")
    payload = "".join(f"data: {line}\n" for line in lines)
    return f"event: {event}\n{payload}\n"


def _chat_stream(query: str, history: list[HistoryTurn]):
    try:
        hist = [h.model_dump() for h in history]
        docs, sources = rag_chat_service.retrieve_for_chat(query, history=hist)
        yield _sse("sources", json.dumps(sources, ensure_ascii=False))

        answer_parts: list[str] = []
        for token in rag_chat_service.stream_answer(query, docs, history=hist):
            answer_parts.append(token)
            yield _sse("token", token)

        answer = "".join(answer_parts)
        used_uids = rag_chat_service.resolve_used_chunk_uids(answer, docs)
        done_payload = {
            "used_chunk_uids": used_uids,
            "context_chunks": rag_chat_service.context_chunks_from_docs(
                docs, used_uids
            ),
        }
        yield _sse("done", json.dumps(done_payload, ensure_ascii=False))
    except RuntimeError as e:
        msg = str(e)
        status = 503 if "GEMINI_API_KEY" in msg else 500
        yield _sse("error", json.dumps({"message": msg, "status": status}))
    except FileNotFoundError as e:
        yield _sse(
            "error",
            json.dumps({"message": str(e), "status": 503}),
        )
    except Exception as e:
        yield _sse(
            "error",
            json.dumps({"message": str(e), "status": 500}),
        )


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """
    Stream RAG answer via SSE.
    Events: sources (JSON array), token (text), error (JSON), done (JSON).
    """
    q = request.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query must not be empty")
    return StreamingResponse(
        _chat_stream(q, request.history),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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

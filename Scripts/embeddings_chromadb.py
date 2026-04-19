"""
Embeddings and ChromaDB vector store.
Creates or loads the vector store for document retrieval.

Uses the ``ollama`` client's ``/api/embed`` (not deprecated /api/embeddings) so
``dimensions`` (Matryoshka width) is honored for models like qwen3-embedding.
"""

from __future__ import annotations

import gc
import json
import sqlite3
import time
from pathlib import Path

import chromadb
import chromadb.errors
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from ollama import Client
from tqdm import tqdm

from . import config

# Vectors held before flush to Chroma (balance RAM vs add frequency).
CHROMA_WRITE_THRESHOLD = 512
# Each collection.add is split into slices this size. Large adds often trigger
# HNSW compaction InternalError on big corpora; small slices + retries are safer.
CHROMA_ADD_SUBBATCH_SIZE = 128
_CHROMA_ADD_RETRIES = 4

# Chroma 1.x: count rows via sqlite (read-only) to avoid loading HNSW into RAM.
# Rows link to the collection via segments; in Chroma 1.5.x they attach to the
# METADATA segment, not VECTOR—filtering scope='VECTOR' returned 0 and broke resume.
_SQLITE_EMBEDDING_COUNT_QUERY = """
SELECT COUNT(*) FROM embeddings AS e
INNER JOIN segments AS s ON s.id = e.segment_id
INNER JOIN collections AS c ON c.id = s.collection
WHERE c.name = ?
"""


def release_chroma_process_cache() -> None:
    """
    Drop Chroma's in-process SharedSystemClient cache (one System per persist path).
    Safe between strategies in a single-threaded loop so vector indexes are not
    all retained until process exit.
    """
    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:
        pass
    gc.collect()


def _chroma_embedding_count_sqlite(
    persist_directory: Path, collection_name: str
) -> int | None:
    """
    Return embedding row count from chroma.sqlite3, or None if unreadable
    (missing file, schema mismatch, etc.).
    """
    sqlite_path = persist_directory / "chroma.sqlite3"
    if not sqlite_path.is_file():
        return None
    try:
        uri = f"file:{sqlite_path.resolve().as_posix()}?mode=ro"
        con = sqlite3.connect(uri, uri=True)
        try:
            row = con.execute(
                _SQLITE_EMBEDDING_COUNT_QUERY, (collection_name,)
            ).fetchone()
            if row is None:
                return None
            return int(row[0])
        finally:
            con.close()
    except Exception:
        return None


def _metadata_for_chroma(meta: dict | None) -> dict:
    """Keep only Chroma-allowed metadata types (same idea as LangChain filter_complex_metadata)."""
    out: dict = {}
    for k, v in (meta or {}).items():
        if isinstance(v, (str, bool, int, float)):
            out[k] = v
    return out


def _collection_add_batched(
    collection,
    client: chromadb.PersistentClient,
    ids: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    documents: list[str],
) -> None:
    """Add records in small slices with retries (mitigates HNSW compaction errors)."""
    n = len(ids)
    if n == 0:
        return
    max_api = max(1, int(client.get_max_batch_size()))
    step = min(CHROMA_ADD_SUBBATCH_SIZE, max_api)
    for i in range(0, n, step):
        sl = slice(i, i + step)
        bid = ids[sl]
        bem = embeddings[sl]
        bmeta = metadatas[sl]
        bdoc = documents[sl]
        last_err: BaseException | None = None
        for attempt in range(_CHROMA_ADD_RETRIES):
            try:
                collection.add(
                    ids=list(bid),
                    embeddings=list(bem),
                    metadatas=list(bmeta),
                    documents=list(bdoc),
                )
                break
            except chromadb.errors.InternalError as e:
                last_err = e
                time.sleep(0.25 * (2**attempt))
        else:
            if last_err is not None:
                raise last_err
            raise RuntimeError("Chroma collection.add failed after retries")


def chroma_collection_count(
    persist_directory: Path | str,
    collection_name: str | None = None,
) -> int | None:
    """
    Return the number of rows in the Chroma collection, or None if the persist
    path is missing, sqlite is absent, or the collection cannot be opened.
    Used to skip complete indices and to detect partial / interrupted embeds.

    Prefers a read-only SQLite count (no HNSW load); falls back to
    PersistentClient if the DB schema is unsupported.
    """
    persist_directory = Path(persist_directory)
    if not persist_directory.exists():
        return None
    sqlite_path = persist_directory / "chroma.sqlite3"
    if not sqlite_path.is_file():
        return None
    name = collection_name or config.CHROMA_COLLECTION
    n_sql = _chroma_embedding_count_sqlite(persist_directory, name)
    if n_sql is not None:
        return n_sql
    try:
        client = chromadb.PersistentClient(path=str(persist_directory))
        coll = client.get_collection(name)
        return int(coll.count())
    except Exception:
        return None


class TqdmOllamaEmbeddings(Embeddings):
    """
    Ollama embeddings via ``Client.embed`` (``/api/embed``), with optional
    ``dimensions`` and the same passage/query prefixes as LangChain's legacy
    community Ollama integration (asymmetric retrieval).
    """

    embed_instruction: str = "passage: "
    query_instruction: str = "query: "

    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        client: Client | None = None,
    ) -> None:
        self.model = model or config.EMBEDDING_MODEL
        self.dimensions = (
            dimensions if dimensions is not None else config.EMBEDDING_DIMENSIONS
        )
        self._client = client or Client()

    def _embed_batch(self, inputs: list[str]) -> list[list[float]]:
        kwargs = {}
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        resp = self._client.embed(self.model, inputs, **kwargs)
        return [list(e) for e in resp.embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        batch_size = config.EMBEDDING_BATCH_SIZE
        prefixed = [f"{self.embed_instruction}{t}" for t in texts]
        all_embeddings: list[list[float]] = []
        for i in tqdm(
            range(0, len(prefixed), batch_size),
            desc="Embedding",
            unit="batch",
        ):
            batch = prefixed[i : i + batch_size]
            all_embeddings.extend(self._embed_batch(batch))
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        one = f"{self.query_instruction}{text}"
        return self._embed_batch([one])[0]


def get_embedding_model() -> TqdmOllamaEmbeddings:
    """Ollama embedder aligned with config (model + dimensions)."""
    return TqdmOllamaEmbeddings()


def persist_chroma_from_chunks_jsonl(
    chunks_jsonl: Path | str,
    persist_directory: Path | str,
    collection_name: str,
    *,
    expected_n: int,
    skip_first: int = 0,
) -> None:
    """
    Stream ``chunks_*.jsonl`` in small batches: embed and write to Chroma without
    loading all chunks into RAM. ``skip_first`` skips that many non-empty lines
    (resume = current Chroma count). Verifies final collection size == ``expected_n``.
    """
    if expected_n < 0:
        raise ValueError(f"expected_n must be >= 0, got {expected_n}")
    if skip_first < 0 or skip_first > expected_n:
        raise ValueError(f"skip_first must be in [0, {expected_n}], got {skip_first}")

    chunks_jsonl = Path(chunks_jsonl)
    if not chunks_jsonl.is_file():
        raise FileNotFoundError(f"Missing chunks JSONL: {chunks_jsonl}")

    n_remaining = expected_n - skip_first
    if n_remaining == 0:
        persist_directory = Path(persist_directory)
        persist_directory.mkdir(parents=True, exist_ok=True)
        final_n = chroma_collection_count(persist_directory, collection_name)
        if final_n != expected_n:
            raise RuntimeError(
                f"Chroma verification: expected {expected_n} vectors, got {final_n!r}"
            )
        return

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    base_emb = get_embedding_model()
    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_or_create_collection(name=collection_name)

    embed_batch = config.EMBEDDING_BATCH_SIZE
    buf_ids: list[str] = []
    buf_embs: list[list[float]] = []
    buf_metas: list[dict] = []
    buf_texts: list[str] = []

    def _flush() -> None:
        nonlocal buf_ids, buf_embs, buf_metas, buf_texts
        if not buf_ids:
            return
        _collection_add_batched(
            collection, client, buf_ids, buf_embs, buf_metas, buf_texts
        )
        buf_ids, buf_embs, buf_metas, buf_texts = [], [], [], []

    desc = (
        f"Embed+stream JSONL {n_remaining} chunks (resume {skip_first}/{expected_n})"
        if skip_first
        else f"Embed+stream JSONL {expected_n} chunks"
    )
    pbar = tqdm(total=n_remaining, desc=desc, unit="chunk")
    processed = 0

    try:
        with open(chunks_jsonl, encoding="utf-8") as f:
            skipped = 0
            while skipped < skip_first:
                line = f.readline()
                if not line:
                    raise ValueError(
                        f"{chunks_jsonl}: EOF while skipping (need {skip_first}, got {skipped})"
                    )
                if line.strip():
                    skipped += 1

            while processed < n_remaining:
                need = n_remaining - processed
                cap = min(embed_batch, need)
                batch_raw: list[str] = []
                while len(batch_raw) < cap:
                    line = f.readline()
                    if not line:
                        break
                    if line.strip():
                        batch_raw.append(line)
                if not batch_raw:
                    break

                abs_start = skip_first + processed
                batch_texts: list[str] = []
                batch_metas: list[dict] = []
                batch_ids: list[str] = []
                for j, raw in enumerate(batch_raw):
                    rec = json.loads(raw)
                    text = rec["page_content"]
                    meta = rec.get("metadata") or {}
                    batch_texts.append(text)
                    batch_metas.append(_metadata_for_chroma(meta))
                    batch_ids.append(
                        str(meta.get("chunk_uid") or (abs_start + j))
                    )

                raw_emb = base_emb.embed_documents(batch_texts)
                batch_embs = [
                    (e.tolist() if hasattr(e, "tolist") else [float(x) for x in e])
                    for e in raw_emb
                ]

                buf_ids.extend(batch_ids)
                buf_embs.extend(batch_embs)
                buf_metas.extend(batch_metas)
                buf_texts.extend(batch_texts)
                processed += len(batch_raw)
                pbar.update(len(batch_raw))

                if len(buf_ids) >= CHROMA_WRITE_THRESHOLD:
                    _flush()

            _flush()

    finally:
        pbar.close()

    if processed != n_remaining:
        raise RuntimeError(
            f"{chunks_jsonl}: expected {n_remaining} chunks to embed after skip, "
            f"got {processed}"
        )

    final_n = chroma_collection_count(persist_directory, collection_name)
    if final_n != expected_n:
        raise RuntimeError(
            f"Chroma verification failed after embed: expected {expected_n} vectors, "
            f"got {final_n!r} (persist_directory={persist_directory!s})"
        )


def persist_chroma_from_documents_one_pass(
    documents: list[Document],
    persist_directory: Path | str,
    collection_name: str,
    skip_first: int = 0,
) -> None:
    """
    Embed chunks via Ollama and write them to a persistent Chroma collection.

    Uses ``get_or_create_collection`` (no delete-then-create) so Chroma's
    path-scoped client cache cannot leave a stale "collection exists" state
    after the on-disk dir was removed and recreated elsewhere.

    ``skip_first``: number of leading documents already present in Chroma
    (same order as ``documents`` / JSONL). Embeds only ``documents[skip_first:]``.

    Memory-efficient: embeds in small batches (EMBEDDING_BATCH_SIZE) and flushes
    to Chroma every WRITE_THRESHOLD chunks.
    """
    if not documents:
        raise ValueError("documents must be non-empty")
    n = len(documents)
    if skip_first < 0 or skip_first > n:
        raise ValueError(f"skip_first must be in [0, {n}], got {skip_first}")

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    base_emb = get_embedding_model()

    client = chromadb.PersistentClient(path=str(persist_directory))
    collection = client.get_or_create_collection(name=collection_name)

    remaining = documents[skip_first:]
    n_remaining = len(remaining)
    embed_batch = config.EMBEDDING_BATCH_SIZE

    buf_ids: list[str] = []
    buf_embs: list[list[float]] = []
    buf_metas: list[dict] = []
    buf_texts: list[str] = []

    def _flush() -> None:
        nonlocal buf_ids, buf_embs, buf_metas, buf_texts
        if not buf_ids:
            return
        _collection_add_batched(
            collection, client, buf_ids, buf_embs, buf_metas, buf_texts
        )
        buf_ids, buf_embs, buf_metas, buf_texts = [], [], [], []

    desc = (
        f"Embed+store {n_remaining} chunks (resume from {skip_first}/{n})"
        if skip_first
        else f"Embed+store {n} chunks"
    )
    pbar = tqdm(total=n_remaining, desc=desc, unit="chunk")
    try:
        for batch_start in range(0, n_remaining, embed_batch):
            batch_docs = remaining[batch_start : batch_start + embed_batch]
            abs_start = skip_first + batch_start
            batch_texts = [d.page_content for d in batch_docs]
            batch_metas = [_metadata_for_chroma(d.metadata) for d in batch_docs]
            batch_ids = [
                str((d.metadata or {}).get("chunk_uid") or (abs_start + j))
                for j, d in enumerate(batch_docs)
            ]

            raw = base_emb.embed_documents(batch_texts)
            batch_embs = [
                (e.tolist() if hasattr(e, "tolist") else [float(x) for x in e])
                for e in raw
            ]

            buf_ids.extend(batch_ids)
            buf_embs.extend(batch_embs)
            buf_metas.extend(batch_metas)
            buf_texts.extend(batch_texts)
            pbar.update(len(batch_docs))

            if len(buf_ids) >= CHROMA_WRITE_THRESHOLD:
                _flush()

        _flush()
    finally:
        pbar.close()

    final_n = chroma_collection_count(persist_directory, collection_name)
    if final_n != n:
        raise RuntimeError(
            f"Chroma verification failed after embed: expected {n} vectors, got {final_n!r} "
            f"(persist_directory={persist_directory!s})"
        )


def load_vectorstore(persist_directory: Path | str) -> Chroma:
    """
    Load an existing Chroma store (eval / inference). Raises if missing.
    """
    chroma_path = Path(persist_directory)
    if not chroma_path.exists() or not (chroma_path / "chroma.sqlite3").exists():
        raise FileNotFoundError(
            f"No Chroma DB at {chroma_path} (expected chroma.sqlite3)."
        )
    return Chroma(
        persist_directory=str(chroma_path),
        embedding_function=get_embedding_model(),
        collection_name=config.CHROMA_COLLECTION,
    )


def get_or_create_vectorstore(
    documents: list[Document] | None = None,
    persist_dir: Path | str | None = None,
) -> Chroma:
    """
    Load existing ChromaDB if present, otherwise create from documents.
    If documents is None and no DB exists, raises ValueError.
    persist_dir overrides config path (e.g. for --limit testing).

    After changing chunk text or embeddings model, remove the persist directory
    and recreate the store so retrieved vectors match the new content.
    """
    persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
    chroma_path = Path(persist_dir)
    chroma_exists = chroma_path.exists() and (
        chroma_path / "chroma.sqlite3"
    ).exists()

    if chroma_exists:
        print("Loading existing ChromaDB vector store...")
        return Chroma(
            persist_directory=str(persist_dir),
            embedding_function=get_embedding_model(),
            collection_name=config.CHROMA_COLLECTION,
        )

    if documents is None or len(documents) == 0:
        raise ValueError(
            "No existing ChromaDB found and no documents provided. "
            "Run the pipeline with data first."
        )

    print("Creating ChromaDB vector store (this may take a while)...")
    chroma_path.mkdir(parents=True, exist_ok=True)
    persist_chroma_from_documents_one_pass(
        documents,
        persist_directory=chroma_path,
        collection_name=config.CHROMA_COLLECTION,
    )
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=get_embedding_model(),
        collection_name=config.CHROMA_COLLECTION,
    )
    print("ChromaDB vector store created and saved successfully!")
    return vectorstore


if __name__ == "__main__":
    from .chunking import chunk_documents
    from .data_extraction_load import main as load_main
    from .preprocess import preprocess_for_rag

    df = preprocess_for_rag(load_main())
    documents = chunk_documents(df)
    vs = get_or_create_vectorstore(documents)
    print(f"Number of documents in vector store: {vs._collection.count()}")

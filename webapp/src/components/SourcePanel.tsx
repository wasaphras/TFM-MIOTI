import { useState } from "react";
import type { DocumentSource } from "../types";

interface SourcePanelProps {
  sources: DocumentSource[];
}

export function SourcePanel({ sources }: SourcePanelProps) {
  const [open, setOpen] = useState(false);
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);
  const [expandedChunk, setExpandedChunk] = useState<string | null>(null);

  if (!sources.length) return null;

  const chunkCount = sources.reduce((n, d) => n + d.chunks.length, 0);

  return (
    <div className="source-panel">
      <button
        type="button"
        className="source-panel-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <svg
          className={`source-chevron${open ? " open" : ""}`}
          width="12"
          height="12"
          viewBox="0 0 12 12"
          fill="currentColor"
          aria-hidden
        >
          <path d="M4 2l4 4-4 4V2z" />
        </svg>
        <span>Sources</span>
        <span className="source-count">
          {sources.length} doc{sources.length !== 1 ? "s" : ""} · {chunkCount}{" "}
          chunk{chunkCount !== 1 ? "s" : ""}
        </span>
      </button>
      {open && (
        <div className="source-list">
          {sources.map((doc) => (
            <article
              key={doc.celex_id || doc.chunks[0]?.chunk_uid}
              className="source-card source-doc-card"
            >
              <header className="source-card-header">
                <code className="source-celex">{doc.celex_id || "—"}</code>
                {doc.eurlex_url && (
                  <a
                    className="source-eurlex-link"
                    href={doc.eurlex_url}
                    target="_blank"
                    rel="noopener noreferrer"
                  >
                    EUR-Lex
                  </a>
                )}
              </header>
              {doc.categories_en && (
                <p className="source-categories">{doc.categories_en}</p>
              )}
              {doc.document_preview && (
                <>
                  <button
                    type="button"
                    className="source-snippet-toggle"
                    onClick={() =>
                      setExpandedDoc((id) =>
                        id === doc.celex_id ? null : doc.celex_id,
                      )
                    }
                  >
                    {expandedDoc === doc.celex_id
                      ? "Hide document preview"
                      : "View full document preview"}
                  </button>
                  {expandedDoc === doc.celex_id && (
                    <pre className="source-doc-preview">{doc.document_preview}</pre>
                  )}
                </>
              )}
              <ul className="source-chunk-list">
                {doc.chunks.map((ch) => (
                  <li
                    key={ch.chunk_uid || ch.rank}
                    className={`source-chunk-item${ch.cited ? " cited" : ""}`}
                  >
                    <div className="source-chunk-header">
                      <span className="source-rank">#{ch.rank}</span>
                      {ch.cited && (
                        <span className="source-cited-badge">cited</span>
                      )}
                      {ch.rerank_score != null && (
                        <span
                          className="source-score"
                          title="Normalized relevance within this query (0–1)"
                        >
                          {ch.rerank_score.toFixed(3)}
                        </span>
                      )}
                    </div>
                    <button
                      type="button"
                      className="source-snippet-toggle"
                      onClick={() =>
                        setExpandedChunk((uid) =>
                          uid === ch.chunk_uid ? null : ch.chunk_uid,
                        )
                      }
                    >
                      {expandedChunk === ch.chunk_uid
                        ? "Hide excerpt"
                        : "View excerpt"}
                    </button>
                    {expandedChunk === ch.chunk_uid && (
                      <pre className="source-snippet">{ch.snippet}</pre>
                    )}
                  </li>
                ))}
              </ul>
            </article>
          ))}
        </div>
      )}
    </div>
  );
}

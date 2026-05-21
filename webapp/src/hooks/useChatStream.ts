import { useCallback, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { sseTokenToPlainText } from "../lib/sseText";
import type {
  ChatMessage,
  DocumentSource,
  DonePayload,
  HistoryTurn,
} from "../types";

const API_BASE =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8000";

function parseSseBlock(block: string): { event: string; data: string } | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trimStart());
    }
  }
  if (dataLines.length === 0 && !block.includes("event:")) {
    return null;
  }
  return { event, data: dataLines.join("\n") };
}

function markCitedSources(
  sources: DocumentSource[],
  usedUids: Set<string>,
): DocumentSource[] {
  return sources.map((doc) => ({
    ...doc,
    chunks: doc.chunks.map((ch) => ({
      ...ch,
      cited: usedUids.has(ch.chunk_uid),
    })),
  }));
}

export function useChatStream(
  setMessages: Dispatch<SetStateAction<ChatMessage[]>>,
) {
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const abortStream = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setLoading(false);
  }, []);

  const sendMessage = useCallback(
    async (query: string, history: HistoryTurn[] = []) => {
      const trimmed = query.trim();
      if (!trimmed || loading) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: trimmed,
      };
      const assistantId = crypto.randomUUID();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        sources: [],
        streaming: true,
      };

      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setLoading(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const res = await fetch(`${API_BASE}/chat/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: trimmed, history }),
          signal: controller.signal,
        });

        if (!res.ok) {
          const text = await res.text();
          throw new Error(text || `HTTP ${res.status}`);
        }

        const reader = res.body?.getReader();
        if (!reader) {
          throw new Error("No response body");
        }

        const decoder = new TextDecoder();
        let buffer = "";

        const updateAssistant = (patch: Partial<ChatMessage>) => {
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, ...patch } : m)),
          );
        };

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });

          const parts = buffer.split("\n\n");
          buffer = parts.pop() ?? "";

          for (const part of parts) {
            if (!part.trim()) continue;
            const parsed = parseSseBlock(part);
            if (!parsed) continue;

            const { event, data } = parsed;

            if (event === "sources") {
              const sources = JSON.parse(data) as DocumentSource[];
              updateAssistant({ sources });
            } else if (event === "token") {
              const piece = sseTokenToPlainText(data);
              if (!piece) continue;
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + piece }
                    : m,
                ),
              );
            } else if (event === "error") {
              const err = JSON.parse(data) as { message?: string };
              updateAssistant({
                error: err.message ?? "Unknown error",
                streaming: false,
              });
            } else if (event === "done") {
              let payload: DonePayload = {
                used_chunk_uids: [],
                context_chunks: [],
              };
              try {
                payload = JSON.parse(data) as DonePayload;
              } catch {
                // legacy empty done
              }
              const usedSet = new Set(payload.used_chunk_uids ?? []);
              setMessages((prev) =>
                prev.map((m) => {
                  if (m.id !== assistantId) return m;
                  return {
                    ...m,
                    streaming: false,
                    usedChunks: payload.context_chunks ?? [],
                    sources: m.sources
                      ? markCitedSources(m.sources, usedSet)
                      : m.sources,
                  };
                }),
              );
            }
          }
        }

        updateAssistant({ streaming: false });
      } catch (e) {
        if ((e as Error).name === "AbortError") return;
        let message = e instanceof Error ? e.message : String(e);
        if (message === "Failed to fetch" || message.includes("NetworkError")) {
          message =
            `Cannot reach API at ${API_BASE}. Start the backend: ` +
            "`conda run -n Data python -m Scripts.api` from the project root.";
        }
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, error: message, streaming: false }
              : m,
          ),
        );
      } finally {
        setLoading(false);
        abortRef.current = null;
      }
    },
    [loading, setMessages],
  );

  return { loading, sendMessage, abortStream };
}

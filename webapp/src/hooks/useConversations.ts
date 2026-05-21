import { useCallback, useEffect, useMemo, useState } from "react";
import type { ChatMessage, Conversation, HistoryTurn } from "../types";
import {
  loadConversations,
  saveConversations,
  titleFromQuery,
} from "../lib/conversationStorage";
import { useChatStream } from "./useChatStream";

function newConversation(): Conversation {
  const now = Date.now();
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
}

function buildHistory(messages: ChatMessage[]): HistoryTurn[] {
  const turns: HistoryTurn[] = [];
  let i = 0;
  while (i < messages.length) {
    const msg = messages[i];
    if (msg.role !== "user" || msg.streaming || msg.error) {
      i += 1;
      continue;
    }
    const userText = msg.content.trim();
    if (!userText) {
      i += 1;
      continue;
    }
    const next = messages[i + 1];
    if (
      !next ||
      next.role !== "assistant" ||
      next.streaming ||
      next.error ||
      !next.content.trim()
    ) {
      i += 1;
      continue;
    }
    turns.push({
      user: userText,
      assistant: next.content.trim(),
      context_chunks: next.usedChunks ?? [],
    });
    i += 2;
  }
  return turns;
}

export function useConversations() {
  const [conversations, setConversations] = useState<Conversation[]>(() => {
    const loaded = loadConversations();
    return loaded.length > 0 ? loaded : [newConversation()];
  });
  const [activeId, setActiveId] = useState<string>(() => conversations[0]?.id ?? "");

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? conversations[0],
    [conversations, activeId],
  );

  const setActiveMessages = useCallback(
    (updater: ChatMessage[] | ((prev: ChatMessage[]) => ChatMessage[])) => {
      setConversations((prev) => {
        const id = activeId || prev[0]?.id;
        if (!id) return prev;
        return prev.map((c) => {
          if (c.id !== id) return c;
          const nextMessages =
            typeof updater === "function" ? updater(c.messages) : updater;
          return {
            ...c,
            messages: nextMessages,
            updatedAt: Date.now(),
          };
        });
      });
    },
    [activeId],
  );

  const { loading, sendMessage: streamSend, abortStream } = useChatStream(
    setActiveMessages,
  );

  useEffect(() => {
    saveConversations(conversations);
  }, [conversations]);

  useEffect(() => {
    if (!conversations.some((c) => c.id === activeId) && conversations[0]) {
      setActiveId(conversations[0].id);
    }
  }, [conversations, activeId]);

  const createConversation = useCallback(() => {
    const c = newConversation();
    setConversations((prev) => [c, ...prev]);
    setActiveId(c.id);
    abortStream();
    return c.id;
  }, [abortStream]);

  const selectConversation = useCallback(
    (id: string) => {
      setActiveId(id);
      abortStream();
    },
    [abortStream],
  );

  const deleteConversation = useCallback(
    (id: string) => {
      setConversations((prev) => {
        const next = prev.filter((c) => c.id !== id);
        if (next.length === 0) {
          const fresh = newConversation();
          setActiveId(fresh.id);
          return [fresh];
        }
        if (id === activeId) {
          setActiveId(next[0].id);
        }
        return next;
      });
      abortStream();
    },
    [activeId, abortStream],
  );

  const sendMessage = useCallback(
    async (query: string) => {
      const trimmed = query.trim();
      if (!trimmed) return;

      const priorMessages = activeConversation?.messages ?? [];
      const history = buildHistory(priorMessages);

      setConversations((prev) =>
        prev.map((c) => {
          if (c.id !== activeId) return c;
          const title =
            c.messages.length === 0 ? titleFromQuery(trimmed) : c.title;
          return { ...c, title, updatedAt: Date.now() };
        }),
      );

      await streamSend(trimmed, history);
    },
    [activeId, activeConversation?.messages, streamSend],
  );

  const sortedConversations = useMemo(
    () => [...conversations].sort((a, b) => b.updatedAt - a.updatedAt),
    [conversations],
  );

  return {
    conversations: sortedConversations,
    activeConversation,
    activeId: activeConversation?.id ?? "",
    loading,
    sendMessage,
    createConversation,
    selectConversation,
    deleteConversation,
  };
}

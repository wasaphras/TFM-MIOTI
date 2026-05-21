import type { Conversation } from "../types";

const STORAGE_KEY = "tfm-rag-conversations-v2";

export function loadConversations(): Conversation[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Conversation[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function saveConversations(conversations: Conversation[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
  } catch {
    // quota exceeded — ignore
  }
}

export function titleFromQuery(query: string): string {
  const t = query.trim().replace(/\s+/g, " ");
  if (t.length <= 48) return t || "New chat";
  return `${t.slice(0, 45)}…`;
}

export interface ChunkSource {
  rank: number;
  chunk_uid: string;
  rerank_score: number | null;
  snippet: string;
  cited?: boolean;
}

export interface DocumentSource {
  celex_id: string;
  categories_en: string;
  eurlex_url: string;
  document_preview: string;
  chunks: ChunkSource[];
}

export interface ContextChunk {
  chunk_uid: string;
  celex_id: string;
  categories_en: string;
  text: string;
}

export interface HistoryTurn {
  user: string;
  assistant: string;
  context_chunks: ContextChunk[];
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: DocumentSource[];
  usedChunks?: ContextChunk[];
  error?: string;
  streaming?: boolean;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
  updatedAt: number;
}

export interface DonePayload {
  used_chunk_uids: string[];
  context_chunks: ContextChunk[];
}

import { useEffect, useRef, useState, type FormEvent } from "react";
import { useConversations } from "../hooks/useConversations";
import { MessageList } from "./MessageList";
import { Sidebar } from "./Sidebar";

export function Chat() {
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  const {
    conversations,
    activeConversation,
    activeId,
    loading,
    sendMessage,
    createConversation,
    selectConversation,
    deleteConversation,
  } = useConversations();

  const messages = activeConversation?.messages ?? [];

  useEffect(() => {
    const el = listRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages, activeId]);

  const submitQuery = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    await sendMessage(q);
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void submitQuery();
  };

  const handleNewChat = () => {
    createConversation();
    setSidebarOpen(false);
  };

  return (
    <div className="app-shell">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelect={selectConversation}
        onNew={handleNewChat}
        onDelete={deleteConversation}
      />

      <div className="main-panel">
        <header className="main-header">
          <button
            type="button"
            className="btn-icon-only sidebar-toggle"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open chat history"
          >
            ☰
          </button>
          <div className="header-titles">
            <h1>{activeConversation?.title ?? "New chat"}</h1>
            <p className="header-sub">
              Grounded answers from EU legal documents with cited sources
            </p>
          </div>
        </header>

        <main className="chat-main" ref={listRef}>
          <div className="chat-main-inner">
            <MessageList messages={messages} loading={loading} />
          </div>
        </main>

        <footer className="composer">
          <form className="composer-form" onSubmit={onSubmit}>
            <div className="composer-box">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about regulations, directives, CELEX documents…"
                rows={1}
                disabled={loading}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    void submitQuery();
                  }
                }}
              />
              <button
                type="submit"
                className="btn-send"
                disabled={loading || !input.trim()}
                aria-label="Send message"
              >
                {loading ? (
                  <span className="spinner" aria-hidden />
                ) : (
                  <span aria-hidden>↑</span>
                )}
              </button>
            </div>
            <p className="composer-hint">
              Shift+Enter for newline · Answers use retrieved context only
            </p>
          </form>
        </footer>
      </div>
    </div>
  );
}

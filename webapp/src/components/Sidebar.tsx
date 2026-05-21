import type { Conversation } from "../types";

interface SidebarProps {
  conversations: Conversation[];
  activeId: string;
  open: boolean;
  onClose: () => void;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

function formatRelativeTime(ts: number): string {
  const diff = Date.now() - ts;
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days}d ago`;
  return new Date(ts).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

export function Sidebar({
  conversations,
  activeId,
  open,
  onClose,
  onSelect,
  onNew,
  onDelete,
}: SidebarProps) {
  return (
    <>
      <div
        className={`sidebar-backdrop${open ? " sidebar-backdrop-visible" : ""}`}
        onClick={onClose}
        aria-hidden={!open}
      />
      <aside className={`sidebar${open ? " sidebar-open" : ""}`}>
        <div className="sidebar-brand">
          <span className="brand-icon" aria-hidden>
            E
          </span>
          <div>
            <span className="brand-title">EURAGLEX</span>
            <span className="brand-sub">EU legislation assistant</span>
          </div>
        </div>

        <button type="button" className="btn-new-chat" onClick={onNew}>
          <span className="btn-icon" aria-hidden>
            +
          </span>
          New conversation
        </button>

        <nav className="sidebar-nav" aria-label="Chat history">
          <p className="sidebar-section-label">Recent</p>
          <ul className="conversation-list">
            {conversations.map((c) => {
              const isActive = c.id === activeId;
              const preview =
                c.messages.find((m) => m.role === "user")?.content ??
                "No messages yet";
              return (
                <li key={c.id}>
                  <button
                    type="button"
                    className={`conversation-item${isActive ? " active" : ""}`}
                    onClick={() => {
                      onSelect(c.id);
                      onClose();
                    }}
                  >
                    <span className="conversation-title">{c.title}</span>
                    <span className="conversation-preview">{preview}</span>
                    <span className="conversation-time">
                      {formatRelativeTime(c.updatedAt)}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="conversation-delete"
                    title="Delete conversation"
                    aria-label={`Delete ${c.title}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      if (window.confirm("Delete this conversation?")) {
                        onDelete(c.id);
                      }
                    }}
                  >
                    ×
                  </button>
                </li>
              );
            })}
          </ul>
        </nav>

        <footer className="sidebar-footer">
          <p>Retrieval: dedup corpus · hybrid + rerank</p>
          <p>Answers: Gemini · sources traceable</p>
        </footer>
      </aside>
    </>
  );
}

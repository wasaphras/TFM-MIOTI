import type { ChatMessage } from "../types";
import { MarkdownContent } from "./MarkdownContent";
import { SourcePanel } from "./SourcePanel";

interface MessageListProps {
  messages: ChatMessage[];
  loading?: boolean;
}

export function MessageList({ messages, loading }: MessageListProps) {
  if (messages.length === 0) {
    return (
      <div className="empty-state">
        <div className="empty-icon" aria-hidden>
          E
        </div>
        <h2>How can I help you today?</h2>
        <p className="empty-lead">
          Ask questions about EU legislation. Each answer is grounded in
          retrieved document chunks with traceable sources.
        </p>
        <ul className="empty-suggestions">
          <li>What are the labeling rules for eggs sold to consumers?</li>
          <li>How is lobster size measured in EU fisheries regulations?</li>
          <li>When can export price be based on resale price from China?</li>
        </ul>
        {loading && <p className="empty-loading">Retrieving context…</p>}
      </div>
    );
  }

  return (
    <div className="message-list">
      {messages.map((msg) => (
        <article
          key={msg.id}
          className={`message message-${msg.role}${msg.error ? " message-error" : ""}`}
        >
          <div className="message-avatar" aria-hidden>
            {msg.role === "user" ? "U" : "AI"}
          </div>
          <div className="message-column">
            <header className="message-header">
              <span className="message-role">
                {msg.role === "user" ? "You" : "Assistant"}
              </span>
              {msg.streaming && (
                <span className="message-status">Generating…</span>
              )}
            </header>
            <div className="message-body">
              {msg.error ? (
                <p className="message-error-text">{msg.error}</p>
              ) : msg.role === "assistant" ? (
                <div className="message-content-wrap">
                  {(msg.content || !msg.streaming) && (
                    <MarkdownContent
                      content={msg.content || "—"}
                    />
                  )}
                  {msg.streaming && (
                    <span className="cursor-blink" aria-hidden>
                      ▌
                    </span>
                  )}
                </div>
              ) : (
                <div className="message-content">{msg.content}</div>
              )}
              {msg.role === "assistant" &&
                msg.sources &&
                msg.sources.length > 0 && (
                  <SourcePanel sources={msg.sources} />
                )}
            </div>
          </div>
        </article>
      ))}
    </div>
  );
}

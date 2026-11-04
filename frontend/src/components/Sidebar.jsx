export default function Sidebar({
  chats = [],
  currentId,
  open,
  onNew,
  onOpen,
  onDelete,
}) {
  return (
    <aside className={`sidebar ${open ? 'open' : ''}`}>

      <div className="sidebar-brand">
        <div className="brand-logo">
          <span>✦</span>
        </div>

        <div className="brand-details">
          <h2>NEXORA <span>AI</span></h2>
          <p>Where Ideas Become Intelligence</p>
        </div>
      </div>

      <button
        type="button"
        className="new-chat"
        onClick={onNew}
      >
        <span className="new-chat-icon">＋</span>
        <span>New Chat</span>
      </button>

      <div className="section-title">
        <span>RECENT CONVERSATIONS</span>
      </div>

      <div className="chat-list">
        {chats.length === 0 ? (
          <div className="empty-history">
            Your conversations will appear here.
          </div>
        ) : (
          chats.map((chat) => (
            <div
              key={chat.id}
              className={`chat-item ${
                chat.id === currentId ? 'active' : ''
              }`}
              onClick={() => onOpen(chat.id)}
              role="button"
              tabIndex={0}
              onKeyDown={(event) => {
                if (event.key === 'Enter') {
                  onOpen(chat.id)
                }
              }}
            >
              <span className="history-icon">◷</span>

              <span className="title">
                {chat.title}
              </span>

              <button
                type="button"
                className="del"
                title="Delete chat"
                aria-label="Delete chat"
                onClick={(event) => {
                  event.stopPropagation()
                  onDelete(chat.id)
                }}
              >
                ×
              </button>
            </div>
          ))
        )}
      </div>

      <div className="sidebar-footer">
        <span className="footer-status"></span>
        <span>NEXORA AI</span>
        <span className="footer-version">AI ASSISTANT</span>
      </div>

    </aside>
  )
}
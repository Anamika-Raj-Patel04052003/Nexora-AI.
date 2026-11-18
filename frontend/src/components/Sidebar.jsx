import { useState } from 'react'

export default function Sidebar({
  chats = [],
  currentId,
  open,
  onNew,
  onOpen,
  onDelete,
}) {
  const [searchQuery, setSearchQuery] = useState('')

  // Filter conversations based on search input
  const filteredChats = chats.filter((chat) =>
    (chat.title || '').toLowerCase().includes(searchQuery.toLowerCase().trim())
  )

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
        onClick={() => {
          setSearchQuery('')
          onNew()
        }}
      >
        <span className="new-chat-icon">＋</span>
        <span>New Chat</span>
      </button>

      {/* Search Bar - Styled to match existing theme */}
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          width: '100%',
          minHeight: '38px',
          padding: '0 10px',
          marginTop: '14px',
          marginBottom: '2px',
          border: '1px solid rgba(151, 126, 255, 0.22)',
          borderRadius: '10px',
          background: 'rgba(255, 255, 255, 0.04)',
          boxSizing: 'border-box',
          transition: 'border-color 0.2s',
        }}
      >
        <span style={{ color: '#858da3', fontSize: '13px' }}>🔍</span>
        <input
          type="text"
          placeholder="Search chats..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{
            flex: 1,
            minWidth: 0,
            background: 'transparent',
            border: 'none',
            outline: 'none',
            color: '#f4f5fb',
            fontSize: '12px',
            padding: '7px 0',
          }}
        />
        {searchQuery && (
          <button
            type="button"
            onClick={() => setSearchQuery('')}
            style={{
              background: 'transparent',
              border: 'none',
              color: '#858da3',
              fontSize: '14px',
              cursor: 'pointer',
              padding: '0 2px',
            }}
            title="Clear search"
          >
            ×
          </button>
        )}
      </div>

      <div className="section-title">
        <span>{searchQuery ? 'SEARCH RESULTS' : 'RECENT CONVERSATIONS'}</span>
      </div>

      <div className="chat-list">
        {chats.length === 0 ? (
          <div className="empty-history">
            Your conversations will appear here.
          </div>
        ) : filteredChats.length === 0 ? (
          <div className="empty-history">
            No chats found matching "{searchQuery}".
          </div>
        ) : (
          filteredChats.map((chat) => (
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
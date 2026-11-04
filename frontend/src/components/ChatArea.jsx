import { useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

export default function ChatArea({
  messages = [],
  busy = false,
  error = '',
  onSend,
  onAskDocument,
  onUploadDocument,
  onUploadMedia,
  documentReady = false,
  documentName = '',
  onMenu,
}) {
  const [text, setText] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const [pendingFile, setPendingFile] = useState(null)
  const [mediaType, setMediaType] = useState('')

  const documentInput = useRef(null)
  const imageInput = useRef(null)
  const audioInput = useRef(null)
  const videoInput = useRef(null)
  const textareaRef = useRef(null)

  function selectMedia(event, type) {
    const file = event.target.files?.[0]
    if (!file) return

    setPendingFile(file)
    setMediaType(type)
    setMenuOpen(false)
    event.target.value = ''
  }

  async function selectDocument(event) {
    const file = event.target.files?.[0]
    if (!file) return

    setMenuOpen(false)

    try {
      await onUploadDocument(file)
    } catch {
      // App.jsx handles the upload error.
    }

    event.target.value = ''
  }

  function removeAttachment() {
    setPendingFile(null)
    setMediaType('')
  }

  async function handleSend() {
    if (busy) return

    const message = text.trim()

    if (!message && !pendingFile) return

    if (pendingFile) {
      const file = pendingFile
      const type = mediaType

      setPendingFile(null)
      setMediaType('')
      setText('')

      await onUploadMedia(file, type, message)
      return
    }

    setText('')

    if (documentReady) {
      await onAskDocument(message)
    } else {
      await onSend(message)
    }
  }

  function handleKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      handleSend()
    }
  }

  function handleTextareaChange(event) {
    setText(event.target.value)

    const textarea = event.target
    textarea.style.height = 'auto'
    textarea.style.height =
      `${Math.min(textarea.scrollHeight, 160)}px`
  }

  const canSend =
    !busy && (text.trim().length > 0 || Boolean(pendingFile))

  return (
    <main className="chat-area">

      {/* Header */}
      <header className="chat-header">
        <button
          type="button"
          className="mobile-menu-btn"
          onClick={onMenu}
          aria-label="Open sidebar"
        >
          ☰
        </button>

        <div className="header-brand">
          <h2>NEXORA <span>AI</span></h2>
          <p>Your intelligent AI workspace</p>
        </div>

        <div className="header-status">
          <span></span>
          AI READY
        </div>
      </header>

      {/* Messages */}
      <section className="messages-container">

        {messages.length === 0 ? (
          <div className="welcome-screen">

            <div className="ai-assistant">

              <div className="assistant-orbit orbit-one"></div>
              <div className="assistant-orbit orbit-two"></div>

              <div className="assistant-glow"></div>

              <div className="assistant-core">
                <div className="assistant-antenna"></div>

                <div className="assistant-face">
                  <div className="assistant-eyes">
                    <span></span>
                    <span></span>
                  </div>

                  <div className="assistant-mouth"></div>
                </div>

                <div className="assistant-side assistant-side-left"></div>
                <div className="assistant-side assistant-side-right"></div>
              </div>

              <span className="assistant-spark spark-one">✦</span>
              <span className="assistant-spark spark-two">✧</span>
              <span className="assistant-spark spark-three">✦</span>

            </div>

            <div className="welcome-brand">
              <span className="welcome-label">
                WELCOME TO NEXORA
              </span>

              <h1>
                Think Beyond.
                <br />
                <span>Create Without Limits.</span>
              </h1>

              <p>
                Your ideas deserve more than ordinary answers.
                <br />
                Let's make something extraordinary.
              </p>
            </div>

            <div className="welcome-capabilities">

              <div className="capability-card">
                <span className="capability-icon">✧</span>
                <span>Ask Anything</span>
              </div>

              <div className="capability-card">
                <span className="capability-icon">▤</span>
                <span>Explore Documents</span>
              </div>

              <div className="capability-card">
                <span className="capability-icon">◎</span>
                <span>Understand Media</span>
              </div>

            </div>

          </div>
        ) : (
          <div className="conversation-content">
            {messages.map((message, index) => (
              <div
                className={`message-row ${message.role}`}
                key={`${index}-${message.role}`}
              >
                <div className="message-avatar">
                  {message.role === 'user' ? 'U' : '✦'}
                </div>

                <div className="message-content">
                  {message.content ? (
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {message.content}
                    </ReactMarkdown>
                  ) : (
                    <span className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {busy && (
          <div className="processing-indicator">
            <span className="processing-dot"></span>
            NEXORA is thinking...
          </div>
        )}

      </section>

      {/* Error */}
      {error && (
        <div className="error-message">
          <span>⚠</span>
          {error}
        </div>
      )}

      {/* Document status */}
      {documentReady && (
        <div className="document-status">
          <span>▤</span>
          <span>Document active:</span>
          <strong>{documentName || 'Uploaded document'}</strong>
        </div>
      )}

      {/* Composer */}
      <section className="chat-input-section">

        {pendingFile && (
          <div className="attachment-preview">
            <div className="attachment-file-icon">
              {mediaType === 'image' ? '▧' :
                mediaType === 'audio' ? '♫' : '▣'}
            </div>

            <div className="attachment-preview-info">
              <strong>{mediaType.toUpperCase()} ATTACHED</strong>
              <span>{pendingFile.name}</span>
            </div>

            <button
              type="button"
              onClick={removeAttachment}
              disabled={busy}
              aria-label="Remove attachment"
            >
              ×
            </button>
          </div>
        )}

        <div className="chat-input-wrapper">

          {/* Attachment menu */}
          <div className="attachment-menu-wrapper">

            <button
              type="button"
              className={`plus-btn ${menuOpen ? 'menu-active' : ''}`}
              onClick={() => setMenuOpen((value) => !value)}
              disabled={busy}
              aria-label="Add attachment"
              title="Attach a file"
            >
              {menuOpen ? '×' : '+'}
            </button>

            {menuOpen && (
              <div className="attachment-menu">

                <button
                  type="button"
                  onClick={() => documentInput.current?.click()}
                >
                  <span className="menu-icon document-icon">▤</span>
                  <span>
                    <strong>Documents</strong>
                    <small>PDF, TXT, DOCX, CSV, MD</small>
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => imageInput.current?.click()}
                >
                  <span className="menu-icon image-icon">▧</span>
                  <span>
                    <strong>Images</strong>
                    <small>JPG, PNG, WEBP</small>
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => audioInput.current?.click()}
                >
                  <span className="menu-icon audio-icon">♫</span>
                  <span>
                    <strong>Audio</strong>
                    <small>MP3, WAV, M4A and more</small>
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => videoInput.current?.click()}
                >
                  <span className="menu-icon video-icon">▷</span>
                  <span>
                    <strong>Videos</strong>
                    <small>MP4, WEBM</small>
                  </span>
                </button>

              </div>
            )}
          </div>

          <textarea
            ref={textareaRef}
            className="chat-textarea"
            placeholder={
              pendingFile
                ? 'Ask NEXORA AI about this file...'
                : documentReady
                  ? 'Ask a question about your document...'
                  : 'Message NEXORA AI...'
            }
            value={text}
            onChange={handleTextareaChange}
            onKeyDown={handleKeyDown}
            disabled={busy}
            rows={1}
          />

          <button
            type="button"
            className="send-btn"
            onClick={handleSend}
            disabled={!canSend}
            aria-label="Send message"
          >
            {busy ? '...' : '↑'}
          </button>

        </div>

        <p className="input-hint">
          NEXORA AI can make mistakes. Verify important information.
          <br />
          Enter to send · Shift + Enter for a new line
        </p>

      </section>

      {/* Hidden file inputs */}
      <input
        ref={documentInput}
        type="file"
        accept=".pdf,.txt,.docx,.csv,.md"
        hidden
        onChange={selectDocument}
      />

      <input
        ref={imageInput}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        hidden
        onChange={(event) => selectMedia(event, 'image')}
      />

      <input
        ref={audioInput}
        type="file"
        accept=".mp3,.wav,.m4a,.ogg,.flac,.webm"
        hidden
        onChange={(event) => selectMedia(event, 'audio')}
      />

      <input
        ref={videoInput}
        type="file"
        accept=".mp4,.webm"
        hidden
        onChange={(event) => selectMedia(event, 'video')}
      />

    </main>
  )
}
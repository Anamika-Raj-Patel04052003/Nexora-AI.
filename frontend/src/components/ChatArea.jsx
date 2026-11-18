import { useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

export default function ChatArea({
  messages = [],
  busy = false,
  error = '',
  onSend,
  onAskDocument,
  onUploadDocument,
  onUploadMedia,
  onGenerateBlog,
  documentReady = false,
  documentName = '',
  onMenu,
}) {
  const [text, setText] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const [pendingFile, setPendingFile] = useState(null)
  const [mediaType, setMediaType] = useState('')

  // Creative Studio Modal State
  const [creativeOpen, setCreativeOpen] = useState(false)
  const [blogTopic, setBlogTopic] = useState('')
  const [blogTone, setBlogTone] = useState('Professional')
  const [blogLength, setBlogLength] = useState('Medium')
  const [includeImages, setIncludeImages] = useState(true)
  const [includeAudio, setIncludeAudio] = useState(false)
  const [includeVideo, setIncludeVideo] = useState(false)

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

  async function handleCreateBlogSubmit(e) {
    e.preventDefault()
    if (!blogTopic.trim() || busy) return

    const params = {
      topic: blogTopic.trim(),
      tone: blogTone,
      length: blogLength,
      include_images: includeImages,
      include_audio: includeAudio,
      include_video: includeVideo,
    }

    setCreativeOpen(false)
    setBlogTopic('')

    if (onGenerateBlog) {
      await onGenerateBlog(params)
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
    textarea.style.height = `${Math.min(textarea.scrollHeight, 160)}px`
  }

  const canSend = !busy && (text.trim().length > 0 || Boolean(pendingFile))

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

              <div className="capability-card" onClick={() => setText('Explain quantum computing with formulas')}>
                <span className="capability-icon">✧</span>
                <span>Ask Anything</span>
              </div>

              <div className="capability-card" onClick={() => documentInput.current?.click()}>
                <span className="capability-icon">▤</span>
                <span>Explore Documents</span>
              </div>

              <div className="capability-card" onClick={() => setCreativeOpen(true)}>
                <span className="capability-icon">✦</span>
                <span>Creative Studio</span>
              </div>

              <div className="capability-card" onClick={() => imageInput.current?.click()}>
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
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm, remarkMath]}
                      rehypePlugins={[rehypeKatex]}
                      components={{
                        // Custom link renderer to embed Audio and Video players inline
                        a: ({ node, href, children, ...props }) => {
                          if (href && (href.endsWith('.mp3') || href.includes('/creative/audio/'))) {
                            return (
                              <div style={{ marginTop: '8px', marginBottom: '8px' }}>
                                <audio controls src={href} style={{ width: '100%', maxWidth: '380px' }} />
                                <div><a href={href} target="_blank" rel="noreferrer" {...props}>{children}</a></div>
                              </div>
                            )
                          }
                          if (href && (href.endsWith('.mp4') || href.includes('/creative/videos/'))) {
                            return (
                              <div style={{ marginTop: '8px', marginBottom: '8px' }}>
                                <video controls src={href} style={{ width: '100%', maxWidth: '520px', borderRadius: '10px' }} />
                                <div><a href={href} target="_blank" rel="noreferrer" {...props}>{children}</a></div>
                              </div>
                            )
                          }
                          return <a href={href} target="_blank" rel="noreferrer" {...props}>{children}</a>
                        }
                      }}
                    >
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
                  onClick={() => {
                    setMenuOpen(false)
                    setCreativeOpen(true)
                  }}
                >
                  <span className="menu-icon" style={{ color: '#ffb355', background: 'rgba(255,179,85,.15)' }}>✦</span>
                  <span>
                    <strong>Creative Studio</strong>
                    <small>Blog, Formulas, Audio, Video</small>
                  </span>
                </button>

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

      {/* Creative Studio Modal - Native Theme */}
      {creativeOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'rgba(0,0,0,0.7)',
            backdropFilter: 'blur(5px)',
            padding: '15px',
          }}
          onClick={() => setCreativeOpen(false)}
        >
          <div
            style={{
              width: '100%',
              maxWidth: '520px',
              background: '#191d29',
              border: '1px solid rgba(151,126,255,.3)',
              borderRadius: '16px',
              padding: '24px',
              boxShadow: '0 20px 50px rgba(0,0,0,0.6)',
              color: '#f4f5fb',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '18px' }}>
              <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '700', color: '#a994ff' }}>
                ✦ NEXORA Creative Studio
              </h3>
              <button
                type="button"
                onClick={() => setCreativeOpen(false)}
                style={{ background: 'transparent', border: 'none', color: '#9ca2b6', fontSize: '20px', cursor: 'pointer' }}
              >
                ×
              </button>
            </div>

            <form onSubmit={handleCreateBlogSubmit}>
              <div style={{ marginBottom: '14px' }}>
                <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', marginBottom: '6px', color: '#c5c9d6' }}>
                  Blog Topic / Concepts:
                </label>
                <input
                  type="text"
                  placeholder="e.g. Deep Learning Transformers, Quantum Computing, Financial ROI..."
                  value={blogTopic}
                  onChange={(e) => setBlogTopic(e.target.value)}
                  required
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    borderRadius: '8px',
                    border: '1px solid rgba(255,255,255,0.12)',
                    background: '#10121a',
                    color: '#fff',
                    outline: 'none',
                    fontSize: '13px',
                  }}
                />
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px', marginBottom: '14px' }}>
                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', marginBottom: '6px', color: '#c5c9d6' }}>
                    Tone:
                  </label>
                  <select
                    value={blogTone}
                    onChange={(e) => setBlogTone(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '9px 10px',
                      borderRadius: '8px',
                      border: '1px solid rgba(255,255,255,0.12)',
                      background: '#10121a',
                      color: '#fff',
                      fontSize: '13px',
                    }}
                  >
                    <option value="Professional">Professional</option>
                    <option value="Engaging">Engaging & Casual</option>
                    <option value="Academic">Academic & Scientific</option>
                    <option value="Executive">Executive & Strategic</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: '12px', fontWeight: '600', marginBottom: '6px', color: '#c5c9d6' }}>
                    Length:
                  </label>
                  <select
                    value={blogLength}
                    onChange={(e) => setBlogLength(e.target.value)}
                    style={{
                      width: '100%',
                      padding: '9px 10px',
                      borderRadius: '8px',
                      border: '1px solid rgba(255,255,255,0.12)',
                      background: '#10121a',
                      color: '#fff',
                      fontSize: '13px',
                    }}
                  >
                    <option value="Short">Short (~500 words)</option>
                    <option value="Medium">Medium (~1000 words)</option>
                    <option value="Long">Long (~1800 words)</option>
                  </select>
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', marginBottom: '20px', padding: '12px', background: '#12141d', borderRadius: '10px' }}>
                <label style={{ display: 'flex', alignItems: 'center', gap: '9px', fontSize: '12px', color: '#d6d9e5', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={includeImages}
                    onChange={(e) => setIncludeImages(e.target.checked)}
                  />
                  <span>Generate In-Between AI Images & Figures</span>
                </label>

                <label style={{ display: 'flex', alignItems: 'center', gap: '9px', fontSize: '12px', color: '#d6d9e5', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={includeAudio}
                    onChange={(e) => setIncludeAudio(e.target.checked)}
                  />
                  <span>Generate Audio Narration Podcast (Edge-TTS)</span>
                </label>

                <label style={{ display: 'flex', alignItems: 'center', gap: '9px', fontSize: '12px', color: '#d6d9e5', cursor: 'pointer' }}>
                  <input
                    type="checkbox"
                    checked={includeVideo}
                    onChange={(e) => setIncludeVideo(e.target.checked)}
                  />
                  <span>Compile MP4 Video Slideshow (MoviePy)</span>
                </label>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                <button
                  type="button"
                  onClick={() => setCreativeOpen(false)}
                  style={{
                    padding: '8px 16px',
                    borderRadius: '8px',
                    border: '1px solid rgba(255,255,255,0.1)',
                    background: 'transparent',
                    color: '#c5c9d6',
                    cursor: 'pointer',
                    fontSize: '13px',
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={!blogTopic.trim() || busy}
                  style={{
                    padding: '8px 20px',
                    borderRadius: '8px',
                    border: 'none',
                    background: 'var(--gradient)',
                    color: '#fff',
                    fontWeight: '600',
                    cursor: !blogTopic.trim() || busy ? 'not-allowed' : 'pointer',
                    opacity: !blogTopic.trim() || busy ? 0.5 : 1,
                    fontSize: '13px',
                  }}
                >
                  {busy ? 'Generating...' : 'Create Publication'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

    </main>
  )
}
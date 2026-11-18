import { useEffect, useState } from 'react'
import Sidebar from './components/Sidebar'
import ChatArea from './components/ChatArea'
import * as api from './api'

const LAST_KEY = 'nexoraLastChat'

// Extract readable text from API responses
function getResponseText(
  result,
  fallback = 'The file was processed successfully.'
) {
  if (typeof result === 'string') {
    return result
  }

  const candidates = [
    result?.answer,
    result?.response,
    result?.transcript,
    result?.text,
    result?.transcription,
    result?.summary,
    result?.message,
  ]

  const value = candidates.find(
    (item) => typeof item === 'string' && item.trim()
  )

  return value || fallback
}

export default function App() {
  const [chats, setChats] = useState([])
  const [currentId, setCurrentId] = useState(null)
  const [messages, setMessages] = useState([])
  const [busy, setBusy] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [error, setError] = useState('')

  const [documentReady, setDocumentReady] = useState(false)
  const [documentName, setDocumentName] = useState('')

  // Refresh chat history
  async function refreshChats() {
    const list = await api.listChats()
    setChats(list)
    return list
  }

  // Open an existing conversation
  async function openChat(id) {
    if (busy) return

    try {
      const msgs = await api.getMessages(id)

      setCurrentId(id)
      localStorage.setItem(LAST_KEY, String(id))

      setMessages(
        msgs.map((message) => ({
          role: message.role,
          content: message.content,
        }))
      )

      const uploadedMessage = msgs.find(
        (message) =>
          message.role === 'user' &&
          message.content?.startsWith('[Document uploaded:')
      )

      setDocumentReady(Boolean(uploadedMessage))

      if (uploadedMessage) {
        const match = uploadedMessage.content.match(
          /^\[Document uploaded: (.+)\]$/
        )

        setDocumentName(
          match ? match[1] : 'Uploaded document'
        )
      } else {
        setDocumentName('')
      }

      setError('')
      setSidebarOpen(false)
    } catch (err) {
      setError('Could not open chat: ' + err.message)
    }
  }

  // Initialize application
  useEffect(() => {
    async function init() {
      try {
        const list = await refreshChats()
        const last = Number(localStorage.getItem(LAST_KEY))

        if (last && list.some((chat) => chat.id === last)) {
          await openChat(last)
        }
      } catch {
        setError(
          'Could not reach the server. Is the backend running?'
        )
      }
    }

    init()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Start a new chat
  function newChat() {
    if (busy) return

    setCurrentId(null)
    setMessages([])
    setDocumentReady(false)
    setDocumentName('')

    localStorage.removeItem(LAST_KEY)

    setSidebarOpen(false)
    setError('')
  }

  // Delete a conversation
  async function removeChat(id) {
    if (busy) return

    if (!window.confirm('Delete this chat permanently?')) {
      return
    }

    try {
      await api.deleteChat(id)

      if (id === currentId) {
        newChat()
      }

      await refreshChats()
    } catch (err) {
      setError('Could not delete chat: ' + err.message)
    }
  }

  // Create a conversation when required
  async function ensureConversation() {
    if (currentId !== null) {
      return currentId
    }

    const conversation = await api.createChat()
    const id = conversation.id

    if (id === undefined || id === null) {
      throw new Error(
        'The server did not return a conversation ID.'
      )
    }

    setCurrentId(id)
    localStorage.setItem(LAST_KEY, String(id))

    await refreshChats().catch(() => {})

    return id
  }

  // Normal text chat with auto document-aware graph streaming
  async function send(text) {
    if (busy || !text.trim()) return

    setBusy(true)
    setError('')

    try {
      const id = await ensureConversation()

      setMessages((prev) => [
        ...prev,
        { role: 'user', content: text },
        { role: 'assistant', content: '' },
      ])

      await api.sendMessage(id, text, (soFar) => {
        setMessages((prev) => {
          const copy = [...prev]

          copy[copy.length - 1] = {
            role: 'assistant',
            content: soFar,
          }

          return copy
        })
      })
    } catch (err) {
      setError('Message failed: ' + err.message)

      setMessages((prev) => {
        const copy = [...prev]

        if (
          copy[copy.length - 1]?.role === 'assistant' &&
          !copy[copy.length - 1].content
        ) {
          copy.pop()
        }

        return copy
      })
    } finally {
      setBusy(false)
      refreshChats().catch(() => {})
    }
  }

  // Upload documents
  async function handleDocumentUpload(file) {
    if (busy || !file) return

    setBusy(true)
    setError('')

    try {
      const id = await ensureConversation()
      const result = await api.uploadDocument(id, file)

      const filename = result.filename || file.name

      setMessages((prev) => [
        ...prev,
        {
          role: 'user',
          content: `[Document uploaded: ${filename}]`,
        },
      ])

      setDocumentReady(true)
      setDocumentName(filename)

      await refreshChats()

      return result
    } catch (err) {
      setError('Document upload failed: ' + err.message)
      throw err
    } finally {
      setBusy(false)
    }
  }

  // Ask questions from uploaded documents
  async function askDocument(text) {
    if (busy || !text.trim()) return

    setBusy(true)
    setError('')

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: text },
      { role: 'assistant', content: '' },
    ])

    try {
      const id = await ensureConversation()
      const result = await api.askDocumentQuestion(id, text)

      let answer = getResponseText(
        result,
        'No answer was returned.'
      )

      if (result.sources?.length) {
        const sourceText = result.sources
          .map(
            (source) =>
              `- 📄 ${source.filename || source.source || 'Unknown'}`
          )
          .join('\n')

        answer += `\n\n**Document Sources:**\n${sourceText}`
      }

      if (result.web_sources?.length) {
        const webText = result.web_sources
          .map(
            (web) =>
              `- 🌐 [${web.title || 'Web Result'}](${web.url})`
          )
          .join('\n')

        answer += `\n\n**Web Sources:**\n${webText}`
      }

      setMessages((prev) => {
        const copy = [...prev]

        copy[copy.length - 1] = {
          role: 'assistant',
          content: answer,
        }

        return copy
      })
    } catch (err) {
      setError('Document question failed: ' + err.message)

      setMessages((prev) => {
        const copy = [...prev]

        copy[copy.length - 1] = {
          role: 'assistant',
          content: 'Unable to answer the document question.',
        }

        return copy
      })
    } finally {
      setBusy(false)
      refreshChats().catch(() => {})
    }
  }

  // Multimodal Image, audio and video processing
  async function handleMediaUpload(file, mediaType, question) {
    if (busy || !file) return

    setBusy(true)
    setError('')

    const userQuestion =
      typeof question === 'string' ? question.trim() : ''

    const userText = userQuestion
      ? `${userQuestion}\n\n[Attached ${mediaType}: ${file.name}]`
      : `[Attached ${mediaType}: ${file.name}]`

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: userText },
      { role: 'assistant', content: '' },
    ])

    try {
      const id = await ensureConversation()
      let result

      if (mediaType === 'image') {
        result = await api.uploadImage(
          id,
          file,
          userQuestion || 'Describe this image in detail.'
        )
      } else if (mediaType === 'audio') {
        result = await api.uploadAudio(id, file)
      } else if (mediaType === 'video') {
        result = await api.uploadVideo(
          id,
          file,
          userQuestion || 'Summarize this video.'
        )
      } else {
        throw new Error('Unsupported attachment type.')
      }

      let answer

      if (mediaType === 'audio') {
        answer = getResponseText(
          result,
          'Audio was processed, but no transcript was returned.'
        )

        answer = `**Audio transcription:**\n\n${answer}`
      } else {
        answer = getResponseText(
          result,
          'The media was processed, but no answer was returned.'
        )
      }

      setMessages((prev) => {
        const copy = [...prev]

        copy[copy.length - 1] = {
          role: 'assistant',
          content: answer,
        }

        return copy
      })
    } catch (err) {
      setError(
        `${mediaType} processing failed: ${err.message}`
      )

      setMessages((prev) => {
        const copy = [...prev]

        copy[copy.length - 1] = {
          role: 'assistant',
          content: `Unable to process this ${mediaType}. Please try again.`,
        }

        return copy
      })
    } finally {
      setBusy(false)
      refreshChats().catch(() => {})
    }
  }

  // Creative Blog & Media Generation Handler
  async function handleGenerateBlog(options) {
    if (busy) return

    setBusy(true)
    setError('')

    const userPromptText = `[Creative Studio: ${options.topic}] (Tone: ${options.tone}, Length: ${options.length})`

    setMessages((prev) => [
      ...prev,
      { role: 'user', content: userPromptText },
      { role: 'assistant', content: '✦ Crafting your publication with formulas, tables and media...' },
    ])

    try {
      const id = await ensureConversation()
      const result = await api.generateBlog(options)

      let answer = result.content || 'Blog generated successfully.'

      if (result.audio_url) {
        answer += `\n\n**🎙 Audio Podcast Narration:**\n[Listen to Narration](${result.audio_url})`
      }

      if (result.video_url) {
        answer += `\n\n**🎬 MP4 Video Slideshow:**\n[Watch Video Slideshow](${result.video_url})`
      }

      setMessages((prev) => {
        const copy = [...prev]

        copy[copy.length - 1] = {
          role: 'assistant',
          content: answer,
        }

        return copy
      })
    } catch (err) {
      setError('Blog generation failed: ' + err.message)

      setMessages((prev) => {
        const copy = [...prev]

        copy[copy.length - 1] = {
          role: 'assistant',
          content: 'Unable to generate blog. Please check backend services.',
        }

        return copy
      })
    } finally {
      setBusy(false)
      refreshChats().catch(() => {})
    }
  }

  // Application UI
  return (
    <div className="app">
      <Sidebar
        chats={chats}
        currentId={currentId}
        open={sidebarOpen}
        onNew={newChat}
        onOpen={openChat}
        onDelete={removeChat}
      />

      <ChatArea
        messages={messages}
        busy={busy}
        error={error}
        onSend={send}
        onAskDocument={askDocument}
        onUploadDocument={handleDocumentUpload}
        onUploadMedia={handleMediaUpload}
        onGenerateBlog={handleGenerateBlog}
        documentReady={documentReady}
        documentName={documentName}
        onMenu={() => setSidebarOpen((value) => !value)}
      />
    </div>
  )
}
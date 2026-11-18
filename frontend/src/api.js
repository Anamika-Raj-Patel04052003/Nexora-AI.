// Line 1 in frontend/src/api.js:
const BASE = import.meta.env.VITE_API_URL || '/api'

async function request(path, options = {}) {
  const res = await fetch(BASE + path, options)

  if (!res.ok) {
    let message = `Request failed (${res.status})`

    try {
      const data = await res.json()
      message = data.detail || message
    } catch {
      // Keep default error message
    }

    throw new Error(message)
  }

  return res
}

// ==========================================
// CONVERSATIONS & CHAT
// ==========================================

export async function listChats() {
  return (await request('/conversations')).json()
}

export async function createChat() {
  return (await request('/conversations', { method: 'POST' })).json()
}

export async function getMessages(id) {
  return (await request(`/conversations/${id}/messages`)).json()
}

export async function deleteChat(id) {
  await request(`/conversations/${id}`, { method: 'DELETE' })
}

// Sends a message and calls onChunk(textSoFar) while the reply streams in
export async function sendMessage(id, message, onChunk) {
  const res = await request(`/conversations/${id}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message }),
  })

  if (!res.body) {
    throw new Error('Streaming response is not supported by the browser.')
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()

  let full = ''

  try {
    while (true) {
      const { done, value } = await reader.read()

      if (done) break

      full += decoder.decode(value, { stream: true })
      onChunk(full)
    }

    full += decoder.decode()
    onChunk(full)

    return full
  } finally {
    reader.releaseLock()
  }
}

// ==========================================
// DOCUMENT UPLOAD & RAG
// ==========================================

export async function uploadDocument(conversationId, file) {
  const formData = new FormData()
  formData.append('file', file)

  const res = await request(
    `/conversations/${conversationId}/documents/upload`,
    {
      method: 'POST',
      body: formData,
    }
  )

  return res.json()
}

export async function askDocumentQuestion(conversationId, question) {
  const res = await request(
    `/conversations/${conversationId}/documents/ask`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        question,
      }),
    }
  )

  return res.json()
}

// ==========================================
// MULTIMODAL MEDIA UPLOAD (VISION & AUDIO)
// ==========================================

export async function uploadImage(conversationId, file, question) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('question', question || 'Describe this image in detail.')

  const res = await request(
    `/conversations/${conversationId}/media/image`,
    {
      method: 'POST',
      body: formData,
    }
  )

  return res.json()
}

export async function uploadAudio(conversationId, file) {
  const formData = new FormData()
  formData.append('file', file)

  const res = await request(
    `/conversations/${conversationId}/media/audio/transcribe`,
    {
      method: 'POST',
      body: formData,
    }
  )

  return res.json()
}

export async function uploadVideo(conversationId, file, question) {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('question', question || 'Summarize this video.')

  const res = await request(
    `/conversations/${conversationId}/media/video`,
    {
      method: 'POST',
      body: formData,
    }
  )

  return res.json()
}

// ==========================================
// CREATIVE STUDIO APIS (BLOG, IMAGE, AUDIO, VIDEO)
// ==========================================

/**
 * Generate a full publication blog with optional formulas, tables, inline images, audio & video
 */
export async function generateBlog({
  topic,
  language = 'English',
  tone = 'Professional',
  length = 'Medium',
  include_images = true,
  image_mode = 'cloud',
  include_audio = false,
  voice = 'en-US-AriaNeural',
  include_video = false,
}) {
  const res = await request('/creative/blog', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      topic,
      language,
      tone,
      length,
      include_images,
      image_mode,
      include_audio,
      voice,
      include_video,
    }),
  })

  return res.json()
}

/**
 * Generate standalone image (cloud or local)
 */
export async function generateStandaloneImage(prompt, mode = 'cloud') {
  const res = await request('/creative/image', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      prompt,
      mode,
    }),
  })

  return res.json()
}

/**
 * Generate standalone audio narration using Edge-TTS
 */
export async function generateStandaloneAudio(text, voice = 'en-US-AriaNeural') {
  const res = await request('/creative/audio', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text,
      voice,
    }),
  })

  return res.json()
}

/**
 * Compile video slideshow from generated images and audio
 */
export async function generateStandaloneVideo(image_filenames, audio_filename) {
  const res = await request('/creative/video', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      image_filenames,
      audio_filename,
    }),
  })

  return res.json()
}
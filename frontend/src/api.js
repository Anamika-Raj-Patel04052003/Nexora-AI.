
const BASE = '/api'

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
    throw new Error('Streaming response is not supported.')
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
// DOCUMENT UPLOAD API
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


// ==========================================
// ASK QUESTION FROM UPLOADED DOCUMENT
// ==========================================

export async function askDocumentQuestion(
  conversationId,
  question
) {

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
// IMAGE UPLOAD API
// ==========================================

export async function uploadImage(conversationId, file, question) {
  const formData = new FormData()

  formData.append('file', file)
  formData.append('question', question || 'Describe this image.')

  const res = await request(
    `/conversations/${conversationId}/media/image`,
    {
      method: 'POST',
      body: formData,
    }
  )

  return res.json()
}


// ==========================================
// AUDIO TRANSCRIPTION API
// ==========================================

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


// ==========================================
// VIDEO UPLOAD API
// ==========================================

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


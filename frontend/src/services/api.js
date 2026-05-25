// frontend/src/services/api.js
const BASE = '/api';   // proxied by Vite to http://localhost:8000

// ── RAG endpoints ─────────────────────────────────────────────────
export async function uploadPolicy(file) {
  const form = new FormData();
  form.append('file', file);
  const res = await fetch(`${BASE}/v1/upload`, { method: 'POST', body: form });
  if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed');
  return res.json(); // { file_id, filename, pages, status }
}

export async function listFiles() {
  const res = await fetch(`${BASE}/v1/files`);
  if (!res.ok) throw new Error('Failed to load files');
  return res.json();
}

export async function queryPolicy(fileId, questions) {
  const res = await fetch(`${BASE}/v1/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ file_id: fileId, questions }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || 'Query failed');
  return res.json();
}

// ── Chat / claims endpoints ───────────────────────────────────────
export async function startSession({ fileId, policyNumber, claimantName }) {
  const res = await fetch(`${BASE}/v1/chat/session`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      file_id:       fileId,
      policy_number: policyNumber,
      claimant_name: claimantName,
    }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || 'Session start failed');
  return res.json(); // { session_id, greeting }
}

export async function sendMessage(sessionId, message) {
  const res = await fetch(`${BASE}/v1/chat/message`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) throw new Error((await res.json()).detail || 'Message failed');
  return res.json();
}

/**
 * Async generator that yields SSE chunks from the stream endpoint.
 * Each chunk: { token, done, type } or { done:true, claim_id, mode }
 */
export async function* streamMessage(sessionId, message) {
  const url = `${BASE}/v1/chat/${sessionId}/stream?message=${encodeURIComponent(message)}`;
  const res  = await fetch(url);
  if (!res.ok) throw new Error('Stream request failed');

  const reader  = res.body.getReader();
  const decoder = new TextDecoder();
  let   buffer  = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try { yield JSON.parse(line.slice(6)); } catch { /* skip */ }
    }
  }
}
const BASE = '/api';

const req = (url, opts = {}) =>
  fetch(BASE + url, opts).then(r => {
    if (!r.ok) return r.json().then(e => { throw new Error(e.detail || 'Request failed'); });
    if (r.status === 204) return null;
    return r.json();
  });

// ── Notebooks ──────────────────────────────────────────────────────────────────
export const fetchNotebooks   = ()            => req('/notebooks').then(d => d.notebooks);
export const createNotebook   = (title, emoji)=> req('/notebooks', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({title, emoji}) }).then(d=>d.notebook);
export const renameNotebook   = (id, title)   => req(`/notebooks/${id}`, { method:'PATCH', headers:{'Content-Type':'application/json'}, body: JSON.stringify({title}) }).then(d=>d.notebook);
export const deleteNotebook   = (id)          => req(`/notebooks/${id}`, { method:'DELETE' });
export const touchNotebook    = (id)          => req(`/notebooks/${id}/touch`, { method:'POST' });

// ── Conversations ──────────────────────────────────────────────────────────────
export const fetchConversations  = (nbId)       => req(`/notebooks/${nbId}/conversations`).then(d=>d.conversations);
export const createConversation  = (nbId)       => req(`/notebooks/${nbId}/conversations`, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({title:'New conversation'}) }).then(d=>d.conversation);
export const renameConversation  = (cId, title) => req(`/conversations/${cId}`, { method:'PATCH', headers:{'Content-Type':'application/json'}, body:JSON.stringify({title}) }).then(d=>d.conversation);
export const deleteConversation = (cid) =>
  req(`/conversations/${cid}`, { method: 'DELETE' });
export const fetchMessages       = (cId)        => req(`/conversations/${cId}/messages`).then(d=>d.messages);
export const sendFeedback = (messageId, rating) =>
  req(`/messages/${messageId}/feedback`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({rating})
  }).then(d => d.feedback);

export const fetchFeedback = (convId) =>
  req(`/conversations/${convId}/feedback`).then(d => d.feedback);

// ── Sources ────────────────────────────────────────────────────────────────────
export const fetchSources = (nbId)       => req(`/sources/${nbId}`).then(d=>d.sources);
export const deleteSource = (sid) =>
  req(`/sources/${sid}`, { method: 'DELETE' });
export const retrySource = (sid) =>
  req(`/sources/${sid}/retry`, { method: 'POST' });
export const uploadFiles  = (nbId, files) => {
  const form = new FormData();
  form.append('notebook_id', nbId);
  files.forEach(f => form.append('files', f));
  return fetch(BASE + '/upload', { method:'POST', body:form })
    .then(r => r.json()).then(d => d.sources);
};

// ── Ask (SSE streaming) ────────────────────────────────────────────────────────
export async function streamAsk({ question, conversationId, mode='standard', topK=5,
  checkedSources=[], mentionedSources=[],
  onStep, onToken, onSources, onDone, onError }) {

  const r = await fetch(BASE + '/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question, conversation_id: conversationId, mode, top_k: topK,
      checked_sources: checkedSources, mentioned_sources: mentionedSources,
    }),
  });

  if (!r.ok) { onError?.('Server error. Is the backend running?'); return; }

  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      try {
        const ev = JSON.parse(line.slice(6));
        if (ev.type === 'step')    onStep?.(ev);
        if (ev.type === 'token')   onToken?.(ev.content);
        if (ev.type === 'sources') onSources?.(ev.content);
        if (ev.type === 'done')    onDone?.();
        if (ev.type === 'error')   onError?.(ev.content);
      } catch {}
    }
  }
}

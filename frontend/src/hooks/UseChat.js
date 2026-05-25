// frontend/src/hooks/useChat.js
import { useState, useCallback, useRef } from 'react';
import { startSession, streamMessage } from '../services/api';

export function useChat() {
  const [sessionId,   setSessionId]   = useState(null);
  const [messages,    setMessages]    = useState([]);
  const [mode,        setMode]        = useState('chat');
  const [isStreaming, setIsStreaming] = useState(false);
  const [isStarting,  setIsStarting] = useState(false);
  const [claimId,     setClaimId]    = useState(null);
  const [error,       setError]      = useState(null);
  const abortRef = useRef(false);

  const start = useCallback(async ({ fileId, policyNumber, claimantName }) => {
    setIsStarting(true); setError(null);
    try {
      const data = await startSession({ fileId, policyNumber, claimantName });
      setSessionId(data.session_id);
      setMessages([{ id: 1, role: 'assistant', content: data.greeting, type: 'text', done: true }]);
      return data.session_id;
    } catch (e) {
      setError(e.message); throw e;
    } finally {
      setIsStarting(false);
    }
  }, []);

  const send = useCallback(async (text) => {
    if (!sessionId || isStreaming) return;
    setIsStreaming(true);
    abortRef.current = false;

    // User bubble
    const userId = Date.now();
    setMessages(prev => [...prev, { id: userId, role: 'user', content: text, type: 'text', done: true }]);

    // Placeholder assistant bubble
    const aiId = userId + 1;
    setMessages(prev => [...prev, { id: aiId, role: 'assistant', content: '', type: 'text', done: false }]);

    try {
      let full = '', meta = null;
      for await (const chunk of streamMessage(sessionId, text)) {
        if (abortRef.current) break;
        if (chunk.done) {
          meta = chunk;
        } else {
          full += chunk.token;
          setMessages(prev => prev.map(m =>
            m.id === aiId ? { ...m, content: full, type: chunk.type || 'text' } : m
          ));
        }
      }
      setMessages(prev => prev.map(m =>
        m.id === aiId ? { ...m, content: full, type: meta?.type || 'text', done: true } : m
      ));
      if (meta?.mode)     setMode(meta.mode);
      if (meta?.claim_id) setClaimId(meta.claim_id);
      return meta;
    } catch (e) {
      setMessages(prev => prev.map(m =>
        m.id === aiId ? { ...m, content: 'Something went wrong. Please try again.', done: true } : m
      ));
      setError(e.message);
    } finally {
      setIsStreaming(false);
    }
  }, [sessionId, isStreaming]);

  const reset = useCallback(() => {
    abortRef.current = true;
    setSessionId(null); setMessages([]); setMode('chat');
    setClaimId(null); setError(null); setIsStreaming(false);
  }, []);

  return { sessionId, messages, mode, isStreaming, isStarting, claimId, error, start, send, reset };
}
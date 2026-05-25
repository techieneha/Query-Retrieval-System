// frontend/src/App.jsx
import { useRef, useEffect, useState } from 'react';
import Sidebar          from './components/Sidebar.jsx';
import TopBar           from './components/TopBar.jsx';
import Message          from './components/Message.jsx';
import TypingIndicator  from './components/TypingIndicator.jsx';
import ChatInput        from './components/ChatInput.jsx';
import EmptyState       from './components/EmptyState.jsx';
import { useChat }      from './hooks/UseChat.js';

const S = {
  app: {
    display: 'grid',
    gridTemplateColumns: '272px 1fr',
    height: '100vh',
    overflow: 'hidden',
  },
  chatArea: {
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    background: '#f7f3ec',
    overflow: 'hidden',
  },
  messages: {
    flex: 1,
    overflowY: 'auto',
    padding: '24px 28px 12px',
    display: 'flex',
    flexDirection: 'column',
    gap: 20,
  },
  toast: (show) => ({
    position: 'fixed', bottom: '1.8rem', right: '1.8rem',
    background: '#12100e', color: '#f7f3ec',
    padding: '10px 18px', borderRadius: 10,
    fontSize: '0.82rem', fontFamily: 'DM Sans, sans-serif',
    borderLeft: '3px solid #c99034',
    opacity: show ? 1 : 0,
    transform: show ? 'translateY(0)' : 'translateY(8px)',
    transition: 'all 0.25s',
    pointerEvents: 'none', zIndex: 99,
    maxWidth: 320,
  }),
};

export default function App() {
  const {
    sessionId, messages, mode, isStreaming,
    isStarting, claimId, error, start, send, reset,
  } = useChat();

  const [sessionMeta, setSessionMeta] = useState({ policyNumber:'', claimantName:'' });
  const [toast,       setToast]       = useState({ show: false, msg: '' });
  const bottomRef = useRef(null);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isStreaming]);

  // Show error as toast
  useEffect(() => {
    if (error) showToast('❌ ' + error);
  }, [error]);

  function showToast(msg, ms = 3500) {
    setToast({ show: true, msg });
    setTimeout(() => setToast({ show: false, msg: '' }), ms);
  }

  async function handleSessionStart({ fileId, policyNumber, claimantName }) {
    try {
      await start({ fileId, policyNumber, claimantName });
      setSessionMeta({ policyNumber, claimantName });
      showToast('✅ Connected! Ask me anything about your policy.');
    } catch (e) {
      showToast('❌ ' + e.message);
      throw e;
    }
  }

  async function handleSend(text) {
    if (!sessionId) return;
    await send(text);
  }

  function handleQuickReply(text) {
    if (sessionId && !isStreaming) send(text);
  }

  return (
    <div style={S.app}>
      {/* Sidebar */}
      <Sidebar
        onSessionStart={handleSessionStart}
        onSuggestion={text => !isStreaming && sessionId && send(text)}
        connected={!!sessionId}
      />

      {/* Main chat */}
      <div style={S.chatArea}>
        <TopBar
          policyNumber={sessionMeta.policyNumber}
          claimantName={sessionMeta.claimantName}
          mode={mode}
          claimId={claimId}
          sessionId={sessionId}
        />

        <div style={S.messages}>
          {messages.length === 0 && !sessionId
            ? <EmptyState />
            : messages.map(msg => (
                <Message
                  key={msg.id}
                  message={msg}
                  onQuickReply={handleQuickReply}
                />
              ))
          }
          {/* Typing indicator while streaming starts */}
          {isStreaming && messages.at(-1)?.role !== 'assistant' && <TypingIndicator />}
          <div ref={bottomRef} />
        </div>

        <ChatInput
          onSend={handleSend}
          disabled={!sessionId || isStreaming}
        />
      </div>

      {/* Toast */}
      <div style={S.toast(toast.show)}>{toast.msg}</div>
    </div>
  );
}
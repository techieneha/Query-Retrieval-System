// frontend/src/components/ChatInput.jsx
import { useState, useRef } from 'react';

const S = {
  area: {
    padding: '14px 24px 18px',
    borderTop: '1px solid #e2e8f0',
    background: 'white',
    flexShrink: 0,
  },
  row: { display: 'flex', gap: 10, alignItems: 'flex-end' },
  textarea: {
    flex: 1,
    background: '#f8fafc',
    border: '1px solid #cbd5e1',
    borderRadius: 12,
    padding: '10px 14px',
    fontSize: '0.9rem',
    fontFamily: 'Inter, system-ui, sans-serif',
    color: '#0f172a',
    resize: 'none',
    maxHeight: 120,
    outline: 'none',
    lineHeight: 1.55,
    transition: 'border-color 0.2s',
  },
  btn: (disabled) => ({
    width: 42,
    height: 42,
    borderRadius: 12,
    flexShrink: 0,
    background: disabled ? '#cbd5e1' : '#3b82f6',
    color: 'white',
    border: 'none',
    cursor: disabled ? 'not-allowed' : 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '1.2rem',
    fontWeight: 600,
    transition: 'background 0.2s',
  }),
  hint: {
    fontSize: '0.7rem',
    color: '#64748b',
    marginTop: 8,
    textAlign: 'center',
  },
};

export default function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState('');
  const ref = useRef();

  function resize() {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
  }

  function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  function submit() {
    const t = text.trim();
    if (!t || disabled) return;
    onSend(t);
    setText('');
    if (ref.current) {
      ref.current.style.height = 'auto';
    }
  }

  return (
    <div style={S.area}>
      <div style={S.row}>
        <textarea
          ref={ref}
          style={S.textarea}
          value={text}
          placeholder={
            disabled
              ? 'Start a session to begin chatting…'
              : 'Ask about your policy or say "I want to file a claim"…'
          }
          disabled={disabled}
          rows={1}
          onChange={(e) => {
            setText(e.target.value);
            resize();
          }}
          onKeyDown={handleKey}
          onFocus={(e) => {
            e.target.style.borderColor = '#3b82f6';
          }}
          onBlur={(e) => {
            e.target.style.borderColor = '#cbd5e1';
          }}
        />
        <button style={S.btn(!text.trim() || disabled)} onClick={submit}>
          ↑
        </button>
      </div>
      <div style={S.hint}>Enter to send · Shift+Enter for new line</div>
    </div>
  );
}
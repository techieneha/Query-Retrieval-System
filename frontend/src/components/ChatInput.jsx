// frontend/src/components/ChatInput.jsx
import { useState, useRef } from 'react';

const S = {
  area: {
    padding: '14px 24px 18px',
    borderTop: '1px solid #d8d3ca',
    background: '#ede9e0',
    flexShrink: 0,
  },
  row: { display:'flex', gap:10, alignItems:'flex-end' },
  textarea: {
    flex: 1, background: 'white', border: '1px solid #c8c3b8',
    borderRadius: 14, padding: '10px 14px',
    fontSize: '0.9rem', fontFamily: 'DM Sans, sans-serif',
    color: '#12100e', resize: 'none', maxHeight: 120, outline: 'none',
    lineHeight: 1.55, transition: 'border-color 0.2s',
  },
  btn: (disabled) => ({
    width: 42, height: 42, borderRadius: 12, flexShrink: 0,
    background: disabled ? '#c8c3b8' : '#1a6b6b',
    color: 'white', border: 'none', cursor: disabled ? 'not-allowed' : 'pointer',
    display: 'flex', alignItems: 'center', justifyContent: 'center',
    fontSize: '1.1rem', transition: 'background 0.2s',
  }),
  hint: { fontSize:'0.7rem', color:'#7a756c', marginTop:6, textAlign:'center' },
};

export default function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState('');
  const ref  = useRef();

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
    if (ref.current) { ref.current.style.height = 'auto'; }
  }

  return (
    <div style={S.area}>
      <div style={S.row}>
        <textarea
          ref={ref}
          style={{ ...S.textarea, borderColor: disabled ? '#e0dbd2' : undefined }}
          value={text}
          placeholder={disabled ? 'Start a session to begin chatting…' : 'Ask about your policy or say "I want to file a claim"…'}
          disabled={disabled}
          rows={1}
          onChange={e => { setText(e.target.value); resize(); }}
          onKeyDown={handleKey}
          onFocus={e => { e.target.style.borderColor = '#1a6b6b'; }}
          onBlur={e  => { e.target.style.borderColor = '#c8c3b8'; }}
        />
        <button style={S.btn(!text.trim() || disabled)} onClick={submit}>↑</button>
      </div>
      <div style={S.hint}>Enter to send · Shift+Enter for new line</div>
    </div>
  );
}
// frontend/src/components/Message.jsx
import { useMemo } from 'react';

/* ── Lightweight markdown → HTML (no external deps) ───────────── */
function renderMd(text) {
  return text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    // Tables
    .replace(/^\|(.+)\|\s*\n\|[-| :]+\|\s*\n((?:\|.+\|\s*\n?)*)/gm, (_, head, body) => {
      const ths = head.split('|').filter(c => c.trim()).map(c => `<th>${c.trim()}</th>`).join('');
      const trs = body.trim().split('\n').filter(Boolean).map(row => {
        const tds = row.split('|').filter(c => c.trim()).map(c => `<td>${c.trim()}</td>`).join('');
        return `<tr>${tds}</tr>`;
      }).join('');
      return `<table><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table>`;
    })
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/_(.+?)_/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^• (.+)$/gm, '<li>$1</li>')
    .replace(/^→ (.+)$/gm, '<li class="arrow">→ $1</li>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br/>')
    .trim();
}

const styles = {
  row: (isUser) => ({
    display: 'flex',
    gap: 10,
    alignItems: 'flex-end',
    flexDirection: isUser ? 'row-reverse' : 'row',
    animation: 'fadeUp 0.22s ease forwards',
  }),
  avatar: (isUser) => ({
    width: 32,
    height: 32,
    borderRadius: '50%',
    flexShrink: 0,
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    fontSize: '0.88rem',
    fontWeight: 700,
    background: isUser ? '#3b82f6' : '#2563eb',
    color: 'white',
    fontFamily: 'Inter, sans-serif',
    border: isUser ? 'none' : '1px solid rgba(59,130,246,0.3)',
  }),
  bubble: (isUser, type) => ({
    maxWidth: '68%',
    padding: '12px 16px',
    borderRadius: isUser ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
    fontSize: '0.9rem',
    lineHeight: 1.68,
    background: isUser ? '#3b82f6' : (
      type === 'coverage_card' ? '#e0f2fe' :   // light blue for coverage card
      type === 'claim_summary' ? '#f0f9ff' :   // very light blue for claim summary
      'white'
    ),
    color: isUser ? 'white' : '#0f172a',
    border: isUser ? 'none' : `1px solid ${
      type === 'coverage_card' ? 'rgba(59,130,246,0.2)' :
      type === 'claim_summary' ? 'rgba(59,130,246,0.15)' : '#e2e8f0'
    }`,
    boxShadow: isUser ? 'none' : '0 1px 6px rgba(0,0,0,0.07)',
    wordBreak: 'break-word',
  }),
  cursor: {
    display: 'inline-block',
    width: 2,
    height: '1em',
    background: '#3b82f6',
    marginLeft: 2,
    animation: 'pulse 0.9s ease infinite',
    verticalAlign: 'text-bottom',
  },
};

// Injected CSS for markdown elements inside bubbles
const mdCSS = `
  .ai-bubble strong { font-weight: 600; }
  .ai-bubble em     { font-style: italic; color: #64748b; }
  .ai-bubble code   { background:#f1f5f9; padding:1px 5px; border-radius:4px;
                       font-family:'JetBrains Mono',monospace; font-size:0.82rem; }
  .ai-bubble table  { border-collapse:collapse; width:100%; margin:8px 0; font-size:0.83rem; }
  .ai-bubble td,
  .ai-bubble th     { padding:5px 10px; border:1px solid #cbd5e1; text-align:left; }
  .ai-bubble th     { background:#f8fafc; font-weight:600; }
  .ai-bubble li     { margin-left:1.2rem; }
  .ai-bubble li.arrow { list-style:none; margin-left:0; }
  .ai-bubble p      { margin:4px 0; }
  @keyframes fadeUp { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
  @keyframes pulse  { 0%,100%{opacity:1} 50%{opacity:0.3} }
  @keyframes typingBounce { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-6px)} }
`;

// Inject once
if (typeof document !== 'undefined' && !document.getElementById('md-css')) {
  const el = document.createElement('style');
  el.id = 'md-css';
  el.textContent = mdCSS;
  document.head.appendChild(el);
}

export default function Message({ message, onQuickReply }) {
  const isUser = message.role === 'user';
  const html = useMemo(() => renderMd(message.content), [message.content]);

  const quickReplies = message.done && !isUser && message.type === 'coverage_card'
    ? ['Yes, submit my claim', 'No, let me change something']
    : message.done && !isUser && message.mode === 'claim_intake' && !message.content.includes('submit')
    ? ['Correct, continue', 'Let me correct that', 'Cancel claim']
    : [];

  return (
    <div style={styles.row(isUser)}>
      <div style={styles.avatar(isUser)}>{isUser ? '👤' : 'P'}</div>
      <div>
        <div
          className={isUser ? '' : 'ai-bubble'}
          style={styles.bubble(isUser, message.type)}
        >
          {isUser ? (
            message.content
          ) : (
            <span dangerouslySetInnerHTML={{ __html: html }} />
          )}
          {!message.done && !isUser && <span style={styles.cursor} />}
        </div>

        {quickReplies.length > 0 && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 7, marginTop: 8 }}>
            {quickReplies.map(r => (
              <button
                key={r}
                onClick={() => onQuickReply?.(r)}
                style={{
                  background: 'white',
                  border: '1px solid #cbd5e1',
                  borderRadius: 99,
                  padding: '4px 13px',
                  fontSize: '0.78rem',
                  cursor: 'pointer',
                  fontFamily: 'Inter, sans-serif',
                  color: '#0f172a',
                  transition: 'all 0.15s',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = '#3b82f6';
                  e.currentTarget.style.color = 'white';
                  e.currentTarget.style.borderColor = '#3b82f6';
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = 'white';
                  e.currentTarget.style.color = '#0f172a';
                  e.currentTarget.style.borderColor = '#cbd5e1';
                }}
              >
                {r}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
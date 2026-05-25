// frontend/src/components/Sidebar.jsx
import { useState, useRef } from 'react';
import { uploadPolicy, listFiles } from '../services/api';

const S = {
  sidebar: {
    width: 272, flexShrink: 0, background: '#12100e',
    display: 'flex', flexDirection: 'column',
    borderRight: '3px solid #b07d2a', overflow: 'hidden',
  },
  header: {
    padding: '26px 22px 18px',
    borderBottom: '1px solid rgba(255,255,255,0.07)',
  },
  logo: {
    fontFamily: 'Instrument Serif, Georgia, serif',
    fontSize: '1.7rem', color: '#f7f3ec', lineHeight: 1,
    letterSpacing: '-0.02em',
  },
  logoAccent: { color: '#c99034' },
  logoSub: {
    fontSize: '0.68rem', textTransform: 'uppercase',
    letterSpacing: '0.13em', color: '#7a756c', marginTop: 4,
  },
  body: { padding: '18px 18px', flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 16 },
  section: { display: 'flex', flexDirection: 'column', gap: 8 },
  sectionTitle: {
    fontSize: '0.68rem', textTransform: 'uppercase',
    letterSpacing: '0.12em', color: '#7a756c',
  },
  dropZone: (drag) => ({
    border: `2px dashed ${drag ? '#c99034' : 'rgba(255,255,255,0.15)'}`,
    borderRadius: 10, padding: '14px 12px', textAlign: 'center',
    cursor: 'pointer', color: drag ? '#f7f3ec' : '#7a756c',
    fontSize: '0.82rem', transition: 'all 0.2s',
  }),
  dropIcon: { fontSize: '1.4rem', marginBottom: 4 },
  field: { display: 'flex', flexDirection: 'column', gap: 5 },
  label: { fontSize: '0.75rem', color: '#c8c3b8' },
  input: {
    background: 'rgba(255,255,255,0.07)',
    border: '1px solid rgba(255,255,255,0.12)',
    borderRadius: 8, padding: '8px 10px',
    color: '#f7f3ec', fontSize: '0.83rem',
    fontFamily: 'DM Sans, sans-serif', outline: 'none',
  },
  btn: (disabled) => ({
    background: disabled ? 'rgba(255,255,255,0.08)' : '#1a6b6b',
    color: disabled ? '#7a756c' : 'white',
    border: 'none', borderRadius: 9, padding: '10px 0',
    fontSize: '0.88rem', fontWeight: 600, cursor: disabled ? 'not-allowed' : 'pointer',
    fontFamily: 'DM Sans, sans-serif', transition: 'background 0.2s', width: '100%',
  }),
  chip: {
    background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)',
    borderRadius: 7, padding: '7px 10px', color: '#c8c3b8',
    fontSize: '0.78rem', cursor: 'pointer', transition: 'all 0.15s',
  },
  statusBar: {
    padding: '12px 18px', borderTop: '1px solid rgba(255,255,255,0.06)',
    display: 'flex', alignItems: 'center', gap: 8,
    fontSize: '0.72rem', color: '#7a756c',
  },
  dot: (live) => ({
    width: 7, height: 7, borderRadius: '50%',
    background: live ? '#22c55e' : '#7a756c',
    boxShadow: live ? '0 0 6px #22c55e' : 'none',
    transition: 'all 0.3s', flexShrink: 0,
  }),
};

const SUGGESTIONS = [
  "What's my deductible?",
  "Am I covered for hospitalization?",
  "I want to file a claim",
  "What's excluded from my plan?",
  "What is my coverage limit?",
];

export default function Sidebar({ onSessionStart, onSuggestion, connected }) {
  const [file,          setFile]          = useState(null);
  const [policy,        setPolicy]        = useState('');
  const [name,          setName]          = useState('');
  const [drag,          setDrag]          = useState(false);
  const [uploading,     setUploading]     = useState(false);
  const [uploadMsg,     setUploadMsg]     = useState('Upload policy PDF');
  const [statusMsg,     setStatusMsg]     = useState('Not connected');
  const fileRef = useRef();

  const ready = file && policy.trim() && name.trim();

  function handleFile(f) {
    if (!f || !f.name.toLowerCase().endsWith('.pdf')) return;
    setFile(f);
    setUploadMsg('✅ ' + f.name);
  }

  async function handleStart() {
    if (!ready || uploading) return;
    setUploading(true);
    setStatusMsg('Uploading…');
    try {
      const meta = await uploadPolicy(file);
      setStatusMsg('Starting session…');
      await onSessionStart({ fileId: meta.file_id, policyNumber: policy.trim(), claimantName: name.trim() });
      setStatusMsg('Connected');
    } catch (e) {
      setStatusMsg('Error: ' + e.message);
      setUploading(false);
    }
  }

  return (
    <aside style={S.sidebar}>
      <div style={S.header}>
        <div style={S.logo}>Policy<span style={S.logoAccent}>AI</span></div>
        <div style={S.logoSub}>Claims Assistant</div>
      </div>

      <div style={S.body}>
        {/* Upload */}
        <div style={S.section}>
          <div style={S.sectionTitle}>Your Policy</div>
          <div
            style={S.dropZone(drag)}
            onClick={() => fileRef.current?.click()}
            onDragOver={e => { e.preventDefault(); setDrag(true); }}
            onDragLeave={() => setDrag(false)}
            onDrop={e => { e.preventDefault(); setDrag(false); handleFile(e.dataTransfer.files[0]); }}
          >
            <div style={S.dropIcon}>📄</div>
            <div>{uploadMsg}</div>
            <div style={{ fontSize: '0.72rem', marginTop: 2, opacity: 0.6 }}>PDF only</div>
          </div>
          <input ref={fileRef} type="file" accept=".pdf" style={{ display: 'none' }}
            onChange={e => handleFile(e.target.files[0])} />
        </div>

        {/* Fields */}
        <div style={S.field}>
          <label style={S.label}>Policy Number</label>
          <input style={S.input} value={policy} placeholder="POL-2024-001"
            onChange={e => setPolicy(e.target.value)} />
        </div>
        <div style={S.field}>
          <label style={S.label}>Your Name</label>
          <input style={S.input} value={name} placeholder="Full name"
            onChange={e => setName(e.target.value)} />
        </div>

        <button style={S.btn(!ready || uploading || connected)} disabled={!ready || uploading || connected}
          onClick={handleStart}>
          {uploading ? '⏳ Setting up…' : connected ? '✓ Connected' : 'Start Conversation'}
        </button>

        {/* Suggestions */}
        <div style={S.section}>
          <div style={S.sectionTitle}>Try asking…</div>
          {SUGGESTIONS.map(s => (
            <div key={s} style={S.chip}
              onClick={() => connected && onSuggestion(s)}
              onMouseEnter={e => { e.currentTarget.style.background='rgba(26,107,107,0.2)'; e.currentTarget.style.color='#f7f3ec'; }}
              onMouseLeave={e => { e.currentTarget.style.background='rgba(255,255,255,0.05)'; e.currentTarget.style.color='#c8c3b8'; }}
            >{s}</div>
          ))}
        </div>
      </div>

      <div style={S.statusBar}>
        <div style={S.dot(connected)} />
        <span>{statusMsg}</span>
      </div>
    </aside>
  );
}
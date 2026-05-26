// frontend/src/components/Sidebar.jsx
import { useState, useRef } from 'react';
import { uploadPolicy } from '../services/api';

const S = {
  sidebar: {
    width: 272,
    flexShrink: 0,
    background: 'white',
    borderRight: '1px solid #e2e8f0',
    display: 'flex',
    flexDirection: 'column',
    overflow: 'hidden',
  },
  header: {
    padding: '20px 22px 16px',
    borderBottom: '1px solid #e2e8f0',
  },
  logo: {
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize: '1.3rem',
    fontWeight: 700,
    color: '#1e40af',
    letterSpacing: '-0.02em',
  },
  logoAccent: { color: '#3b82f6' },
  logoSub: {
    fontSize: '0.68rem',
    textTransform: 'uppercase',
    letterSpacing: '0.12em',
    color: '#64748b',
    marginTop: 4,
  },
  body: {
    padding: '18px',
    flex: 1,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    gap: 20,
  },
  section: {
    display: 'flex',
    flexDirection: 'column',
    gap: 10,
  },
  sectionTitle: {
    fontSize: '0.7rem',
    textTransform: 'uppercase',
    letterSpacing: '0.1em',
    color: '#64748b',
    fontWeight: 600,
  },
  dropZone: (drag) => ({
    border: `2px dashed ${drag ? '#3b82f6' : '#cbd5e1'}`,
    borderRadius: 10,
    padding: '16px 12px',
    textAlign: 'center',
    cursor: 'pointer',
    color: drag ? '#1e40af' : '#475569',
    fontSize: '0.82rem',
    transition: 'all 0.2s',
    background: '#f8fafc',
  }),
  dropIcon: { fontSize: '1.3rem', marginBottom: 4 },
  field: {
    display: 'flex',
    flexDirection: 'column',
    gap: 4,
  },
  label: {
    fontSize: '0.75rem',
    color: '#475569',
    fontWeight: 500,
  },
  input: {
    background: '#f8fafc',
    border: '1px solid #cbd5e1',
    borderRadius: 8,
    padding: '8px 10px',
    color: '#0f172a',
    fontSize: '0.85rem',
    fontFamily: 'Inter, sans-serif',
    outline: 'none',
    transition: 'border-color 0.2s',
  },
  btn: (disabled) => ({
    background: disabled ? '#cbd5e1' : '#3b82f6',
    color: disabled ? '#64748b' : 'white',
    border: 'none',
    borderRadius: 8,
    padding: '10px 0',
    fontSize: '0.88rem',
    fontWeight: 600,
    cursor: disabled ? 'not-allowed' : 'pointer',
    fontFamily: 'Inter, sans-serif',
    transition: 'background 0.2s',
    width: '100%',
    marginTop: 6,
  }),
  chip: {
    background: '#f8fafc',
    border: '1px solid #e2e8f0',
    borderRadius: 8,
    padding: '8px 12px',
    color: '#1e293b',
    fontSize: '0.8rem',
    cursor: 'pointer',
    transition: 'all 0.15s',
  },
  statusBar: {
    padding: '12px 18px',
    borderTop: '1px solid #e2e8f0',
    display: 'flex',
    alignItems: 'center',
    gap: 8,
    fontSize: '0.75rem',
    color: '#64748b',
    background: '#f8fafc',
  },
  dot: (live) => ({
    width: 7,
    height: 7,
    borderRadius: '50%',
    background: live ? '#22c55e' : '#94a3b8',
    boxShadow: live ? '0 0 6px #22c55e' : 'none',
    transition: 'all 0.3s',
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
  const [file, setFile] = useState(null);
  const [policy, setPolicy] = useState('');
  const [name, setName] = useState('');
  const [drag, setDrag] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState('Upload policy PDF');
  const [statusMsg, setStatusMsg] = useState('Not connected');
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
        <div style={S.logo}>
          Policy<span style={S.logoAccent}>AI</span>
        </div>
        <div style={S.logoSub}>Claims Assistant</div>
      </div>

      <div style={S.body}>
        {/* Upload */}
        <div style={S.section}>
          <div style={S.sectionTitle}>Your Policy</div>
          <div
            style={S.dropZone(drag)}
            onClick={() => fileRef.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setDrag(true);
            }}
            onDragLeave={() => setDrag(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDrag(false);
              handleFile(e.dataTransfer.files[0]);
            }}
          >
            <div style={S.dropIcon}>📄</div>
            <div>{uploadMsg}</div>
            <div style={{ fontSize: '0.7rem', marginTop: 2, opacity: 0.6 }}>PDF only</div>
          </div>
          <input
            ref={fileRef}
            type="file"
            accept=".pdf"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>

        {/* Fields */}
        <div style={S.field}>
          <label style={S.label}>Policy Number</label>
          <input
            style={S.input}
            value={policy}
            placeholder="POL-2024-001"
            onChange={(e) => setPolicy(e.target.value)}
            onFocus={(e) => (e.target.style.borderColor = '#3b82f6')}
            onBlur={(e) => (e.target.style.borderColor = '#cbd5e1')}
          />
        </div>
        <div style={S.field}>
          <label style={S.label}>Your Name</label>
          <input
            style={S.input}
            value={name}
            placeholder="Full name"
            onChange={(e) => setName(e.target.value)}
            onFocus={(e) => (e.target.style.borderColor = '#3b82f6')}
            onBlur={(e) => (e.target.style.borderColor = '#cbd5e1')}
          />
        </div>

        <button
          style={S.btn(!ready || uploading || connected)}
          disabled={!ready || uploading || connected}
          onClick={handleStart}
        >
          {uploading ? '⏳ Setting up…' : connected ? '✓ Connected' : 'Start Conversation'}
        </button>

        {/* Suggestions */}
        <div style={S.section}>
          <div style={S.sectionTitle}>Try asking…</div>
          {SUGGESTIONS.map((s) => (
            <div
              key={s}
              style={S.chip}
              onClick={() => connected && onSuggestion(s)}
              onMouseEnter={(e) => {
                e.currentTarget.style.background = '#eff6ff';
                e.currentTarget.style.borderColor = '#3b82f6';
                e.currentTarget.style.color = '#1e40af';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.background = '#f8fafc';
                e.currentTarget.style.borderColor = '#e2e8f0';
                e.currentTarget.style.color = '#1e293b';
              }}
            >
              {s}
            </div>
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
// frontend/src/components/TopBar.jsx
const S = {
  bar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '14px 24px', borderBottom: '1px solid #d8d3ca',
    background: '#ede9e0', flexShrink: 0,
  },
  info: { fontSize: '0.82rem', color: '#7a756c' },
  strong: { color: '#12100e', fontWeight: 600 },
  badge: (mode) => ({
    padding: '4px 13px', borderRadius: 99,
    fontSize: '0.7rem', fontWeight: 700,
    letterSpacing: '0.06em', textTransform: 'uppercase',
    background: mode === 'claim_intake' ? 'rgba(176,125,42,0.12)' : 'rgba(26,107,107,0.1)',
    color:      mode === 'claim_intake' ? '#b07d2a' : '#1a6b6b',
  }),
  claimPill: {
    background: '#e6f2eb', border: '1px solid rgba(26,88,48,0.2)',
    color: '#1a5830', borderRadius: 99, padding: '3px 11px',
    fontSize: '0.72rem', fontFamily: "'JetBrains Mono', monospace",
  },
};

export default function TopBar({ policyNumber, claimantName, mode, claimId, sessionId }) {
  const info = claimantName
    ? <span><strong style={S.strong}>{claimantName}</strong> · {policyNumber}</span>
    : <span><strong style={S.strong}>PolicyAI</strong> — Upload a policy to begin</span>;

  return (
    <div style={S.bar}>
      <div style={S.info}>{info}</div>
      <div style={{ display:'flex', gap:8, alignItems:'center' }}>
        {claimId && <span style={S.claimPill}>{claimId}</span>}
        <span style={S.badge(mode)}>
          {mode === 'claim_intake' ? 'Filing Claim' : 'Chat'}
        </span>
      </div>
    </div>
  );
}
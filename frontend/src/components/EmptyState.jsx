// frontend/src/components/EmptyState.jsx
const S = {
  wrap: {
    flex: 1, display:'flex', flexDirection:'column',
    alignItems:'center', justifyContent:'center',
    textAlign:'center', padding:'3rem', color:'#7a756c',
  },
  title: {
    fontFamily: 'Instrument Serif, serif',
    fontSize: '2.2rem', color: '#12100e',
    letterSpacing: '-0.02em', marginBottom: '0.5rem',
  },
  sub: { fontSize: '0.9rem', maxWidth: 300, lineHeight: 1.75 },
  items: { marginTop:'1.5rem', display:'flex', flexDirection:'column', gap:10 },
  item: {
    display:'flex', alignItems:'center', gap:10,
    background:'white', border:'1px solid #e0dbd2',
    borderRadius:12, padding:'10px 16px',
    fontSize:'0.85rem', color:'#12100e', maxWidth:280,
    boxShadow:'0 1px 4px rgba(0,0,0,0.05)',
  },
  icon: { fontSize:'1.1rem', flexShrink:0 },
};

const FEATURES = [
  { icon:'💬', text:'Ask anything about your policy in plain language' },
  { icon:'📋', text:'File a claim step-by-step through conversation' },
  { icon:'🔍', text:'Instant coverage verification before you submit' },
  { icon:'📊', text:'Track claim status with your Claim ID' },
];

export default function EmptyState() {
  return (
    <div style={S.wrap}>
      <div style={S.title}>Hello! 👋</div>
      <p style={S.sub}>Upload your insurance policy PDF on the left to get started.</p>
      <div style={S.items}>
        {FEATURES.map(f => (
          <div key={f.text} style={S.item}>
            <span style={S.icon}>{f.icon}</span>
            <span>{f.text}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
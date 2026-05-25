// frontend/src/components/TypingIndicator.jsx
const S = {
  row:    { display:'flex', gap:10, alignItems:'flex-end' },
  avatar: {
    width:32, height:32, borderRadius:'50%', flexShrink:0,
    display:'flex', alignItems:'center', justifyContent:'center',
    background:'#12100e', color:'#c99034', fontSize:'0.88rem',
    fontFamily:'Instrument Serif, serif', fontWeight:700,
    border:'1px solid rgba(201,144,52,0.3)',
  },
  bubble: {
    background:'white', border:'1px solid #e0dbd2',
    borderRadius:'16px 16px 16px 4px',
    padding:'12px 16px', display:'flex', gap:5, alignItems:'center',
    boxShadow:'0 1px 6px rgba(0,0,0,0.07)',
  },
  dot: (delay) => ({
    width:6, height:6, borderRadius:'50%', background:'#c8c3b8',
    animation:`typingBounce 1.2s ${delay}s infinite`,
  }),
};

export default function TypingIndicator() {
  return (
    <div style={S.row}>
      <div style={S.avatar}>P</div>
      <div style={S.bubble}>
        <span style={S.dot(0)} />
        <span style={S.dot(0.2)} />
        <span style={S.dot(0.4)} />
      </div>
    </div>
  );
}
// frontend/src/components/TopBar.jsx
export default function TopBar({ metrics }) {
  return (
    <div className="top-bar">
      <div className="logo-area">
        <div className="logo-icon">
          <svg viewBox="0 0 24 24">
            <path d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
          </svg>
        </div>
        <div className="logo-text">
          <h1>PolicyAI</h1>
          <p>Insurance Document Intelligence</p>
        </div>
      </div>
      <div className="metrics">
        <div className="metric-card">
          <div className="metric-label">Status</div>
          <div className="metric-value">{metrics.status || 'Ready'}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Latency</div>
          <div className="metric-value">{metrics.latency || '--'}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Cache Hit</div>
          <div className="metric-value">{metrics.cacheHit || '0%'}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Queries</div>
          <div className="metric-value">{metrics.queries || '0'}</div>
        </div>
      </div>
    </div>
  );
}
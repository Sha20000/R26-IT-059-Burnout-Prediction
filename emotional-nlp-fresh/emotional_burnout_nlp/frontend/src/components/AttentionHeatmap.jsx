function getTokenStyle(intensity) {
  if (intensity < 0.2) {
    return { bg: 'rgba(51,65,85,0.7)',   border: 'rgba(51,65,85,0.5)',    text: '#475569', glow: null }
  } else if (intensity < 0.4) {
    return { bg: 'rgba(59,130,246,0.55)', border: 'rgba(59,130,246,0.35)', text: '#93c5fd', glow: 'rgba(59,130,246,0.5)' }
  } else if (intensity < 0.6) {
    return { bg: 'rgba(234,179,8,0.55)',  border: 'rgba(234,179,8,0.35)',  text: '#fde047', glow: 'rgba(234,179,8,0.5)' }
  } else if (intensity < 0.8) {
    return { bg: 'rgba(249,115,22,0.65)', border: 'rgba(249,115,22,0.4)',  text: '#fdba74', glow: 'rgba(249,115,22,0.6)' }
  } else {
    return { bg: 'rgba(239,68,68,0.85)',  border: 'rgba(239,68,68,0.6)',   text: '#ffffff', glow: 'rgba(239,68,68,0.7)' }
  }
}

export default function AttentionHeatmap({ tokens, prediction }) {
  if (!tokens || tokens.length === 0) return null

  const maxWeight = Math.max(...tokens.map(t => t.weight))

  return (
    <div style={{
      background: '#1e293b',
      border: '1px solidrgba(51, 65, 85, 0.01)',
      borderRadius: 10,
      padding: '20px 24px',
    }}>
      <div style={{ marginBottom: 14 }}>
        <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b' }}>
          ATTENTION · XAI EXPLAINABILITY
        </div>
        <div style={{ fontSize: 13, color: '#94a3b8', marginTop: 4 }}>
          Words driving the <span style={{ color: '#f1f5f9', fontWeight: 600 }}>"{prediction}"</span> prediction
        </div>
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, lineHeight: 2 }}>
        {tokens.map(({ token, weight }, i) => {
          const intensity = maxWeight > 0 ? weight / maxWeight : 0
          const s = getTokenStyle(intensity)

          return (
            <span
              key={i}
              title={`"${token}" — attention: ${(weight * 100).toFixed(2)}%`}
              style={{
                background: s.bg,
                color: s.text,
                border: `1px solid ${s.border}`,
                borderRadius: 5,
                padding: '3px 8px',
                fontSize: 13,
                fontFamily: 'monospace',
                cursor: 'default',
                userSelect: 'none',
                transition: 'transform 0.15s, box-shadow 0.15s',
                boxShadow: s.glow ? `0 0 8px ${s.glow}` : 'none',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.transform = 'scale(1.12)'
                e.currentTarget.style.boxShadow = s.glow ? `0 0 16px ${s.glow}` : '0 0 8px rgba(100,116,139,0.4)'
              }}
              onMouseLeave={e => {
                e.currentTarget.style.transform = 'scale(1)'
                e.currentTarget.style.boxShadow = s.glow ? `0 0 8px ${s.glow}` : 'none'
              }}
            >
              {token}
            </span>
          )
        })}
      </div>

      <div style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
        <div style={{ fontSize: 10, color: '#475569' }}>LOW</div>
        <div style={{
          height: 8,
          width: 200,
          borderRadius: 4,
          background: 'linear-gradient(to right, #334155, #3b82f6, #eab308, #f97316, #ef4444)',
          border: '1px solid #334155',
        }} />
        <div style={{ fontSize: 10, color: '#f87171' }}>HIGH ATTENTION</div>
        <div style={{ fontSize: 10, color: '#475569', marginLeft: 4 }}>Hover token for exact %</div>
      </div>

      <div style={{ marginTop: 8, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
        {[
          { label: '0–20%',  color: '#475569', bg: 'rgba(51,65,85,0.7)' },
          { label: '20–40%', color: '#93c5fd', bg: 'rgba(59,130,246,0.55)' },
          { label: '40–60%', color: '#fde047', bg: 'rgba(234,179,8,0.55)' },
          { label: '60–80%', color: '#fdba74', bg: 'rgba(249,115,22,0.65)' },
          { label: '80–100%',color: '#ffffff', bg: 'rgba(239,68,68,0.85)' },
        ].map(({ label, color, bg }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: bg }} />
            <span style={{ fontSize: 10, color }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function AttentionHeatmap({ tokens, prediction }) {
  if (!tokens || tokens.length === 0) return null

  const maxWeight = Math.max(...tokens.map(t => t.weight))

  return (
    <div style={{
      background: '#1e293b',
      border: '1px solid #334155',
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

      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 6,
        lineHeight: 2,
      }}>
        {tokens.map(({ token, weight }, i) => {
          const intensity = maxWeight > 0 ? weight / maxWeight : 0
          const alpha = 0.12 + intensity * 0.88
          const isHigh = intensity > 0.6

          return (
            <span
              key={i}
              title={`"${token}" — attention: ${(weight * 100).toFixed(2)}%`}
              style={{
                background: `rgba(239, 68, 68, ${alpha})`,
                color: isHigh ? '#fff' : '#cbd5e1',
                border: `1px solid rgba(239, 68, 68, ${alpha * 0.5})`,
                borderRadius: 5,
                padding: '3px 8px',
                fontSize: 13,
                fontFamily: 'monospace',
                cursor: 'default',
                userSelect: 'none',
                transition: 'transform 0.15s, box-shadow 0.15s',
                boxShadow: isHigh ? `0 0 8px rgba(239,68,68,${alpha * 0.6})` : 'none',
              }}
              onMouseEnter={e => {
                e.currentTarget.style.transform = 'scale(1.12)'
                e.currentTarget.style.boxShadow = `0 0 14px rgba(239,68,68,0.7)`
              }}
              onMouseLeave={e => {
                e.currentTarget.style.transform = 'scale(1)'
                e.currentTarget.style.boxShadow = isHigh ? `0 0 8px rgba(239,68,68,${alpha * 0.6})` : 'none'
              }}
            >
              {token}
            </span>
          )
        })}
      </div>

      <div style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{ fontSize: 10, color: '#475569' }}>LOW ATTENTION</div>
        <div style={{
          height: 8,
          width: 160,
          borderRadius: 4,
          background: 'linear-gradient(to right, rgba(239,68,68,0.12), rgba(239,68,68,1))',
          border: '1px solid #334155',
        }} />
        <div style={{ fontSize: 10, color: '#f87171' }}>HIGH ATTENTION</div>
        <div style={{ fontSize: 10, color: '#475569', marginLeft: 8 }}>Hover token for exact %</div>
      </div>
    </div>
  )
}

const CLASS_COLORS = {
  Normal:                '#22d3ee',
  Anxiety:               '#fbbf24',
  Stress:                '#f97316',
  Depression:            '#f87171',
  Bipolar:               '#a855f7',
  'Personality disorder': '#ec4899',
  Suicidal:              '#ff2020',
}

export default function ProbabilityBars({ probabilities }) {
  if (!probabilities) return null
  const sorted = Object.entries(probabilities).sort(([, a], [, b]) => b - a)

  return (
    <div style={{
      background: '#1e293b',
      border: '1px solid #334155',
      borderRadius: 10,
      padding: '20px 24px',
    }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 16 }}>
        ALL CLASS PROBABILITIES
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {sorted.map(([label, prob]) => (
          <div key={label}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 5 }}>
              <span style={{ fontSize: 12, color: '#94a3b8' }}>{label}</span>
              <span style={{ fontSize: 12, fontWeight: 600, color: CLASS_COLORS[label] || '#94a3b8' }}>
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ background: '#0f172a', borderRadius: 999, height: 8, overflow: 'hidden' }}>
              <div style={{
                width: `${prob * 100}%`,
                background: CLASS_COLORS[label] || '#94a3b8',
                height: '100%',
                borderRadius: 999,
                transition: 'width 0.7s cubic-bezier(0.4, 0, 0.2, 1)',
                boxShadow: prob > 0.5 ? `0 0 8px ${CLASS_COLORS[label]}80` : 'none',
              }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

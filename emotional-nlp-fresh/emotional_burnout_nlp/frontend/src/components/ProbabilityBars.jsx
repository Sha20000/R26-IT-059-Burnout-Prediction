const CLASS_COLORS = {
  Normal:                '#0891b2',
  Anxiety:               '#b45309',
  Stress:                '#f97316',
  Depression:            '#dc2626',
  Bipolar:               '#a855f7',
  'Personality disorder': '#ec4899',
  Suicidal:              '#ff2020',
}

export default function ProbabilityBars({ probabilities }) {
  if (!probabilities) return null
  const sorted = Object.entries(probabilities).sort(([, a], [, b]) => b - a)

  return (
    <div style={{
      background: '#ffffff',
      border: '1px solid #cbd5e1',
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
              <span style={{ fontSize: 12, color: '#475569' }}>{label}</span>
              <span style={{ fontSize: 12, fontWeight: 600, color: CLASS_COLORS[label] || '#475569' }}>
                {(prob * 100).toFixed(1)}%
              </span>
            </div>
            <div style={{ background: '#f1f5f9', borderRadius: 999, height: 8, overflow: 'hidden' }}>
              <div style={{
                width: `${prob * 100}%`,
                background: CLASS_COLORS[label] || '#cbd5e1',
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

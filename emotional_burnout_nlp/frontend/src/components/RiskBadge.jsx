const RISK_STYLES = {
  Low:      { color: '#22d3ee', bg: '#22d3ee10', border: '#22d3ee40', shadow: '0 0 12px #22d3ee30', label: 'LOW RISK' },
  Medium:   { color: '#fbbf24', bg: '#fbbf2410', border: '#fbbf2440', shadow: '0 0 12px #fbbf2430', label: 'MEDIUM RISK' },
  High:     { color: '#f87171', bg: '#f8717110', border: '#f8717140', shadow: '0 0 12px #f8717130', label: 'HIGH RISK' },
  Critical: { color: '#ff2020', bg: '#ff000015', border: '#ff000060', shadow: 'pulse', label: 'CRITICAL RISK' },
}

export default function RiskBadge({ riskLevel }) {
  const s = RISK_STYLES[riskLevel] || RISK_STYLES.Medium
  const isCritical = riskLevel === 'Critical'

  return (
    <div style={{
      background: s.bg,
      border: `1px solid ${s.border}`,
      borderRadius: 12,
      padding: '20px 28px',
      textAlign: 'center',
      minWidth: 160,
      animation: isCritical ? 'pulse-critical 1.5s ease-in-out infinite' : undefined,
      boxShadow: isCritical ? undefined : s.shadow,
    }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 6 }}>
        RISK LEVEL
      </div>
      <div style={{ fontSize: 22, fontWeight: 800, color: s.color, letterSpacing: 1 }}>
        {isCritical ? '⚠ ' : ''}{s.label}
      </div>
      {isCritical && (
        <div style={{ fontSize: 11, color: '#f87171', marginTop: 8, lineHeight: 1.4 }}>
          Seek immediate mental health support
        </div>
      )}
    </div>
  )
}

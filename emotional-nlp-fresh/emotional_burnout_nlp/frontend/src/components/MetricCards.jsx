function Card({ label, value, sub }) {
  return (
    <div style={{
      background: '#ffffff',
      border: '1px solid #cbd5e1',
      borderRadius: 10,
      padding: '16px 24px',
      flex: 1,
      minWidth: 130,
    }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 8 }}>
        {label}
      </div>
      <div style={{ fontSize: 26, fontWeight: 800, color: '#0f172a' }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: '#64748b', marginTop: 4 }}>{sub}</div>}
    </div>
  )
}

export default function MetricCards({ prediction, confidence, stressScore }) {
  return (
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
      <Card label="PREDICTION" value={prediction} sub="Mental health category" />
      <Card label="CONFIDENCE" value={`${(confidence * 100).toFixed(1)}%`} sub="Model certainty" />
      <Card label="STRESS SCORE" value={`${stressScore} / 100`} sub="Distress intensity" />
    </div>
  )
}

import { useState, useEffect } from 'react'

export default function Header() {
  const [apiLive, setApiLive] = useState(true)

  useEffect(() => {
    fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: 'test ping check' }) })
      .then(() => setApiLive(true))
      .catch(() => setApiLive(false))
  }, [])

  return (
    <header style={{
      background: '#f8fafc',
      borderBottom: '1px solid #ffffff',
      padding: '0 24px',
      height: 56,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      position: 'sticky',
      top: 0,
      zIndex: 100,
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
        <span style={{
          background: '#ef4444',
          color: '#fff',
          fontSize: 11,
          fontWeight: 700,
          padding: '3px 8px',
          borderRadius: 4,
          letterSpacing: 1,
        }}>R26-IT-059</span>
        <div>
          <div style={{ fontSize: 15, fontWeight: 700, color: '#0f172a', lineHeight: 1.2 }}>
            Emotional Burnout Detection System
          </div>
          <div style={{ fontSize: 11, color: '#64748b' }}>
            Explainable NLP · IT22196392 · Induwara K.P.Y.
          </div>
        </div>
      </div>

      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{
          background: '#ffffff',
          border: '1px solid #e2e8f0',
          color: '#64748b',
          fontSize: 12,
          padding: '4px 12px',
          borderRadius: 6,
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          lineHeight: 1.3,
        }}>
          <span style={{ fontWeight: 700, color: '#c4b5fd' }}>BERT+RoBERTa Ensemble</span>
          <span style={{ fontSize: 10, color: '#64748b' }}>Accuracy: 82.32%</span>
        </span>

        <span style={{
          display: 'flex',
          alignItems: 'center',
          gap: 6,
          background: '#ffffff',
          border: `1px solid ${apiLive ? '#22c55e40' : '#ef444440'}`,
          color: apiLive ? '#22c55e' : '#ef4444',
          fontSize: 12,
          padding: '4px 12px',
          borderRadius: 6,
        }}>
          <span style={{
            width: 8, height: 8,
            borderRadius: '50%',
            background: apiLive ? '#22c55e' : '#ef4444',
            boxShadow: apiLive ? '0 0 6px #22c55e' : '0 0 6px #ef4444',
          }} />
          {apiLive ? 'API Live' : 'API Down'}
        </span>
      </div>
    </header>
  )
}

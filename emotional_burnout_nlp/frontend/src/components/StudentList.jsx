import { useState } from 'react'

const RISK_DOT = {
  Low:      '#22d3ee',
  Medium:   '#fbbf24',
  High:     '#f87171',
  Critical: '#ff2020',
}

const RISK_ORDER = { Critical: 0, High: 1, Medium: 2, Low: 3 }

export default function StudentList({ students, selected, onSelect, results }) {
  const [filter, setFilter] = useState('All')

  const filtered = students
    .filter(s => {
      if (filter === 'All') return true
      const r = results[s.id]
      return r && r.risk_level === filter
    })
    .sort((a, b) => {
      const ra = results[a.id]?.risk_level
      const rb = results[b.id]?.risk_level
      return (RISK_ORDER[ra] ?? 4) - (RISK_ORDER[rb] ?? 4)
    })

  const counts = { Critical: 0, High: 0, Medium: 0, Low: 0 }
  Object.values(results).forEach(r => {
    if (r?.risk_level && counts[r.risk_level] !== undefined) counts[r.risk_level]++
  })

  return (
    <div style={{
      width: 260,
      minWidth: 260,
      background: '#0f172a',
      borderRight: '1px solid #1e293b',
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
    }}>
      <div style={{ padding: '16px 16px 8px' }}>
        <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 10 }}>
          STUDENT LIST
        </div>

        <select
          value={filter}
          onChange={e => setFilter(e.target.value)}
          style={{
            width: '100%',
            background: '#1e293b',
            border: '1px solid #334155',
            color: '#94a3b8',
            fontSize: 12,
            padding: '6px 10px',
            borderRadius: 6,
            cursor: 'pointer',
            outline: 'none',
          }}
        >
          <option value="All">All Risk Levels</option>
          <option value="Critical">Critical</option>
          <option value="High">High</option>
          <option value="Medium">Medium</option>
          <option value="Low">Low</option>
        </select>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '4px 8px' }}>
        {filtered.map(s => {
          const r = results[s.id]
          const risk = r?.risk_level
          const dotColor = RISK_DOT[risk] || '#334155'
          const isSelected = selected?.id === s.id
          const isAnalyzing = r === 'loading'

          return (
            <div
              key={s.id}
              onClick={() => onSelect(s)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '10px 12px',
                borderRadius: 8,
                marginBottom: 2,
                cursor: 'pointer',
                background: isSelected ? '#1e293b' : 'transparent',
                border: isSelected ? '1px solid #334155' : '1px solid transparent',
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => {
                if (!isSelected) e.currentTarget.style.background = '#1e293b80'
              }}
              onMouseLeave={e => {
                if (!isSelected) e.currentTarget.style.background = 'transparent'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{
                  width: 28, height: 28,
                  borderRadius: '50%',
                  background: '#1e293b',
                  border: `1px solid ${dotColor}40`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 10, fontWeight: 700, color: '#64748b',
                  flexShrink: 0,
                }}>
                  {s.id.slice(-2)}
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#e2e8f0' }}>{s.name}</div>
                  {r && r !== 'loading' && (
                    <div style={{ fontSize: 11, color: '#475569', marginTop: 1 }}>
                      {r.prediction}
                    </div>
                  )}
                </div>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                {isAnalyzing && (
                  <div style={{ fontSize: 10, color: '#475569' }}>...</div>
                )}
                {risk && (
                  <div style={{
                    width: 8, height: 8,
                    borderRadius: '50%',
                    background: dotColor,
                    boxShadow: `0 0 6px ${dotColor}`,
                    flexShrink: 0,
                  }} />
                )}
              </div>
            </div>
          )
        })}
      </div>

      <div style={{
        padding: '12px 16px',
        borderTop: '1px solid #1e293b',
        display: 'flex',
        flexDirection: 'column',
        gap: 4,
      }}>
        <div style={{ fontSize: 10, color: '#475569' }}>
          {students.length} students analyzed
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {counts.Critical > 0 && (
            <span style={{ fontSize: 10, color: '#ff2020' }}>● {counts.Critical} Critical</span>
          )}
          {counts.High > 0 && (
            <span style={{ fontSize: 10, color: '#f87171' }}>● {counts.High} High</span>
          )}
          {counts.Medium > 0 && (
            <span style={{ fontSize: 10, color: '#fbbf24' }}>● {counts.Medium} Medium</span>
          )}
          {counts.Low > 0 && (
            <span style={{ fontSize: 10, color: '#22d3ee' }}>● {counts.Low} Low</span>
          )}
        </div>
      </div>
    </div>
  )
}

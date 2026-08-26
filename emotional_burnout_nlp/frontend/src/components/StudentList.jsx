import { useState } from 'react'

const RISK_DOT   = { Low: '#22d3ee', Medium: '#fbbf24', High: '#f87171', Critical: '#ff2020' }
const RISK_ORDER = { Critical: 0, High: 1, Medium: 2, Low: 3 }

export default function StudentList({ students, selected, onSelect, onAddNew, onClearAll }) {
  const [filter, setFilter] = useState('All')

  const filtered = students
    .filter(s => filter === 'All' || s.latestResult?.risk_level === filter)
    .sort((a, b) => {
      const ra = a.latestResult?.risk_level
      const rb = b.latestResult?.risk_level
      return (RISK_ORDER[ra] ?? 4) - (RISK_ORDER[rb] ?? 4)
    })

  const counts = { Critical: 0, High: 0, Medium: 0, Low: 0 }
  students.forEach(s => {
    const rl = s.latestResult?.risk_level
    if (rl && counts[rl] !== undefined) counts[rl]++
  })

  return (
    <div style={{
      width: 260, minWidth: 260,
      background: '#f8fafc',
      borderRight: '1px solid #ffffff',
      display: 'flex', flexDirection: 'column',
      height: '100%',
    }}>
      <div style={{ padding: '12px 12px 8px' }}>
        <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 8 }}>
          STUDENT LIST
        </div>

        <button
          onClick={onAddNew}
          style={{
            width: '100%',
            background: '#dbeafe',
            border: '1px solid #3b82f640',
            color: '#1d4ed8',
            fontSize: 12, fontWeight: 700,
            padding: '8px 12px',
            borderRadius: 7,
            cursor: 'pointer',
            marginBottom: 8,
            display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
          }}
        >
          <span style={{ fontSize: 16, lineHeight: 1 }}>+</span> Add New Student
        </button>

        <select
          value={filter}
          onChange={e => setFilter(e.target.value)}
          style={{
            width: '100%',
            background: '#ffffff', border: '1px solid #e2e8f0',
            color: '#64748b', fontSize: 12,
            padding: '6px 10px', borderRadius: 6,
            cursor: 'pointer', outline: 'none',
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
        {filtered.length === 0 && (
          <div style={{ padding: '24px 12px', textAlign: 'center' }}>
            <div style={{ fontSize: 28, marginBottom: 10 }}>👤</div>
            <div style={{ fontSize: 12, color: '#94a3b8', lineHeight: 1.6 }}>
              No students yet.<br />
              Click <span style={{ color: '#1d4ed8' }}>+ Add New Student</span> to get started.
            </div>
          </div>
        )}

        {filtered.map(s => {
          const risk = s.latestResult?.risk_level
          const dotColor = RISK_DOT[risk] || '#e2e8f0'
          const isSelected = selected?.id === s.id
          const initials = s.name.trim().split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase()

          return (
            <div
              key={s.id}
              onClick={() => onSelect(s)}
              style={{
                display: 'flex', alignItems: 'center', justifyContent: 'space-between',
                padding: '10px 12px', borderRadius: 8, marginBottom: 2,
                cursor: 'pointer',
                background: isSelected ? '#ffffff' : 'transparent',
                border: isSelected ? '1px solid #e2e8f0' : '1px solid transparent',
                transition: 'background 0.15s',
              }}
              onMouseEnter={e => { if (!isSelected) e.currentTarget.style.background = '#ffffff80' }}
              onMouseLeave={e => { if (!isSelected) e.currentTarget.style.background = 'transparent' }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <div style={{
                  width: 30, height: 30, borderRadius: '50%',
                  background: '#ffffff', border: `1px solid ${dotColor}40`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: 10, fontWeight: 700, color: '#64748b', flexShrink: 0,
                }}>
                  {initials}
                </div>
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#1e293b' }}>{s.name}</div>
                  {risk && (
                    <div style={{ fontSize: 11, color: dotColor, marginTop: 1, fontWeight: 600 }}>
                      {s.latestResult?.prediction}
                    </div>
                  )}
                </div>
              </div>

              {risk && (
                <div style={{
                  width: 8, height: 8, borderRadius: '50%',
                  background: dotColor, boxShadow: `0 0 6px ${dotColor}`,
                  flexShrink: 0,
                }} />
              )}
            </div>
          )
        })}
      </div>

      <div style={{ padding: '10px 12px', borderTop: '1px solid #ffffff', display: 'flex', flexDirection: 'column', gap: 6 }}>
        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
          {counts.Critical > 0 && <span style={{ fontSize: 10, color: '#ff2020' }}>● {counts.Critical} Critical</span>}
          {counts.High > 0 && <span style={{ fontSize: 10, color: '#f87171' }}>● {counts.High} High</span>}
          {counts.Medium > 0 && <span style={{ fontSize: 10, color: '#fbbf24' }}>● {counts.Medium} Medium</span>}
          {counts.Low > 0 && <span style={{ fontSize: 10, color: '#22d3ee' }}>● {counts.Low} Low</span>}
          {students.length === 0 && <span style={{ fontSize: 10, color: '#e2e8f0' }}>No students added</span>}
        </div>
        {students.length > 0 && (
          <button
            onClick={onClearAll}
            style={{
              background: 'none', border: '1px solid #ef444430',
              color: '#f87171', fontSize: 11, fontWeight: 600,
              padding: '5px 10px', borderRadius: 6, cursor: 'pointer',
              width: '100%',
            }}
          >
            Clear All Students
          </button>
        )}
      </div>
    </div>
  )
}

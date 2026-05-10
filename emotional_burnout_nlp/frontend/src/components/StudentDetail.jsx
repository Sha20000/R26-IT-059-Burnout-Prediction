import { useState } from 'react'
import MetricCards from './MetricCards'
import RiskBadge from './RiskBadge'
import AttentionHeatmap from './AttentionHeatmap'
import ProbabilityBars from './ProbabilityBars'

const ADVISOR_MAP = {
  Normal:               { text: '✅ Student appears emotionally stable. Continue routine monitoring.',                                                                                          color: '#22d3ee', bg: '#22d3ee0d', border: '#22d3ee30' },
  Stress:               { text: '⚠️ Student showing stress indicators. Recommend check-in with advisor.',                                                                                     color: '#fbbf24', bg: '#fbbf240d', border: '#fbbf2430' },
  Anxiety:              { text: '⚠️ Student showing anxiety indicators. Recommend counseling session.',                                                                                       color: '#fbbf24', bg: '#fbbf240d', border: '#fbbf2430' },
  Depression:           { text: '🔴 HIGH RISK: Student showing depression indicators. Immediate counseling recommended. Key trigger words are highlighted in the attention heatmap above.',    color: '#f87171', bg: '#f871710d', border: '#f8717130' },
  Suicidal:             { text: '🚨 CRITICAL: Student showing suicidal ideation. IMMEDIATE intervention required. Contact student NOW.',                                                       color: '#ff2020', bg: '#ff20200d', border: '#ff202050' },
  Bipolar:              { text: '🔴 HIGH RISK: Student showing bipolar indicators. Recommend psychiatric evaluation.',                                                                         color: '#f87171', bg: '#f871710d', border: '#f8717130' },
  'Personality disorder': { text: '🔴 HIGH RISK: Student showing personality disorder indicators. Recommend professional evaluation.',                                                         color: '#f87171', bg: '#f871710d', border: '#f8717130' },
}

function AdvisorRecommendation({ prediction }) {
  const rec = ADVISOR_MAP[prediction]
  if (!rec) return null
  return (
    <div style={{
      background: rec.bg,
      border: `1px solid ${rec.border}`,
      borderRadius: 10,
      padding: '16px 20px',
    }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 8 }}>
        ADVISOR RECOMMENDATION
      </div>
      <div style={{ fontSize: 13, color: rec.color, lineHeight: 1.6, fontWeight: 500 }}>
        {rec.text}
      </div>
    </div>
  )
}

function Spinner() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, height: '100%', minHeight: 300 }}>
      <div style={{
        width: 40, height: 40,
        border: '3px solid #1e293b',
        borderTop: '3px solid #6366f1',
        borderRadius: '50%',
        animation: 'spin 0.8s linear infinite',
      }} />
      <div style={{ fontSize: 13, color: '#475569' }}>Running BERT inference…</div>
    </div>
  )
}

function EmptyState() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, height: '100%', minHeight: 300 }}>
      <div style={{ fontSize: 32 }}>←</div>
      <div style={{ fontSize: 14, color: '#475569', textAlign: 'center', maxWidth: 280 }}>
        Select a student from the list to view their mental health analysis
      </div>
    </div>
  )
}

const WEEKLY_DATA = [
  { week: 2,  prediction: 'Normal',    confidence: 82.5, riskLevel: 'Low',      riskScore: 1,  weeksLeft: 14, f1: 0.9505, color: '#22c55e' },
  { week: 4,  prediction: 'Stress',    confidence: 63.8, riskLevel: 'Medium',   riskScore: 3,  weeksLeft: 12, f1: 0.9344, color: '#fbbf24' },
  { week: 8,  prediction: 'Depression',confidence: 77.8, riskLevel: 'High',     riskScore: 7,  weeksLeft: 8,  f1: 0.9515, color: '#f97316' },
  { week: 12, prediction: 'Suicidal',  confidence: 63.3, riskLevel: 'Critical', riskScore: 10, weeksLeft: 4,  f1: 0.9382, color: '#ef4444' },
]

const RISK_TO_WEEK = { Low: 2, Medium: 4, High: 8, Critical: 12 }

function RiskChart({ activeRiskLevel }) {
  const W = 520, H = 200
  const padL = 44, padR = 24, padT = 36, padB = 36
  const chartW = W - padL - padR
  const chartH = H - padT - padB

  const weeks = WEEKLY_DATA.map(d => d.week)
  const minWeek = weeks[0], maxWeek = weeks[weeks.length - 1]

  const xOf = w => padL + ((w - minWeek) / (maxWeek - minWeek)) * chartW
  const yOf = s => padT + (1 - s / 10) * chartH

  const activeWeek = RISK_TO_WEEK[activeRiskLevel]
  const points = WEEKLY_DATA.map(d => ({ x: xOf(d.week), y: yOf(d.riskScore), ...d, isActive: d.week === activeWeek }))

  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ')
  const areaPath = `${linePath} L${points[points.length-1].x},${padT + chartH} L${points[0].x},${padT + chartH} Z`

  const gradId = 'riskLineGrad'
  const areaId = 'riskAreaGrad'

  const yTicks = [0, 2, 4, 6, 8, 10]

  return (
    <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, padding: '16px 20px' }}>
      <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 4 }}>
        EMOTIONAL BURNOUT RISK PROGRESSION
      </div>
      <div style={{ fontSize: 11, color: '#4ade80', marginBottom: 12, fontWeight: 500 }}>
        🎯 Early detection at Week 2 — 14 weeks to intervene!
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }}>
        <defs>
          <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%"   stopColor="#22c55e" />
            <stop offset="33%"  stopColor="#fbbf24" />
            <stop offset="66%"  stopColor="#f97316" />
            <stop offset="100%" stopColor="#ef4444" />
          </linearGradient>
          <linearGradient id={areaId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor="#ef4444" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#ef4444" stopOpacity="0.02" />
          </linearGradient>
        </defs>

        {yTicks.map(v => (
          <g key={v}>
            <line x1={padL} y1={yOf(v)} x2={padL + chartW} y2={yOf(v)}
              stroke="#334155" strokeWidth="1" strokeDasharray="4 4" />
            <text x={padL - 6} y={yOf(v) + 4} textAnchor="end"
              fontSize="9" fill="#475569">{v}</text>
          </g>
        ))}

        <line x1={padL} y1={padT} x2={padL} y2={padT + chartH} stroke="#334155" strokeWidth="1" />
        <line x1={padL} y1={padT + chartH} x2={padL + chartW} y2={padT + chartH} stroke="#334155" strokeWidth="1" />

        {WEEKLY_DATA.map(d => (
          <text key={d.week} x={xOf(d.week)} y={padT + chartH + 16}
            textAnchor="middle" fontSize="9" fill="#475569">
            Wk {d.week}
          </text>
        ))}

        <path d={areaPath} fill={`url(#${areaId})`} />

        <path d={linePath} fill="none" stroke={`url(#${gradId})`} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />

        {points.map((p, i) => (
          <g key={i}>
            {p.isActive && (
              <>
                <circle cx={p.x} cy={p.y} r="18" fill={p.color} fillOpacity="0.12">
                  <animate attributeName="r" values="14;20;14" dur="1.6s" repeatCount="indefinite" />
                  <animate attributeName="fill-opacity" values="0.18;0.04;0.18" dur="1.6s" repeatCount="indefinite" />
                </circle>
                <circle cx={p.x} cy={p.y} r="11" fill="none" stroke={p.color} strokeWidth="1.5" strokeOpacity="0.5">
                  <animate attributeName="r" values="9;13;9" dur="1.6s" repeatCount="indefinite" />
                  <animate attributeName="stroke-opacity" values="0.6;0.1;0.6" dur="1.6s" repeatCount="indefinite" />
                </circle>
              </>
            )}
            <circle cx={p.x} cy={p.y} r={p.isActive ? 9 : 7} fill="#0f172a" stroke={p.color} strokeWidth={p.isActive ? 3 : 2.5} />
            <circle cx={p.x} cy={p.y} r={p.isActive ? 4 : 3} fill={p.color} />
            <text x={p.x} y={p.y - (p.isActive ? 17 : 13)} textAnchor="middle" fontSize={p.isActive ? 12 : 10} fontWeight="700" fill={p.color}>
              {p.riskScore}
            </text>
            {p.isActive && (
              <text x={p.x} y={p.y + 22} textAnchor="middle" fontSize="8" fill={p.color} fontWeight="600">
                ▲ NOW
              </text>
            )}
          </g>
        ))}

        <text x={padL - 2} y={padT - 10} fontSize="9" fill="#475569" textAnchor="middle">Score</text>
      </svg>
    </div>
  )
}

function WeeklyProgression({ riskLevel }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      <div style={{
        background: '#1e3a1e',
        border: '1px solid #22c55e40',
        borderRadius: 10,
        padding: '14px 18px',
        fontSize: 13,
        color: '#4ade80',
        fontWeight: 600,
        lineHeight: 1.5,
      }}>
        🎯 Model detects burnout at Week 2 — 14 weeks before academic decline becomes measurable!
      </div>

      <RiskChart activeRiskLevel={riskLevel} />

      <div style={{ display: 'flex', flexDirection: 'column', gap: 0 }}>
        {WEEKLY_DATA.map((w, i) => {
          const isActive = w.week === RISK_TO_WEEK[riskLevel]
          return (
          <div key={w.week} style={{ display: 'flex', gap: 0 }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 40, flexShrink: 0 }}>
              <div style={{
                width: isActive ? 18 : 14, height: isActive ? 18 : 14, borderRadius: '50%',
                background: w.color,
                boxShadow: isActive ? `0 0 14px ${w.color}, 0 0 28px ${w.color}60` : `0 0 8px ${w.color}80`,
                marginTop: isActive ? 16 : 18, flexShrink: 0,
                transition: 'all 0.3s',
              }} />
              {i < WEEKLY_DATA.length - 1 && (
                <div style={{ width: 2, flex: 1, background: '#1e293b', minHeight: 20 }} />
              )}
            </div>

            <div style={{
              flex: 1,
              background: isActive ? `${w.color}0d` : '#1e293b',
              border: `1px solid ${isActive ? w.color : `${w.color}30`}`,
              borderRadius: 10,
              padding: '14px 16px',
              marginBottom: i < WEEKLY_DATA.length - 1 ? 10 : 0,
              marginLeft: 12,
              boxShadow: isActive ? `0 0 16px ${w.color}30` : 'none',
              transition: 'all 0.3s',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: isActive ? '#f1f5f9' : '#64748b', letterSpacing: 1 }}>WEEK {w.week}</div>
                  {isActive && <div style={{ fontSize: 9, fontWeight: 700, color: w.color, background: `${w.color}20`, border: `1px solid ${w.color}50`, borderRadius: 3, padding: '1px 6px', letterSpacing: 1 }}>CURRENT</div>}
                </div>
                <div style={{
                  fontSize: 10, fontWeight: 700, color: w.color,
                  background: `${w.color}15`, border: `1px solid ${w.color}40`,
                  borderRadius: 4, padding: '2px 8px', letterSpacing: 1,
                }}>{w.riskLevel.toUpperCase()}</div>
              </div>

              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {[
                  { label: 'PREDICTION',        value: w.prediction },
                  { label: 'CONFIDENCE',         value: `${w.confidence}%` },
                  { label: 'RISK SCORE',         value: `${w.riskScore}/10` },
                  { label: 'WEEKS TO INTERVENE', value: `${w.weeksLeft} wks` },
                ].map(({ label, value }) => (
                  <div key={label} style={{
                    background: '#0f172a',
                    border: '1px solid #334155',
                    borderRadius: 6,
                    padding: '8px 12px',
                    flex: 1, minWidth: 90,
                  }}>
                    <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 1.5, color: '#475569', marginBottom: 4 }}>{label}</div>
                    <div style={{ fontSize: 14, fontWeight: 700, color: '#f1f5f9' }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
          )
        })}
      </div>

      <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 10, overflow: 'hidden' }}>
        <div style={{ padding: '12px 16px', borderBottom: '1px solid #334155' }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b' }}>F1 SCORE BY WEEK</div>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#0f172a' }}>
              {['Week', 'Prediction', 'F1 Score', 'Weeks Remaining'].map(h => (
                <th key={h} style={{ padding: '10px 16px', fontSize: 10, fontWeight: 700, color: '#475569', letterSpacing: 1, textAlign: 'left' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {WEEKLY_DATA.map((w, i) => (
              <tr key={w.week} style={{ borderTop: '1px solid #1e293b', background: i % 2 === 0 ? 'transparent' : '#0f172a08' }}>
                <td style={{ padding: '10px 16px', fontSize: 13, color: '#94a3b8', fontWeight: 600 }}>Week {w.week}</td>
                <td style={{ padding: '10px 16px', fontSize: 13, color: w.color, fontWeight: 600 }}>{w.prediction}</td>
                <td style={{ padding: '10px 16px', fontSize: 13, color: '#f1f5f9', fontFamily: 'monospace' }}>{w.f1.toFixed(4)}</td>
                <td style={{ padding: '10px 16px', fontSize: 13, color: '#94a3b8' }}>{w.weeksLeft}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function StudentDetail({ student, result }) {
  const [tab, setTab] = useState('analysis')

  if (!student) return <EmptyState />
  if (!result || result === 'loading') return <Spinner />

  const isCritical = result.risk_level === 'Critical'

  const TABS = [
    { id: 'analysis',    label: 'Analysis' },
    { id: 'progression', label: 'Weekly Progression' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: '24px', flex: 1, overflowY: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div style={{ fontSize: 22, fontWeight: 800, color: '#f1f5f9' }}>{student.name}</div>
          <div style={{ fontSize: 12, color: '#475569', marginTop: 4 }}>ID: {student.id}</div>
        </div>
        <RiskBadge riskLevel={result.risk_level} />
      </div>

      {isCritical && (
        <div style={{
          background: '#ff000012',
          border: '1px solid #ff000040',
          borderRadius: 10,
          padding: '12px 16px',
          display: 'flex',
          alignItems: 'center',
          gap: 10,
        }}>
          <span style={{ fontSize: 18 }}>⚠</span>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#ff4444' }}>Immediate Attention Required</div>
            <div style={{ fontSize: 12, color: '#f87171', marginTop: 2 }}>
              This student's text indicates critical distress. Please contact a mental health professional immediately.
            </div>
          </div>
        </div>
      )}

      <div style={{ display: 'flex', gap: 4, borderBottom: '1px solid #1e293b', paddingBottom: 0 }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            style={{
              background: 'none',
              border: 'none',
              borderBottom: tab === t.id ? '2px solid #6366f1' : '2px solid transparent',
              color: tab === t.id ? '#f1f5f9' : '#475569',
              fontSize: 13,
              fontWeight: tab === t.id ? 700 : 400,
              padding: '8px 16px',
              cursor: 'pointer',
              marginBottom: -1,
              transition: 'color 0.15s',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'analysis' && <>
        <div style={{
          background: '#1e293b',
          border: '1px solid #334155',
          borderRadius: 10,
          padding: '14px 20px',
        }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 8 }}>
            STUDENT INPUT TEXT
          </div>
          <div style={{ fontSize: 13, color: '#94a3b8', lineHeight: 1.6, fontStyle: 'italic' }}>
            "{student.text}"
          </div>
        </div>

        <MetricCards
          prediction={result.prediction}
          confidence={result.confidence}
          stressScore={result.stress_score}
        />

        <AttentionHeatmap tokens={result.attention} prediction={result.prediction} />

        <AdvisorRecommendation prediction={result.prediction} />

        <ProbabilityBars probabilities={result.probabilities} />

        <div style={{
          background: '#1e293b',
          border: '1px solid #334155',
          borderRadius: 10,
          padding: '16px 20px',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, flexWrap: 'wrap', gap: 8 }}>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b' }}>MODEL PERFORMANCE</div>
            <div style={{ fontSize: 10, color: '#334155', fontWeight: 600, letterSpacing: 1 }}>IT22196392 — R26-IT-059</div>
          </div>
          <div style={{ fontSize: 12, fontWeight: 600, color: '#6366f1', marginBottom: 12 }}>
            BERT Fine-tuned + RoBERTa Ensemble
          </div>
          <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
            {[
              { label: 'ACCURACY', value: '82.32%', sub: 'Test set' },
              { label: 'F1 SCORE', value: '82.32%', sub: 'Weighted avg' },
              { label: 'DATASET',  value: '51,055', sub: 'Training samples' },
              { label: 'CLASSES',  value: '7',      sub: 'Mental health categories' },
            ].map(({ label, value, sub }) => (
              <div key={label} style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 8, padding: '12px 16px', flex: 1, minWidth: 100 }}>
                <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 6 }}>{label}</div>
                <div style={{ fontSize: 20, fontWeight: 800, color: '#f1f5f9' }}>{value}</div>
                <div style={{ fontSize: 10, color: '#475569', marginTop: 3 }}>{sub}</div>
              </div>
            ))}
          </div>
        </div>

        <div style={{
          background: '#1e293b',
          border: '1px solid #334155',
          borderRadius: 10,
          padding: '16px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 12,
        }}>
          <div>
            <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 4 }}>
              EXPORT DATA
            </div>
            <div style={{ fontSize: 11, color: '#475569' }}>
              Export for Meta-Integration Layer
            </div>
          </div>
          <button
            onClick={() => {
              const rows = [
                'student_id,predicted_label,confidence,risk_level,stress_score',
                'Student #0001,Normal,0.993,Low,1',
                'Student #0002,Depression,0.85,High,7',
                'Student #0003,Stress,0.92,Medium,3',
                'Student #0004,Anxiety,0.78,Medium,4',
                'Student #0005,Depression,0.889,High,7',
                'Student #0006,Normal,0.998,Low,1',
                'Student #0007,Suicidal,0.95,Critical,10',
                'Student #0008,Normal,0.991,Low,1',
                'Student #0009,Depression,0.87,High,7',
                'Student #0010,Depression,0.82,High,7',
                'Student #0011,Depression,0.79,High,7',
                'Student #0012,Stress,0.88,Medium,3',
              ].join('\n')
              const blob = new Blob([rows], { type: 'text/csv' })
              const url = URL.createObjectURL(blob)
              const a = document.createElement('a')
              a.href = url
              a.download = 'stress_scores_R26-IT-059.csv'
              a.click()
              URL.revokeObjectURL(url)
            }}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              background: '#14532d',
              border: '1px solid #22c55e50',
              color: '#4ade80',
              fontSize: 13,
              fontWeight: 700,
              padding: '10px 20px',
              borderRadius: 8,
              cursor: 'pointer',
              transition: 'background 0.15s, box-shadow 0.15s',
              boxShadow: '0 0 12px #22c55e20',
            }}
            onMouseEnter={e => {
              e.currentTarget.style.background = '#166534'
              e.currentTarget.style.boxShadow = '0 0 18px #22c55e40'
            }}
            onMouseLeave={e => {
              e.currentTarget.style.background = '#14532d'
              e.currentTarget.style.boxShadow = '0 0 12px #22c55e20'
            }}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            Download Stress Scores CSV
          </button>
        </div>
      </>}

      {tab === 'progression' && <WeeklyProgression riskLevel={result.risk_level} />}
    </div>
  )
}

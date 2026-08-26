import { useState, useEffect } from 'react'
import MetricCards from './MetricCards'
import RiskBadge from './RiskBadge'
import AttentionHeatmap from './AttentionHeatmap'
import ProbabilityBars from './ProbabilityBars'

// ─── Constants ────────────────────────────────────────────────────────────────

const WEEK_NUMS = [2, 4, 8, 18]

const RISK_COLOR = { Low: '#22c55e', Medium: '#fbbf24', High: '#f97316', Critical: '#ef4444', Unknown: '#64748b' }
const RISK_SCORE = { Low: 1, Medium: 3, High: 7, Critical: 10, Unknown: 0 }
const RISK_LEFT  = { Low: 14, Medium: 12, High: 8, Critical: 4, Unknown: 0 }

const ADVISOR_MAP = {
  Normal:               { text: '✅ Student appears emotionally stable. Continue routine monitoring.',                                                                                          color: '#22d3ee', bg: '#22d3ee0d', border: '#22d3ee30' },
  Stress:               { text: '⚠️ Student showing stress indicators. Recommend check-in with advisor.',                                                                                     color: '#fbbf24', bg: '#fbbf240d', border: '#fbbf2430' },
  Anxiety:              { text: '⚠️ Student showing anxiety indicators. Recommend counseling session.',                                                                                       color: '#fbbf24', bg: '#fbbf240d', border: '#fbbf2430' },
  Depression:           { text: '🔴 HIGH RISK: Student showing depression indicators. Immediate counseling recommended. Key trigger words are highlighted in the attention heatmap above.',    color: '#f87171', bg: '#f871710d', border: '#f8717130' },
  Suicidal:             { text: '🚨 CRITICAL: Student showing suicidal ideation. IMMEDIATE intervention required. Contact student NOW.',                                                       color: '#ff2020', bg: '#ff20200d', border: '#ff202050' },
  Bipolar:              { text: '🔴 HIGH RISK: Student showing bipolar indicators. Recommend psychiatric evaluation.',                                                                         color: '#f87171', bg: '#f871710d', border: '#f8717130' },
  'Personality disorder': { text: '🔴 HIGH RISK: Student showing personality disorder indicators. Recommend professional evaluation.',                                                         color: '#f87171', bg: '#f871710d', border: '#f8717130' },
}

// ─── Small helpers ─────────────────────────────────────────────────────────────

function Spinner({ label = 'Running BERT inference…' }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, minHeight: 200 }}>
      <div style={{ width: 40, height: 40, border: '3px solid #ffffff', borderTop: '3px solid #6366f1', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
      <div style={{ fontSize: 13, color: '#94a3b8' }}>{label}</div>
    </div>
  )
}

function AdvisorRecommendation({ prediction }) {
  const rec = ADVISOR_MAP[prediction]
  if (!rec) return null
  return (
    <div style={{ background: rec.bg, border: `1px solid ${rec.border}`, borderRadius: 10, padding: '16px 20px' }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 8 }}>ADVISOR RECOMMENDATION</div>
      <div style={{ fontSize: 13, color: rec.color, lineHeight: 1.6, fontWeight: 500 }}>{rec.text}</div>
    </div>
  )
}

// ─── Risk Chart ────────────────────────────────────────────────────────────────

function RiskChart({ data }) {
  const W = 520, H = 200
  const padL = 44, padR = 24, padT = 36, padB = 36
  const chartW = W - padL - padR
  const chartH = H - padT - padB
  const minWeek = WEEK_NUMS[0], maxWeek = WEEK_NUMS[WEEK_NUMS.length - 1]
  const xOf = w => padL + ((w - minWeek) / (maxWeek - minWeek)) * chartW
  const yOf = s => padT + (1 - s / 10) * chartH
  const yTicks = [0, 2, 4, 6, 8, 10]

  const points = data.map(d => ({ x: xOf(d.week), y: yOf(d.riskScore), color: d.color, riskScore: d.riskScore, week: d.week }))
  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ')
  const areaPath = `${linePath} L${points[points.length-1].x},${padT+chartH} L${points[0].x},${padT+chartH} Z`
  const maxScore = Math.max(...data.map(d => d.riskScore))
  const areaColor = maxScore >= 7 ? '#ef4444' : maxScore >= 3 ? '#f97316' : '#22c55e'

  return (
    <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 10, padding: '16px 20px' }}>
      <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 12 }}>
        EMOTIONAL BURNOUT RISK PROGRESSION
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', display: 'block' }}>
        <defs>
          <linearGradient id="chartLineGrad" x1="0%" y1="0%" x2="100%" y2="0%">
            {points.map((p, i) => <stop key={i} offset={`${(i/(points.length-1))*100}%`} stopColor={p.color} />)}
          </linearGradient>
          <linearGradient id="chartAreaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%"   stopColor={areaColor} stopOpacity="0.25" />
            <stop offset="100%" stopColor={areaColor} stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {yTicks.map(v => (
          <g key={v}>
            <line x1={padL} y1={yOf(v)} x2={padL+chartW} y2={yOf(v)} stroke="#e2e8f0" strokeWidth="1" strokeDasharray="4 4" />
            <text x={padL-6} y={yOf(v)+4} textAnchor="end" fontSize="9" fill="#94a3b8">{v}</text>
          </g>
        ))}
        <line x1={padL} y1={padT} x2={padL} y2={padT+chartH} stroke="#e2e8f0" strokeWidth="1" />
        <line x1={padL} y1={padT+chartH} x2={padL+chartW} y2={padT+chartH} stroke="#e2e8f0" strokeWidth="1" />
        {WEEK_NUMS.map(w => (
          <text key={w} x={xOf(w)} y={padT+chartH+16} textAnchor="middle" fontSize="9" fill="#94a3b8">Wk {w}</text>
        ))}
        <path d={areaPath} fill="url(#chartAreaGrad)" />
        <path d={linePath} fill="none" stroke="url(#chartLineGrad)" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
        {points.map((p, i) => (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r="7" fill="#f8fafc" stroke={p.color} strokeWidth="2.5" />
            <circle cx={p.x} cy={p.y} r="3" fill={p.color} />
            <text x={p.x} y={p.y-13} textAnchor="middle" fontSize="10" fontWeight="700" fill={p.color}>{p.riskScore}</text>
          </g>
        ))}
        <text x={padL-2} y={padT-10} fontSize="9" fill="#94a3b8" textAnchor="middle">Score</text>
      </svg>
    </div>
  )
}

// ─── Add New Student Form (inside Weekly Progression tab) ─────────────────────

function AddStudentForm({ onSaved }) {
  const [name, setName]         = useState('')
  // Each week holds an array of text strings (1 or 2 entries)
  const [texts, setTexts]       = useState({ 2: [''], 4: [''], 8: [''], 18: [''] })
  const [saving, setSaving]     = useState(false)
  const [progress, setProgress] = useState('')
  const [error, setError]       = useState('')

  const hasValidEntry = (w) => texts[w].some(t => t.trim().split(/\s+/).length >= 3)
  const canSave = name.trim().length > 0 && WEEK_NUMS.some(w => hasValidEntry(w))

  const addSecondEntry = (w) => {
    setTexts(p => ({ ...p, [w]: [p[w][0], ''] }))
  }
  const removeSecondEntry = (w) => {
    setTexts(p => ({ ...p, [w]: [p[w][0]] }))
  }
  const updateText = (w, idx, val) => {
    setTexts(p => {
      const arr = [...p[w]]
      arr[idx] = val
      return { ...p, [w]: arr }
    })
  }

  const handleSave = async () => {
    if (!canSave) return
    setSaving(true)
    setError('')
    const weeks = {}

    for (const w of WEEK_NUMS) {
      const entries = texts[w].filter(t => t.trim().split(/\s+/).length >= 3)
      if (entries.length === 0) { weeks[w] = null; continue }

      setProgress(`Analyzing Week ${w}…`)
      const results = []
      for (let i = 0; i < entries.length; i++) {
        if (entries.length > 1) setProgress(`Analyzing Week ${w} — entry ${i + 1} of ${entries.length}…`)
        try {
          const res = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: entries[i].trim() }),
          })
          const data = await res.json()
          results.push({ text: entries[i].trim(), result: data })
        } catch {
          results.push({ text: entries[i].trim(), result: null })
        }
      }

      // Use the worst result (highest risk score) among the entries for this week
      const validResults = results.filter(r => r.result && !r.result.error)
      if (validResults.length === 0) { weeks[w] = null; continue }
      const worstEntry = validResults.sort(
        (a, b) => (RISK_SCORE[b.result.risk_level] ?? 0) - (RISK_SCORE[a.result.risk_level] ?? 0)
      )[0]
      weeks[w] = { text: worstEntry.text, result: worstEntry.result, entries: results }
    }

    const filled = WEEK_NUMS.filter(w => weeks[w]?.result)
    if (filled.length === 0) {
      setError('All analyses failed. Check the server is running.')
      setSaving(false)
      setProgress('')
      return
    }

    const latestResult = weeks[filled[filled.length - 1]].result
    const worstResult  = filled
      .map(w => weeks[w].result)
      .sort((a, b) => (RISK_SCORE[b.risk_level] ?? 0) - (RISK_SCORE[a.risk_level] ?? 0))[0]

    const student = {
      id: `s_${Date.now()}`,
      name: name.trim(),
      weeks,
      latestResult,
      worstResult,
    }

    setSaving(false)
    setProgress('')
    onSaved(student)
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14, padding: '24px' }}>
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b' }}>ADD NEW STUDENT</div>
      <div style={{ fontSize: 13, color: '#94a3b8' }}>
        Enter the student's name and their written text from each week. Each week supports up to 2 submissions.
      </div>

      <div>
        <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', letterSpacing: 1, marginBottom: 6 }}>STUDENT NAME</div>
        <input
          value={name}
          onChange={e => setName(e.target.value)}
          placeholder="e.g. John Silva"
          style={{
            width: '100%', boxSizing: 'border-box',
            background: '#ffffff', border: '1px solid #e2e8f0',
            color: '#0f172a', fontSize: 13, padding: '10px 12px',
            borderRadius: 8, outline: 'none', fontFamily: 'inherit',
          }}
        />
      </div>

      {WEEK_NUMS.map(w => (
        <div key={w} style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 10, padding: '14px 16px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: '#64748b', letterSpacing: 1 }}>
              WEEK {w} <span style={{ color: '#e2e8f0', fontWeight: 400 }}>(optional)</span>
            </div>
            {texts[w].length === 1
              ? (
                <button
                  onClick={() => addSecondEntry(w)}
                  title="Add second submission for this week"
                  style={{
                    background: '#f8fafc', border: '1px solid #e2e8f0',
                    color: '#64748b', fontSize: 12, fontWeight: 700,
                    width: 24, height: 24, borderRadius: 6,
                    cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                    lineHeight: 1, padding: 0,
                  }}
                  onMouseEnter={e => { e.currentTarget.style.borderColor = '#6366f1'; e.currentTarget.style.color = '#4338ca' }}
                  onMouseLeave={e => { e.currentTarget.style.borderColor = '#e2e8f0'; e.currentTarget.style.color = '#64748b' }}
                >
                  +
                </button>
              ) : (
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ fontSize: 10, color: '#6366f1', fontWeight: 600 }}>2 submissions</span>
                  <button
                    onClick={() => removeSecondEntry(w)}
                    title="Remove second submission"
                    style={{
                      background: '#f8fafc', border: '1px solid #e2e8f0',
                      color: '#64748b', fontSize: 14, fontWeight: 700,
                      width: 24, height: 24, borderRadius: 6,
                      cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center',
                      lineHeight: 1, padding: 0,
                    }}
                    onMouseEnter={e => { e.currentTarget.style.borderColor = '#ef4444'; e.currentTarget.style.color = '#f87171' }}
                    onMouseLeave={e => { e.currentTarget.style.borderColor = '#e2e8f0'; e.currentTarget.style.color = '#64748b' }}
                  >
                    ×
                  </button>
                </div>
              )
            }
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {texts[w].map((val, idx) => (
              <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                {texts[w].length > 1 && (
                  <div style={{ fontSize: 9, fontWeight: 700, color: '#94a3b8', letterSpacing: 1 }}>
                    SUBMISSION {idx + 1}
                  </div>
                )}
                <textarea
                  value={val}
                  onChange={e => updateText(w, idx, e.target.value)}
                  placeholder={`Enter student's written text from Week ${w}… (min 3 words)`}
                  rows={2}
                  style={{
                    width: '100%', boxSizing: 'border-box',
                    background: '#f8fafc', border: '1px solid #e2e8f0',
                    color: '#1e293b', fontSize: 12, padding: '8px 10px',
                    borderRadius: 6, resize: 'vertical',
                    fontFamily: 'inherit', outline: 'none',
                  }}
                />
              </div>
            ))}
          </div>
        </div>
      ))}

      {error && <div style={{ fontSize: 12, color: '#f87171' }}>{error}</div>}

      {saving
        ? <Spinner label={progress || 'Analyzing…'} />
        : (
          <button
            onClick={handleSave}
            disabled={!canSave}
            style={{
              background: canSave ? '#e0e7ff' : '#ffffff',
              border: `1px solid ${canSave ? '#6366f1' : '#e2e8f0'}`,
              color: canSave ? '#4338ca' : '#94a3b8',
              fontSize: 13, fontWeight: 700,
              padding: '12px 20px', borderRadius: 8,
              cursor: canSave ? 'pointer' : 'not-allowed',
            }}
          >
            Analyze & Save Student
          </button>
        )
      }
    </div>
  )
}

// ─── Weekly Progression for a saved student ───────────────────────────────────

function SavedProgression({ student }) {
  const chartData = WEEK_NUMS.map(w => {
    const r = student.weeks?.[w]?.result
    const rl = r?.risk_level || 'Unknown'
    return {
      week: w,
      prediction: r?.prediction || '—',
      confidence: r ? (r.confidence * 100).toFixed(1) : '—',
      riskLevel: rl,
      riskScore: RISK_SCORE[rl] ?? 0,
      weeksLeft: RISK_LEFT[rl] ?? 0,
      color: RISK_COLOR[rl],
    }
  })

  const hasAny = chartData.some(d => d.riskScore > 0)

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {hasAny && <RiskChart data={chartData} />}

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
        {chartData.map((w, i) => {
          const hasData = student.weeks?.[w.week]?.result
          const col = w.color
          return (
            <div key={w.week} style={{ display: 'flex', gap: 0 }}>
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', width: 36, flexShrink: 0 }}>
                <div style={{ width: 12, height: 12, borderRadius: '50%', background: hasData ? col : '#e2e8f0', boxShadow: hasData ? `0 0 6px ${col}80` : 'none', marginTop: 16, flexShrink: 0 }} />
                {i < chartData.length - 1 && <div style={{ width: 2, flex: 1, background: '#ffffff', minHeight: 16 }} />}
              </div>
              <div style={{ flex: 1, background: '#ffffff', border: `1px solid ${hasData ? `${col}30` : '#ffffff'}`, borderRadius: 10, padding: '12px 14px', marginLeft: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: hasData ? 8 : 0 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: hasData ? '#64748b' : '#e2e8f0', letterSpacing: 1 }}>WEEK {w.week}</div>
                  {hasData
                    ? <div style={{ fontSize: 10, fontWeight: 700, color: col, background: `${col}15`, border: `1px solid ${col}40`, borderRadius: 4, padding: '2px 8px', letterSpacing: 1 }}>{w.riskLevel.toUpperCase()}</div>
                    : <div style={{ fontSize: 10, color: '#e2e8f0' }}>No data</div>
                  }
                </div>
                {hasData && (
                  <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                    {[['PREDICTION', w.prediction], ['CONFIDENCE', `${w.confidence}%`], ['RISK SCORE', `${w.riskScore}/10`], ['WEEKS TO INTERVENE', `${w.weeksLeft} wks`]].map(([label, value]) => (
                      <div key={label} style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 6, padding: '8px 12px', flex: 1, minWidth: 90 }}>
                        <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 1.5, color: '#94a3b8', marginBottom: 4 }}>{label}</div>
                        <div style={{ fontSize: 13, fontWeight: 700, color: '#0f172a' }}>{value}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )
        })}
      </div>

      <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 10, overflow: 'hidden' }}>
        <div style={{ padding: '12px 16px', borderBottom: '1px solid #e2e8f0' }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b' }}>RESULTS BY WEEK</div>
        </div>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ background: '#f8fafc' }}>
              {['Week', 'Prediction', 'Confidence', 'Risk Score', 'F1 Score', 'Weeks Left'].map(h => (
                <th key={h} style={{ padding: '10px 16px', fontSize: 10, fontWeight: 700, color: '#94a3b8', letterSpacing: 1, textAlign: 'left' }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {chartData.map((w, i) => {
              const F1_MAP = { Normal: '92%', Stress: '75%', Anxiety: '82%', Depression: '85%', Bipolar: '82%', 'Personality disorder': '77%', Suicidal: '86%' }
              const f1 = F1_MAP[w.prediction] || '—'
              return (
                <tr key={w.week} style={{ borderTop: '1px solid #ffffff', background: i % 2 === 0 ? 'transparent' : '#f8fafc08' }}>
                  <td style={{ padding: '10px 16px', fontSize: 13, color: '#64748b', fontWeight: 600 }}>Week {w.week}</td>
                  <td style={{ padding: '10px 16px', fontSize: 13, color: w.color, fontWeight: 600 }}>{w.prediction}</td>
                  <td style={{ padding: '10px 16px', fontSize: 13, color: '#0f172a', fontFamily: 'monospace' }}>{w.confidence !== '—' ? `${w.confidence}%` : '—'}</td>
                  <td style={{ padding: '10px 16px', fontSize: 13, color: w.color, fontWeight: 700 }}>{w.riskScore > 0 ? `${w.riskScore}/10` : '—'}</td>
                  <td style={{ padding: '10px 16px', fontSize: 13, color: w.riskScore > 0 ? '#c4b5fd' : '#94a3b8', fontFamily: 'monospace', fontWeight: 600 }}>{f1}</td>
                  <td style={{ padding: '10px 16px', fontSize: 13, color: '#64748b' }}>{w.weeksLeft > 0 ? w.weeksLeft : '—'}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ─── Analysis Tab ─────────────────────────────────────────────────────────────

function WeekAnalysisCard({ weekNum, weekData, isLatest }) {
  const [open, setOpen] = useState(false)
  const result = weekData?.result
  const text   = weekData?.text
  if (!result) return null

  const col = RISK_COLOR[result.risk_level] || '#64748b'

  return (
    <div style={{
      background: '#ffffff',
      border: `1px solid ${open ? `${col}50` : '#e2e8f0'}`,
      borderRadius: 10,
      overflow: 'hidden',
      transition: 'border-color 0.2s',
    }}>
      {/* Card header — always visible, clickable */}
      <div
        onClick={() => setOpen(o => !o)}
        style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '14px 16px', cursor: 'pointer', userSelect: 'none' }}
      >
        {/* Week label */}
        <div style={{ fontSize: 11, fontWeight: 700, color: '#64748b', letterSpacing: 1, minWidth: 52 }}>
          WEEK {weekNum}
          {isLatest && <span style={{ marginLeft: 6, fontSize: 9, color: '#6366f1', background: '#6366f115', border: '1px solid #6366f130', borderRadius: 4, padding: '1px 5px', letterSpacing: 0.5 }}>LATEST</span>}
        </div>

        {/* Risk badge */}
        <div style={{ fontSize: 10, fontWeight: 700, color: col, background: `${col}15`, border: `1px solid ${col}40`, borderRadius: 4, padding: '2px 8px', letterSpacing: 1 }}>
          {result.risk_level?.toUpperCase()}
        </div>

        {/* Prediction */}
        <div style={{ fontSize: 13, fontWeight: 700, color: col }}>{result.prediction}</div>

        {/* Confidence */}
        <div style={{ fontSize: 12, color: '#64748b', marginLeft: 2 }}>{(result.confidence * 100).toFixed(1)}%</div>

        {/* Text preview */}
        <div style={{ flex: 1, fontSize: 12, color: '#94a3b8', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontStyle: 'italic' }}>
          "{text}"
        </div>

        {/* Chevron */}
        <div style={{ fontSize: 12, color: '#94a3b8', transform: open ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s', flexShrink: 0 }}>▼</div>
      </div>

      {/* Expanded content */}
      {open && (
        <div style={{ borderTop: `1px solid ${col}30`, padding: '16px', display: 'flex', flexDirection: 'column', gap: 14 }}>
          {/* Full text */}
          <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '12px 14px' }}>
            <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 1.5, color: '#94a3b8', marginBottom: 6 }}>STUDENT TEXT</div>
            <div style={{ fontSize: 13, color: '#64748b', lineHeight: 1.7, fontStyle: 'italic' }}>"{text}"</div>
          </div>

          <MetricCards prediction={result.prediction} confidence={result.confidence} stressScore={result.stress_score} />
          <AttentionHeatmap tokens={result.attention} prediction={result.prediction} />
          <ProbabilityBars probabilities={result.probabilities} />
        </div>
      )}
    </div>
  )
}

function AnalysisTab({ student, latest, worst }) {
  if (!latest) {
    return <div style={{ fontSize: 13, color: '#94a3b8', padding: 20 }}>No analysis data available for this student.</div>
  }

  const filledWeeks = WEEK_NUMS.filter(w => student.weeks?.[w]?.result)
  const latestWeek  = filledWeeks[filledWeeks.length - 1]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>

      {/* Overall summary row */}
      <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 10, padding: '14px 18px' }}>
        <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 10 }}>OVERALL SUMMARY</div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {[
            ['WEEKS ANALYZED', `${filledWeeks.length} / ${WEEK_NUMS.length}`],
            ['WORST PREDICTION', worst.prediction],
            ['WORST RISK', worst.risk_level],
            ['LATEST STRESS SCORE', `${latest.stress_score} / 100`],
          ].map(([label, value]) => (
            <div key={label} style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '10px 14px', flex: 1, minWidth: 110 }}>
              <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 1.5, color: '#94a3b8', marginBottom: 4 }}>{label}</div>
              <div style={{ fontSize: 14, fontWeight: 700, color: '#0f172a' }}>{value}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Advisor recommendation based on worst week */}
      <AdvisorRecommendation prediction={worst.prediction} />

      {/* Per-week collapsible cards */}
      <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b' }}>
        WEEKLY SUBMISSIONS — click any week to view XAI heatmap
      </div>
      {WEEK_NUMS.map(w => (
        <WeekAnalysisCard
          key={w}
          weekNum={w}
          weekData={student.weeks?.[w]}
          isLatest={w === latestWeek}
        />
      ))}

      {/* Model performance */}
      <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 10, padding: '16px 20px' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, flexWrap: 'wrap', gap: 8 }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b' }}>MODEL PERFORMANCE</div>
          <div style={{ fontSize: 10, color: '#e2e8f0', fontWeight: 600, letterSpacing: 1 }}>IT22196392 — R26-IT-059</div>
        </div>
        <div style={{ fontSize: 12, fontWeight: 600, color: '#6366f1', marginBottom: 12 }}>BERT Fine-tuned + RoBERTa Ensemble</div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {[['ACCURACY', '82.32%', 'Test set'], ['F1 SCORE', '82.32%', 'Weighted avg'], ['DATASET', '51,055', 'Training samples'], ['CLASSES', '7', 'Mental health categories']].map(([label, value, sub]) => (
            <div key={label} style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '12px 16px', flex: 1, minWidth: 100 }}>
              <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 6 }}>{label}</div>
              <div style={{ fontSize: 20, fontWeight: 800, color: '#0f172a' }}>{value}</div>
              <div style={{ fontSize: 10, color: '#94a3b8', marginTop: 3 }}>{sub}</div>
            </div>
          ))}
        </div>
      </div>

      {/* CSV export */}
      <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 10, padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b', marginBottom: 4 }}>EXPORT DATA</div>
          <div style={{ fontSize: 11, color: '#94a3b8' }}>Export for Meta-Integration Layer</div>
        </div>
        <button
          onClick={() => {
            const rows = [
              'student_name,week,predicted_label,confidence,risk_level,stress_score',
              ...WEEK_NUMS.flatMap(w => {
                const r = student.weeks?.[w]?.result
                if (!r) return []
                return [`${student.name},${w},${r.prediction},${r.confidence},${r.risk_level},${r.stress_score}`]
              }),
            ].join('\n')
            const a = document.createElement('a')
            a.href = URL.createObjectURL(new Blob([rows], { type: 'text/csv' }))
            a.download = `stress_scores_${student.name.replace(/\s+/g, '_')}.csv`
            a.click()
          }}
          style={{ display: 'flex', alignItems: 'center', gap: 8, background: '#dcfce7', border: '1px solid #22c55e50', color: '#15803d', fontSize: 13, fontWeight: 700, padding: '10px 20px', borderRadius: 8, cursor: 'pointer', boxShadow: '0 0 12px #22c55e20' }}
          onMouseEnter={e => e.currentTarget.style.background = '#bbf7d0'}
          onMouseLeave={e => e.currentTarget.style.background = '#dcfce7'}
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
          </svg>
          Download Stress Scores CSV
        </button>
      </div>
    </div>
  )
}

// ─── Empty state ───────────────────────────────────────────────────────────────

function WelcomeState({ onAddNew }) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 16, height: '100%', minHeight: 400, padding: 40 }}>
      <div style={{ fontSize: 48 }}>👤</div>
      <div style={{ fontSize: 18, fontWeight: 700, color: '#0f172a' }}>No student selected</div>
      <div style={{ fontSize: 14, color: '#94a3b8', textAlign: 'center', maxWidth: 340, lineHeight: 1.7 }}>
        Select a student from the left panel, or add a new student to begin analyzing their emotional progression.
      </div>
      <button
        onClick={onAddNew}
        style={{
          background: '#dbeafe', border: '1px solid #3b82f640',
          color: '#1d4ed8', fontSize: 13, fontWeight: 700,
          padding: '10px 24px', borderRadius: 8, cursor: 'pointer',
          marginTop: 8,
        }}
      >
        + Add New Student
      </button>
    </div>
  )
}

// ─── Main StudentDetail ────────────────────────────────────────────────────────

export default function StudentDetail({ student, addingNew, forcedTab, onTabChange, onStudentSaved }) {
  const [tab, setTab] = useState('analysis')

  useEffect(() => {
    if (forcedTab) setTab(forcedTab)
  }, [forcedTab, student?.id])

  const switchTab = (t) => {
    setTab(t)
    onTabChange?.(t)
  }

  // No student selected and not adding new → welcome screen
  if (!student && !addingNew) {
    return <WelcomeState onAddNew={() => { switchTab('progression'); onStudentSaved?.({ id: '__new__' }) }} />
  }

  // Adding new student → go straight to the form
  if (addingNew || !student) {
    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: 0, height: '100%' }}>
        <div style={{ padding: '16px 24px 0' }}>
          <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: 2, color: '#64748b' }}>NEW STUDENT</div>
        </div>
        <AddStudentForm onSaved={onStudentSaved} />
      </div>
    )
  }

  // Saved student selected
  const latest = student.latestResult
  const worst  = student.worstResult || latest
  const isCritical = worst?.risk_level === 'Critical'

  const TABS = [
    { id: 'analysis',    label: 'Analysis' },
    { id: 'progression', label: 'Weekly Progression' },
  ]

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16, padding: '24px', flex: 1, overflowY: 'auto' }}>

      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
        <div>
          <div style={{ fontSize: 22, fontWeight: 800, color: '#0f172a' }}>{student.name}</div>
          <div style={{ fontSize: 12, color: '#94a3b8', marginTop: 4 }}>ID: {student.id}</div>
        </div>
        {worst && <RiskBadge riskLevel={worst.risk_level} />}
      </div>

      {/* Critical alert */}
      {isCritical && (
        <div style={{ background: '#ff000012', border: '1px solid #ff000040', borderRadius: 10, padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 10 }}>
          <span style={{ fontSize: 18 }}>⚠</span>
          <div>
            <div style={{ fontSize: 13, fontWeight: 700, color: '#ff4444' }}>Immediate Attention Required</div>
            <div style={{ fontSize: 12, color: '#f87171', marginTop: 2 }}>This student's text indicates critical distress. Contact a mental health professional immediately.</div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div style={{ display: 'flex', gap: 4, borderBottom: '1px solid #ffffff' }}>
        {TABS.map(t => (
          <button
            key={t.id}
            onClick={() => switchTab(t.id)}
            style={{
              background: 'none', border: 'none',
              borderBottom: tab === t.id ? '2px solid #6366f1' : '2px solid transparent',
              color: tab === t.id ? '#0f172a' : '#94a3b8',
              fontSize: 13, fontWeight: tab === t.id ? 700 : 400,
              padding: '8px 16px', cursor: 'pointer', marginBottom: -1,
              transition: 'color 0.15s',
            }}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Analysis tab ── */}
      {tab === 'analysis' && <AnalysisTab student={student} latest={latest} worst={worst} />}

      {/* ── Weekly Progression tab ── */}
      {tab === 'progression' && <SavedProgression student={student} />}
    </div>
  )
}

import MetricCards from './MetricCards'
import RiskBadge from './RiskBadge'
import AttentionHeatmap from './AttentionHeatmap'
import ProbabilityBars from './ProbabilityBars'

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

export default function StudentDetail({ student, result }) {
  if (!student) return <EmptyState />
  if (!result || result === 'loading') return <Spinner />

  const isCritical = result.risk_level === 'Critical'

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

      <ProbabilityBars probabilities={result.probabilities} />
    </div>
  )
}

function Box({ label, color, bg, border, star, sub }) {
  return (
    <div style={{
      background: bg,
      border: `1px solid ${border}`,
      borderRadius: 8,
      padding: '10px 14px',
      textAlign: 'center',
      minWidth: 130,
      flex: 1,
      boxShadow: star ? `0 0 18px ${border}` : `0 0 6px ${border}40`,
      position: 'relative',
    }}>
      {star && (
        <div style={{
          position: 'absolute', top: -10, left: '50%', transform: 'translateX(-50%)',
          background: '#7c3aed', border: '1px solid #a855f7',
          borderRadius: 4, fontSize: 9, fontWeight: 800,
          color: '#e9d5ff', padding: '1px 7px', letterSpacing: 1, whiteSpace: 'nowrap',
        }}>⭐ BEST MODEL</div>
      )}
      <div style={{ fontSize: 12, fontWeight: 700, color, lineHeight: 1.4 }}>{label}</div>
      {sub && <div style={{ fontSize: 9, color: `${color}99`, marginTop: 3 }}>{sub}</div>}
    </div>
  )
}

function Arrow({ down, label }) {
  return down ? (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0, margin: '4px 0' }}>
      <div style={{ width: 2, height: 20, background: '#cbd5e1' }} />
      <div style={{ width: 0, height: 0, borderLeft: '5px solid transparent', borderRight: '5px solid transparent', borderTop: '7px solid #cbd5e1' }} />
      {label && <div style={{ fontSize: 9, color: '#64748b', marginTop: 3 }}>{label}</div>}
    </div>
  ) : (
    <div style={{ display: 'flex', alignItems: 'center', gap: 0, flexShrink: 0, padding: '0 4px' }}>
      <div style={{ height: 2, width: 20, background: '#cbd5e1' }} />
      <div style={{ width: 0, height: 0, borderTop: '5px solid transparent', borderBottom: '5px solid transparent', borderLeft: '7px solid #cbd5e1' }} />
    </div>
  )
}

function SectionLabel({ children }) {
  return (
    <div style={{ fontSize: 9, fontWeight: 700, letterSpacing: 2, color: '#64748b', textAlign: 'center', marginBottom: 6 }}>
      {children}
    </div>
  )
}

const THEME = {
  input:      { color: '#1d4ed8', bg: '#dbeafe', border: '#3b82f640' },
  baseline:   { color: '#475569', bg: '#ffffff', border: '#cbd5e1' },
  pretrained: { color: '#a16207', bg: '#fef9c3', border: '#ca8a0440' },
  finetuned:  { color: '#15803d', bg: '#dcfce7', border: '#22c55e40' },
  ensemble:   { color: '#7e22ce', bg: '#f3e8ff', border: '#a855f7'   },
  xai:        { color: '#c2410c', bg: '#ffedd5', border: '#f9731640' },
  output:     { color: '#b91c1c', bg: '#fee2e2', border: '#ef444440' },
}

export default function ArchitectureDiagram() {
  return (
    <div style={{
      padding: '32px 40px',
      background: '#f1f5f9',
      minHeight: '100%',
      fontFamily: "'Inter','Segoe UI',system-ui,sans-serif",
    }}>
      <div style={{ textAlign: 'center', marginBottom: 32 }}>
        <div style={{ fontSize: 20, fontWeight: 800, color: '#0f172a', marginBottom: 6 }}>
          Emotional Burnout Detection System Architecture
        </div>
        <div style={{ fontSize: 12, color: '#64748b' }}>
          IT22196392 — Induwara K.P.Y. &nbsp;·&nbsp; R26-IT-059
        </div>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 0, maxWidth: 960, margin: '0 auto' }}>

        {/* Row 1 — Input */}
        <div style={{ width: '100%' }}>
          <SectionLabel>INPUT</SectionLabel>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0 }}>
            <Box label="Student Text Input" sub="Raw journal / survey text" {...THEME.input} />
            <Arrow />
            <Box label="Text Preprocessing & Cleaning" sub="Tokenization · stopwords · lowercasing" {...THEME.input} />
          </div>
        </div>

        <Arrow down label="feed to models" />

        {/* Row 2 — Models */}
        <div style={{ width: '100%' }}>
          <SectionLabel>MODEL PIPELINE — 10 MODELS</SectionLabel>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {/* Baseline row */}
            <div style={{ display: 'flex', gap: 8 }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: '#64748b', writingMode: 'vertical-rl', textOrientation: 'mixed', transform: 'rotate(180deg)', letterSpacing: 1, alignSelf: 'center', marginRight: 2 }}>BASELINE ML</div>
              <Box label="Logistic Regression" sub="Model 1 · 72.4%" {...THEME.baseline} />
              <Box label="Random Forest" sub="Model 2 · 62.0%" {...THEME.baseline} />
              <Box label="SVM" sub="Model 3 · 73.0%" {...THEME.baseline} />
              <Box label="LSTM" sub="Model 4 · 78.0%" {...THEME.baseline} />
            </div>

            {/* Pretrained / Zero-shot row */}
            <div style={{ display: 'flex', gap: 8 }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: '#64748b', writingMode: 'vertical-rl', textOrientation: 'mixed', transform: 'rotate(180deg)', letterSpacing: 1, alignSelf: 'center', marginRight: 2 }}>ZERO-SHOT</div>
              <Box label="VADER" sub="Model 5 · 34.3%" {...THEME.pretrained} />
              <Box label="RoBERTa Zero-Shot" sub="Model 6 · 38.8%" {...THEME.pretrained} />
            </div>

            {/* Fine-tuned row */}
            <div style={{ display: 'flex', gap: 8 }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: '#64748b', writingMode: 'vertical-rl', textOrientation: 'mixed', transform: 'rotate(180deg)', letterSpacing: 1, alignSelf: 'center', marginRight: 2 }}>FINE-TUNED</div>
              <Box label="BERT Fine-tuned" sub="Model 7 · 82.3%" {...THEME.finetuned} />
              <Box label="RoBERTa Fine-tuned" sub="Model 8 · 81.7%" {...THEME.finetuned} />
              <Box label="Mental-RoBERTa" sub="Model 9 · domain-adapted" {...THEME.finetuned} />
            </div>

            {/* Ensemble row */}
            <div style={{ display: 'flex', gap: 8 }}>
              <div style={{ fontSize: 9, fontWeight: 700, color: '#64748b', writingMode: 'vertical-rl', textOrientation: 'mixed', transform: 'rotate(180deg)', letterSpacing: 1, alignSelf: 'center', marginRight: 2 }}>ENSEMBLE</div>
              <Box label="BERT + RoBERTa Ensemble" sub="Model 10 · 82.32% · Weighted Soft Voting" {...THEME.ensemble} star />
            </div>
          </div>
        </div>

        <Arrow down label="XAI layer" />

        {/* Row 3 — XAI */}
        <div style={{ width: '100%' }}>
          <SectionLabel>EXPLAINABILITY (XAI)</SectionLabel>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0 }}>
            <Box label="Attention Weight Extraction" sub="Avg 12 layers × 12 heads · CLS row" {...THEME.xai} />
            <Arrow />
            <Box label="Phrase-Level Heatmap Visualization" sub="Token intensity · gray→blue→yellow→orange→red" {...THEME.xai} />
          </div>
        </div>

        <Arrow down label="output" />

        {/* Row 4 — Output */}
        <div style={{ width: '100%' }}>
          <SectionLabel>OUTPUT</SectionLabel>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 0, flexWrap: 'wrap' }}>
            <Box label="7-Class Prediction" sub="Normal · Anxiety · Stress · Depression · Bipolar · PD · Suicidal" {...THEME.output} />
            <Arrow />
            <Box label="Risk Level" sub="Low · Medium · High · Critical" {...THEME.output} />
            <Arrow />
            <Box label="Stress Score CSV" sub="Weighted distress index · export" {...THEME.output} />
            <Arrow />
            <Box label="Meta-Integration Layer" sub="Dashboard · advisor alert · early intervention" {...THEME.output} />
          </div>
        </div>

      </div>

      {/* Legend */}
      <div style={{
        display: 'flex', gap: 12, flexWrap: 'wrap', justifyContent: 'center',
        marginTop: 36, padding: '14px 20px',
        background: '#ffffff', border: '1px solid #cbd5e1', borderRadius: 10,
        maxWidth: 960, margin: '36px auto 0',
      }}>
        {[
          { label: 'Input / Preprocessing', ...THEME.input },
          { label: 'Baseline ML Models',    ...THEME.baseline },
          { label: 'Zero-Shot / Rule-Based', ...THEME.pretrained },
          { label: 'Fine-tuned Transformers', ...THEME.finetuned },
          { label: 'Ensemble (Best)',        ...THEME.ensemble },
          { label: 'XAI Explainability',    ...THEME.xai },
          { label: 'Output Layer',          ...THEME.output },
        ].map(({ label, color, bg, border }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <div style={{ width: 12, height: 12, borderRadius: 3, background: bg, border: `1px solid ${border}` }} />
            <span style={{ fontSize: 11, color }}>{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

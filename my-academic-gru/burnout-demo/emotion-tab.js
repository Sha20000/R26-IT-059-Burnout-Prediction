/* ============================================================================
 * Emotional Analysis tab — integrated dashboard
 * IT22196392 — Induwara K.P.Y. | R26-IT-059
 *
 * Vanilla-JS port of the standalone Next.js frontend
 * (emotional-nlp-fresh/emotional_burnout_nlp/frontend), so the integration
 * layer keeps the full XAI depth of the original: attention heatmap,
 * all-class probabilities, advisor recommendation, weekly progression and
 * lead-time analysis.
 *
 * Talks directly to serve_model.py (port 5004) — the warm BERT service —
 * the same way the other members' tabs talk to 5001/5002/5003.
 * ========================================================================== */

(function () {
  'use strict';

  const EMO_API = 'http://localhost:5004';

  // ─── Constants (ported from StudentDetail.jsx) ────────────────────────────

  const WEEK_NUMS = [2, 4, 8, 18];

  const RISK_COLOR = {
    Low: '#16a34a', Medium: '#b45309', High: '#f97316',
    Critical: '#ef4444', Unknown: '#475569',
  };

  // Risk severity on a 0-10 scale, used for the progression chart
  const RISK_SCORE = { Low: 1, Medium: 3, High: 7, Critical: 10, Unknown: 0 };

  // Lead-time analysis: estimated weeks remaining to intervene before burnout
  const RISK_LEFT = { Low: 14, Medium: 12, High: 8, Critical: 4, Unknown: 0 };

  const CLASS_COLORS = {
    Normal: '#0891b2', Anxiety: '#b45309', Stress: '#f97316',
    Depression: '#dc2626', Bipolar: '#a855f7',
    'Personality disorder': '#ec4899', Suicidal: '#ff2020',
  };

  const PER_CLASS_F1 = {
    Normal: '92%', Stress: '75%', Anxiety: '82%', Depression: '85%',
    Bipolar: '82%', 'Personality disorder': '77%', Suicidal: '86%',
  };

  const ADVISOR_MAP = {
    Normal: {
      text: '✅ Student appears emotionally stable. Continue routine monitoring.',
      color: '#0891b2', bg: '#ecfeff', border: '#22d3ee55',
    },
    Stress: {
      text: '⚠️ Student showing stress indicators. Recommend check-in with advisor.',
      color: '#b45309', bg: '#fffbeb', border: '#fbbf2455',
    },
    Anxiety: {
      text: '⚠️ Student showing anxiety indicators. Recommend counseling session.',
      color: '#b45309', bg: '#fffbeb', border: '#fbbf2455',
    },
    Depression: {
      text: '🔴 HIGH RISK: Student showing depression indicators. Immediate counseling recommended. Key trigger words are highlighted in the attention heatmap above.',
      color: '#dc2626', bg: '#fef2f2', border: '#f8717155',
    },
    Suicidal: {
      text: '🚨 CRITICAL: Student showing suicidal ideation. IMMEDIATE intervention required. Contact student NOW.',
      color: '#ff2020', bg: '#fef2f2', border: '#ff202070',
    },
    Bipolar: {
      text: '🔴 HIGH RISK: Student showing bipolar indicators. Recommend psychiatric evaluation.',
      color: '#dc2626', bg: '#fef2f2', border: '#f8717155',
    },
    'Personality disorder': {
      text: '🔴 HIGH RISK: Student showing personality disorder indicators. Recommend professional evaluation.',
      color: '#dc2626', bg: '#fef2f2', border: '#f8717155',
    },
  };

  // Tokens that are never informative as "top words" — the original frontend
  // showed raw top-3 attention tokens, which surfaced things like "i, don, '".
  const STOPWORDS = new Set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'you', 'your', 'he', 'him', 'his',
    'she', 'her', 'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
    'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'into', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'then', 'once', 'here',
    'there', 'all', 'any', 'both', 'each', 'more', 'most', 'other', 'some',
    'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
    'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'm',
    're', 've', 'll', 'd', 'o', 'y', 'get', 'got', 'like', 'even',
  ]);

  const isContentWord = (tok) =>
    tok && tok.length > 2 && /[a-z]/i.test(tok) && !STOPWORDS.has(tok.toLowerCase());

  // ─── Utilities ─────────────────────────────────────────────────────────────

  const esc = (s) => String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');

  const el = (id) => document.getElementById(id);

  const riskOf = (level) => RISK_COLOR[level] || RISK_COLOR.Unknown;

  /** Top N attention tokens, content words preferred, falling back to raw. */
  function topWords(attention, n) {
    if (!attention || !attention.length) return [];
    const sorted = [...attention].sort((a, b) => b.weight - a.weight);
    const content = sorted.filter((t) => isContentWord(t.token));
    return (content.length >= n ? content : sorted).slice(0, n).map((t) => t.token);
  }

  // ─── Card shell ────────────────────────────────────────────────────────────

  function card(label, bodyHtml, extraStyle) {
    return `
      <div class="emo-card" style="${extraStyle || ''}">
        ${label ? `<div class="emo-card-label">${esc(label)}</div>` : ''}
        ${bodyHtml}
      </div>`;
  }

  // ─── Metric cards (MetricCards.jsx) ────────────────────────────────────────

  function renderMetricCards(result) {
    const items = [
      ['PREDICTION', result.prediction, 'Mental health category'],
      ['CONFIDENCE', `${(result.confidence * 100).toFixed(1)}%`, 'Model certainty'],
      ['STRESS SCORE', `${result.stress_score} / 100`, 'Distress intensity'],
      ['RISK LEVEL', result.risk_level, 'Intervention tier'],
    ];
    return `
      <div class="emo-metrics">
        ${items.map(([label, value, sub]) => `
          <div class="emo-metric">
            <div class="emo-metric-label">${esc(label)}</div>
            <div class="emo-metric-value">${esc(value)}</div>
            <div class="emo-metric-sub">${esc(sub)}</div>
          </div>`).join('')}
      </div>`;
  }

  // ─── Attention heatmap (AttentionHeatmap.jsx) ──────────────────────────────

  function tokenStyle(intensity) {
    if (intensity < 0.2) return { bg: 'rgba(226,232,240,0.9)', border: 'rgba(203,213,225,0.9)', text: '#64748b', glow: null };
    if (intensity < 0.4) return { bg: 'rgba(59,130,246,0.55)', border: 'rgba(59,130,246,0.35)', text: '#1d4ed8', glow: 'rgba(59,130,246,0.5)' };
    if (intensity < 0.6) return { bg: 'rgba(234,179,8,0.55)', border: 'rgba(234,179,8,0.35)', text: '#a16207', glow: 'rgba(234,179,8,0.5)' };
    if (intensity < 0.8) return { bg: 'rgba(249,115,22,0.65)', border: 'rgba(249,115,22,0.4)', text: '#c2410c', glow: 'rgba(249,115,22,0.6)' };
    return { bg: 'rgba(239,68,68,0.85)', border: 'rgba(239,68,68,0.6)', text: '#ffffff', glow: 'rgba(239,68,68,0.7)' };
  }

  function renderAttentionHeatmap(tokens, prediction) {
    if (!tokens || !tokens.length) return '';
    const maxWeight = Math.max(...tokens.map((t) => t.weight));

    const chips = tokens.map(({ token, weight }) => {
      const intensity = maxWeight > 0 ? weight / maxWeight : 0;
      const s = tokenStyle(intensity);
      return `<span class="emo-token"
        title="&quot;${esc(token)}&quot; — attention: ${(weight * 100).toFixed(2)}%"
        style="background:${s.bg};color:${s.text};border:1px solid ${s.border};${s.glow ? `box-shadow:0 0 8px ${s.glow};` : ''}"
      >${esc(token)}</span>`;
    }).join('');

    const bands = [
      ['0–20%', '#64748b', 'rgba(226,232,240,0.9)'],
      ['20–40%', '#1d4ed8', 'rgba(59,130,246,0.55)'],
      ['40–60%', '#a16207', 'rgba(234,179,8,0.55)'],
      ['60–80%', '#c2410c', 'rgba(249,115,22,0.65)'],
      ['80–100%', '#b91c1c', 'rgba(239,68,68,0.85)'],
    ].map(([label, color, bg]) => `
      <div class="emo-band">
        <div class="emo-band-swatch" style="background:${bg}"></div>
        <span style="color:${color}">${label}</span>
      </div>`).join('');

    return card('ATTENTION · XAI EXPLAINABILITY', `
      <div class="emo-card-sub">
        Words driving the <strong>"${esc(prediction)}"</strong> prediction
      </div>
      <div class="emo-tokens">${chips}</div>
      <div class="emo-scale">
        <span class="emo-scale-end">LOW</span>
        <div class="emo-scale-bar"></div>
        <span class="emo-scale-end" style="color:#dc2626">HIGH ATTENTION</span>
        <span class="emo-scale-hint">Hover any token for its exact %</span>
      </div>
      <div class="emo-bands">${bands}</div>
    `);
  }

  // ─── Probability bars (ProbabilityBars.jsx) ────────────────────────────────

  function renderProbabilityBars(probabilities) {
    if (!probabilities) return '';
    const sorted = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    const rows = sorted.map(([label, prob]) => {
      const color = CLASS_COLORS[label] || '#475569';
      return `
        <div class="emo-prob-row">
          <div class="emo-prob-head">
            <span>${esc(label)}</span>
            <span style="color:${color};font-weight:600">${(prob * 100).toFixed(1)}%</span>
          </div>
          <div class="emo-prob-track">
            <div class="emo-prob-fill" style="width:${prob * 100}%;background:${color};${prob > 0.5 ? `box-shadow:0 0 8px ${color}80;` : ''}"></div>
          </div>
        </div>`;
    }).join('');

    return card('ALL CLASS PROBABILITIES', `<div class="emo-probs">${rows}</div>`);
  }

  // ─── Advisor recommendation ────────────────────────────────────────────────

  function renderAdvisor(prediction) {
    const rec = ADVISOR_MAP[prediction];
    if (!rec) return '';
    return `
      <div class="emo-advisor" style="background:${rec.bg};border:1px solid ${rec.border}">
        <div class="emo-card-label">ADVISOR RECOMMENDATION</div>
        <div style="color:${rec.color};font-size:13px;line-height:1.6;font-weight:500">${esc(rec.text)}</div>
      </div>`;
  }

  // ─── Model performance panel ───────────────────────────────────────────────

  function renderModelPerformance() {
    const stats = [
      ['ACCURACY', '82.32%', 'Test set'],
      ['F1 SCORE', '82.32%', 'Weighted avg'],
      ['DATASET', '51,055', 'Training samples'],
      ['CLASSES', '7', 'Mental health categories'],
    ];
    return card(null, `
      <div class="emo-perf-head">
        <div class="emo-card-label" style="margin:0">MODEL PERFORMANCE</div>
        <div class="emo-perf-author">IT22196392 — R26-IT-059</div>
      </div>
      <div class="emo-perf-model">BERT Fine-tuned + RoBERTa Ensemble</div>
      <div class="emo-metrics">
        ${stats.map(([label, value, sub]) => `
          <div class="emo-metric">
            <div class="emo-metric-label">${esc(label)}</div>
            <div class="emo-metric-value">${esc(value)}</div>
            <div class="emo-metric-sub">${esc(sub)}</div>
          </div>`).join('')}
      </div>
    `);
  }

  // ─── Risk progression chart (RiskChart in StudentDetail.jsx) ───────────────

  function renderRiskChart(data) {
    const W = 520, H = 200;
    const padL = 44, padR = 24, padT = 36, padB = 36;
    const chartW = W - padL - padR;
    const chartH = H - padT - padB;
    const minWeek = WEEK_NUMS[0];
    const maxWeek = WEEK_NUMS[WEEK_NUMS.length - 1];
    const xOf = (w) => padL + ((w - minWeek) / (maxWeek - minWeek)) * chartW;
    const yOf = (s) => padT + (1 - s / 10) * chartH;
    const yTicks = [0, 2, 4, 6, 8, 10];

    const points = data.map((d) => ({
      x: xOf(d.week), y: yOf(d.riskScore),
      color: d.color, riskScore: d.riskScore,
    }));

    const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');
    const areaPath = `${linePath} L${points[points.length - 1].x},${padT + chartH} L${points[0].x},${padT + chartH} Z`;
    const maxScore = Math.max(...data.map((d) => d.riskScore));
    const areaColor = maxScore >= 7 ? '#ef4444' : maxScore >= 3 ? '#f97316' : '#16a34a';

    const grid = yTicks.map((v) => `
      <line x1="${padL}" y1="${yOf(v)}" x2="${padL + chartW}" y2="${yOf(v)}"
            stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 4" />
      <text x="${padL - 6}" y="${yOf(v) + 4}" text-anchor="end" font-size="9" fill="#64748b">${v}</text>
    `).join('');

    const weekLabels = WEEK_NUMS.map((w) =>
      `<text x="${xOf(w)}" y="${padT + chartH + 16}" text-anchor="middle" font-size="9" fill="#64748b">Wk ${w}</text>`
    ).join('');

    const stops = points.map((p, i) =>
      `<stop offset="${(i / Math.max(1, points.length - 1)) * 100}%" stop-color="${p.color}" />`
    ).join('');

    const dots = points.map((p) => `
      <circle cx="${p.x}" cy="${p.y}" r="7" fill="#ffffff" stroke="${p.color}" stroke-width="2.5" />
      <circle cx="${p.x}" cy="${p.y}" r="3" fill="${p.color}" />
      <text x="${p.x}" y="${p.y - 13}" text-anchor="middle" font-size="10" font-weight="700" fill="${p.color}">${p.riskScore}</text>
    `).join('');

    return card('EMOTIONAL BURNOUT RISK PROGRESSION', `
      <svg viewBox="0 0 ${W} ${H}" style="width:100%;display:block" role="img"
           aria-label="Risk score across weeks 2, 4, 8 and 18">
        <defs>
          <linearGradient id="emoLineGrad" x1="0%" y1="0%" x2="100%" y2="0%">${stops}</linearGradient>
          <linearGradient id="emoAreaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="${areaColor}" stop-opacity="0.25" />
            <stop offset="100%" stop-color="${areaColor}" stop-opacity="0.02" />
          </linearGradient>
        </defs>
        ${grid}
        <line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + chartH}" stroke="#cbd5e1" />
        <line x1="${padL}" y1="${padT + chartH}" x2="${padL + chartW}" y2="${padT + chartH}" stroke="#cbd5e1" />
        ${weekLabels}
        <path d="${areaPath}" fill="url(#emoAreaGrad)" />
        <path d="${linePath}" fill="none" stroke="url(#emoLineGrad)" stroke-width="2.5"
              stroke-linejoin="round" stroke-linecap="round" />
        ${dots}
        <text x="${padL - 2}" y="${padT - 10}" font-size="9" fill="#64748b" text-anchor="middle">Score</text>
      </svg>
    `);
  }

  // ─── Weekly timeline + lead-time table ─────────────────────────────────────

  function weekChartData(student) {
    return WEEK_NUMS.map((w) => {
      const r = student.weeks && student.weeks[w] ? student.weeks[w].result : null;
      const rl = (r && r.risk_level) || 'Unknown';
      return {
        week: w,
        prediction: (r && r.prediction) || '—',
        confidence: r ? (r.confidence * 100).toFixed(1) : '—',
        riskLevel: rl,
        riskScore: RISK_SCORE[rl] != null ? RISK_SCORE[rl] : 0,
        weeksLeft: RISK_LEFT[rl] != null ? RISK_LEFT[rl] : 0,
        color: riskOf(rl),
        hasData: !!r,
      };
    });
  }

  function renderTimeline(chartData) {
    const items = chartData.map((w, i) => {
      const col = w.color;
      const tiles = [
        ['PREDICTION', w.prediction],
        ['CONFIDENCE', `${w.confidence}%`],
        ['RISK SCORE', `${w.riskScore}/10`],
        ['WEEKS TO INTERVENE', `${w.weeksLeft} wks`],
      ].map(([label, value]) => `
        <div class="emo-tile">
          <div class="emo-tile-label">${esc(label)}</div>
          <div class="emo-tile-value">${esc(value)}</div>
        </div>`).join('');

      return `
        <div class="emo-tl-row">
          <div class="emo-tl-rail">
            <div class="emo-tl-dot" style="background:${w.hasData ? col : '#cbd5e1'};${w.hasData ? `box-shadow:0 0 6px ${col}80` : ''}"></div>
            ${i < chartData.length - 1 ? '<div class="emo-tl-line"></div>' : ''}
          </div>
          <div class="emo-tl-body" style="border-color:${w.hasData ? col + '55' : '#e2e8f0'}">
            <div class="emo-tl-head">
              <span class="emo-tl-week">WEEK ${w.week}</span>
              ${w.hasData
                ? `<span class="emo-pill" style="color:${col};background:${col}15;border:1px solid ${col}40">${esc(w.riskLevel.toUpperCase())}</span>`
                : '<span class="emo-tl-nodata">No data</span>'}
            </div>
            ${w.hasData ? `<div class="emo-tiles">${tiles}</div>` : ''}
          </div>
        </div>`;
    }).join('');

    return `<div class="emo-timeline">${items}</div>`;
  }

  function renderResultsTable(chartData) {
    const rows = chartData.map((w) => `
      <tr>
        <td style="font-weight:600">Week ${w.week}</td>
        <td style="color:${w.color};font-weight:600">${esc(w.prediction)}</td>
        <td style="font-family:monospace">${w.confidence !== '—' ? `${w.confidence}%` : '—'}</td>
        <td style="color:${w.color};font-weight:700">${w.riskScore > 0 ? `${w.riskScore}/10` : '—'}</td>
        <td style="font-family:monospace;color:${w.hasData ? '#6d28d9' : '#64748b'};font-weight:600">${PER_CLASS_F1[w.prediction] || '—'}</td>
        <td>${w.weeksLeft > 0 ? w.weeksLeft : '—'}</td>
      </tr>`).join('');

    return card('RESULTS BY WEEK — LEAD-TIME ANALYSIS', `
      <div class="table-wrapper" style="margin-top:10px">
        <table>
          <thead>
            <tr>
              <th>Week</th><th>Prediction</th><th>Confidence</th>
              <th>Risk Score</th><th>F1 Score</th><th>Weeks Left</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `);
  }

  // ─── Expose for the tab controller ─────────────────────────────────────────

  window.EmotionTab = {
    EMO_API, WEEK_NUMS, RISK_COLOR, RISK_SCORE, RISK_LEFT, PER_CLASS_F1,
    esc, el, riskOf, topWords, card,
    renderMetricCards, renderAttentionHeatmap, renderProbabilityBars,
    renderAdvisor, renderModelPerformance, renderRiskChart,
    weekChartData, renderTimeline, renderResultsTable,
  };
})();

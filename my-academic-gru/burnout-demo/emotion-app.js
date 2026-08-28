/* ============================================================================
 * Emotional Analysis tab — controller
 * IT22196392 — Induwara K.P.Y. | R26-IT-059
 *
 * State, student persistence, sub-tabs and the weekly-progression form.
 * Rendering primitives live in emotion-tab.js.
 * ========================================================================== */

(function () {
  'use strict';

  const T = window.EmotionTab;
  const { esc, el, riskOf, topWords, WEEK_NUMS, RISK_SCORE, EMO_API } = T;

  const STORE_KEY = 'burnout_students_integrated';

  // ─── State ─────────────────────────────────────────────────────────────────

  const state = {
    view: 'single',        // 'single' | 'analysis' | 'progression' | 'add'
    students: [],
    selectedId: null,
    filter: 'All',         // sidebar risk filter
    lastSingle: null,      // result of the quick single-text analysis
    busy: false,
  };

  // ─── Persistence ───────────────────────────────────────────────────────────

  const SEEDED_KEY = 'burnout_demo_seeded_v1';

  const demoCohort = () =>
    (window.EMOTION_DEMO_STUDENTS || []).map((s) => JSON.parse(JSON.stringify(s)));

  function loadStudents() {
    let saved = [];
    try {
      saved = JSON.parse(localStorage.getItem(STORE_KEY) || '[]');
    } catch {
      saved = [];
    }

    // Seed the demo cohort once, on first visit only. After that the user's
    // list is theirs: clearing it stays cleared, and the sidebar's "Restore
    // demo students" button is the only way they come back.
    let seeded = false;
    try {
      seeded = localStorage.getItem(SEEDED_KEY) === '1';
    } catch { /* storage unavailable — fall through and seed in memory */ }

    if (!seeded && !saved.length) {
      saved = demoCohort();
      saveStudents(saved);
      try { localStorage.setItem(SEEDED_KEY, '1'); } catch { /* ignore */ }
    }
    return saved;
  }

  /** Re-add any demo students that are missing, leaving the user's own alone. */
  function restoreDemo() {
    const have = new Set(state.students.map((s) => s.id));
    const missing = demoCohort().filter((s) => !have.has(s.id));
    if (!missing.length) return 0;
    state.students = state.students.concat(missing);
    saveStudents(state.students);
    return missing.length;
  }

  function saveStudents(students) {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(students));
    } catch (e) {
      console.warn('Could not persist students:', e.message);
    }
  }

  const selectedStudent = () =>
    state.students.find((s) => s.id === state.selectedId) || null;

  // ─── API ───────────────────────────────────────────────────────────────────

  async function analyze(text) {
    const res = await fetch(`${EMO_API}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `API error ${res.status}`);
    return data;
  }

  async function checkHealth() {
    const dot = el('emoApiStatus');
    if (!dot) return;
    try {
      const res = await fetch(`${EMO_API}/api/health`);
      const data = await res.json();
      const ok = data.status === 'online';
      dot.className = ok ? 'emo-status emo-status-live' : 'emo-status emo-status-down';
      dot.textContent = ok ? '● BERT service live' : '● model not loaded';
    } catch {
      dot.className = 'emo-status emo-status-down';
      dot.textContent = '● BERT service offline (port 5004)';
    }
  }

  // ─── Error / notice helpers ────────────────────────────────────────────────

  function showError(msg, hint) {
    el('emotionError').innerHTML = `
      <div class="error">
        ⚠️ ${esc(msg)}
        ${hint ? `<br/><span style="font-size:12px;opacity:0.85">${esc(hint)}</span>` : ''}
      </div>`;
  }

  const clearError = () => { el('emotionError').innerHTML = ''; };

  const SERVICE_HINT =
    'Start it with:  cd emotional-nlp-fresh/emotional_burnout_nlp  →  .venv/Scripts/python serve_model.py';

  // ─── Single-text analysis (the quick path) ─────────────────────────────────

  async function runSingleAnalysis() {
    const input = el('emotionText');
    const text = input.value.trim();

    if (text.split(/\s+/).filter(Boolean).length < 3) {
      showError('Please enter at least a few words for a meaningful prediction.');
      return;
    }

    clearError();
    state.busy = true;
    el('emotionBody').innerHTML =
      '<div class="emo-spinner-wrap"><div class="emo-spinner"></div><div>Running BERT inference…</div></div>';

    try {
      const result = await analyze(text);
      state.lastSingle = { text, result };
      state.view = 'single';
      render();
    } catch (e) {
      el('emotionBody').innerHTML = '';
      showError(`Emotional-NLP API not available (${EMO_API}) — ${e.message}`, SERVICE_HINT);
    } finally {
      state.busy = false;
    }
  }

  function renderSingle() {
    const s = state.lastSingle;
    if (!s) {
      return `
        <div class="emo-empty">
          <div class="emo-empty-icon">🧠</div>
          <div class="emo-empty-title">Analyze student text</div>
          <div class="emo-empty-sub">
            Enter a student's written text above to see the prediction, the attention
            heatmap showing which words drove it, and all seven class probabilities.
          </div>
        </div>`;
    }

    const { text, result } = s;
    const col = riskOf(result.risk_level);
    const words = topWords(result.attention, 3);

    return `
      <div class="emo-stack">
        <div class="emo-quote">
          <div class="emo-card-label">STUDENT TEXT</div>
          <div class="emo-quote-body">"${esc(text)}"</div>
        </div>

        ${T.renderMetricCards(result)}
        ${T.renderAdvisor(result.prediction)}
        ${T.renderAttentionHeatmap(result.attention, result.prediction)}
        ${T.renderProbabilityBars(result.probabilities)}

        ${T.card('SUMMARY', `
          <div class="table-wrapper" style="margin-top:10px">
            <table>
              <thead>
                <tr>
                  <th>Emotion Class</th><th>Stress Score</th><th>Confidence</th>
                  <th>Risk Level</th><th>Top Words</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td><strong>${esc(result.prediction)}</strong></td>
                  <td class="risk-score">${result.stress_score.toFixed(1)}</td>
                  <td>${(result.confidence * 100).toFixed(1)}%</td>
                  <td><span class="emo-pill" style="color:${col};background:${col}15;border:1px solid ${col}40">${esc(result.risk_level.toUpperCase())}</span></td>
                  <td class="feature">${words.length ? esc(words.join(', ')) : '—'}</td>
                </tr>
              </tbody>
            </table>
          </div>`)}

        ${T.renderModelPerformance()}
      </div>`;
  }

  // ─── Add-student form (weekly progression input) ───────────────────────────

  const formTexts = { 2: [''], 4: [''], 8: [''], 18: [''] };
  let formName = '';

  function renderAddForm() {
    const weekBlocks = WEEK_NUMS.map((w) => {
      const entries = formTexts[w];
      const boxes = entries.map((val, idx) => `
        <textarea class="emo-textarea" data-week="${w}" data-idx="${idx}"
          placeholder="${idx === 0 ? `Student's written text in week ${w}…` : 'Second submission (optional)…'}"
          rows="2">${esc(val)}</textarea>`).join('');

      return `
        <div class="emo-week-block">
          <div class="emo-week-head">
            <span class="emo-tile-label">WEEK ${w} <span style="color:#94a3b8;font-weight:400">(optional)</span></span>
            ${entries.length === 1
              ? `<button class="emo-mini-btn" data-add-entry="${w}" title="Add second submission">+</button>`
              : `<button class="emo-mini-btn" data-del-entry="${w}" title="Remove second submission">−</button>`}
          </div>
          ${boxes}
        </div>`;
    }).join('');

    return `
      <div class="emo-stack">
        <div class="emo-card">
          <div class="emo-card-label">ADD NEW STUDENT</div>
          <div class="emo-card-sub">
            Enter the student's name and their written text from each week.
            Each week supports up to 2 submissions — the higher-risk one is kept.
          </div>

          <div style="margin:14px 0">
            <div class="emo-tile-label" style="margin-bottom:6px">STUDENT NAME</div>
            <input type="text" id="emoStudentName" class="emo-input"
                   placeholder="e.g. John Silva" value="${esc(formName)}" />
          </div>

          ${weekBlocks}

          <div id="emoFormProgress" class="emo-progress"></div>

          <button id="emoSaveStudent" class="emo-btn emo-btn-primary" style="margin-top:14px">
            Analyze &amp; Save Student
          </button>
        </div>
      </div>`;
  }

  function bindAddForm() {
    const nameInput = el('emoStudentName');
    if (nameInput) {
      nameInput.addEventListener('input', (e) => { formName = e.target.value; });
    }

    document.querySelectorAll('#emotionResult .emo-textarea').forEach((ta) => {
      ta.addEventListener('input', (e) => {
        const w = e.target.dataset.week;
        const i = Number(e.target.dataset.idx);
        formTexts[w][i] = e.target.value;
      });
    });

    document.querySelectorAll('#emotionResult [data-add-entry]').forEach((b) => {
      b.addEventListener('click', () => {
        formTexts[b.dataset.addEntry].push('');
        render();
      });
    });

    document.querySelectorAll('#emotionResult [data-del-entry]').forEach((b) => {
      b.addEventListener('click', () => {
        formTexts[b.dataset.delEntry] = [formTexts[b.dataset.delEntry][0]];
        render();
      });
    });

    const save = el('emoSaveStudent');
    if (save) save.addEventListener('click', saveNewStudent);
  }

  const wordCount = (t) => t.trim().split(/\s+/).filter(Boolean).length;

  async function saveNewStudent() {
    if (state.busy) return;

    const name = formName.trim();
    if (!name) { showError('Please enter a student name.'); return; }

    const hasAny = WEEK_NUMS.some((w) => formTexts[w].some((t) => wordCount(t) >= 3));
    if (!hasAny) { showError('Enter text for at least one week (3+ words).'); return; }

    clearError();
    state.busy = true;
    const progress = el('emoFormProgress');
    const setProgress = (msg) => { if (progress) progress.textContent = msg; };

    const weeks = {};
    let failures = 0;

    for (const w of WEEK_NUMS) {
      const entries = formTexts[w].filter((t) => wordCount(t) >= 3);
      if (!entries.length) { weeks[w] = null; continue; }

      const results = [];
      for (let i = 0; i < entries.length; i++) {
        setProgress(entries.length > 1
          ? `Analyzing Week ${w} — entry ${i + 1} of ${entries.length}…`
          : `Analyzing Week ${w}…`);
        try {
          results.push({ text: entries[i].trim(), result: await analyze(entries[i].trim()) });
        } catch (e) {
          failures++;
          results.push({ text: entries[i].trim(), result: null });
        }
      }

      // Keep the highest-risk submission for the week
      const valid = results.filter((r) => r.result && !r.result.error);
      if (!valid.length) { weeks[w] = null; continue; }
      const worst = valid.sort((a, b) =>
        (RISK_SCORE[b.result.risk_level] || 0) - (RISK_SCORE[a.result.risk_level] || 0))[0];
      weeks[w] = { text: worst.text, result: worst.result, entries: results };
    }

    setProgress('');
    state.busy = false;

    const filled = WEEK_NUMS.filter((w) => weeks[w] && weeks[w].result);
    if (!filled.length) {
      showError('All analyses failed — the BERT service is not reachable.', SERVICE_HINT);
      return;
    }

    const latestResult = weeks[filled[filled.length - 1]].result;
    const worstResult = filled
      .map((w) => weeks[w].result)
      .sort((a, b) => (RISK_SCORE[b.risk_level] || 0) - (RISK_SCORE[a.risk_level] || 0))[0];

    const student = {
      id: `s_${Date.now()}`,
      name,
      weeks,
      latestResult,
      worstResult,
    };

    state.students.push(student);
    saveStudents(state.students);
    state.selectedId = student.id;
    state.view = 'analysis';

    // Reset the form
    formName = '';
    WEEK_NUMS.forEach((w) => { formTexts[w] = ['']; });

    render();
    if (failures) {
      showError(`Saved, but ${failures} submission(s) failed to analyze.`);
    }
  }

  // ─── Saved-student views ───────────────────────────────────────────────────

  function renderStudentAnalysis(student) {
    const latest = student.latestResult;
    const worst = student.worstResult || latest;
    if (!latest) return '<div class="emo-empty">No analysis data for this student.</div>';

    const filled = WEEK_NUMS.filter((w) => student.weeks[w] && student.weeks[w].result);
    const latestWeek = filled[filled.length - 1];
    const isCritical = worst.risk_level === 'Critical';

    const summary = [
      ['WEEKS ANALYZED', `${filled.length} / ${WEEK_NUMS.length}`],
      ['WORST PREDICTION', worst.prediction],
      ['WORST RISK', worst.risk_level],
      ['LATEST STRESS SCORE', `${latest.stress_score} / 100`],
    ].map(([label, value]) => `
      <div class="emo-tile">
        <div class="emo-tile-label">${esc(label)}</div>
        <div class="emo-tile-value">${esc(value)}</div>
      </div>`).join('');

    const weekCards = WEEK_NUMS.map((w) => {
      const wd = student.weeks[w];
      if (!wd || !wd.result) return '';
      const r = wd.result;
      const col = riskOf(r.risk_level);
      return `
        <div class="emo-week-card" data-week-card="${w}">
          <div class="emo-week-card-head" data-toggle-week="${w}">
            <span class="emo-tl-week">WEEK ${w}${w === latestWeek ? ' <span class="emo-latest">LATEST</span>' : ''}</span>
            <span class="emo-pill" style="color:${col};background:${col}15;border:1px solid ${col}40">${esc(r.risk_level.toUpperCase())}</span>
            <span style="color:${col};font-weight:700;font-size:13px">${esc(r.prediction)}</span>
            <span style="color:#64748b;font-size:12px">${(r.confidence * 100).toFixed(1)}%</span>
            <span class="emo-week-preview">"${esc(wd.text)}"</span>
            <span class="emo-chevron">▼</span>
          </div>
          <div class="emo-week-card-body" hidden>
            <div class="emo-quote">
              <div class="emo-card-label">STUDENT TEXT</div>
              <div class="emo-quote-body">"${esc(wd.text)}"</div>
            </div>
            ${T.renderMetricCards(r)}
            ${T.renderAttentionHeatmap(r.attention, r.prediction)}
            ${T.renderProbabilityBars(r.probabilities)}
          </div>
        </div>`;
    }).join('');

    return `
      <div class="emo-stack">
        ${isCritical ? `
          <div class="emo-critical">
            <span style="font-size:18px">⚠</span>
            <div>
              <div style="font-weight:700;color:#dc2626">Immediate Attention Required</div>
              <div style="font-size:12px;color:#dc2626;margin-top:2px">
                This student's text indicates critical distress. Contact a mental health professional immediately.
              </div>
            </div>
          </div>` : ''}

        ${T.card('OVERALL SUMMARY', `<div class="emo-tiles">${summary}</div>`)}
        ${T.renderAdvisor(worst.prediction)}

        <div class="emo-card-label">WEEKLY SUBMISSIONS — click any week to view its XAI heatmap</div>
        ${weekCards}

        ${T.renderModelPerformance()}

        <div class="emo-card emo-export">
          <div>
            <div class="emo-card-label">EXPORT DATA</div>
            <div style="font-size:11px;color:#64748b">Export for Meta-Integration Layer</div>
          </div>
          <button id="emoExportCsv" class="emo-btn emo-btn-green">⬇ Download Stress Scores CSV</button>
        </div>
      </div>`;
  }

  function renderStudentProgression(student) {
    const chartData = T.weekChartData(student);
    const hasAny = chartData.some((d) => d.riskScore > 0);

    return `
      <div class="emo-stack">
        ${hasAny ? T.renderRiskChart(chartData) : ''}
        ${T.renderTimeline(chartData)}
        ${T.renderResultsTable(chartData)}
        ${T.card('HOW LEAD TIME IS DERIVED', `
          <div class="emo-card-sub">
            "Weeks to intervene" maps each risk tier to the window remaining before
            predicted burnout, so an advisor can prioritise:
            <strong>Low 14</strong> · <strong>Medium 12</strong> ·
            <strong>High 8</strong> · <strong>Critical 4</strong> weeks.
          </div>`)}
      </div>`;
  }

  function exportCsv(student) {
    const rows = ['student_name,week,predicted_label,confidence,risk_level,stress_score'];
    WEEK_NUMS.forEach((w) => {
      const r = student.weeks[w] && student.weeks[w].result;
      if (r) rows.push(`${student.name},${w},${r.prediction},${r.confidence},${r.risk_level},${r.stress_score}`);
    });
    const blob = new Blob([rows.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `stress_scores_${student.name.replace(/\s+/g, '_')}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ─── Student list sidebar (StudentList.jsx) ────────────────────────────────

  const RISK_DOT = { Low: '#0891b2', Medium: '#b45309', High: '#dc2626', Critical: '#ff2020' };
  const RISK_ORDER = { Critical: 0, High: 1, Medium: 2, Low: 3 };

  const initialsOf = (name) => name.trim().split(/\s+/)
    .map((w) => w[0]).join('').slice(0, 2).toUpperCase();

  function renderSidebar() {
    const filtered = state.students
      .filter((s) => state.filter === 'All' ||
        ((s.latestResult || {}).risk_level === state.filter))
      .sort((a, b) => {
        const ra = (a.latestResult || {}).risk_level;
        const rb = (b.latestResult || {}).risk_level;
        return (RISK_ORDER[ra] != null ? RISK_ORDER[ra] : 4)
             - (RISK_ORDER[rb] != null ? RISK_ORDER[rb] : 4);
      });

    const counts = { Critical: 0, High: 0, Medium: 0, Low: 0 };
    state.students.forEach((s) => {
      const rl = (s.latestResult || {}).risk_level;
      if (rl && counts[rl] !== undefined) counts[rl]++;
    });

    const options = ['All', 'Critical', 'High', 'Medium', 'Low'].map((v) =>
      `<option value="${v}"${state.filter === v ? ' selected' : ''}>${v === 'All' ? 'All Risk Levels' : v}</option>`
    ).join('');

    const rows = filtered.map((s) => {
      const risk = (s.latestResult || {}).risk_level;
      const dot = RISK_DOT[risk] || '#cbd5e1';
      const active = s.id === state.selectedId;
      return `
        <div class="emo-student-row${active ? ' active' : ''}" data-student="${esc(s.id)}">
          <div class="emo-student-main">
            <div class="emo-avatar" style="border-color:${dot}40">${esc(initialsOf(s.name))}</div>
            <div>
              <div class="emo-student-name">${esc(s.name)}</div>
              ${risk ? `<div class="emo-student-pred" style="color:${dot}">${esc(s.latestResult.prediction)}</div>` : ''}
            </div>
          </div>
          ${risk ? `<div class="emo-student-dot" style="background:${dot};box-shadow:0 0 6px ${dot}"></div>` : ''}
        </div>`;
    }).join('');

    const empty = `
      <div class="emo-sidebar-empty">
        <div style="font-size:28px;margin-bottom:10px">👤</div>
        <div>No students yet.<br/>Click <span style="color:#1d4ed8">+ Add New Student</span> to get started.</div>
      </div>`;

    const countPills = [
      ['Critical', counts.Critical, '#ff2020'],
      ['High', counts.High, '#dc2626'],
      ['Medium', counts.Medium, '#b45309'],
      ['Low', counts.Low, '#0891b2'],
    ].filter(([, n]) => n > 0)
      .map(([label, n, col]) => `<span style="font-size:10px;color:${col}">● ${n} ${label}</span>`)
      .join('');

    const demoMissing = demoCohort()
      .some((d) => !state.students.some((s) => s.id === d.id));

    return `
      <div class="emo-sidebar-top">
        <div class="emo-card-label">STUDENT LIST</div>
        <button class="emo-add-btn" data-add-student>
          <span style="font-size:16px;line-height:1">+</span> Add New Student
        </button>
        <select id="emoRiskFilter" class="emo-select">${options}</select>
      </div>

      <div class="emo-sidebar-list">
        ${filtered.length ? rows : empty}
      </div>

      <div class="emo-sidebar-foot">
        <div class="emo-counts">
          ${state.students.length ? countPills : '<span style="font-size:10px;color:#94a3b8">No students added</span>'}
        </div>
        ${demoMissing ? '<button class="emo-restore-btn" data-restore-demo>↺ Restore demo students</button>' : ''}
        ${state.students.length ? '<button class="emo-clear-btn" data-clear-all>Clear All Students</button>' : ''}
      </div>`;
  }

  // ─── Student header + risk badge (RiskBadge.jsx) ───────────────────────────

  const RISK_STYLES = {
    Low: { color: '#0891b2', bg: '#ecfeff', border: '#22d3ee40', shadow: '0 0 12px #22d3ee30', label: 'LOW RISK' },
    Medium: { color: '#b45309', bg: '#fffbeb', border: '#fbbf2440', shadow: '0 0 12px #fbbf2430', label: 'MEDIUM RISK' },
    High: { color: '#dc2626', bg: '#fef2f2', border: '#f8717140', shadow: '0 0 12px #f8717130', label: 'HIGH RISK' },
    Critical: { color: '#ff2020', bg: '#fef2f2', border: '#ff000060', shadow: null, label: 'CRITICAL RISK' },
  };

  function renderRiskBadge(riskLevel) {
    const s = RISK_STYLES[riskLevel] || RISK_STYLES.Medium;
    const isCritical = riskLevel === 'Critical';
    return `
      <div class="emo-riskbadge${isCritical ? ' critical' : ''}"
           style="background:${s.bg};border:1px solid ${s.border};${s.shadow ? `box-shadow:${s.shadow}` : ''}">
        <div class="emo-card-label" style="margin-bottom:6px">RISK LEVEL</div>
        <div class="emo-riskbadge-label" style="color:${s.color}">${isCritical ? '⚠ ' : ''}${s.label}</div>
        ${isCritical ? '<div class="emo-riskbadge-note">Seek immediate mental health support</div>' : ''}
      </div>`;
  }

  function renderHeader() {
    const student = selectedStudent();

    if (state.view === 'add') {
      return `<div class="emo-head"><div><div class="emo-head-name">New Student</div>
        <div class="emo-head-id">Enter weekly text to build a progression</div></div></div>`;
    }

    if (state.view === 'single' || !student) {
      return `<div class="emo-head"><div>
        <div class="emo-head-name">Emotional Analysis</div>
        <div class="emo-head-id">Explainable NLP · IT22196392 · Induwara K.P.Y.</div>
      </div></div>`;
    }

    const worst = student.worstResult || student.latestResult;
    return `
      <div class="emo-head">
        <div>
          <div class="emo-head-name">${esc(student.name)}</div>
          <div class="emo-head-id">ID: ${esc(student.id)}</div>
        </div>
        ${worst ? renderRiskBadge(worst.risk_level) : ''}
      </div>`;
  }

  // ─── Sub-tabs ──────────────────────────────────────────────────────────────

  function renderSubTabs() {
    const student = selectedStudent();
    if (state.view === 'add') return '';

    const tabs = [
      { id: 'analysis', label: 'Analysis', on: !!student },
      { id: 'progression', label: 'Weekly Progression', on: !!student },
      { id: 'single', label: 'Quick Analysis', on: true },
    ];
    return `
      <div class="emo-subtabs">
        ${tabs.map((t) => `
          <button class="emo-subtab${state.view === t.id ? ' active' : ''}"
                  data-subtab="${t.id}" ${t.on ? '' : 'disabled title="Select or add a student first"'}>
            ${esc(t.label)}
          </button>`).join('')}
      </div>`;
  }

  // ─── Main render ───────────────────────────────────────────────────────────

  function render() {
    const student = selectedStudent();
    let body;

    if (state.view === 'add') body = renderAddForm();
    else if (state.view === 'analysis' && student) body = renderStudentAnalysis(student);
    else if (state.view === 'progression' && student) body = renderStudentProgression(student);
    else body = renderSingle();

    el('emotionSidebar').innerHTML = renderSidebar();
    el('emotionHeader').innerHTML = renderHeader();
    el('emotionTabs').innerHTML = renderSubTabs();

    // The quick-analysis input is static markup; show it only on that tab
    el('emotionQuick').style.display = state.view === 'single' ? '' : 'none';

    // emotionBody holds the transient spinner; clear it whenever we paint
    el('emotionBody').innerHTML = '';
    el('emotionResult').innerHTML = body;

    bindDynamic();
  }

  function bindDynamic() {
    const root = el('emotionResult');
    const panes = [el('emotionSidebar'), el('emotionTabs'), root];
    const each = (sel, fn) =>
      panes.forEach((p) => p.querySelectorAll(sel).forEach(fn));

    each('[data-subtab]', (b) => {
      b.addEventListener('click', () => {
        if (b.disabled) return;
        state.view = b.dataset.subtab;
        render();
      });
    });

    each('[data-student]', (b) => {
      b.addEventListener('click', () => {
        state.selectedId = b.dataset.student;
        state.view = 'analysis';
        render();
      });
    });

    each('[data-add-student]', (b) => {
      b.addEventListener('click', () => { state.view = 'add'; render(); });
    });

    each('[data-clear-all]', (b) => {
      b.addEventListener('click', () => {
        if (!confirm('Remove all saved students? This cannot be undone.')) return;
        state.students = [];
        state.selectedId = null;
        saveStudents([]);
        state.view = 'single';
        render();
      });
    });

    each('[data-restore-demo]', (b) => {
      b.addEventListener('click', () => {
        const n = restoreDemo();
        if (n && !state.selectedId) state.selectedId = state.students[0].id;
        render();
      });
    });

    const filterSel = el('emoRiskFilter');
    if (filterSel) {
      filterSel.addEventListener('change', (e) => {
        state.filter = e.target.value;
        render();
      });
    }

    root.querySelectorAll('[data-toggle-week]').forEach((h) => {
      h.addEventListener('click', () => {
        const cardEl = h.closest('[data-week-card]');
        const bodyEl = cardEl.querySelector('.emo-week-card-body');
        const chev = h.querySelector('.emo-chevron');
        const open = bodyEl.hasAttribute('hidden');
        if (open) bodyEl.removeAttribute('hidden'); else bodyEl.setAttribute('hidden', '');
        chev.style.transform = open ? 'rotate(180deg)' : 'rotate(0deg)';
      });
    });

    const exportBtn = el('emoExportCsv');
    if (exportBtn) {
      const s = selectedStudent();
      if (s) exportBtn.addEventListener('click', () => exportCsv(s));
    }

    if (state.view === 'add') bindAddForm();
  }

  // ─── Init ──────────────────────────────────────────────────────────────────

  function init() {
    state.students = loadStudents();
    if (state.students.length) state.selectedId = state.students[0].id;

    const btn = el('emotionAnalyzeBtn');
    if (btn) btn.addEventListener('click', runSingleAnalysis);

    const input = el('emotionText');
    if (input) {
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); runSingleAnalysis(); }
      });
    }

    render();
    checkHealth();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Kept global: the dashboard's markup calls this from an onclick attribute
  window.analyzeEmotion = runSingleAnalysis;
})();

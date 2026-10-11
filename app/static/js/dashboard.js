/**
 * Burnout Guard — Student Support & Triage Station
 * Plain-English Counselor Interface, Bounded Caseload Manager,
 * Inline Visual Sparklines, Explainable AI "Why" Breakdown, and Meeting Logging
 */

// Application State
const state = {
  records: window.initialRecords || [],
  capacity: 5,
  selected: null,
  isStreaming: false,
  streamInterval: null,
  activeFilter: "all",
  activeSort: "opportunity_desc",
  filters: {
    query: "",
    programme: "",
  },
};

// DOM References
const dom = {
  // Stats
  statTotal: document.getElementById("stat-total"),
  statP1: document.getElementById("stat-p1"),
  statP2: document.getElementById("stat-p2"),
  statResolved: document.getElementById("stat-resolved"),
  statDiscordant: document.getElementById("stat-discordant"),
  navP1Badge: document.getElementById("nav-p1-badge"),
  
  // Kanban columns
  cardsP1: document.getElementById("cards-p1"),
  cardsP2: document.getElementById("cards-p2"),
  cardsResolved: document.getElementById("cards-resolved"),
  countP1: document.getElementById("count-p1"),
  countP2: document.getElementById("count-p2"),
  countResolved: document.getElementById("count-resolved"),
  
  // Capacity Controls
  capacitySlider: document.getElementById("capacity-slider"),
  capacityDisplay: document.getElementById("capacity-display"),
  sliderGlowBar: document.getElementById("slider-glow-bar"),
  knapsackOverflowCount: document.getElementById("knapsack-overflow-count"),
  btnCapDecrement: document.getElementById("btn-cap-decrement"),
  btnCapIncrement: document.getElementById("btn-cap-increment"),
  
  // Toolbar
  searchInput: document.getElementById("search-input"),
  btnClearSearch: document.getElementById("btn-clear-search"),
  filterProgramme: document.getElementById("filter-programme"),
  sortSelector: document.getElementById("sort-selector"),
  dataModeBadge: document.getElementById("data-mode-badge"),
  
  // Live Ingestion & Presets
  btnStreamToggle: document.getElementById("btn-stream-toggle"),
  streamBtnText: document.getElementById("stream-btn-text"),
  presetDiscordant: document.getElementById("preset-discordant"),
  presetCapacity: document.getElementById("preset-capacity"),
  presetReset: document.getElementById("preset-reset"),
  
  // Drawer
  drawer: document.getElementById("detail-drawer"),
  drawerClose: document.getElementById("drawer-close"),
  drawerName: document.getElementById("drawer-student-name"),
  drawerMeta: document.getElementById("drawer-student-meta"),
  drawerDiscordantBanner: document.getElementById("drawer-discordant-banner"),
  drawerPriorityBadge: document.getElementById("drawer-priority-badge"),
  drawerEffectiveBadge: document.getElementById("drawer-effective-badge"),
  drawerConfidenceBadge: document.getElementById("drawer-confidence-badge"),
  drawerPriorityScore: document.getElementById("drawer-priority-score"),
  
  // Sparklines
  drawerAcademicState: document.getElementById("drawer-academic-state"),
  svgAcademicSparkline: document.getElementById("svg-academic-sparkline"),
  drawerBehVal: document.getElementById("drawer-beh-val"),
  svgBehaviorSparkline: document.getElementById("svg-behavior-sparkline"),
  drawerComplianceTrend: document.getElementById("drawer-compliance-trend"),
  drawerStressLabel: document.getElementById("drawer-stress-label"),
  drawerStressMeter: document.getElementById("drawer-stress-meter"),
  
  // TreeSHAP & Opp
  drawerShapSummary: document.getElementById("drawer-shap-summary"),
  drawerShapBars: document.getElementById("drawer-shap-bars"),
  drawerOppProb: document.getElementById("drawer-opp-prob"),
  drawerOppBand: document.getElementById("drawer-opp-band"),
  drawerOppExpl: document.getElementById("drawer-opp-expl"),
  
  // Case Resolution Form
  resolveForm: document.getElementById("resolve-form"),
  advisorNotes: document.getElementById("advisor-notes"),
  toastMessage: document.getElementById("toast-message"),
};

// Technical feature to Human Label dictionary
const HUMAN_FEATURE_NAMES = {
  week17_risk: "Final Exam Risk",
  week12_risk: "Late-Semester Grade Risk",
  week8_risk: "Midterm Assessment Risk",
  week4_risk: "Early Assessment Score",
  risk_change: "Grade Drop Speed",
  recent_risk_change: "Recent Performance Drop",
  behavior_risk: "Online Study Disruption",
  compliance_mean: "Quiz & LMS Attendance",
  anomaly_mean: "Irregular Study Schedule",
  high_anomaly_weeks: "Weeks with Skipped Modules",
  compliance_trend: "Recent Study Momentum",
  emotional_stress: "Student Emotional Stress",
  evidence_coverage: "Data Signals Available",
};

// =============================================================================
// Helper Functions: Initials & Mini-Sparklines
// =============================================================================
function getInitials(name) {
  if (!name) return "ST";
  const parts = name.trim().split(" ");
  if (parts.length >= 2) {
    return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
  }
  return name.slice(0, 2).toUpperCase();
}

function createMiniSparklineSvg(risks) {
  const vals = (risks && risks.length === 4) ? risks : [0.2, 0.2, 0.2, 0.2];
  const w = 90;
  const h = 24;
  const pts = vals.map((v, i) => {
    const x = (i / 3) * (w - 12) + 6;
    const y = h - (Math.max(0, Math.min(1, v)) * (h - 8) + 4);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  
  const polylineStr = pts.join(" ");
  const w4 = vals[0];
  const w17 = vals[3];
  
  let strokeColor = "#f59e0b"; // amber neutral
  if (w17 > w4 + 0.12) {
    strokeColor = "#f43f5e"; // crimson deteriorating
  } else if (w17 < w4 - 0.12) {
    strokeColor = "#10b981"; // emerald improving
  }

  return `
    <svg class="sparkline-svg-mini" viewBox="0 0 ${w} ${h}">
      <polyline points="${polylineStr}" fill="none" stroke="${strokeColor}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="${pts[pts.length - 1].split(",")[0]}" cy="${pts[pts.length - 1].split(",")[1]}" r="2.5" fill="${strokeColor}"/>
    </svg>
  `;
}

// =============================================================================
// Mathematical Capacity Knapsack Re-ranking (Pure In-Memory, Sub-Millisecond)
// =============================================================================
function applyCapacityKnapsack(records, capacityLimit) {
  const p1Candidates = [];
  const otherStudents = [];

  records.forEach((student) => {
    const baseP = student.base_priority_level || student.priority_level || "P3";
    student.base_priority_level = baseP;
    if (baseP === "P1") {
      p1Candidates.push(student);
    } else {
      otherStudents.push(student);
    }
  });

  // Sort P1 candidates by predicted success probability descending
  p1Candidates.sort((a, b) => {
    const probA = (a.intervention_opportunity && a.intervention_opportunity.intervention_success_probability) || a.priority_score || 0;
    const probB = (b.intervention_opportunity && b.intervention_opportunity.intervention_success_probability) || b.priority_score || 0;
    if (probB !== probA) return probB - probA;
    return (b.priority_score || 0) - (a.priority_score || 0);
  });

  let overflowCount = 0;

  // Allocate top K to P1, remainder deferred to P2
  p1Candidates.forEach((student, index) => {
    if (index < capacityLimit) {
      student.effective_priority = "P1";
      student.capacity_status = "In Your Top Priority Slots";
      student.is_overflow = false;
      student.capacity_rank = index + 1;
    } else {
      student.effective_priority = "P2";
      student.capacity_status = "Queued for Next Week (Caseload Capped)";
      student.is_overflow = true;
      student.capacity_rank = index + 1;
      overflowCount++;
    }
  });

  otherStudents.forEach((student) => {
    student.effective_priority = student.base_priority_level || "P3";
    student.is_overflow = false;
  });

  if (dom.knapsackOverflowCount) {
    dom.knapsackOverflowCount.textContent = overflowCount > 0 
      ? `${overflowCount} Queued For Next Week` 
      : "All Urgent Cases Fitted";
  }

  return records;
}

// =============================================================================
// Filtering, Sorting & Card Generation
// =============================================================================
function getFilteredRecords() {
  const query = state.filters.query.trim().toLowerCase();
  const prog = state.filters.programme;
  const filterType = state.activeFilter;

  let filtered = state.records.filter((rec) => {
    // Search query
    if (query) {
      const name = (rec.student_name || "").toLowerCase();
      const id = (rec.student_id || "").toLowerCase();
      const p = (rec.programme || "").toLowerCase();
      if (!name.includes(query) && !id.includes(query) && !p.includes(query)) {
        return false;
      }
    }

    // Programme filter
    if (prog && rec.programme !== prog) {
      return false;
    }

    // Filter pills
    if (filterType === "p1" && rec.effective_priority !== "P1") {
      return false;
    }
    if (filterType === "discordant" && !rec.is_multi_modal_override) {
      return false;
    }
    if (filterType === "high_stress" && (rec.emotional_stress || 0) < 3.0) {
      return false;
    }
    if (filterType === "high_opp") {
      const opp = rec.intervention_opportunity ? rec.intervention_opportunity.intervention_success_probability : 0;
      if (opp < 0.70) return false;
    }
    if (filterType === "resolved" && rec.status !== "Resolved") {
      return false;
    }

    return true;
  });

  // Sorting
  filtered.sort((a, b) => {
    if (state.activeSort === "opportunity_desc") {
      const oppA = (a.intervention_opportunity && a.intervention_opportunity.intervention_success_probability) || 0;
      const oppB = (b.intervention_opportunity && b.intervention_opportunity.intervention_success_probability) || 0;
      return oppB - oppA;
    }
    if (state.activeSort === "priority_desc") {
      return (b.priority_score || 0) - (a.priority_score || 0);
    }
    if (state.activeSort === "academic_risk_desc") {
      const riskA = (a.academic_risks && a.academic_risks[3]) || a.academic_risk || 0;
      const riskB = (b.academic_risks && b.academic_risks[3]) || b.academic_risk || 0;
      return riskB - riskA;
    }
    if (state.activeSort === "name_asc") {
      return (a.student_name || "").localeCompare(b.student_name || "");
    }
    return 0;
  });

  return filtered;
}

function renderKanbanBoard() {
  applyCapacityKnapsack(state.records, state.capacity);
  const records = getFilteredRecords();

  const p1List = [];
  const p2List = [];
  const resolvedList = [];

  records.forEach((r) => {
    if (r.status === "Resolved") {
      resolvedList.push(r);
    } else if (r.effective_priority === "P1") {
      p1List.push(r);
    } else {
      p2List.push(r);
    }
  });

  // Update Telemetry Header
  const totalCount = state.records.length;
  const p1Active = state.records.filter((r) => r.status !== "Resolved" && r.effective_priority === "P1").length;
  const p2Active = state.records.filter((r) => r.status !== "Resolved" && r.effective_priority !== "P1").length;
  const resolvedActive = state.records.filter((r) => r.status === "Resolved").length;
  const discordantActive = state.records.filter((r) => r.is_multi_modal_override).length;

  if (dom.statTotal) dom.statTotal.textContent = totalCount;
  if (dom.statP1) dom.statP1.textContent = p1Active;
  if (dom.statP2) dom.statP2.textContent = p2Active;
  if (dom.statResolved) dom.statResolved.textContent = resolvedActive;
  if (dom.statDiscordant) dom.statDiscordant.textContent = discordantActive;
  if (dom.navP1Badge) dom.navP1Badge.textContent = `${p1Active} Urgent`;

  if (dom.countP1) dom.countP1.textContent = p1List.length;
  if (dom.countP2) dom.countP2.textContent = p2List.length;
  if (dom.countResolved) dom.countResolved.textContent = resolvedList.length;

  // Render Columns
  dom.cardsP1.innerHTML = p1List.length ? p1List.map((r) => createCardHtml(r)).join("") : emptyColumnHtml("No urgent cases requiring immediate meetings this week.");
  dom.cardsP2.innerHTML = p2List.length ? p2List.map((r) => createCardHtml(r)).join("") : emptyColumnHtml("No students currently on the watchlist.");
  dom.cardsResolved.innerHTML = resolvedList.length ? resolvedList.map((r) => createCardHtml(r)).join("") : emptyColumnHtml("No resolved meetings yet. Open a student file to log notes.");

  // Attach card click handlers
  document.querySelectorAll(".triage-card").forEach((card) => {
    card.addEventListener("click", () => {
      const studentId = card.dataset.id;
      selectStudent(studentId);
    });
  });
}

function emptyColumnHtml(message) {
  return `<div class="empty-column-msg"><p>${message}</p></div>`;
}

function createCardHtml(r) {
  const isSelected = state.selected && state.selected.student_id === r.student_id;
  const isDiscordant = r.is_multi_modal_override;
  const isOverflow = r.is_overflow;
  const oppProb = r.intervention_opportunity ? Math.round((r.intervention_opportunity.intervention_success_probability || 0) * 100) : null;
  const initials = getInitials(r.student_name);
  const miniSparklineHtml = createMiniSparklineSvg(r.academic_risks);

  let priorityClass = "p3";
  let avatarClass = "avatar-p2";
  let priorityLabel = "WATCHLIST";

  if (r.status === "Resolved") {
    priorityClass = "resolved";
    avatarClass = "avatar-resolved";
    priorityLabel = "HANDLED";
  } else if (r.effective_priority === "P1") {
    priorityClass = "p1";
    avatarClass = "avatar-p1";
    priorityLabel = "CALL THIS WEEK";
  } else if (r.effective_priority === "P2") {
    priorityClass = "p2";
    avatarClass = "avatar-p2";
    priorityLabel = "WATCHLIST";
  }

  const finalRisk = (r.academic_risks && r.academic_risks.length === 4) ? r.academic_risks[3] : (r.academic_risk !== undefined ? r.academic_risk : 0.2);
  const stress = r.emotional_stress || 0;

  return `
    <article class="triage-card ${priorityClass} ${isSelected ? "selected" : ""} ${isOverflow ? "card-overflow" : ""}" data-id="${r.student_id}">
      <div class="card-topline">
        <div class="card-avatar-wrap">
          <div class="student-avatar ${avatarClass}">${initials}</div>
          <div class="card-identity">
            <strong>${r.student_name}</strong>
            <small>${r.student_id} · ${r.programme}</small>
          </div>
        </div>
        <span class="priority-pill ${priorityClass}">${priorityLabel}</span>
      </div>

      ${isDiscordant ? `<div class="discordant-badge">⚡ HIDDEN DISTRESS (High Stress, Passing Marks)</div>` : ""}
      ${isOverflow ? `<div class="overflow-badge">⏱ Queued for Next Week (Caseload Capped)</div>` : ""}

      <!-- Inline Mini-Sparkline -->
      <div class="card-mini-sparkline">
        ${miniSparklineHtml}
        <div class="sparkline-meta">
          <small>Course Risk</small>
          <strong>${finalRisk.toFixed(2)}</strong>
        </div>
      </div>

      <div class="signals-mini-row">
        <div class="signal-chip">
          <span>LMS Activity</span>
          <strong>${Math.round((r.compliance_mean !== undefined ? r.compliance_mean : 0.8) * 100)}%</strong>
        </div>
        <div class="signal-chip">
          <span>Schedule Disruption</span>
          <strong>${(r.anomaly_mean !== undefined ? r.anomaly_mean : 0.1).toFixed(2)}</strong>
        </div>
        <div class="signal-chip">
          <span>Stress Level</span>
          <strong style="color: ${stress >= 3.0 ? "var(--p1-crimson)" : stress >= 2.0 ? "var(--p2-amber)" : "inherit"}">${stress.toFixed(1)} / 4</strong>
        </div>
      </div>

      <div class="card-footer">
        ${oppProb !== null ? `<span class="opp-badge">🎯 ${oppProb}% Chance to Recover with Support</span>` : `<span class="opp-badge">Monitoring Active</span>`}
        <span class="action-link">Open Student File →</span>
      </div>
    </article>
  `;
}

// =============================================================================
// Student Selection & Drawer Breakdown
// =============================================================================
function selectStudent(studentId) {
  state.selected = state.records.find((r) => r.student_id === studentId);
  if (!state.selected) return;

  const r = state.selected;

  // Header info
  dom.drawerName.textContent = r.student_name;
  dom.drawerMeta.textContent = `ID: ${r.student_id} · Degree: ${r.programme} · Cohort: ${r.cohort_id || "2026-S1"}`;

  // Discordant Alert
  if (r.is_multi_modal_override) {
    dom.drawerDiscordantBanner.hidden = false;
  } else {
    dom.drawerDiscordantBanner.hidden = true;
  }

  // Priority badges
  const baseP = r.base_priority_level || r.priority_level;
  dom.drawerPriorityBadge.textContent = baseP === "P1" ? "Urgent Priority" : "Regular Watchlist";
  dom.drawerPriorityBadge.className = `badge badge-priority ${baseP.toLowerCase()}`;

  dom.drawerEffectiveBadge.textContent = r.capacity_status || (r.effective_priority === "P1" ? "In Your Top 5 Slots" : "Queued for Next Week");
  dom.drawerConfidenceBadge.textContent = `${r.confidence || "High"} Confidence`;
  dom.drawerPriorityScore.textContent = `${Math.round((r.priority_score || 0.7) * 100)}%`;

  // Sparklines
  renderAcademicSparkline(r.academic_risks || [0.2, 0.2, 0.2, 0.2]);
  dom.drawerAcademicState.textContent = (r.trajectory || "stable_low").replaceAll("_", " ").toUpperCase();

  renderBehaviorSparkline(r);
  dom.drawerBehVal.textContent = `LMS Attendance: ${Math.round((r.compliance_mean || 0.8) * 100)}% · Schedule Disruption: ${(r.anomaly_mean || 0.2).toFixed(2)}`;
  const compTrendVal = r.compliance_trend !== undefined ? r.compliance_trend : 0.0;
  dom.drawerComplianceTrend.textContent = `Study Momentum: ${compTrendVal >= 0 ? "Positive (+" : "Negative ("}${compTrendVal.toFixed(2)})`;

  renderEmotionalMeter(r.emotional_stress || 0);

  // TreeSHAP "Why" Panel
  renderShapPanel(r);

  // Recovery Opportunity
  if (r.intervention_opportunity) {
    const p = Math.round((r.intervention_opportunity.intervention_success_probability || 0.5) * 100);
    dom.drawerOppProb.textContent = `${p}%`;
    dom.drawerOppBand.textContent = `${(r.intervention_opportunity.opportunity_band || "high").toUpperCase()} RECOVERY OPPORTUNITY`;
    dom.drawerOppExpl.textContent = r.intervention_opportunity.why_explanation || "Historical data confirms that students exhibiting this exact pattern recover rapidly when an advisor provides timely support.";
  } else {
    dom.drawerOppProb.textContent = "—";
    dom.drawerOppBand.textContent = "AWAITING DATA";
  }

  // Case notes & Outcome form
  if (r.advisor_notes) {
    dom.advisorNotes.value = r.advisor_notes;
  } else {
    dom.advisorNotes.value = "";
  }
  dom.toastMessage.hidden = true;

  // Open drawer
  dom.drawer.classList.add("open");
  dom.drawer.setAttribute("aria-hidden", "false");

  // Re-render board to show card selection highlight
  renderKanbanBoard();
}

function closeDrawer() {
  dom.drawer.classList.remove("open");
  dom.drawer.setAttribute("aria-hidden", "true");
  state.selected = null;
  renderKanbanBoard();
}

// =============================================================================
// Sparkline SVG Renderers (Drawer)
// =============================================================================
function renderAcademicSparkline(risks) {
  const width = 280;
  const height = 50;
  const points = risks.map((val, idx) => {
    const x = (idx / (risks.length - 1)) * (width - 24) + 12;
    const y = height - (Math.max(0, Math.min(1, val)) * (height - 18) + 8);
    return { x, y, val };
  });

  const polylineStr = points.map((p) => `${p.x},${p.y}`).join(" ");
  const fillPolylineStr = `12,${height} ${polylineStr} ${width - 12},${height}`;

  dom.svgAcademicSparkline.innerHTML = `
    <defs>
      <linearGradient id="acadGradient" x1="0" y1="0" x2="0" y2="1">
        <stop offset="0%" stop-color="#3b82f6" stop-opacity="0.35"/>
        <stop offset="100%" stop-color="#3b82f6" stop-opacity="0.0"/>
      </linearGradient>
    </defs>
    <polygon points="${fillPolylineStr}" fill="url(#acadGradient)"/>
    <polyline points="${polylineStr}" fill="none" stroke="#60a5fa" stroke-width="2.5" stroke-linecap="round"/>
    ${points.map((p) => `
      <circle cx="${p.x}" cy="${p.y}" r="4" fill="#0e172a" stroke="#60a5fa" stroke-width="2"/>
      <text x="${p.x}" y="${p.y - 7}" font-size="8" fill="#94a3b8" text-anchor="middle">${p.val.toFixed(2)}</text>
    `).join("")}
  `;
}

function renderBehaviorSparkline(rec) {
  const comp = rec.compliance_mean || 0.8;
  const anom = rec.anomaly_mean || 0.2;
  const width = 280;

  dom.svgBehaviorSparkline.innerHTML = `
    <rect x="0" y="8" width="${width}" height="6" rx="3" fill="rgba(255,255,255,0.08)"/>
    <rect x="0" y="8" width="${Math.round(width * comp)}" height="6" rx="3" fill="#34d399"/>
    
    <rect x="0" y="24" width="${width}" height="6" rx="3" fill="rgba(255,255,255,0.08)"/>
    <rect x="0" y="24" width="${Math.round(width * anom)}" height="6" rx="3" fill="${anom >= 0.5 ? "#f43f5e" : "#f59e0b"}"/>
  `;
}

function renderEmotionalMeter(stress) {
  let label = "Normal Emotional State (0.0 - 1.0)";
  if (stress >= 3.0) {
    label = `Critical Stress (${stress.toFixed(1)} / 4.0)`;
  } else if (stress >= 2.0) {
    label = `Elevated Stress (${stress.toFixed(1)} / 4.0)`;
  }
  dom.drawerStressLabel.textContent = label;

  const segments = dom.drawerStressMeter.querySelectorAll(".stress-segment");
  segments.forEach((seg, idx) => {
    seg.className = "stress-segment";
    if (stress >= idx + 0.8) {
      seg.classList.add(`active-${idx + 1}`);
    }
  });
}

// =============================================================================
// Plain-English "Why Flagged" Breakdown Renderer
// =============================================================================
function renderShapPanel(r) {
  const opp = r.intervention_opportunity || {};
  const drivers = opp.feature_contributions || [];

  if (opp.why_explanation) {
    dom.drawerShapSummary.textContent = opp.why_explanation;
    dom.drawerShapSummary.hidden = false;
  } else {
    dom.drawerShapSummary.textContent = "Analyzing student profile and assessing support options...";
  }

  if (drivers.length > 0) {
    const top4 = drivers.slice(0, 4);
    dom.drawerShapBars.innerHTML = top4.map((d) => {
      const pct = Math.abs(d.percentage);
      const isPositive = d.percentage >= 0;
      const sign = isPositive ? "+" : "-";
      const cleanName = HUMAN_FEATURE_NAMES[d.feature] || d.display_name || d.feature;
      const explanation = isPositive ? "Increases Need for Meeting" : "Protective Buffer";

      return `
        <div class="shap-bar-item">
          <div class="shap-bar-label-row">
            <div>
              <span class="shap-feat-name">${cleanName}</span>
              <small style="color: var(--text-dim); display: block; font-size: 10px;">${explanation}</small>
            </div>
            <span class="shap-pct-val ${isPositive ? "positive" : "negative"}">${sign}${pct.toFixed(1)}% Impact</span>
          </div>
          <div class="shap-track">
            <div class="shap-fill ${isPositive ? "pos" : "neg"}" style="width: ${Math.min(100, pct)}%"></div>
          </div>
        </div>
      `;
    }).join("");
  } else {
    dom.drawerShapBars.innerHTML = `<p style="color: var(--text-dim); font-size: 11px;">Awaiting live stream student features.</p>`;
  }
}

// =============================================================================
// Closed-Loop Case Resolution Form Submission
// =============================================================================
async function handleResolveCase(e) {
  e.preventDefault();
  if (!state.selected) return;

  const selectedOutcome = document.querySelector('input[name="intervention-outcome"]:checked').value;
  const notes = dom.advisorNotes.value.trim();

  const payload = {
    status: "Resolved",
    intervention_outcome: selectedOutcome,
    notes: notes,
  };

  try {
    const response = await fetch(`/api/students/${state.selected.student_id}/case`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (response.ok) {
      const updated = await response.json();
      
      const idx = state.records.findIndex((r) => r.student_id === updated.student_id);
      if (idx !== -1) {
        state.records[idx] = updated;
      }
      state.selected = updated;

      dom.toastMessage.textContent = `✔ Meeting documented successfully! Outcome saved to faculty records.`;
      dom.toastMessage.hidden = false;

      renderKanbanBoard();
      selectStudent(updated.student_id);
    } else {
      dom.toastMessage.textContent = "Error saving notes. Please try again.";
      dom.toastMessage.hidden = false;
    }
  } catch (err) {
    dom.toastMessage.textContent = `Failed to connect: ${err.message}`;
    dom.toastMessage.hidden = false;
  }
}

// =============================================================================
// Viva Demo Controls: Cold Start, Presets & Live Ingestion
// =============================================================================
async function resetWorkspace() {
  if (!confirm("Clear student workspace? This will empty the queue so you can show a live demo stream from scratch.")) return;

  try {
    const response = await fetch("/api/reset", { method: "POST" });
    if (response.ok) {
      state.records = [];
      state.selected = null;
      if (dom.dataModeBadge) dom.dataModeBadge.textContent = "FRESH BOARD (READY FOR STREAM)";
      closeDrawer();
      renderKanbanBoard();
    }
  } catch (err) {
    alert(`Reset failed: ${err.message}`);
  }
}

function triggerDiscordantPreset() {
  const discordantStudent = state.records.find((r) => r.is_multi_modal_override);
  if (discordantStudent) {
    selectStudent(discordantStudent.student_id);
    switchDrawerTab("tab-trajectories");
  } else {
    alert("No discordant students currently in memory. Please stream student records first.");
  }
}

function triggerCapacityPreset() {
  updateCapacity(3);
}

function updateCapacity(newCap) {
  const cap = Math.max(1, Math.min(25, newCap));
  state.capacity = cap;
  if (dom.capacitySlider) dom.capacitySlider.value = cap;
  if (dom.capacityDisplay) dom.capacityDisplay.textContent = `${cap} Meetings Available`;
  
  // Update glow bar width
  if (dom.sliderGlowBar) {
    const pct = ((cap - 1) / 24) * 100;
    dom.sliderGlowBar.style.width = `${pct}%`;
  }

  // Update quick pills
  document.querySelectorAll(".pill-btn").forEach((p) => {
    p.classList.toggle("active", parseInt(p.dataset.cap, 10) === cap);
  });

  renderKanbanBoard();
  fetch(`/api/capacity?limit=${cap}`, { method: "POST" }).catch(() => {});
}

async function fetchLiveQueue() {
  try {
    const response = await fetch("/api/students");
    if (response.ok) {
      const freshRecords = await response.json();
      if (freshRecords.length !== state.records.length) {
        state.records = freshRecords;
        renderKanbanBoard();
        if (state.selected) {
          const updatedSel = state.records.find((r) => r.student_id === state.selected.student_id);
          if (updatedSel) selectStudent(updatedSel.student_id);
        }
      }
    }
  } catch (e) {
    console.debug("Live fetch error", e);
  }
}

function toggleLiveStreaming() {
  state.isStreaming = !state.isStreaming;

  if (state.isStreaming) {
    if (dom.streamBtnText) dom.streamBtnText.textContent = "Live Sync: Active";
    if (dom.btnStreamToggle) dom.btnStreamToggle.classList.add("btn-live-stream-active");
    state.streamInterval = setInterval(fetchLiveQueue, 1500);
  } else {
    if (dom.streamBtnText) dom.streamBtnText.textContent = "Live Sync: Paused";
    if (dom.btnStreamToggle) dom.btnStreamToggle.classList.remove("btn-live-stream-active");
    clearInterval(state.streamInterval);
    state.streamInterval = null;
  }
}

// Drawer Tab Switching
function switchDrawerTab(tabId) {
  document.querySelectorAll(".drawer-tab").forEach((t) => {
    t.classList.toggle("active", t.dataset.tab === tabId);
  });
  document.querySelectorAll(".drawer-tab-pane").forEach((p) => {
    p.classList.toggle("active", p.id === tabId);
  });
}

// =============================================================================
// Initialization & Event Binding
// =============================================================================
function init() {
  // Capacity Controls
  if (dom.capacitySlider) {
    dom.capacitySlider.addEventListener("input", (e) => {
      updateCapacity(parseInt(e.target.value, 10));
    });
  }
  if (dom.btnCapDecrement) {
    dom.btnCapDecrement.addEventListener("click", () => updateCapacity(state.capacity - 1));
  }
  if (dom.btnCapIncrement) {
    dom.btnCapIncrement.addEventListener("click", () => updateCapacity(state.capacity + 1));
  }
  document.querySelectorAll(".pill-btn").forEach((btn) => {
    btn.addEventListener("click", () => updateCapacity(parseInt(btn.dataset.cap, 10)));
  });

  // Filter Pills
  document.querySelectorAll(".filter-pill").forEach((pill) => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".filter-pill").forEach((p) => p.classList.remove("active"));
      pill.classList.add("active");
      state.activeFilter = pill.dataset.filter;
      renderKanbanBoard();
    });
  });

  // Search & Clear
  if (dom.searchInput) {
    dom.searchInput.addEventListener("input", (e) => {
      state.filters.query = e.target.value;
      if (dom.btnClearSearch) {
        dom.btnClearSearch.hidden = e.target.value.length === 0;
      }
      renderKanbanBoard();
    });
  }
  if (dom.btnClearSearch) {
    dom.btnClearSearch.addEventListener("click", () => {
      dom.searchInput.value = "";
      state.filters.query = "";
      dom.btnClearSearch.hidden = true;
      renderKanbanBoard();
    });
  }

  // Selects
  if (dom.filterProgramme) {
    dom.filterProgramme.addEventListener("change", (e) => {
      state.filters.programme = e.target.value;
      renderKanbanBoard();
    });
  }
  if (dom.sortSelector) {
    dom.sortSelector.addEventListener("change", (e) => {
      state.activeSort = e.target.value;
      renderKanbanBoard();
    });
  }

  // Viva Presets in Sidebar
  if (dom.presetDiscordant) dom.presetDiscordant.addEventListener("click", triggerDiscordantPreset);
  if (dom.presetCapacity) dom.presetCapacity.addEventListener("click", triggerCapacityPreset);
  if (dom.presetReset) dom.presetReset.addEventListener("click", resetWorkspace);

  // Live Stream Toggle
  if (dom.btnStreamToggle) dom.btnStreamToggle.addEventListener("click", toggleLiveStreaming);

  // Drawer
  if (dom.drawerClose) dom.drawerClose.addEventListener("click", closeDrawer);
  document.querySelectorAll(".drawer-tab").forEach((tab) => {
    tab.addEventListener("click", () => switchDrawerTab(tab.dataset.tab));
  });

  // Quick Notes Snippets in Tab 3
  document.querySelectorAll(".btn-snippet").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (dom.advisorNotes) {
        dom.advisorNotes.value = btn.dataset.text;
      }
    });
  });

  if (dom.resolveForm) dom.resolveForm.addEventListener("submit", handleResolveCase);

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && dom.drawer.classList.contains("open")) {
      closeDrawer();
    }
  });

  // Initial Render & Polling
  updateCapacity(5);
  toggleLiveStreaming();
}

// Hydrate on DOM ready
document.addEventListener("DOMContentLoaded", init);

/**
 * Burnout Guard — Clinical Triage Dashboard
 * Phase 4: Kanban Board, 3 Parallel Sparklines, TreeSHAP "Why" Panel, Capacity Knapsack, Feedback Loop
 */

// Application State
const state = {
  records: window.initialRecords || [],
  capacity: 5,
  selected: null,
  isStreaming: false,
  streamInterval: null,
  filters: {
    query: "",
    programme: "",
    signal: "",
  },
};

// DOM References
const dom = {
  // Stats
  statTotal: document.getElementById("stat-total"),
  statP1: document.getElementById("stat-p1"),
  statP2: document.getElementById("stat-p2"),
  statResolved: document.getElementById("stat-resolved"),
  
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
  
  // Toolbar
  searchInput: document.getElementById("search-input"),
  filterProgramme: document.getElementById("filter-programme"),
  filterCatch: document.getElementById("filter-catch"),
  dataModeBadge: document.getElementById("data-mode-badge"),
  
  // Actions
  btnReset: document.getElementById("btn-reset"),
  btnStreamToggle: document.getElementById("btn-stream-toggle"),
  streamBtnText: document.getElementById("stream-btn-text"),
  
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
  
  // SHAP & Opp
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

  // Sort P1 candidates by XGBoost success probability descending
  p1Candidates.sort((a, b) => {
    const probA = (a.intervention_opportunity && a.intervention_opportunity.intervention_success_probability) || a.priority_score || 0;
    const probB = (b.intervention_opportunity && b.intervention_opportunity.intervention_success_probability) || b.priority_score || 0;
    if (probB !== probA) return probB - probA;
    return (b.priority_score || 0) - (a.priority_score || 0);
  });

  // Allocate top K to P1, remainder deferred to P2
  p1Candidates.forEach((student, index) => {
    if (index < capacityLimit) {
      student.effective_priority = "P1";
      student.capacity_status = "Allocated P1 (High Impact)";
      student.is_overflow = false;
      student.capacity_rank = index + 1;
    } else {
      student.effective_priority = "P2";
      student.capacity_status = "Deferred to P2 (Capacity Limit Reached)";
      student.is_overflow = true;
      student.capacity_rank = index + 1;
    }
  });

  otherStudents.forEach((student) => {
    student.effective_priority = student.base_priority_level || "P3";
    student.is_overflow = false;
  });

  return records;
}

// =============================================================================
// Filtering & Card Generation
// =============================================================================
function getFilteredRecords() {
  const query = state.filters.query.trim().toLowerCase();
  const prog = state.filters.programme;
  const signal = state.filters.signal;

  return state.records.filter((rec) => {
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

    // Signal filter
    if (signal === "discordant" && !rec.is_multi_modal_override) {
      return false;
    }
    if (signal === "high_stress" && (rec.emotional_stress || 0) < 3.0) {
      return false;
    }
    if (signal === "anomaly" && (rec.high_anomaly_weeks || 0) < 1 && (rec.anomaly_mean || 0) < 0.4) {
      return false;
    }

    return true;
  });
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
  dom.statTotal.textContent = state.records.length;
  dom.statP1.textContent = state.records.filter((r) => r.status !== "Resolved" && r.effective_priority === "P1").length;
  dom.statP2.textContent = state.records.filter((r) => r.status !== "Resolved" && r.effective_priority !== "P1").length;
  dom.statResolved.textContent = state.records.filter((r) => r.status === "Resolved").length;

  dom.countP1.textContent = p1List.length;
  dom.countP2.textContent = p2List.length;
  dom.countResolved.textContent = resolvedList.length;

  // Render Columns
  dom.cardsP1.innerHTML = p1List.length ? p1List.map((r) => createCardHtml(r)).join("") : emptyColumnHtml("No active P1 cases requiring immediate action.");
  dom.cardsP2.innerHTML = p2List.length ? p2List.map((r) => createCardHtml(r)).join("") : emptyColumnHtml("No students currently under monitoring.");
  dom.cardsResolved.innerHTML = resolvedList.length ? resolvedList.map((r) => createCardHtml(r)).join("") : emptyColumnHtml("No cases resolved yet. Click 'Resolve Case' in the drawer to close a case.");

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
  const oppBand = r.intervention_opportunity ? r.intervention_opportunity.opportunity_band : "medium";

  let priorityClass = "p3";
  if (r.status === "Resolved") {
    priorityClass = "resolved";
  } else if (r.effective_priority === "P1") {
    priorityClass = "p1";
  } else if (r.effective_priority === "P2") {
    priorityClass = "p2";
  }

  const priorityLabel = r.status === "Resolved" ? "RESOLVED" : (r.effective_priority || r.priority_level);

  return `
    <article class="triage-card ${priorityClass} ${isSelected ? "selected" : ""} ${isOverflow ? "card-overflow" : ""}" data-id="${r.student_id}">
      <div class="card-topline">
        <div class="card-identity">
          <strong>${r.student_name}</strong>
          <small>${r.student_id} · ${r.programme}</small>
        </div>
        <span class="priority-pill ${priorityClass}">${priorityLabel}</span>
      </div>

      ${isDiscordant ? `<div class="discordant-badge">⚡ MULTI-MODAL CATCH</div>` : ""}
      ${isOverflow ? `<div class="overflow-badge">⚠ Deferred to P2 (Capacity Limit)</div>` : ""}

      <div class="signals-mini-row">
        <div class="signal-chip">
          <span>W17 Risk</span>
          <strong>${(r.academic_risk !== undefined ? r.academic_risk : 0.2).toFixed(2)}</strong>
        </div>
        <div class="signal-chip">
          <span>Behavior</span>
          <strong>${(r.behavior_risk !== undefined ? r.behavior_risk : 0.1).toFixed(2)}</strong>
        </div>
        <div class="signal-chip">
          <span>Stress</span>
          <strong style="color: ${(r.emotional_stress || 0) >= 3 ? "var(--p1-crimson)" : "inherit"}">${(r.emotional_stress || 0).toFixed(1)}/4</strong>
        </div>
      </div>

      <div class="card-footer">
        ${oppProb !== null ? `<span class="opp-badge">⚡ ${oppProb}% Opp (${oppBand})</span>` : `<span class="opp-badge">Meta-XAI Active</span>`}
        <span class="action-text" title="${r.recommended_action}">${r.status === "Resolved" ? (r.intervention_outcome || "Resolved") : r.recommended_action}</span>
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
  dom.drawerMeta.textContent = `ID: ${r.student_id} · Programme: ${r.programme} · Cohort: ${r.cohort_id || "2026-S1"}`;

  // Discordant Alert
  if (r.is_multi_modal_override) {
    dom.drawerDiscordantBanner.hidden = false;
  } else {
    dom.drawerDiscordantBanner.hidden = true;
  }

  // Priority badges
  dom.drawerPriorityBadge.textContent = r.base_priority_level || r.priority_level;
  dom.drawerPriorityBadge.className = `badge badge-priority ${(r.base_priority_level || r.priority_level).toLowerCase()}`;

  dom.drawerEffectiveBadge.textContent = r.capacity_status || (r.effective_priority === "P1" ? "Allocated P1" : "Monitoring Queue");
  dom.drawerConfidenceBadge.textContent = `${r.confidence || "High"} Confidence`;
  dom.drawerPriorityScore.textContent = `${Math.round((r.priority_score || 0.7) * 100)}%`;

  // Sparklines
  renderAcademicSparkline(r.academic_risks || [0.2, 0.2, 0.2, 0.2]);
  dom.drawerAcademicState.textContent = (r.trajectory || "stable_low").replaceAll("_", " ");

  renderBehaviorSparkline(r);
  dom.drawerBehVal.textContent = `Compliance: ${Math.round((r.compliance_mean || 0.8) * 100)}% · Anomaly: ${(r.anomaly_mean || 0.2).toFixed(2)}`;
  const compTrendVal = r.compliance_trend !== undefined ? r.compliance_trend : 0.0;
  dom.drawerComplianceTrend.textContent = `Compliance Trend: ${compTrendVal >= 0 ? "+" : ""}${compTrendVal.toFixed(2)}`;

  renderEmotionalMeter(r.emotional_stress || 0);

  // TreeSHAP "Why" Panel
  renderShapPanel(r);

  // XGBoost Opportunity
  if (r.intervention_opportunity) {
    const p = Math.round((r.intervention_opportunity.intervention_success_probability || 0.5) * 100);
    dom.drawerOppProb.textContent = `${p}%`;
    dom.drawerOppBand.textContent = `${(r.intervention_opportunity.opportunity_band || "medium").toUpperCase()} BAND`;
    dom.drawerOppExpl.textContent = r.intervention_opportunity.why_explanation || "XGBoost predicts responsiveness based on multi-modal temporal features.";
  } else {
    dom.drawerOppProb.textContent = "—";
    dom.drawerOppBand.textContent = "NOT TRAINED";
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
// Sparkline SVG Renderers
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
      <circle cx="${p.x}" cy="${p.y}" r="4" fill="#1e293b" stroke="#60a5fa" stroke-width="2"/>
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
  let label = "Normal Stress (0-1)";
  if (stress >= 3.0) {
    label = `Severe Stress (${stress.toFixed(1)} / 4.0)`;
  } else if (stress >= 2.0) {
    label = `Elevated Stress (${stress.toFixed(1)} / 4.0)`;
  }
  dom.drawerStressLabel.textContent = label;

  const segments = dom.drawerStressMeter.querySelectorAll(".stress-segment");
  segments.forEach((seg, idx) => {
    seg.className = "stress-segment";
    if (stress >= idx + 1) {
      if (idx >= 2) seg.classList.add("active-high");
      else if (idx === 1) seg.classList.add("active-med");
      else seg.classList.add("active-low");
    }
  });
}

// =============================================================================
// TreeSHAP "Why" Panel Renderer
// =============================================================================
function renderShapPanel(r) {
  const opp = r.intervention_opportunity || {};
  const drivers = opp.feature_contributions || [];

  if (opp.why_explanation) {
    dom.drawerShapSummary.textContent = opp.why_explanation;
    dom.drawerShapSummary.hidden = false;
  } else {
    dom.drawerShapSummary.textContent = "TreeSHAP attribution will compute once case is fully ingested.";
  }

  if (drivers.length > 0) {
    // Show top 4 drivers
    const top4 = drivers.slice(0, 4);
    dom.drawerShapBars.innerHTML = top4.map((d) => {
      const pct = Math.abs(d.percentage);
      const isPositive = d.percentage >= 0;
      const sign = isPositive ? "+" : "-";

      return `
        <div class="shap-row">
          <div class="shap-row-header">
            <span class="shap-feat-name">${d.display_name || d.feature} (${d.value})</span>
            <span class="shap-feat-pct ${isPositive ? "positive" : "negative"}">${sign}${pct.toFixed(1)}%</span>
          </div>
          <div class="shap-bar-track">
            <div class="shap-bar-fill ${isPositive ? "positive" : "negative"}" style="width: ${Math.min(100, pct)}%"></div>
          </div>
        </div>
      `;
    }).join("");
  } else {
    dom.drawerShapBars.innerHTML = `<p style="color: var(--text-dim); font-size: 11px;">Awaiting live stream feature contributions.</p>`;
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
      
      // Update local record
      const idx = state.records.findIndex((r) => r.student_id === updated.student_id);
      if (idx !== -1) {
        state.records[idx] = updated;
      }
      state.selected = updated;

      // Toast feedback
      dom.toastMessage.textContent = `✔ Case resolved as "${selectedOutcome}". Data routed to model retraining database.`;
      dom.toastMessage.hidden = false;

      // Re-render board (card moves to Resolved column)
      renderKanbanBoard();
      selectStudent(updated.student_id);
    } else {
      dom.toastMessage.textContent = "Error saving resolution. Please try again.";
      dom.toastMessage.hidden = false;
    }
  } catch (err) {
    dom.toastMessage.textContent = `Failed to connect: ${err.message}`;
    dom.toastMessage.hidden = false;
  }
}

// =============================================================================
// Viva Demo Controls: Cold Start & Live Ingestion
// =============================================================================
async function resetWorkspace() {
  if (!confirm("Reset workspace to Cold Start? This will clear all student cases from the queue.")) return;

  try {
    const response = await fetch("/api/reset", { method: "POST" });
    if (response.ok) {
      state.records = [];
      state.selected = null;
      dom.dataModeBadge.textContent = "COLD START (LIVE READY)";
      closeDrawer();
      renderKanbanBoard();
    }
  } catch (err) {
    alert(`Reset failed: ${err.message}`);
  }
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
    dom.streamBtnText.textContent = "Live Polling: ON";
    dom.btnStreamToggle.classList.add("btn-primary");
    state.streamInterval = setInterval(fetchLiveQueue, 1500);
  } else {
    dom.streamBtnText.textContent = "Live Polling: PAUSED";
    dom.btnStreamToggle.classList.remove("btn-primary");
    clearInterval(state.streamInterval);
    state.streamInterval = null;
  }
}

// Dynamic Capacity Slider Handler
function handleCapacityChange(e) {
  const newCap = parseInt(e.target.value, 10);
  state.capacity = newCap;
  dom.capacityDisplay.textContent = `${newCap} Slots`;

  // Instant 60 FPS re-render
  renderKanbanBoard();

  // Async server update
  fetch(`/api/capacity?limit=${newCap}`, { method: "POST" }).catch(() => {});
}

// =============================================================================
// Event Listeners & Initialization
// =============================================================================
function init() {
  // Slider
  dom.capacitySlider.addEventListener("input", handleCapacityChange);

  // Search & Filters
  dom.searchInput.addEventListener("input", (e) => {
    state.filters.query = e.target.value;
    renderKanbanBoard();
  });
  dom.filterProgramme.addEventListener("change", (e) => {
    state.filters.programme = e.target.value;
    renderKanbanBoard();
  });
  dom.filterCatch.addEventListener("change", (e) => {
    state.filters.signal = e.target.value;
    renderKanbanBoard();
  });

  // Action Buttons
  dom.btnReset.addEventListener("click", resetWorkspace);
  dom.btnStreamToggle.addEventListener("click", toggleLiveStreaming);
  dom.drawerClose.addEventListener("click", closeDrawer);
  dom.resolveForm.addEventListener("submit", handleResolveCase);

  // Keyboard shortcut
  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && dom.drawer.classList.contains("open")) {
      closeDrawer();
    }
  });

  // Initial Render
  renderKanbanBoard();

  // Start background live poller so streamer events appear automatically
  toggleLiveStreaming();
}

// Hydrate on DOM ready
document.addEventListener("DOMContentLoaded", init);

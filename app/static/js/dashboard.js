const state = {
  records: window.initialRecords || [],
  selected: null,
};

const elements = {
  rows: document.getElementById("queue-body"),
  search: document.getElementById("search"),
  priority: document.getElementById("priority-filter"),
  status: document.getElementById("status-filter"),
  resultCount: document.getElementById("result-count"),
  empty: document.getElementById("empty-state"),
  name: document.getElementById("detail-name"),
  meta: document.getElementById("detail-meta"),
  avatar: document.getElementById("detail-avatar"),
  confidence: document.getElementById("detail-confidence"),
  priority: document.getElementById("detail-priority"),
  priorityPill: document.getElementById("detail-priority-pill"),
  trajectory: document.getElementById("detail-trajectory"),
  sparkline: document.getElementById("sparkline"),
  explanation: document.getElementById("detail-explanation"),
  reasons: document.getElementById("reason-list"),
  evidence: document.getElementById("evidence-items"),
  action: document.getElementById("detail-action"),
  owner: document.getElementById("detail-owner"),
  caseStatus: document.getElementById("case-status"),
  saveCase: document.getElementById("save-case"),
  saveMessage: document.getElementById("save-message"),
};

function initials(name) {
  return name.split(" ").map((part) => part[0]).join("").slice(0, 2).toUpperCase();
}

function visibleRecords() {
  const query = elements.search.value.trim().toLowerCase();
  return state.records.filter((record) => {
    const matchesQuery = !query || record.student_name.toLowerCase().includes(query) || record.student_id.toLowerCase().includes(query);
    const matchesPriority = !elements.priority.value || record.priority_level === elements.priority.value;
    const matchesStatus = !elements.status.value || record.status === elements.status.value;
    return matchesQuery && matchesPriority && matchesStatus;
  });
}

function renderRows() {
  const records = visibleRecords();
  elements.resultCount.textContent = `${records.length} cases`;
  elements.empty.hidden = records.length !== 0;
  elements.rows.innerHTML = records.map((record) => `
    <tr class="queue-row ${state.selected && state.selected.student_id === record.student_id ? "selected" : ""}" data-id="${record.student_id}">
      <td><strong>${record.student_name}</strong><small>${record.student_id} · ${record.programme}</small></td>
      <td><span class="priority-pill ${record.priority_level.toLowerCase()}">${record.priority_level}</span><small>${Math.round(record.priority_score * 100)} priority</small></td>
      <td><span class="pattern-label">${record.trajectory.replaceAll("_", " ")}</span><small>${record.trajectory_explanation}</small></td>
      <td><span class="coverage-value">${Math.round(record.evidence_coverage * 100)}%</span><small>${record.available_sources}/${record.expected_sources} sources</small></td>
      <td><span class="action-label">${record.recommended_action}</span><small>Due ${record.due_date}</small></td>
      <td><span class="status-pill ${record.status.toLowerCase().replaceAll(" ", "-")}">${record.status}</span></td>
    </tr>`).join("");
  document.querySelectorAll(".queue-row").forEach((row) => row.addEventListener("click", () => selectRecord(row.dataset.id)));
}

function selectRecord(studentId) {
  state.selected = state.records.find((record) => record.student_id === studentId);
  if (!state.selected) return;
  const record = state.selected;
  elements.name.textContent = record.student_name;
  elements.meta.textContent = `${record.student_id} · ${record.programme}`;
  elements.avatar.textContent = initials(record.student_name);
  elements.confidence.textContent = `${record.confidence} confidence`;
  elements.priority.textContent = record.priority_level;
  elements.priorityPill.textContent = record.priority_level;
  elements.priorityPill.className = `priority-pill ${record.priority_level.toLowerCase()}`;
  elements.trajectory.textContent = record.trajectory.replaceAll("_", " ");
  elements.explanation.textContent = record.trajectory_explanation;
  elements.action.textContent = record.recommended_action;
  elements.owner.textContent = `Owner: ${record.owner_role} · Due ${record.due_date}`;
  elements.caseStatus.value = record.status;
  elements.reasons.innerHTML = record.reason_codes.map((reason) => `<span class="reason-chip">${reason.replaceAll("_", " ")}</span>`).join("");
  elements.evidence.innerHTML = Object.entries(record.model_evidence).map(([source, value]) => `<div class="evidence-row"><span>${source}</span><span>${value}</span></div>`).join("");
  elements.sparkline.innerHTML = record.academic_risks.map((risk) => `<span style="height: ${Math.max(12, Math.round(risk * 100))}%"></span>`).join("");
  renderRows();
}

async function saveCase() {
  if (!state.selected) return;
  elements.saveCase.disabled = true;
  const response = await fetch(`/api/students/${state.selected.student_id}/case`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status: elements.caseStatus.value }),
  });
  if (response.ok) {
    const updated = await response.json();
    state.records = state.records.map((record) => record.student_id === updated.student_id ? updated : record);
    state.selected = updated;
    elements.saveMessage.textContent = "Case update saved.";
    renderRows();
  } else {
    elements.saveMessage.textContent = "The case could not be updated.";
  }
  elements.saveCase.disabled = false;
}

[elements.search, elements.priority, elements.status].forEach((element) => element.addEventListener("input", renderRows));
elements.saveCase.addEventListener("click", saveCase);
selectRecord(state.records[0]?.student_id);

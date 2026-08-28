# Academic Burnout & Dropout Risk Detection — Frontend Demo

**Project:** R26-IT-059 | **Student:** IT22916426 Mahavitha S.M.
**Purpose:** PP2 Presentation Demo — May 13, 2026

---

## What This Is

A browser-based demo dashboard that visualises the output of the
**Unified Multi-Horizon GRU** model (Model 8). It shows per-student
risk scores, XAI feature attributions, and advisor recommendations.

No server. No installation. No internet required.
Just open `index.html` in any browser.

---

## Folder Structure

```
burnout-demo/
│
├── index.html   ← open this to run the demo
├── style.css    ← all visual styling
├── data.js      ← student predictions from model8_predictions.csv
├── app.js       ← all interaction and rendering logic
└── README.md    ← this file
```

---

## How to Run

```
1. Download or unzip the burnout-demo folder
2. Double-click index.html
3. It opens in your browser — done
```

Works in Chrome, Firefox, Safari, Edge.
Tested on macOS. Works on Windows and Linux too.

---

## How the Frontend Connects to the Model

```
model_8_unified_gru.py
        ↓ runs and writes
results/metrics/model8_predictions.csv
        ↓ rows manually copied into
burnout-demo/data.js   (STUDENTS array)
        ↓ read by
burnout-demo/app.js    (renders everything)
```

The frontend reads from `data.js` which contains a JavaScript array of
student objects. These are real predictions from the trained GRU model.

### To update data.js with new model output

After rerunning `model_8_unified_gru.py`, open `model8_predictions.csv`
and copy student rows into `data.js` following this format:

```javascript
{ student_id:"OULAD_TEST_0007",
  actual_label:1,
  academic_risk:0.9922,
  week4_risk:0.4417,
  week8_risk:0.7664,
  week12_risk:0.8518,
  week17_risk:0.9922,
  alert_level:"HIGH",
  top_feature_1:"F1_login_count",
  top_feature_1_pct:29.3,
  top_feature_2:"F6_inactive_days",
  top_feature_2_pct:11.4,
  top_feature_3:"F2_vle_clicks",
  top_feature_3_pct:9.6,
  most_important_week:13,
  main_reason:"F1 login count (29.3%) and F6 inactive days (11.4%)"
}
```

---

## What the Demo Shows

| Panel | Description |
|---|---|
| Left sidebar | All students with HIGH / MEDIUM / LOW badges |
| Filter dropdown | Filter by alert level — useful for supervisor demo |
| Student header | Student ID and overall Week 17 risk percentage |
| 4 Horizon cards | Risk at Week 4, 8, 12, 17 with colour-coded bars |
| Week 17 highlight | Marked as optimal (F1=0.7193, AUC=0.8809) |
| XAI reason | Human-readable explanation of why the student was flagged |
| Feature bars | Top 3 features driving this student's prediction |
| Attention chart | 17-week bar chart showing which week the model focused on |
| Ground truth | Actual OULAD label for honest evaluation |
| Advisor box | Actionable recommendation — changes colour per risk level |
| Footer stats | F1, AUC, baseline count — always visible |

---

## Alert Level Colour Coding

| Level | Colour | Meaning |
|---|---|---|
| HIGH | Red | Risk > 0.70 — immediate intervention needed |
| MEDIUM | Amber | Risk 0.40–0.70 — monitor closely |
| LOW | Green | Risk < 0.40 — on track |

---

## PP2 Demo Script (May 13, 2026)

### Opening (30 seconds)
> "This is the Academic Burnout and Dropout Risk Detection component.
> The model predicts at-risk students at 4 time horizons using a
> Unified Multi-Horizon GRU trained on the OULAD dataset."

### Show a HIGH risk student (1 minute)
- Click any student with a HIGH badge
- Point to Week 17 card: "Week 17 is our optimal horizon — F1 = 0.7193"
- Point to XAI feature bars: "Login count drives 29% of this prediction"
- Point to Advisor box: "The system gives actionable recommendations"

### Use the filter (30 seconds)
- Change dropdown to HIGH only
- "An academic advisor sees only critical cases at a glance"

### Show a LOW risk student (30 seconds)
- Click any LOW student
- "This student has 2% risk — the system correctly identifies them as safe"

### Show OULAD_TEST_2542 — false negative (30 seconds)
- Click student 2542 (actual=1 but predicted LOW)
- "This is a false negative — a known model limitation we discuss in the paper.
  Week 17 prediction is 37%, below our 0.55 threshold. This is why we
  report precision and recall alongside F1."

### Close (30 seconds)
> "The CSV output from this model feeds directly into IT22253194's
> meta-integration FNN as the academic_risk feature."

---

## Files Explained

### index.html
The page structure. Contains the header, sidebar, detail panel, and
footer. Loads `style.css`, `data.js`, and `app.js` in that order.
No external libraries — pure HTML.

### style.css
All visual styling. Uses CSS variables for colours so the theme is
consistent. Key sections: header, sidebar student rows, detail panel,
horizon cards, XAI feature bars, attention chart, advisor box, footer.

### data.js
Declares the `STUDENTS` constant — a JavaScript array where each object
is one student's model output. This is the only file you need to update
when you rerun the model. All column names match `model8_predictions.csv`
exactly.

### app.js
All interaction logic:
- `buildStudentList()` — renders the sidebar
- `filterStudents()` — handles the dropdown filter
- `selectStudent()` — renders the detail panel for the clicked student
- `getAdvisorText()` — generates advisor recommendation per alert level
- `riskColor()` — maps risk score to red/amber/green
- `updateSummary()` — populates the footer stats

---

## No Dependencies

This frontend has zero external dependencies:
- No React, Vue, or Angular
- No npm or node_modules
- No CDN links
- No Python or Flask server
- No internet connection required during demo

This means it will work on any laptop, in any venue, even with no WiFi.

---

## Suggested Demo Order for Supervisor

1. Open the page — it auto-selects the first HIGH risk student
2. Walk through the 4 horizon cards from left to right
3. Point out Week 17 is highlighted as optimal
4. Explain the XAI bar chart (gradient attribution, not black box)
5. Filter to HIGH only — show the list shrinks
6. Click a LOW student — show the advisor text changes colour and tone
7. Mention the CSV feeds IT22253194's FNN

---

*R26-IT-059 | SLIIT | 2025–2026*
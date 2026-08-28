# Academic Burnout & Dropout Risk Detection (GRU)

**Project:** R26-IT-059 — Explainable Multi-Modal AI for Academic Burnout Prediction
**Student:** IT22916426 Mahavitha S.M.
**Supervisor:** Prof. Anuradha Karunasena | **Co-supervisor:** Mrs. Malithi Nawarathne

---

## What This Component Does

This is the **academic-risk** component of the project's three-signal system. It trains a
**Unified Multi-Horizon GRU** on the OULAD dataset to predict student dropout/failure risk at
four time horizons simultaneously (Week 4, 8, 12, 17), with gradient-based XAI explanations per
student, and serves the results through a live REST API + browser dashboard for demos and PP2.

**Key Results (Model 8 — Unified Multi-Horizon GRU):**

| Horizon | F1     | AUC    | Threshold |
|---------|--------|--------|-----------|
| Week 4  | 0.5196 | 0.6880 | 0.45      |
| Week 8  | 0.6033 | 0.7789 | 0.45      |
| Week 12 | 0.6283 | 0.8102 | 0.50      |
| Week 17 | 0.7193 | 0.8809 | 0.55      |

**Optimal horizon: Week 17 — beats all 7 baselines** (Random Forest, Logistic Regression,
XGBoost, Gradient Boosting, Simple LSTM, GRU-baseline, BiLSTM). See
`Multi Horizon Time Series Single Window/README.MD` for the full baseline comparison table.

---

## Folder Structure

```
my-academic-gru/
│
├── Multi Horizon Time Series Single Window/   ← model training (see its own README.MD)
│   ├── src/
│   │   ├── model_8_unified_gru.py   ← MAIN MODEL — trains the GRU, writes predictions
│   │   ├── data_loader.py           ← loads and splits OULAD data
│   │   └── app.py                   ← standalone Streamlit demo (optional)
│   ├── models/saved/                ← trained weights + scaler
│   ├── results/
│   │   ├── figures/                 ← accuracy curve, attention heatmap, XAI plots
│   │   └── metrics/                 ← model8_predictions.csv / .json, results.json
│   ├── data/raw/                    ← OULAD CSVs go here
│   └── notebooks/                   ← exploratory notebooks
│
└── burnout-demo/                    ← live REST API + dashboard (see below)
    ├── app_api.py                   ← Flask API, port 5001
    ├── api-index-expanded.html      ← dashboard UI (open this in a browser)
    ├── data/
    │   ├── model8_predictions.csv   ← Test Students tab data source
    │   └── pp2_final_results.csv    ← PP2 Concurrent tab data source
    └── FLASK_SETUP.md               ← original API setup notes
```

---

## Part 1 — Training the GRU Model

```bash
cd "Multi Horizon Time Series Single Window/src"
python model_8_unified_gru.py
```

Loads OULAD data, trains the Unified Multi-Horizon GRU (up to 500 epochs, early stopping —
historically triggers around epoch 106, best epoch 56), evaluates all 4 horizons, generates
the figures in `results/figures/`, and writes `model8_predictions.csv` / `.json` to
`results/metrics/`. Full architecture, 13-feature list, and XAI methodology are documented in
`Multi Horizon Time Series Single Window/README.MD`.

Requirements: Python 3.10+, `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `shap`,
`streamlit`.

---

## Part 2 — Live Dashboard (`burnout-demo`)

The dashboard is now a **live Flask API + browser frontend**, not the static `data.js` version
described in `burnout-demo`'s own older README. `app_api.py` reads CSVs at startup and serves
them as JSON; `api-index-expanded.html` fetches from those endpoints and renders them into tabs.

### Run it

```bash
cd burnout-demo
python3 app_api.py        # starts the API on http://localhost:5001
```

Then open `api-index-expanded.html` in a browser.

### Tabs and what each one needs running

| Tab | Data source | Server required |
|---|---|---|
| Test Students | `data/model8_predictions.csv` | `app_api.py` (port 5001) |
| PP2 Concurrent | `data/pp2_final_results.csv` | `app_api.py` (port 5001) |
| Model Metrics | cross-year validation + model performance | `app_api.py` (port 5001) |
| Academic Risk | GRU academic-risk view | `app_api.py` (port 5001) |
| SLIIT Synthetic | `sliit-synthetic/SLIIT BurnoutDetector/results/metrics/sliit_anomaly_scores_final.csv` (teammate IT22215710's synthetic-cohort validation) | `app_api.py` (port 5001) |
| Behavioral Anomaly | `sliit-synthetic/Behavioral Anomaly Detection VAE/results/metrics/behavior_prediction.csv` | **separate** — `vae_api.py` in `sliit-synthetic/Behavioral Anomaly Detection VAE/src/` (port 5003) |
| Meta-FNN Risk | fused GRU + VAE + NLP score | **separate** — `meta_api.py` in `meta-integration/` (port 5002) |
| Emotional Analysis | NLP text classifier | **separate** — Next.js frontend in `emotional-nlp-fresh/emotional_burnout_nlp/frontend/` (port 3000) |

Only `app_api.py` is required for this module's own tabs (Test Students, PP2, Model Metrics,
Academic Risk, SLIIT Synthetic). The Behavioral Anomaly, Meta-FNN Risk, and Emotional Analysis
tabs pull from other teammates' modules and need their servers running separately if you want
those tabs populated too.

### How many terminals do I need to open?

**Minimum (this module's own tabs only): 1 terminal.**

```bash
# Terminal 1 — required
cd my-academic-gru/burnout-demo
python3 app_api.py                    # http://localhost:5001
```
Then open `api-index-expanded.html` in a browser. This alone populates Test Students, PP2
Concurrent, Model Metrics, Academic Risk, and SLIIT Synthetic.

**Full demo (every tab populated): 4 terminals**, one per teammate's API — run whichever ones
correspond to the tabs you want live:

```bash
# Terminal 1 — academic risk (this module)
cd my-academic-gru/burnout-demo
python3 app_api.py                    # http://localhost:5001

# Terminal 2 — behavioral anomaly (VAE), teammate IT22215710
cd sliit-synthetic/"Behavioral Anomaly Detection VAE"/src
python3 vae_api.py                    # http://localhost:5003

# Terminal 3 — fused Meta-FNN risk score, teammate IT22253194
cd meta-integration
python3 meta_api.py                   # http://localhost:5002

# Terminal 4 — emotional-text analysis, teammate IT22196392
cd emotional-nlp-fresh/emotional_burnout_nlp/frontend
npm run dev                           # http://localhost:3000
```

Each terminal is independent — start only the ones you need for the tabs you're demoing that
day, and leave the rest closed. A tab whose server isn't running just shows a "not available"
error in that tab; it doesn't break the rest of the dashboard.

### REST API reference (`app_api.py`, port 5001)

```
GET  /api/health
GET  /api/test-students[?alert_level=HIGH|MEDIUM|LOW][&limit=N]
GET  /api/test-students/<id>
GET  /api/test-students/stats
GET  /api/pp2-students[?agreement=true|false][&alert_level=...]
GET  /api/pp2-students/<id>
GET  /api/pp2-students/stats
GET  /api/cross-year-validation
GET  /api/model-performance
GET  /api/summary
GET  /api/sliit-students[?student_type=...][&module_code=...][&limit=N]
GET  /api/sliit-students/<id>
GET  /api/sliit-students/stats
```

---

## Model Architecture (summary)

```
Input: (batch, 17_weeks, 13_features)
        ↓
GRU Encoder (hidden=128, layers=2, dropout=0.3)
        ↓
Attention Layer (learns which weeks matter most)
        ↓
4 Prediction Heads (Week 4 / 8 / 12 / 17), simple equal-weighted loss averaging across heads

Total parameters: 170,757
```

Full 13-feature table, XAI methodology (gradient attribution — SHAP is unreliable for GRU and
kept only as an approximate reference), and output columns for the Meta-Integration FNN are in
`Multi Horizon Time Series Single Window/README.MD`.

---

## Known Limitations

- The Behavioral Anomaly and Meta-FNN Risk tabs in `burnout-demo` depend on other modules' APIs
  (`vae_api.py`, `meta_api.py`) being run separately — they are not part of this module.
- The Meta-Integration FNN's fused score currently uses a **simulated** behavioral signal in
  place of a real per-student VAE merge, due to a student-ID mismatch between this module's
  `OULAD_TEST_XXXX`-style IDs and the VAE module's real OULAD integer IDs (`vae_real_data: false`
  in `meta-integration/meta_results/meta_config.json`). This is a known, disclosed limitation of
  the fusion layer, not of this module's own GRU results.

---

*R26-IT-059 | SLIIT | 2025–2026*

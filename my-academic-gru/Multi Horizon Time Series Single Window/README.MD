# Academic Burnout & Dropout Risk Detection — Backend

**Project:** R26-IT-059 | **Student:** IT22916426 Mahavitha S.M.
**Supervisor:** Prof. Anuradha Karunasena | **Co-supervisor:** Mrs. Malithi Nawarathne

---

## What This Component Does

This is the **Academic Burnout and Dropout Risk Detection** component of the
Explainable Multi-Modal AI for Academic Burnout Prediction system.

It trains a **Unified Multi-Horizon GRU** model on the OULAD dataset and
predicts student failure/dropout risk at 4 time horizons simultaneously
(Week 4, 8, 12, 17), with XAI explanations per student.

**Key Results:**

| Horizon | F1     | AUC    | Threshold |
|---------|--------|--------|-----------|
| Week 4  | 0.5196 | 0.6880 | 0.45      |
| Week 8  | 0.6033 | 0.7789 | 0.45      |
| Week 12 | 0.6283 | 0.8102 | 0.50      |
| Week 17 | 0.7193 | 0.8809 | 0.55      |

**Optimal horizon: Week 17 — beats all 7 baselines.**

---

## Folder Structure

```
Multi Horizon Time Series Single Window/
│
├── src/
│   ├── model_8_unified_gru.py   ← MAIN MODEL — run this
│   ├── data_loader.py           ← loads and splits OULAD data
│   └── app.py                   ← Streamlit demo app
│
├── models/
│   └── saved/
│       ├── model8_unified_gru_best.pt   ← saved best weights
│       └── scaler.pkl                   ← feature scaler
│
├── results/
│   ├── figures/
│   │   ├── unified_gru_accuracy_curve.png
│   │   ├── unified_gru_comparison.png
│   │   ├── unified_gru_attention_heatmap.png
│   │   ├── model_8_xai_features.png
│   │   ├── model8_shap.png
│   │   └── model_8_xai_student.png
│   └── metrics/
│       ├── model8_unified_gru_results.json
│       ├── model8_predictions.csv    ← give this to IT22253194
│       └── model8_predictions.json  ← give this to IT22253194
│
├── data/
│   └── raw/                     ← OULAD CSV files go here
│
├── notebooks/                   ← exploratory notebooks
│
└── .venv/                       ← Python virtual environment
```

---

## Requirements

```
Python 3.10+
torch
numpy
pandas
scikit-learn
matplotlib
shap
streamlit
```

Install all dependencies:

```bash
cd "Multi Horizon Time Series Single Window"
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

pip install torch numpy pandas scikit-learn matplotlib shap streamlit
```

---

## How to Run

### Step 1 — Train the model

```bash
cd "Multi Horizon Time Series Single Window/src"
python model_8_unified_gru.py
```

This will:
- Load OULAD data via `data_loader.py`
- Train the Unified Multi-Horizon GRU (up to 500 epochs, early stopping)
- Evaluate on test set at all 4 horizons
- Generate all 5 charts in `results/figures/`
- Generate XAI explanations per student
- Save `model8_predictions.csv` in `results/metrics/`
- Print final summary with F1 and AUC per horizon

Training takes approximately **5–15 minutes** depending on hardware.
Early stopping triggered at epoch 106 (best epoch: 56).

### Step 2 — Run Streamlit demo (optional)

```bash
cd "Multi Horizon Time Series Single Window/src"
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## Model Architecture

```
Input: (batch, 17_weeks, 13_features)
         ↓
GRU Encoder (hidden=128, layers=2, dropout=0.3)
         ↓
Attention Layer (learns which weeks matter most)
         ↓
4 Prediction Heads (one per horizon)
    Head 1 → Week 4  risk
    Head 2 → Week 8  risk
    Head 3 → Week 12 risk
    Head 4 → Week 17 risk (optimal)

Total parameters: 170,757
```

## 13 Features (F1–F13)

| Feature | Description |
|---------|-------------|
| F1_login_count | Weekly VLE login count |
| F2_vle_clicks | Total VLE click interactions |
| F3_avg_score | Average assessment score |
| F4_num_submitted | Number of assessments submitted |
| F5_on_time_rate | On-time submission rate |
| F6_inactive_days | Days with no platform activity |
| F7_score_change | Week-over-week score change |
| F8_running_avg | Cumulative running average score |
| F9_score_variance | Variance in scores |
| F10_min_score | Minimum score recorded |
| F11_max_score | Maximum score recorded |
| F12_late_count | Number of late submissions |
| F13_weeks_since_active | Weeks since last activity |

---

## XAI Methods

Three levels of explainability are implemented:

**Level 1 — Temporal attention:** The attention layer shows which of the
17 weeks the model focused on when making predictions.

**Level 2 — Feature attribution:** Gradient-based importance scores show
which of the 13 features drove the prediction across the test set.

**Level 3 — Single student:** `explain_single_student()` produces a
per-student explanation showing feature importance and attention weights
for that specific student.

**Note on SHAP:** SHAP DeepExplainer is included but unreliable for GRU
(GRU recurrent connections cause additivity errors). Gradient attribution
is the primary XAI method. SHAP output is kept as an approximate reference
only — do not cite SHAP results in the paper.

---

## Output Files for IT22253194

After running `model_8_unified_gru.py`, give these two files to IT22253194:

```
results/metrics/model8_predictions.csv
results/metrics/model8_predictions.json
```

**Columns IT22253194 uses for FNN training:**

```
student_id      ← join key across components
academic_risk   ← main FNN input (week 17 risk, 0–1)
week4_risk      ← optional additional input
week8_risk      ← optional additional input
week12_risk     ← optional additional input
week17_risk     ← same as academic_risk
actual_label    ← ground truth (0=safe, 1=at-risk)
```

**Columns for dashboard UI only (IT22253194 ignores for training):**

```
alert_level, top_feature_1/2/3, top_feature_1/2/3_pct,
most_important_week, main_reason
```

---

## Research Novelty

This is the **first unified multi-horizon GRU with attention and XAI
for educational burnout/dropout prediction** using the OULAD dataset.

Research gap confirmed by: **Jin et al. 2024, AIED conference.**

The accuracy-vs-lead-time curve (Chart 1) is the key novel contribution —
it shows how prediction accuracy improves as the model observes more weeks,
and identifies Week 17 as the optimal intervention window leaving 16–21
weeks for academic advisors to act.

---

## Key Configuration

```python
SEED        = 42
EPOCHS      = 500
PATIENCE    = 50
BATCH_SIZE  = 64
LR          = 0.0003
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3

HORIZON_WEEKS   = [4, 8, 12, 17]
HORIZON_INDICES = [3, 7, 11, 16]   # 0-based indices
```

---

## Baselines Beaten

| Model | F1 | vs Model 8 |
|---|---|---|
| Random Forest | 0.6647 | +0.0546 |
| Logistic Regression | 0.6800 | +0.0393 |
| XGBoost | 0.6890 | +0.0303 |
| Gradient Boosting | 0.6965 | +0.0228 |
| Simple LSTM | 0.7087 | +0.0106 |
| GRU (baseline) | 0.7185 | +0.0008 |
| BiLSTM | 0.7107 | +0.0086 |

---

*R26-IT-059 | SLIIT | 2025–2026*
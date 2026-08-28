# R26-IT-059 — Explainable Multi-Modal AI System for Early Academic Burnout and Dropout Prediction

**Student:** Mahavitha S.M. | IT22916426  
**Component:** Academic Behavioural Prediction — Unified Multi-Horizon GRU  
**Supervisor:** Prof. Anuradha Karunasena  
**Co-Supervisor:** Ms. Malithi Nawarathne  
**Institute:** SLIIT Faculty of Computing | 2025–2026  
**Branch:** `feature/academic-trend-IT22916426`

---

## Project Overview

This repository contains the individual component for **Academic Behavioural Prediction** within the R26-IT-059 multi-modal burnout detection system.

The component trains a **Unified Multi-Horizon Gated Recurrent Unit (GRU)** model on the Open University Learning Analytics Dataset (OULAD) to simultaneously predict academic burnout and dropout risk at four time horizons — Week 4, Week 8, Week 12, and Week 17. The model produces an **accuracy-versus-lead-time curve** that empirically identifies the optimal intervention window for academic advisors.

### Key Results

| Horizon | F1 Score | AUC-ROC | Weeks Remaining |
|---|---|---|---|
| Week 4 | 0.5178 | 0.6876 | 29 weeks |
| Week 8 | 0.6081 | 0.7813 | 25 weeks |
| Week 12 | 0.6352 | 0.8110 | 21 weeks |
| **Week 17 ★ Optimal** | **0.7227** | **0.8831** | **16 weeks** |

Beats all 7 baselines: Random Forest, Logistic Regression, XGBoost, Gradient Boosting, Simple LSTM, BiLSTM, and GRU.

---

## Repository Structure

```
R26-IT-059-Burnout-Prediction/
│
├── Multi Horizon Time Series Single Window/
│   ├── src/
│   │   ├── model_8_unified_gru.py     ← main model (train + evaluate)
│   │   ├── data_loader.py             ← data loading and splitting
│   │   └── app_api.py                 ← Flask REST API (port 5001)
│   │
│   ├── data/
│   │   └── processed/
│   │       ├── X_sequences.npy        ← feature tensor (16950, 17, 13)
│   │       └── y_labels.npy           ← binary labels (16950,)
│   │
│   ├── models/
│   │   └── saved/
│   │       ├── model8_unified_gru_best.pt  ← trained model weights
│   │       └── scaler.pkl                  ← feature scaler
│   │
│   └── results/
│       ├── figures/                   ← all generated charts
│       └── metrics/
│           ├── model8_predictions.csv ← 2543 student predictions
│           └── model8_results.json    ← evaluation metrics
│
├── burnout-demo/
│   ├── index.html                     ← frontend dashboard
│   ├── style.css                      ← dashboard styles
│   ├── app.js                         ← API connection + rendering
│   └── data.js                        ← static fallback data
│
└── README.md                          ← this file
```

---

## Dataset

**Open University Learning Analytics Dataset (OULAD)**  
- Source: Open University, United Kingdom  
- Licence: Creative Commons CC-BY 4.0 (free for research)  
- Students: 32,593 → 16,950 after preprocessing  
- Features: 17 weeks × 13 features per student  
- Label: Binary — 0 = Pass/Distinction, 1 = Fail/Withdrawn  
- At-risk rate: 31.4%

### 13 Input Features

| Feature | Source | Type |
|---|---|---|
| F1 Login Count | studentVle | Engagement |
| F2 VLE Clicks | studentVle | Engagement |
| F3 Avg Score | studentAssessment | Score |
| F4 Num Submitted | studentAssessment | Engagement |
| F5 On-Time Rate | studentAssessment | Engagement |
| F6 Inactive Days | studentVle | Engagement |
| F7 Score Change | studentAssessment | Score |
| F8 Running Avg | studentAssessment | Score |
| F9 Score Variance | studentAssessment | Score |
| F10 Min Score | studentAssessment | Score |
| F11 Max Score | studentAssessment | Score |
| F12 Late Count | studentAssessment | Engagement |
| F13 Weeks Since Active | studentVle | Engagement |

---

## Model Architecture

```
Input: (batch, 17 weeks, 13 features)
    ↓
GRU Encoder
  hidden_size = 256
  num_layers  = 2
  dropout     = 0.3
    ↓
Attention Layer
  learns which weeks matter most (XAI Level 1)
    ↓
4 Independent Prediction Heads
  Head 1 → Week 4  risk score
  Head 2 → Week 8  risk score
  Head 3 → Week 12 risk score
  Head 4 → Week 17 risk score
    ↓
Output: 4 risk probabilities (0.0–1.0)
```

### Final Training Configuration

| Parameter | Value |
|---|---|
| Learning Rate | 0.0003 |
| Hidden Size | 256 |
| Batch Size | 128 |
| Dropout | 0.3 |
| Patience | 80 |
| Horizon Weights | 0.1 / 0.2 / 0.3 / 0.4 |
| Label Smoothing | 0.1 |
| Best Epoch | 87 |

---

## Three-Level XAI

| Level | Method | What It Shows |
|---|---|---|
| Level 1 — Temporal | Attention weights | Which weeks the model focused on |
| Level 2 — Feature | Gradient attribution | Which features drove the prediction |
| Level 3 — Student | Per-student gradient | Why THIS specific student was flagged |

---

## Setup and Installation

### Requirements

- Python 3.10 or higher
- Mac (MPS) or Windows (CPU/CUDA)

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-org/R26-IT-059-Burnout-Prediction.git
cd R26-IT-059-Burnout-Prediction
```

### Step 2 — Create virtual environment

```bash
python -m venv .venv

# Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install torch torchvision torchaudio
pip install pandas numpy scikit-learn matplotlib shap flask flask-cors
```

### Step 4 — Download OULAD data

Download from: https://analyse.kmi.open.ac.uk/open_dataset

Place the CSV files in:
```
Multi Horizon Time Series Single Window/data/raw/
```

### Step 5 — Run feature engineering notebook

```bash
# Open and run notebook 02 to generate processed data
jupyter notebook
# Run: 02_feature_engineering.ipynb
```

This produces `X_sequences.npy` and `y_labels.npy` in `data/processed/`.

---

## Running the Model

### Train and evaluate

```bash
cd "Multi Horizon Time Series Single Window/src"
python model_8_unified_gru.py
```

Training takes approximately 30–60 minutes on Mac MPS. Output includes:
- 5 result charts saved to `results/figures/`
- Evaluation metrics saved to `results/metrics/`
- Trained model saved to `models/saved/`
- Predictions CSV for IT22253194 meta-integration layer

---

## Running the Frontend Demo

### Step 1 — Start the Flask API

```bash
cd "Multi Horizon Time Series Single Window/src"
python app_api.py
# Runs at http://localhost:5001
```

### Step 2 — Open the dashboard

```bash
# Mac
open burnout-demo/index.html

# Windows
start burnout-demo/index.html
```

The dashboard shows:
- All 2543 test student predictions
- HIGH / MEDIUM / LOW risk filtering
- Four horizon risk scores per student
- XAI feature attribution bars
- Advisor recommendation text
- 🟢 API Live / 🟡 Static Mode indicator

### Available API Routes

| Route | Description |
|---|---|
| `/health` | API status and model info |
| `/students` | All student predictions |
| `/students?alert=HIGH` | Filtered by alert level |
| `/students/7` | Single student by index |
| `/summary` | Counts and model metrics |

---

## Output CSV — for Meta-Integration Layer

The model generates `results/metrics/model8_predictions.csv` with 2543 rows.

Key columns passed to IT22253194:

| Column | Description |
|---|---|
| `student_id` | Student identifier |
| `actual_label` | Ground truth (0=safe, 1=at-risk) |
| `academic_risk` | Week 17 risk score (primary signal) |
| `week4_risk` | Week 4 horizon risk |
| `week8_risk` | Week 8 horizon risk |
| `week12_risk` | Week 12 horizon risk |
| `week17_risk` | Week 17 horizon risk |
| `alert_level` | HIGH / MEDIUM / LOW |
| `top_feature_1` | Most important XAI feature |
| `main_reason` | Plain English explanation |

---

## Research Novelty

1. **First unified multi-horizon GRU** trained simultaneously on 4 prediction horizons from one training run on OULAD — gap confirmed by Aljaloud et al. 2024 systematic review

2. **Accuracy-versus-lead-time curve** — novel research instrument empirically identifying the optimal intervention window

3. **Three-level XAI** combining temporal attention, gradient feature attribution, and per-student explanation

---

## Common Errors

| Error | Fix |
|---|---|
| `FileNotFoundError: X_sequences.npy` | Run notebook 02 first |
| `ModuleNotFoundError: torch` | Run `pip install torch` |
| `ModuleNotFoundError: flask` | Run `pip install flask flask-cors` |
| `MPS device not found` | Change `device = torch.device('cpu')` |
| Dashboard shows Static Mode | Start `app_api.py` in a separate terminal |

---

## References

- Waheed et al. (2023). Expert Systems with Applications. Q1 Elsevier.
- Aljaloud et al. (2024). Computers and Education: AI. Q1 Elsevier.
- Nagy & Molontay (2023). Int. J. AI in Education. Q2 Springer.
- Oguche et al. (2025). Discover Computing. Q2 Springer.
- Torkhani & Rezgui (2024). ICODAI Conference.

---

*R26-IT-059 | SLIIT Faculty of Computing | IT22916426 Mahavitha S.M. | 2025–2026*
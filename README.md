# R26-IT-059 — Emotional Burnout Detection System

**Explainable NLP for Academic Mental Health Monitoring**  
SLIIT Final Year Research · IT22196392 · Induwara K.P.Y.

---

## Overview

A full-stack mental health detection system that classifies student text into 7 emotional states using fine-tuned BERT and RoBERTa models, with word-level attention heatmaps for explainability (XAI). An academic advisor dashboard enables real-time risk monitoring across multiple students with weekly progression tracking.

**Best model accuracy: 82.32% (BERT + RoBERTa Ensemble)**

---

## System Architecture

```
Browser (Next.js Dashboard)
        │
        │  POST /api/analyze  { text }
        ▼
Next.js API Route (analyze.js)
        │
        │  spawn python predict.py "<text>"
        ▼
Python Inference (predict.py)
        │
        │  loads BERT model, runs inference, extracts attention
        ▼
BERT Fine-tuned Model (models/saved/bert_burnout)
        │
        │  returns JSON → stdout
        ▼
React Components render results
```

**One command to run:** `cd emotional_burnout_nlp/frontend && npm run dev`

---

## 7 Emotional Categories

| Label | Risk Level | Stress Weight |
|-------|-----------|---------------|
| Normal | Low | 0.0 |
| Stress | Medium | 0.5 |
| Anxiety | Medium | 0.6 |
| Depression | High | 0.8 |
| Bipolar | High | 0.75 |
| Personality disorder | High | 0.7 |
| Suicidal | **Critical** | 1.0 |

**Stress Score formula:**  
`(Suicidal×1.0 + Depression×0.8 + Bipolar×0.75 + PD×0.7 + Anxiety×0.6 + Stress×0.5) × 100`

---

## Model Results

| # | Model | Accuracy | Notes |
|---|-------|----------|-------|
| 1 | Logistic Regression (baseline) | ~65% | TF-IDF features |
| 2 | Random Forest (baseline) | ~62% | TF-IDF features |
| 3 | SVM (baseline) | ~68% | TF-IDF + class_weight='balanced' |
| 4 | LSTM | ~72% | Custom tokenizer, 128-dim embeddings |
| 5 | BERT fine-tuned | ~85% | bert-base-uncased, 3 epochs |
| 6 | VADER + RoBERTa zero-shot | ~55% | No training required |
| 7 | RoBERTa fine-tuned | 81.72% | Weighted loss for class imbalance |
| 8 | Mental-RoBERTa fine-tuned | 80.37% | Domain-specific pre-training |
| **9** | **BERT + RoBERTa Ensemble** | **82.32%** | **Best model — used in production** |

---

## Project Structure

```
R26-IT-059-Burnout-Prediction/
├── README.md
└── emotional_burnout_nlp/
    ├── app.py                          # Streamlit testing UI
    ├── predict.py                      # Python BERT inference script
    ├── data/
    │   ├── Combined Data.csv           # Raw dataset
    │   └── cleaned_data.csv            # 87,250 rows — cleaned, labelled
    ├── models/
    │   └── saved/
    │       ├── bert_burnout/           # Fine-tuned BERT (main model)
    │       ├── mental_roberta/         # Fine-tuned Mental-RoBERTa
    │       ├── lstm_vocab.pkl          # LSTM vocabulary
    │       ├── model1_logistic.pkl     # Logistic Regression
    │       ├── model2_random_forest.pkl
    │       ├── model3_svm.pkl
    │       └── model4_lstm.pt
    ├── notebooks/
    │   ├── 01_data_exploration.ipynb
    │   ├── 02_baseline_models.ipynb    # LR, RF, SVM + k-fold CV
    │   ├── 03_lstm_model.ipynb
    │   ├── 04_bert_finetuning.ipynb
    │   ├── 05_attention_visualization.ipynb
    │   ├── 06_vader_roberta_zeroshot.ipynb
    │   ├── 07_roberta_finetuning.ipynb # WeightedTrainer for class imbalance
    │   ├── 08_mental_roberta_finetuning.ipynb
    │   ├── 09_ensemble_model.ipynb
    │   └── 10_lead_time_analysis.ipynb # Weekly risk progression
    ├── results/
    │   ├── figures/                    # Confusion matrices, charts
    │   └── metrics/                    # JSON results per model
    └── frontend/                       # Next.js 16 dashboard
        ├── package.json
        ├── next.config.js
        ├── pages/
        │   ├── _app.js
        │   ├── index.js                # Main dashboard page
        │   └── api/
        │       └── analyze.js          # API route → Python subprocess
        └── src/
            ├── components/
            │   ├── Header.jsx          # Title bar + API status
            │   ├── StudentList.jsx     # Left sidebar, filter, add/clear
            │   ├── StudentDetail.jsx   # Analysis + progression tabs
            │   ├── MetricCards.jsx     # Prediction / Confidence / Stress
            │   ├── RiskBadge.jsx       # Glowing risk level badge
            │   ├── AttentionHeatmap.jsx # XAI word-level heatmap
            │   ├── ProbabilityBars.jsx # Per-class probability bars
            │   └── ArchitectureDiagram.jsx
            ├── data/
            │   └── students.js         # localStorage read/write
            └── styles/
                └── globals.css         # Dark theme + animations
```

---

## Dataset

- **Source:** Combined mental health text dataset
- **Size:** 87,250 labelled samples
- **Format:** `text`, `label`, `text_clean`, `label_id`
- **Preprocessing:** Lowercased, punctuation removed, whitespace normalized
- **Split:** Stratified 80/10/10 train/val/test

---

## Setup & Running

### Prerequisites

- Python (Anaconda recommended) with `torch` and `transformers`
- Node.js 18+

### Install Python dependencies

```bash
pip install torch transformers scikit-learn pandas numpy matplotlib seaborn
```

### Install frontend dependencies

```bash
cd emotional_burnout_nlp/frontend
npm install
```

### Run the Next.js dashboard

```bash
cd emotional_burnout_nlp/frontend
npm run dev
```

Open `http://localhost:3000`

### Run the Streamlit app (alternative)

```bash
cd emotional_burnout_nlp
streamlit run app.py
```

---

## Dashboard Features

| Feature | Description |
|---------|-------------|
| Student list | Add/remove students, filter by risk level |
| Risk badge | Color-coded: cyan (Low) → amber (Medium) → red (High) → pulsing red (Critical) |
| Attention heatmap | Word-level XAI — shows which words drove the prediction |
| Probability bars | All 7 class probabilities with animated bars |
| Metric cards | Prediction, confidence %, stress score /100 |
| Advisor recommendation | Auto-generated guidance based on prediction |
| Weekly progression | Risk chart across weeks 2, 4, 8, 18 |
| Persistent storage | Student data saved in browser localStorage |

---

## API

### `POST /api/analyze`

**Request:**
```json
{ "text": "I feel overwhelmed and can't sleep anymore" }
```

**Response:**
```json
{
  "prediction": "Anxiety",
  "confidence": 0.74,
  "risk_level": "Medium",
  "stress_score": 57.2,
  "probabilities": {
    "Anxiety": 0.74,
    "Stress": 0.12,
    "Depression": 0.06,
    "Normal": 0.05,
    "Suicidal": 0.01,
    "Bipolar": 0.01,
    "Personality disorder": 0.01
  },
  "attention": [
    { "token": "overwhelmed", "weight": 0.21 },
    { "token": "sleep", "weight": 0.18 }
  ]
}
```

---

## Key Design Decisions

**Why BERT over RoBERTa for inference?**  
BERT fine-tuned achieved higher accuracy (~85%) on this dataset. The ensemble is the most accurate but requires both models loaded simultaneously — BERT alone is used in the real-time API for speed.

**Why attention for XAI?**  
Attention weights from all 12 BERT layers are averaged across all 12 heads, then the CLS row is extracted and normalized. This gives a per-token importance score that directly explains the model's prediction in human-readable form.

**Why class-weighted loss?**  
The dataset has significant class imbalance (Suicidal and Bipolar are underrepresented). `WeightedTrainer` uses inverse-frequency weights so the model does not overfit to majority classes.

---

## Ethical Disclaimer

This system is a **research prototype** intended to assist academic advisors — it README.md is done. It covers:

System architecture diagram (Browser → Next.js → Python → BERT)
All 9 models with accuracy results in a table
Full project file tree with descriptions
Dataset info (87,250 rows, 7 classes)
Setup instructions (Python + Node.js)
Dashboard features table
API request/response example
Key design decisions (why BERT, why attention XAI, why weighted loss)
Ethical disclaimer
Your author details (IT22196392, Induwara K.P.Y., SLIIT)is **not** a diagnostic tool. Predictions should never replace professional README.md is done. It covers:

System architecture diagram (Browser → Next.js → Python → BERT)
All 9 models with accuracy results in a table
Full project file tree with descriptions
Dataset info (87,250 rows, 7 classes)
Setup instructions (Python + Node.js)
Dashboard features table
API request/response example
Key design decisions (why BERT, why attention XAI, why weighted loss)
Ethical disclaimer
Your author details (IT22196392, Induwara K.P.Y., SLIIT)mental health evaluation. Any Critical/Suicidal prediction must be escalated to a qualified professional immediately.

---

## Author

**Induwara K.P.Y.**  
Student ID: IT22196392  
Research Group: R26  
Project Code: IT-059  
Institution: SLIIT (Sri Lanka Institute of Information Technology)  
Branch: `feature/emotional-burnout-IT22196392`

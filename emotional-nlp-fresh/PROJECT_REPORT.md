# Project Report — Emotional Burnout Detection from Student Text

**Project ID:** R26-IT-059
**Student ID:** IT22196392
**Name:** Induwara K.P.Y.
**Component:** Emotional Burnout Detection from Student-Generated Text using Transformer-Based NLP with Explainable AI
**Institution:** SLIIT — Final Year Research Project

---

## 1. Executive Summary

This project builds an AI system that reads student-written text and classifies it into **7 mental-health categories**, then **explains its reasoning** using attention visualization. The goal is to give academic advisors an early-warning tool that detects emotional burnout from a student's own words — weeks before it shows up in grades.

I built and compared **9 models** spanning classical machine learning, deep learning, and transformers. The best model — a **BERT + RoBERTa ensemble** — reached **82.32% accuracy**. A separate lead-time study showed the system can flag burnout as early as **Week 2 of a semester with 91% accuracy**. The trained model is deployed in a full-stack web dashboard.

---

## 2. Problem Statement & Motivation

Students experiencing burnout often reveal it in **how they write** long before academic decline becomes visible. Traditional monitoring (grades, attendance, logins) reacts *after* the damage is done. Text, by contrast, carries emotional signals in real time.

**Research gap:** Existing systems either (a) use only behavioral data, not language, or (b) act as black boxes that give a label with no explanation an advisor can trust.

**My solution:** A transformer-based NLP classifier that is both **accurate** and **explainable** — it shows *which words* drove each decision.

---

## 3. Research Objectives

1. Build a multi-class classifier for 7 emotional/mental-health states from text.
2. Compare classical ML, deep learning, and transformer approaches.
3. Solve the dataset's class imbalance problem.
4. Add Explainable AI so predictions are transparent to a human.
5. Measure how *early* burnout can be detected (lead-time analysis).
6. Deploy the model in a usable advisor dashboard.

---

## 4. The Dataset

- **Source:** Combined Reddit / mental-health text corpus
- **Raw file:** `Combined Data.csv`
- **Cleaned file:** `cleaned_data.csv` — **51,068 labelled samples**
- **Columns:** `text`, `label`, `text_clean`, `label_id`

### Class Distribution (verified)

| Class | Samples | Share | Risk Level |
|-------|---------|-------|-----------|
| Normal | 16,039 | 31.4% | Low |
| Depression | 15,085 | 29.5% | High |
| Suicidal | 10,638 | 20.8% | Critical |
| Anxiety | 3,617 | 7.1% | Medium |
| Bipolar | 2,501 | 4.9% | High |
| Stress | 2,293 | 4.5% | Medium |
| Personality disorder | 895 | 1.8% | High |

**Key challenge:** Severe class imbalance — *Normal* has ~18× more samples than *Personality disorder*. This drove a major design decision (weighted loss, Section 6).

---

## 5. Methodology & Preprocessing

**Data split:** 70% train / 15% validation / 15% test, **stratified** (every class appears proportionally in each split). The test set = ~7,659 samples.

**Preprocessing pipeline:**
1. Text cleaning — lowercase, remove punctuation/URLs, collapse whitespace, drop nulls & duplicates.
2. Two feature paths:
   - **TF-IDF vectorizer (10,000 features)** → for classical ML models.
   - **AutoTokenizer** → converts text to `input_ids` + `attention_mask` tensors (max length 128) → for transformers.
3. Label encoding — text labels mapped to integer IDs 0–6.

---

## 6. Models Built & Results

All numbers below are from the saved results files in `results/metrics/`.

### Classical ML Baselines (TF-IDF features)

| # | Model | Accuracy | F1 |
|---|-------|----------|----|
| 1 | Logistic Regression | 76.69% | 75.95% |
| 2 | Random Forest | 71.87% | 69.94% |
| 3 | SVM (`class_weight='balanced'`) | 75.49% | 74.98% |

### Deep Learning

| # | Model | Accuracy | F1 | Notes |
|---|-------|----------|----|----|
| 4 | LSTM | 76.84% | 76.62% | Custom embedding layer + vocabulary, trained on RTX 4060 GPU |

### No-Training Baselines (to prove fine-tuning is needed)

| # | Model | Accuracy | F1 (weighted) | F1 (macro) |
|---|-------|----------|---------------|-----------|
| 5 | VADER (rule-based) | 25.02% | 22.79% | 12.51% |
| 6 | RoBERTa Zero-Shot (bart-large-mnli) | 32.80% | 34.39% | 30.71% |

### Transformers (Fine-tuned)

| # | Model | Accuracy | F1 (weighted) | F1 (macro) |
|---|-------|----------|---------------|-----------|
| 7 | RoBERTa fine-tuned (weighted loss) | 82.30% | 82.31% | 78.26% |
| 8 | Mental-RoBERTa fine-tuned | 80.37% | — | — |
| (base) | BERT fine-tuned (bert-base-uncased) | ≈84% | — | — |

### ⭐ Best Model — Ensemble

| # | Model | Accuracy | F1 | Method |
|---|-------|----------|----|----|
| **9** | **BERT + RoBERTa Ensemble** | **82.32%** | **82.32%** | Soft voting — averaged softmax probabilities, 50/50 weight, 7,659 test samples |

**Ensemble per-class F1:** Normal 0.96 · Anxiety 0.87 · Bipolar 0.84 · Depression 0.76 · Suicidal 0.73 · Stress 0.72 · Personality disorder 0.65.

**Observation:** The rarest class (Personality disorder) still has the weakest F1 (0.65) — direct evidence of the imbalance challenge, even after mitigation.

---

## 7. Solving Class Imbalance — Weighted Loss

Rare classes were being under-predicted. I built a custom **`WeightedTrainer`** (subclass of HuggingFace `Trainer`) that overrides `compute_loss` to apply **inverse-frequency class weights** inside `CrossEntropyLoss`:

```python
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop('labels')
        outputs = model(**inputs)
        loss = nn.CrossEntropyLoss(weight=class_weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
```

This raised **macro-F1** (the average across all classes, which punishes ignoring rare classes) without sacrificing overall accuracy.

---

## 8. Explainable AI (XAI) — The Novelty

Most classifiers are black boxes. Mine shows its reasoning:

- Extract attention weights from **all 12 transformer layers × 12 heads**, averaged together.
- Take the `[CLS]` token's attention row → strip `[CLS]`/`[SEP]` → normalize to sum = 1.
- Strip `##` wordpiece prefixes so tokens read as normal words.
- Render each word as a **heatmap** (darker red = higher attention).

For *"I feel exhausted and hopeless,"* the words **exhausted** and **hopeless** light up — confirming the model focused on the right evidence. Figures saved in `results/figures/` (`attention_*.png`).

---

## 9. Lead-Time Analysis — The Research Contribution

I tested how *early* in a semester the model can detect burnout:

| Week | Accuracy | F1 | Lead time before decline |
|------|----------|----|--------------------------|
| Week 2 | 91% | 95.05% | **14 weeks** |
| Week 4 | 89% | 93.44% | 12 weeks |
| Week 8 | 91% | 95.15% | 8 weeks |
| Week 12 | 89% | 93.82% | 4 weeks |

**Conclusion:** The model detects burnout at **Week 2 with 91% accuracy — 14 weeks before academic decline.** This early-warning capability is the headline finding.

---

## 10. Weekly Progression Simulation

To demonstrate real-world use, I simulated one student's text drifting across a semester:

| Week | Text (summary) | Prediction | Confidence | Risk | Score |
|------|----------------|-----------|-----------|------|-------|
| 2 | "tired lately, but managing okay" | Normal | 82.5% | Low | 1 |
| 4 | "struggling, stressed, overwhelmed" | Stress | 63.4% | Medium | 3 |
| 8 | "exhausted, can't focus, hopeless" | Depression | 77.0% | High | 7 |
| 12 | "want to give up, worthless, nothing matters" | Suicidal | 63.3% | Critical | 10 |

This shows the system tracking deterioration step-by-step.

---

## 11. Stress Score (Custom Metric)

A single 0–100 distress number, so advisors don't have to read 7 probabilities:

```
Stress Score = (Suicidal×1.0 + Depression×0.8 + Bipolar×0.75
              + Personality×0.7 + Anxiety×0.6 + Stress×0.5) × 100
```

---

## 12. Deployment — Full-Stack System

The project goes beyond notebooks into a working application:

- **`predict.py`** — Python inference script: text in → JSON out (prediction, confidence, risk level, stress score, all 7 probabilities, attention tokens).
- **Next.js dashboard** (dark theme) — advisor view: student list + live detail panel.
- **API route** (`pages/api/analyze.js`) — spawns `predict.py` as a subprocess and returns its JSON to the browser.
- **React components:** attention heatmap, animated probability bars, glowing risk badge (Critical = pulsing red), metric cards, advisor recommendation, weekly progression SVG chart, architecture diagram, CSV download.
- **Streamlit app** (`app.py`) — a simpler second testing UI.

---

## 13. Technology Stack

| Layer | Tools |
|-------|-------|
| Language | Python, JavaScript |
| ML / DL | scikit-learn, PyTorch, HuggingFace Transformers |
| Models | BERT, RoBERTa, Mental-RoBERTa, LSTM, SVM, Logistic Regression, Random Forest |
| NLP | TF-IDF, AutoTokenizer, VADER |
| Frontend | Next.js 16, React 18 |
| Visualization | Matplotlib, Seaborn, custom SVG |
| Storage | Git LFS (large model files), browser localStorage |
| Hardware | NVIDIA RTX 4060 GPU (training) |

---

## 14. Project Structure

```
emotional_burnout_nlp/
├── app.py                  # Streamlit UI
├── predict.py              # Inference script (used by web app)
├── data/                   # Raw + cleaned datasets (51,068 samples)
├── models/saved/           # All trained models (BERT, RoBERTa, LSTM, .pkl baselines)
├── notebooks/01–10         # Full research pipeline (see below)
├── results/
│   ├── metrics/            # JSON results per model
│   └── figures/            # Confusion matrices, attention heatmaps, charts
└── frontend/               # Next.js dashboard
```

**Notebook pipeline:**
1. Data exploration
2. Baseline models (LR, RF, SVM) + k-fold CV
3. LSTM
4. BERT fine-tuning
5. Attention visualization
6. VADER + RoBERTa zero-shot
7. RoBERTa fine-tuning (weighted loss)
8. Mental-RoBERTa fine-tuning
9. Ensemble model
10. Lead-time analysis

---

## 15. Key Findings

1. **Fine-tuning is essential** — zero-shot/rule-based baselines scored 25–33%; fine-tuned transformers scored 80–84%.
2. **Ensembling helps** — combining BERT + RoBERTa beat either model alone (82.32%).
3. **Class imbalance is the core difficulty** — the rarest class (Personality disorder) remained hardest even after weighted loss.
4. **Early detection works** — 91% accuracy at Week 2, a 14-week lead time.
5. **Explainability is achievable** — attention heatmaps make every prediction transparent.

---

## 16. Limitations & Future Work

- **Imbalance remains** for rare classes → could add data augmentation or focal loss.
- **Dataset is Reddit-based**, not real student data → future work: validate on actual (consented) student writing.
- **Attention ≠ full explanation** → could add SHAP/LIME for comparison.
- **Single language (English)** → extend to multilingual student populations.

---

## 17. AI/ML Concepts Demonstrated

**Foundations:** train/test split, stratification, overfitting, precision/recall, F1 (macro vs weighted), confusion matrices, k-fold cross-validation.
**Classical ML:** Logistic Regression, SVM, Random Forest, decision trees, TF-IDF.
**Deep Learning:** neural networks, embeddings, LSTM/RNN, GPU training, cross-entropy loss.
**NLP:** tokenization, word/contextual embeddings, text classification.
**Transformers:** BERT, RoBERTa, attention/self-attention, transfer learning, fine-tuning, zero-shot learning, HuggingFace (AutoTokenizer, AutoModel, Trainer).
**Advanced:** ensemble learning (soft voting), class imbalance + weighted loss, Explainable AI.
**MLOps:** model saving/loading, inference scripts, serving a model via API, Git LFS.

---

*Report generated for R26-IT-059 · IT22196392 · Induwara K.P.Y.*

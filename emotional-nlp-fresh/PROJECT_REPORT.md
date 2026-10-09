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

**Update — weekly dataset v2 (current deployed model).** The component was later retrained on a weekly version of the dataset (13,171 simulated students × weeks 2, 4, 8 and 12), split by student so that no student appears in both training and test data. On unseen students the deployed ensemble reaches **85.25% accuracy and 0.843 macro-F1** (see *Weekly Dataset v2 — Retrain*). Its known weaknesses, most importantly a bias towards labelling short texts as *Normal*, are measured in *Limitations & Future Work*.

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

## 15. Weekly Dataset v2 — Retrain (current deployed model)

**Dataset.** `Combined_Data_all_weeks_v2.csv` arranges the same corpus as 13,171 students, each with one text at weeks 2, 4, 8 and 12 (52,681 texts). After cleaning, 52,582 texts remain. The 99 removed rows are 17 Excel error values (`#NAME?`), 11 texts with no readable words, and 71 identical texts that carried conflicting labels.

**Preprocessing** (`preprocess_weekly.py`):
- Light cleaning that keeps case and punctuation, which transformers use. It repairs broken characters, HTML codes, links and markdown.
- Fixed label IDs, the same as the deployed models.
- A **70/15/15 split by student**, stratified by label. Students who share an identical text are kept in the same split, and a check confirms zero student or text overlap between splits.
- **Maximum length 256 tokens** with head+tail truncation (first 128 and last 126 tokens). With 128 tokens only 66–67% of texts fit; with 256, 85–86% fit.
- Square-root class weights in the loss to handle class imbalance.

**Results** on the test split (7,887 texts from 1,974 students not seen in training; 95% confidence intervals from a student-level bootstrap):

| Model | Accuracy | Macro-F1 (95% CI) | Weighted F1 |
|-------|----------|-------------------|-------------|
| Logistic Regression (TF-IDF word + character n-grams) | 79.85% | 0.774 (0.759–0.786) | 0.798 |
| BERT fine-tuned | 84.16% | 0.830 (0.818–0.840) | 0.843 |
| RoBERTa fine-tuned | 85.15% | 0.840 (0.827–0.852) | 0.852 |
| **BERT + RoBERTa ensemble (0.45 / 0.55)** | **85.25%** | **0.843 (0.831–0.854)** | **0.853** |

- **Per-class F1 (ensemble):** Normal 0.96 · Anxiety 0.89 · Bipolar 0.88 · Personality disorder 0.81 · Depression 0.80 · Stress 0.79 · Suicidal 0.77.
- **Main error:** Depression and Suicidal are confused with each other. 21% of Depression texts are predicted Suicidal, and 16% of Suicidal texts are predicted Depression.
- **Early-warning view:** 98.6% of non-Normal texts are flagged as at risk (precision 98.1%).
- **Per week:** macro-F1 is 0.83–0.86 at every week.
- **Significance** (paired student-level bootstrap): RoBERTa beats Logistic Regression by +0.066 macro-F1 (p < 0.001) and beats BERT by +0.010 (p = 0.02). The ensemble's +0.003 over RoBERTa alone is **not** significant (p = 0.11).

**Deployment and integration.** `serve_model.py` (port 5004) serves this ensemble with the same cleaning and truncation used in training. `export_weekly_stress_scores.py` writes a weekly 0–4 emotional stress score for every student for the Meta-FNN layer.

**Scripts:** `preprocess_weekly.py`, `train_baseline_weekly.py`, `train_transformer_weekly.py`, `ensemble_weekly.py`, `export_weekly_stress_scores.py`. Results are saved in `results/metrics/weekly_v2_*.json`.

---

## 16. Key Findings

1. **Fine-tuning is essential** — zero-shot/rule-based baselines scored 25–33%; fine-tuned transformers scored 80–85%.
2. **Ensembling helps only slightly** — in the first version the ensemble beat either model alone (82.32%). In v2 its gain over RoBERTa alone is not statistically significant.
3. **Class imbalance is a core difficulty** — rare classes stayed hardest after weighted loss. In v2 the largest remaining error is Depression vs Suicidal.
4. **Early detection (simulated)** — 91% accuracy at Week 2 in the lead-time study. The week labels are simulated, so see Limitations before reading this as real early detection.
5. **Explainability is achievable** — attention heatmaps make every prediction transparent.

---

## 17. Limitations & Future Work

**1. Short texts are biased towards *Normal* (most important).**
- In the training data, 82% of *Normal* texts have 20 words or fewer, compared with 8% of texts in the other six classes. The model therefore partly learned that a short text is a *Normal* text.
- On the v2 test set, non-Normal texts of 10 words or fewer are predicted *Normal* 13.4% of the time, compared with 0.5% for texts longer than 50 words.
- On a check with 12 short student-style sentences (for example *"I cry every night before going to class"* and *"I want to drop out, I can't take it"*), the deployed model predicted *Normal* for 10.
- **Consequence:** short student messages are under-flagged, and the 85% test accuracy overstates performance on them, because the test set contains few short distressed texts.
- *Future work:* add labelled short student-style texts, or train with length-balanced sampling, and evaluate on a separate short-text test set.

**2. The text is not student language.** The corpus is Reddit and Twitter posts, so the reported scores measure performance on that kind of text. *Future work:* validate on real, consented student writing.

**3. The weekly structure is simulated.**
- Texts were assigned to students and weeks without a real timeline; for example, 1,036 students jump directly between *Suicidal* and *Normal* from one week to the next.
- The per-week results therefore show that the model is equally accurate at every week. They do not show that it detects burnout earlier in the semester.
- For the same reason, the lead-time study (Section 9) and the weekly progression (Section 10) are demonstrations, not evidence of real early detection.

**4. Meta-layer scores are partly in-sample.** In the exported stress-score file, the 70% of students in the NLP training split are scored on texts the model was trained on: accuracy 93%, against 85% for test-split students. Only test-split students (`nlp_split = test`) are out-of-sample.

**5. The NLP students are not the academic students.** In the Meta-FNN layer, emotional scores are joined to academic students by ID number only; they are different simulated people.
- The new file gives 2,542 of the 2,543 academic students a real NLP score, up from 500.
- That score's correlation with the dropout label is about 0 (−0.004). Before, it was 0.46, but only because the 2,043 unmatched students received a score generated from their academic risk.
- After the swap the Meta-FNN changed little: test F1 0.681 → 0.696, AUC 0.855 → 0.856.
- The NLP contribution to the meta prediction can only be measured properly on students who have both text and academic records.

**6. The first version and v2 are not directly comparable.** Both use the same corpus, and the first version split by row rather than by student. The v2 numbers are the reliable ones.

**7. Attention is not a full explanation.** *Future work:* compare with SHAP or LIME.

**8. English only.** *Future work:* extend to multilingual student populations.

---

## 18. AI/ML Concepts Demonstrated

**Foundations:** train/test split, stratification, overfitting, precision/recall, F1 (macro vs weighted), confusion matrices, k-fold cross-validation.
**Classical ML:** Logistic Regression, SVM, Random Forest, decision trees, TF-IDF.
**Deep Learning:** neural networks, embeddings, LSTM/RNN, GPU training, cross-entropy loss.
**NLP:** tokenization, word/contextual embeddings, text classification.
**Transformers:** BERT, RoBERTa, attention/self-attention, transfer learning, fine-tuning, zero-shot learning, HuggingFace (AutoTokenizer, AutoModel, Trainer).
**Advanced:** ensemble learning (soft voting), class imbalance + weighted loss, Explainable AI.
**MLOps:** model saving/loading, inference scripts, serving a model via API, Git LFS.

---

*Report generated for R26-IT-059 · IT22196392 · Induwara K.P.Y.*

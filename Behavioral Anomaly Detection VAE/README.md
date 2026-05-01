# Behavioral Anomaly Detection — VAE on LMS Data

Project: R26-IT-059 | Student: IT22215710 Karunarathne D C
Supervisor: Prof. Anuradha Karunasena | Co-supervisor: Mrs. Malithi Nawarathne

## What This Component Does

This is the Behavioral Pattern Anomaly Detection component of the
Explainable Multi-Modal AI for Academic Burnout Prediction system.

It trains an unsupervised Variational Autoencoder (VAE) on OULAD 
LMS interaction data to detect behavioral anomalies in student 
engagement patterns — without requiring any burnout labels during 
training. The system identifies students whose LMS behavior deviates 
significantly from normal engagement patterns, generating weekly 
anomaly scores that feed into the group meta-integration layer.

## Key Results

| Metric | Value |
|--------|-------|
| ROC-AUC (combined) | 0.5840 |
| Random baseline | 0.5000 |
| Improvement over random | +16.8% |
| VAE training loss | 0.1365 |
| Students analyzed | 26,050 |
| LMS behavioral features | 10 |
| Weeks tracked | 17 |
| Anomaly records generated | 250,483 |
| Normal students trained on | 21,118 |

## SHAP Feature Importance

| Rank | Feature | SHAP Value | Meaning |
|------|---------|------------|---------|
| 1 | weekly_total_clicks | 0.0182 | Total LMS interactions per week |
| 2 | content_diversity | 0.0168 | Number of different activity types accessed |
| 3 | night_activity | 0.0150 | Late night LMS usage patterns |
| 4 | days_since_login | 0.0125 | Days since last platform access |
| 5 | active_days | 0.0100 | Number of active days per week |
| 6 | forum_posts | 0.0083 | Forum engagement per week |
| 7 | forum_replies | 0.0072 | Forum reply activity per week |
| 8 | quiz_attempts | 0.0047 | Quiz attempt frequency |
| 9 | resource_views | 0.0045 | Resource access frequency |
| 10 | inactive_weeks_count | 0.0000 | Cumulative inactive weeks |

## Temporal Lag Analysis

| Lag (Weeks Ahead) | ROC-AUC |
|-------------------|---------|
| 0 weeks | 0.5865 |
| 1 week | 0.5738 |
| 2 weeks | 0.5704 |
| 3 weeks | 0.5655 |
| 4 weeks | 0.5589 |
| 5 weeks | 0.5549 |
| 6 weeks | 0.5607 |
| 7 weeks | 0.5584 |
| 8 weeks | 0.5584 |

All lag windows remain above random baseline (0.5000).
System maintains meaningful predictive signal up to 8 weeks ahead.

## Folder Structure

Behavioral Anomaly Detection VAE/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    ← OULAD data exploration
│   ├── 02_feature_engineering.ipynb ← 10 LMS feature extraction
│   ├── 03_vae_model.ipynb           ← VAE training + anomaly scores
│   └── 04_evaluation.ipynb          ← SHAP + temporal lag + results
│
├── src/                             ← production code (coming soon)
│
├── models/
│   └── saved/
│       ├── vae_behavioral.pt        ← trained VAE weights
│       └── scaler.pkl               ← fitted StandardScaler
│
├── results/
│   ├── figures/
│   │   ├── withdrawal_timing.png
│   │   ├── feature_distributions.png
│   │   ├── vae_training_loss.png
│   │   ├── roc_curve_combined.png
│   │   ├── temporal_lag_analysis.png
│   │   └── shap_feature_importance.png
│   └── metrics/
│       └── anomaly_scores.csv       ← give this to IT22253194
│
└── data/
    └── raw/                         ← OULAD CSV files (not in git)

## VAE Architecture

Input: (batch, 10_features)
         ↓
Encoder
  Linear(10 → 32) + ReLU
  Linear(32 → 16) + ReLU
         ↓
Latent Space (dim=6)
  fc_mu     → mean vector
  fc_logvar → variance vector
  z = mu + sigma * epsilon  ← reparameterization trick
         ↓
Decoder
  Linear(6 → 16)  + ReLU
  Linear(16 → 32) + ReLU
  Linear(32 → 10) ← reconstruction
         ↓
Output: reconstructed behavioral feature vector

Total parameters: 2,070
Loss = Reconstruction Loss + 0.1 * KL Divergence

## 10 Behavioral Features (F1-F10)

| Feature | Description | Burnout Signal |
|---------|-------------|----------------|
| F1: weekly_total_clicks | Total VLE clicks per week | Reduced engagement |
| F2: active_days | Days with LMS activity per week | Irregular access |
| F3: content_diversity | Unique activity types accessed | Narrowing focus |
| F4: forum_posts | Forum post count per week | Social withdrawal |
| F5: forum_replies | Forum reply count per week | Reduced interaction |
| F6: quiz_attempts | Quiz attempt frequency | Academic disengagement |
| F7: resource_views | Resource access per week | Content avoidance |
| F8: night_activity | Late night LMS usage | Sleep disruption proxy |
| F9: days_since_login | Days since last platform access | Inactivity signal |
| F10: inactive_weeks_count | Cumulative inactive weeks | Sustained withdrawal |

## How Anomaly Detection Works

1. VAE trained ONLY on normal (not at-risk) students baseline
2. VAE learns what normal LMS engagement looks like
3. When monitoring student behavior is passed through VAE:
   - Normal behavior → VAE reconstructs well → LOW anomaly score
   - Abnormal behavior → VAE reconstructs poorly → HIGH anomaly score
4. Combined score = VAE reconstruction error + cohort deviation
5. High anomaly flag set at 75th percentile threshold

## Dataset

| Property | Value |
|----------|-------|
| Name | OULAD (Open University Learning Analytics) |
| Source | Open University, United Kingdom |
| License | CC-BY 4.0 (free for academic research) |
| Students | 32,593 |
| Courses | 22 modules |
| VLE interactions | 10,655,280 |
| Date range | Day -25 to Day 269 |
| At-risk rate | 31.2% (Withdrawn) |

Download: https://analyse.kmi.open.ac.uk/open_dataset

## Output Files for IT22253194

After running notebooks, give this file to IT22253194:
results/metrics/anomaly_scores.csv

Columns IT22253194 uses:
  student_id            ← join key across components
  behavioral_risk_score ← main FNN input (0-1 scale)
  high_anomaly_flag     ← binary flag (1=high risk)
  actual_label          ← ground truth for evaluation

## Research Novelty

This is the first unsupervised VAE-based behavioral anomaly 
detection system applied to LMS interaction logs for academic 
burnout prediction.

No existing paper combines:
  ✅ Unsupervised VAE (zero labels during training)
  ✅ LMS behavioral logs (Moodle/OULAD data)
  ✅ 10 engagement behavioral features
  ✅ Burnout/dropout detection target
  ✅ SHAP-based behavioral explainability
  ✅ Temporal lag analysis across 8 weeks

Research gap confirmed by literature review of 13 papers
across IEEE, ACM, Nature and MDPI journals.

## Setup

conda create -n vae_burnout python=3.10
conda activate vae_burnout
pip install torch pandas numpy scikit-learn shap matplotlib seaborn jupyter ipykernel

## How to Run

Step 1 — Download OULAD dataset
Place CSV files in data/raw/

Step 2 — Run notebooks in order
  01_data_exploration.ipynb
  02_feature_engineering.ipynb
  03_vae_model.ipynb
  04_evaluation.ipynb

Training takes approximately 90 minutes for 23,526
per-student models OR 2 minutes for shared VAE approach.

## Key Configuration

BASELINE_WEEKS  = 3      ← first 3 weeks = personal normal
MONITORING_WEEKS = 4-16  ← monitoring period
LATENT_DIM      = 6      ← VAE latent space dimensions
EPOCHS          = 200    ← training epochs
LR              = 0.001  ← learning rate
BATCH_SIZE      = 256    ← training batch size
BETA            = 0.1    ← KL divergence weight
THRESHOLD       = 0.75   ← high anomaly percentile

R26-IT-059 | SLIIT | 2025-2026

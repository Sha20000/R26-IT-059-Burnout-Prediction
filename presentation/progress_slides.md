# Progress Presentation Slides (Draft)

## Slide 1 — Title + Sub-Problem
**Title:** Explainable Multi-Modal AI for Early Academic Burnout and Dropout Prediction

**Sub-Problem (one line):**
Detect at-risk students early and explain why, by fusing academic trends with behavior/engagement signals into a single actionable risk score.

**Why it matters:**
- Early warning enables timely academic intervention and reduces dropout costs.
- Explanations build trust for advisors and administrators.


## Slide 2 — System Overview (End-to-End)
**Pipeline:**
- Academic trend model (GRU multi-horizon) -> academic risk signals
- Behavioral anomaly model (VAE) -> behavioral risk signals
- Engagement/other signals (OULAD activity)
- Meta-Integration FNN -> final burnout risk + alert level
- Dashboard + API -> decision support for staff

**Key outcome:**
- Single, explainable risk score per student for early intervention.


## Slide 3 — Model + Architecture
**Meta-Integration FNN (current component):**
- Input features (10): academic_risk, week4_risk, week8_risk, week12_risk, week17_risk, behavior_risk_mean, behavior_risk_max, compliance_mean, anomaly_mean, high_anomaly_weeks
- Architecture: 10 -> 32 -> 16 -> 8 -> 1 (ReLU + Dropout 0.3), sigmoid output
- Output: final_burnout_risk (0–1) + alert (HIGH/MEDIUM/LOW)

**Explainability:**
- Meta-level: shows contribution by component signals
- Model-level: academic GRU uses attention + feature attribution


## Slide 4 — Results + Evidence
**Meta-Integration performance (current run):**
- F1: 0.6914
- AUC-ROC: 0.8522
- Accuracy: 0.7827
- Precision: 0.6242, Recall: 0.7750

**Academic GRU evidence (to insert from academic model outputs):**
- AUC-ROC (Week 17)
- Baseline comparisons (RF, LR, XGB, GRU, BiLSTM)
- Lead-time curve (accuracy vs weeks)

**Screenshot placeholders:**
- Accuracy vs lead time curve (Unified GRU)
- GRU vs baselines comparison chart
- XAI feature importance plot


## Slide 5 — Dashboard + User Requirements
**User requirements addressed:**
- Single risk score with alert tiers (HIGH/MEDIUM/LOW)
- Transparent explanation and top contributing signals
- Filterable list of students and export support
- API endpoints for integration with existing systems

**Dashboard demo highlights:**
- Risk distribution overview
- Per-student record details
- Explanation summary and signal breakdown

**Screenshot placeholder:**
- Dashboard overview and per-student detail view


## Slide 6 — Dataset Used
**Primary dataset:**
- OULAD (Open University Learning Analytics Dataset)

**Data sources used in this component:**
- Academic prediction CSV (GRU outputs)
- Behavioral prediction CSV (VAE outputs)
- Engagement signals (OULAD activity logs)

**Note:**
- If behavioral IDs do not align, simulated signals were used for PP1; real merge is planned after ID alignment.


## Slide 7 — Final Training Config (Meta FNN)
- Seed: 42
- Epochs: 300 (early stopping)
- Batch size: 64
- LR: 0.001
- Hidden size: 32
- Dropout: 0.3
- Threshold: 0.50


## Slide 8 — Design Excellence / Contribution
- Multi-horizon GRU captures early and late risk patterns
- Unified meta-integration reduces model silos into one actionable risk
- Explainability at both model-level (attention + gradients) and system-level (component contributions)
- Operational dashboard + API make the research deployable


## Slide 9 — Commercialization / Sustainability
**Ideal buyers / stakeholders:**
- University leadership (VC, Deans, Registrar)
- Retention/Quality assurance units
- Learning analytics / IT offices
- Ministries or national education boards (for scaled adoption)

**Value / ROI for institutions:**
- Reduced dropout and repeat-course costs
- Improved retention and graduation KPIs
- Early identification reduces remediation cost per student

**Student benefits:**
- Earlier academic support and counseling
- Personalized interventions reduce burnout risk


## Slide 10 — Addressing Panel Questions (Q&A Ready)
**What is Meta XAI?**
- Explainability across multiple models and the fusion layer. It clarifies not just individual model logic, but which model signals contributed to the final risk.

**Legal/data bottlenecks:**
- Use consent-based, privacy-preserving pipelines
- Data sharing agreements (MoU, DSA), anonymization, and governance
- Option to deploy on-premise to avoid data transfer

**Why implement meta-AI vs a single model?**
- Single models miss critical signals; fusion improves robustness and reduces false positives.
- Meta layer provides actionable confidence and explanation for staff decisions.

**Financial benefits:**
- Lower dropout = higher tuition continuity
- Reduced support cost through targeted interventions
- Stronger institutional reputation and ranking metrics


## Slide 11 — User Feedback on Prototype Demo
**Observed feedback themes (update with actual quotes if available):**
- Dashboard is easy to interpret for advisors
- Risk categories are clear and actionable
- Explainability improves trust and adoption

**Next iteration plan:**
- Integrate real behavioral IDs and live data feeds
- Add intervention tracking and outcome feedback


## Slide 12 — Appendices (Optional)
**Insert actual figures here when ready:**
- Accuracy vs lead time curve
- GRU baseline comparison plot
- XAI feature attribution plot
- Dashboard screenshots

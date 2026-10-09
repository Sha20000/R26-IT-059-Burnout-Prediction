# CONTENT PACK — Draft Final Report (Individual Component)
## IT22196392 · Induwara K.P.Y. · R26-IT-059

---

## PART A — FORMATTING INSTRUCTIONS (not part of the report text)

Create a SLIIT-style **Draft Final Report** PDF from the content in Part B. Follow these rules:

- **Page:** A4, a thin black border around every page, and a page number at the bottom right in bold.
- **Font:** Times New Roman, 12 pt body, 1.5 line spacing, justified text.
- **Headings:** bold. Chapter headings are 14 pt ("1 INTRODUCTION"); sub-headings are 12 pt bold ("1.1 Background Literature"). Keep the numbering exactly as given.
- **Start each of these on a new page:** title page, second title page, Declaration, Abstract, Acknowledgement, Table of Contents, lists, and each chapter.
- **Captions:** figure captions go *below* the figure; table captions go *below* the table. Both are italic, centred and dark blue, for example *Figure 3 – Emotional NLP component architecture* and *Table 6 – Test case 01*.
- **Figures:** where the text says **[FIGURE n: …]**, insert the image file named there, or leave a clearly marked box if the image is not supplied.
- **Test cases:** use the two-part layout:
  1. A small header box with *Test case ID*, *Test title*, *Test priority*, *Module name*.
  2. Below it, a box with *Description* and *Pre-conditions*, then a table with columns **Test ID | Test Steps | Expected Output | Actual Output | Result (Pass/Fail)**.
- **References:** IEEE numeric style. Citation numbers in the text, such as [11], must match the reference list.
- **Table of Contents, List of Figures and List of Tables:** build them from the headings and captions, with page numbers.
- **Placeholders:** text in **[square brackets with CAPITALS]** is for the student to fill in. Keep it visible and highlighted so it is not missed.

---

## PART B — REPORT CONTENT

---

### PAGE 1 — TITLE PAGE

**EMOTIONAL BURNOUT DETECTION FROM STUDENT-GENERATED TEXT USING TRANSFORMER-BASED NATURAL LANGUAGE PROCESSING WITH EXPLAINABLE AI**

Project ID: R26-IT-059

Draft Final Report

Induwara K.P.Y. — IT22196392

BSc (Hons) Degree in Information Technology specialization in **[SPECIALIZATION — CONFIRM]**

Department of Information Technology

Sri Lanka Institute of Information Technology

Sri Lanka

**[MONTH YEAR]**

---

### PAGE 2 — SECOND TITLE PAGE

**Draft Final Report**

Project ID: R26-IT-059

Main project: Explainable Multi-Modal AI System for Early Academic Burnout and Dropout Prediction

Component: Emotional Burnout Detection from Student-Generated Text

Supervisor: Prof. Anuradha Karunasena

Co-Supervisor: Ms. Malithi Nawarathne **[CONFIRM TITLE: Ms./Mrs.]**

BSc (Hons) Degree in Information Technology specialization in **[SPECIALIZATION — CONFIRM]**

Department of Information Technology

Sri Lanka Institute of Information Technology

Sri Lanka

**[MONTH YEAR]**

---

### PAGE 3 — DECLARATION

**DECLARATION**

The following declaration should be made by the candidate following the signature and the date. A candidate, after a discussion with the supervisor/s, can request an embargo for a particular dissertation for a given work for a given time or indefinitely. Such an embargo may override the statement made in the dissertation itself.

"I declare that this is my own work and this dissertation does not incorporate without acknowledgement any material previously submitted for a Degree or Diploma in any other University or institute of higher learning and to the best of my knowledge and belief it does not contain any material previously published or written by another person except where the acknowledgement is made in the text.

Also, I hereby grant to Sri Lanka Institute of Information Technology, the non-exclusive right to reproduce and distribute my dissertation, in whole or in part in print, electronic or other medium. I retain the right to use this content in whole or part in future works (such as articles or books)."

| Name | Student ID | Signature |
|---|---|---|
| Induwara K.P.Y. | IT22196392 | **[SIGNATURE]** |

The supervisor/s should certify the dissertation with the following declaration. The above candidate has carried out research for the bachelor's degree Dissertation under my supervision.

Signature of the supervisor: **[SIGNATURE]**  Date: **[DATE]**

---

### PAGE 4–5 — ABSTRACT

**ABSTRACT**

Emotional burnout is a growing concern among university students, and it is closely linked to disengagement, poor academic performance and dropout. Students often express their emotional state in their own words long before the effects appear in grades or attendance, yet universities rarely analyse this text in a systematic way. Manual review of student writing is slow, inconsistent and impossible at scale, and most automated approaches either classify only a single emotion or act as black boxes that give advisors no reason to trust their output.

This research presents an explainable Natural Language Processing (NLP) component that detects emotional burnout signals in student-generated text and classifies each text into seven mental-health categories: Normal, Stress, Anxiety, Depression, Bipolar, Personality disorder and Suicidal. Each prediction is mapped to a four-level risk tier (Low, Medium, High, Critical), a 0–100 distress score and an advisor recommendation. The component is part of a multi-modal early-warning system, in which its weekly emotional stress score is combined with academic (GRU) and behavioural (VAE) signals by a Meta-FNN integration layer.

The study uses a public mental-health text corpus organised as a weekly semester cohort of 13,171 students observed at weeks 2, 4, 8 and 12 (52,582 texts after cleaning). The data was split 70/15/15 at the student level, so that no student appears in more than one split. Classical baselines (TF-IDF with Logistic Regression, Linear SVM, Naive Bayes and Random Forest) were compared with fine-tuned BERT and RoBERTa transformers trained with class-weighted loss and a 256-token head-and-tail input window. A weighted soft-voting ensemble of BERT and RoBERTa achieved **85.25% accuracy and a macro-F1 of 0.843** on unseen students, compared with 0.774 for the strongest classical baseline. It detected **98.6% of at-risk texts** and performed consistently across all four semester checkpoints (macro-F1 0.83–0.86). Attention-based explanations highlight the words that drive each prediction.

The trained ensemble is deployed as a REST API that analyses a text in under 0.3 seconds on a standard CPU. It is integrated into a web dashboard for weekly student monitoring and into the Meta-FNN layer of the overall system. The results show that transformer-based, explainable text analysis can provide a reliable emotional signal for early burnout and dropout prediction.

**Keywords:** Natural Language Processing, Emotional Burnout, Mental Health Text Classification, BERT, RoBERTa, Transformer Ensemble, Explainable AI, Attention Visualization, Student Well-being, Early Warning System, Multi-Modal Integration

---

### PAGE 6 — ACKNOWLEDGEMENT

**ACKNOWLEDGEMENT**

This research would not have been possible without the guidance, support and encouragement of many people. First, I express my sincere gratitude to my supervisor, Prof. Anuradha Karunasena, and my co-supervisor, Ms. Malithi Nawarathne. Their expertise, constructive feedback and continuous encouragement shaped the direction and quality of this study.

I also thank my research group members, Mahavitha S.M., Karunarathne D.C. and **[NAME OF IT22253194]**, for their collaboration in building the integrated burnout and dropout prediction system. Their academic and behavioural components and the integration layer gave this work its wider purpose.

I am grateful to the Sri Lanka Institute of Information Technology for the facilities and academic environment that supported this research. I also thank the open research community whose publicly available datasets, pre-trained language models and software libraries made this work possible. Finally, I thank my family and friends for their patience and support throughout the project.

---

### PAGE 7 — TABLE OF CONTENTS (generate with page numbers)

- DECLARATION
- ABSTRACT
- ACKNOWLEDGEMENT
- LIST OF FIGURES
- LIST OF TABLES
- LIST OF APPENDICES
- LIST OF ABBREVIATIONS
- 1 INTRODUCTION
  - 1.1 Background Literature
  - 1.2 Research Gap
  - 1.3 Research Problem
  - 1.4 Research Objectives
    - 1.4.1 Main Objective
    - 1.4.2 Specific Objectives
- 2 METHODOLOGY
  - 2.1 System Architecture
    - 2.1.1 Overall System Architecture Diagram
    - 2.1.2 Software Solution
    - 2.1.3 Requirements Gathering
    - 2.1.4 Commercialization Aspects of the Product
    - 2.1.5 Testing and Implementation
- 3 RESULTS & DISCUSSION
  - 3.1 Results
  - 3.2 Research Findings
  - 3.3 Discussion
  - 3.4 Summary of Each Student's Contribution
- 4 CONCLUSION
- 5 REFERENCES
- 6 APPENDICES

---

### PAGE 8 — LISTS

**LIST OF FIGURES**

- Figure 1 – Overall architecture of the multi-modal burnout and dropout prediction system
- Figure 2 – Emotional NLP component architecture
- Figure 3 – The Software Development Life Cycle (Agile)
- Figure 4 – Label distribution by semester week and by data split
- Figure 5 – Token length distribution by class (training split)
- Figure 6 – Classical baseline comparison on the test set
- Figure 7 – Comparison of all final models on the test set and by semester week
- Figure 8 – Confusion matrix of the BERT + RoBERTa ensemble (test set)
- Figure 9 – Emotional Analysis dashboard: attention-based explanation (Quick Analysis)
- Figure 10 – Emotional Analysis dashboard: student list and weekly progression
- Figure 11 – Meta-FNN Risk view showing the integrated emotional stress score

**LIST OF TABLES**

- Table 1 – Class distribution of the dataset after cleaning
- Table 2 – Student-level data split
- Table 3 – Records removed during cleaning
- Table 4 – Transformer training configuration
- Table 5 – Risk level, distress weight and emotional stress score per class
- Table 6 – Test case 01
- Table 7 – Test case 02
- Table 8 – Test case 03
- Table 9 – Test case 04
- Table 10 – Test case 05
- Table 11 – Test case 06
- Table 12 – Test case 07
- Table 13 – Test case 08
- Table 14 – Test case 09
- Table 15 – Test case 10
- Table 16 – Test case 11
- Table 17 – Test case 12
- Table 18 – Initial model exploration results
- Table 19 – Classical baseline results (student-level test set)
- Table 20 – Final model results (student-level test set)
- Table 21 – Per-class results of the ensemble
- Table 22 – Macro-F1 by semester week
- Table 23 – Statistical comparison of models (paired student-level bootstrap)
- Table 24 – Emotional stress trend of test-split students across the semester
- Table 25 – Meta-FNN integration results

**LIST OF APPENDICES**

- Appendix A: Plagiarism report
- Appendix B: Sample API request and response
- Appendix C: Training environment and hyperparameters
- Appendix D: System screenshots

**LIST OF ABBREVIATIONS**

| Abbreviation | Description |
|---|---|
| AI | Artificial Intelligence |
| API | Application Programming Interface |
| AUC | Area Under the ROC Curve |
| BERT | Bidirectional Encoder Representations from Transformers |
| CI | Confidence Interval |
| CLS | Classification token of a transformer input |
| CPU / GPU | Central / Graphics Processing Unit |
| F1 | Harmonic mean of precision and recall |
| FNN | Feed-forward Neural Network |
| GRU | Gated Recurrent Unit |
| LMS | Learning Management System |
| LR | Logistic Regression |
| LSTM | Long Short-Term Memory |
| ML | Machine Learning |
| NLP | Natural Language Processing |
| REST | Representational State Transfer |
| RoBERTa | Robustly Optimized BERT Pretraining Approach |
| SDLC | Software Development Life Cycle |
| SVM | Support Vector Machine |
| TF-IDF | Term Frequency – Inverse Document Frequency |
| UAT | User Acceptance Testing |
| VAE | Variational Autoencoder |
| XAI | Explainable Artificial Intelligence |

---

## 1 INTRODUCTION

University life places heavy academic, social and financial demands on students. When these demands persist without enough recovery, many students develop burnout, a state of emotional exhaustion, cynicism and reduced sense of accomplishment [1]. Student burnout is associated with lower engagement, declining academic performance and a higher risk of dropping out of a degree programme [2]. Because these outcomes develop gradually across a semester, the value of an early-warning system lies in recognising risk early enough for advisors and counsellors to act.

Most institutional monitoring relies on academic and behavioural records such as grades, assignment submissions, attendance and learning-management-system activity. These signals are useful, but they mostly reflect problems after they have affected performance. By contrast, the language students use when they write reflections, forum posts, messages or check-in responses often carries emotional signals in real time: exhaustion, hopelessness, worry, pressure or withdrawal. Earlier research has shown that language in social media posts can reveal depression and other mental-health conditions [3], [4], which suggests that student-generated text is a valuable and largely unused source of early-warning information.

Analysing such text manually is not practical. It is time-consuming, depends on the reader's judgement, and cannot scale to thousands of students every week. Natural Language Processing (NLP) offers a way to automate this analysis. Transformer-based language models such as BERT [11] and RoBERTa [12], built on the self-attention mechanism [10], have set new standards in text classification and can be fine-tuned on relatively small labelled datasets. However, in a sensitive domain such as student mental health, accuracy alone is not enough: advisors need to understand *why* a text was flagged before they can trust and act on the result. This makes explainability a core requirement.

This research develops the emotional NLP component of the project "Explainable Multi-Modal AI System for Early Academic Burnout and Dropout Prediction" (R26-IT-059). The component:

- reads student-written text and classifies it into seven mental-health categories;
- maps each prediction to a risk level, a distress score and an advisor recommendation;
- explains each prediction by highlighting the words the model attended to;
- tracks students week by week across a semester.

Its weekly emotional stress score is passed to a Meta-FNN integration layer, where it is combined with an academic risk model (GRU) and a behavioural anomaly model (VAE). Together they produce a single burnout and dropout risk for each student.

---

### 1.1 Background Literature

**Burnout and student well-being.** Burnout was first described in occupational settings as a combination of exhaustion, cynicism and inefficacy [1]. Later research showed that university students experience the same syndrome in relation to their studies, and that it is linked to lower engagement and academic performance [2]. Because burnout develops over weeks and months, repeated measurement across a semester is more informative than a single assessment.

**Detecting mental-health signals from text.** A growing body of work uses language to detect mental-health conditions:
- De Choudhury et al. showed that the language and activity of social media users can predict the onset of depression [3].
- Coppersmith et al. quantified signals of several mental-health conditions in Twitter data [4].
- Shared datasets have supported this research, including Dreaddit for stress detection in Reddit posts [5] and the eRisk collection for early depression detection [6].
- A recent survey summarises the wide range of mental-health conditions, datasets and methods studied in social media text [7].

These studies confirm that written language carries measurable signals of stress, anxiety, depression and suicidal ideation.

**From lexicons to deep learning.** Early approaches relied on sentiment lexicons and rule-based tools such as VADER [8], which score text by matching words against a curated dictionary. These tools are fast and transparent but cannot capture context, negation or the varied ways people describe their feelings. Classical machine learning models trained on TF-IDF features improved accuracy by learning from labelled data. Recurrent neural networks such as LSTM [9] then modelled word order and longer context.

**Transformers and transfer learning.** The transformer architecture [10] replaced recurrence with self-attention, allowing each word to be interpreted in the context of every other word.
- BERT [11] pre-trains a bidirectional transformer on large text collections and can then be fine-tuned for classification.
- RoBERTa [12] improves BERT's pre-training procedure, with larger data, longer training and dynamic masking, and generally performs better on downstream tasks.
- Domain-adapted models such as MentalBERT [13] show that transformers are well suited to mental-health text.
- Practical guidance on fine-tuning shows that, for long documents, keeping both the beginning and the end of the text (head-and-tail truncation) is an effective strategy [14].
- Zero-shot classification with natural-language-inference models [15], [16] and emotion-pretrained models [17] offer alternatives when labelled data is limited.

**Ensembles.** Combining several models often produces more accurate and more stable predictions than any single model, because their errors are partly independent [18]. Soft voting, which averages the predicted class probabilities, is a simple and effective ensemble method for neural classifiers.

**Explainability.** Explainable AI aims to make model decisions understandable to people. For transformers, attention weights provide a direct view of which input words the model focused on. Their exact interpretation is debated [19], [20], but they remain a practical, low-cost way to show users which words drove a prediction. Model-agnostic methods such as LIME [21] and SHAP [22] provide complementary explanations.

---

### 1.2 Research Gap

Although text-based mental-health detection is an active research area, several gaps remain when it is applied to student burnout and early dropout prediction.

1. **Single-point analysis instead of weekly monitoring.** Most studies classify individual posts at one point in time [3]–[7]. Few approaches follow the same students across a semester and report how their emotional state and model performance change from week to week, which is what an early-warning system needs.

2. **Text analysed in isolation.** Emotional text analysis is usually studied separately from academic and behavioural data. Existing early-warning systems in education mostly use grades, attendance and learning-platform activity. There is a gap in systems that feed a text-based emotional signal into a multi-modal model alongside academic and behavioural predictions.

3. **Limited explainability for end users.** Many high-accuracy text classifiers return only a label and a probability. Advisors and counsellors cannot see which words caused a student to be flagged, which limits trust and makes it hard to act on the prediction.

4. **Coarse output categories.** Many studies detect a single condition, such as depression or stress, or a binary at-risk label. Fine-grained classification across several mental-health categories, combined with an actionable risk tier, is less common.

5. **Evaluation that does not reflect deployment.** Random row-level splits can place texts from the same person in both training and test data, which can overstate performance. Evaluation that holds out entire students, and reports confidence intervals, gives a more realistic picture of how a model will perform on new students.

6. **Few deployable systems.** Many models remain in research notebooks. There are few real-time services that an advisor can actually use through a dashboard, and fewer that are integrated with other components of a larger system.

The key research gap is the absence of an explainable, fine-grained, week-by-week emotional text analysis component that is evaluated at the student level, deployed in real time, and integrated with academic and behavioural signals for early burnout and dropout prediction.

---

### 1.3 Research Problem

Universities need to identify students at risk of burnout and dropout early in the semester, while support can still make a difference. Academic and behavioural records reveal problems mainly after performance has already declined. Students' own words often show emotional strain much earlier, but this text is not analysed systematically. Manual reading is slow, subjective and impossible to scale to thousands of students every week.

Automated text classification can address the scale problem, but several difficulties must be solved together:

- Student emotional states are diverse and overlap. Stress, anxiety, depression and suicidal ideation share vocabulary, and the categories are highly imbalanced: severe but rare categories have far fewer examples than common ones.
- Texts vary from a few words to several hundred, so the model must handle both short messages and long posts without losing key information.
- In a sensitive domain, a prediction must come with an explanation that a non-technical advisor can understand.
- The output must be useful to the wider system, which means a weekly, numeric emotional signal that the integration layer can combine with academic and behavioural risk.

The research problem can therefore be stated as follows:

> *How can student-generated text be analysed automatically, accurately and explainably, week by week across a semester, to detect emotional burnout signals across multiple mental-health categories, and how can this emotional signal be delivered in real time to advisors and integrated with academic and behavioural predictions for early burnout and dropout risk detection?*

---

### 1.4 Research Objectives

#### 1.4.1 Main Objective

To design, develop and evaluate an explainable transformer-based NLP component that detects emotional burnout signals in student-generated text. The component classifies each text into seven mental-health categories, maps it to an actionable risk level and distress score, explains each prediction through attention-based word highlighting, and tracks students week by week. Its weekly emotional stress score is delivered in real time to an advisor dashboard and to the Meta-FNN integration layer of the multi-modal burnout and dropout prediction system.

#### 1.4.2 Specific Objectives

- **Prepare a reliable, student-level dataset.** Clean and normalise a large weekly cohort of mental-health texts, map labels to numeric classes, and split the data at the student level (70/15/15) so that evaluation reflects performance on new students.

- **Establish baseline performance.** Train and tune classical machine learning models (TF-IDF with Logistic Regression, Linear SVM, Naive Bayes and Random Forest) to provide a reference point for the transformer models.

- **Fine-tune transformer models for seven-class classification.** Fine-tune BERT and RoBERTa to classify text as Normal, Stress, Anxiety, Depression, Bipolar, Personality disorder or Suicidal, using a 256-token head-and-tail input window to retain key information in long texts.

- **Handle class imbalance.** Apply class-weighted loss and use macro-F1 as the main selection metric, so that rare but important categories are learned well.

- **Build a weighted ensemble.** Combine BERT and RoBERTa through weighted soft voting, with the weights chosen on validation data.

- **Provide explainable predictions.** Extract attention weights to highlight the words that drove each prediction, and present them as a colour-coded heatmap that advisors can understand.

- **Translate predictions into actionable outputs.** Map each prediction to a risk level (Low, Medium, High, Critical), a 0–100 distress score, a 0–4 emotional stress score and an advisor recommendation.

- **Support week-by-week student monitoring.** Track each student's emotional state at weeks 2, 4, 8 and 12, and evaluate model performance at each checkpoint.

- **Deploy the model as a real-time service.** Serve the ensemble through a REST API that responds fast enough for interactive use on standard hardware.

- **Integrate with the multi-modal system.** Deliver the weekly emotional stress score to the Meta-FNN integration layer and the shared dashboard, alongside the academic (GRU) and behavioural (VAE) components.

- **Evaluate performance rigorously.** Report accuracy, macro-F1, per-class results, confusion matrices, per-week results, confidence intervals and statistical comparisons between models.

---

## 2 METHODOLOGY

### 2.1 System Architecture

#### 2.1.1 Overall System Architecture Diagram

**[FIGURE 1: Overall architecture of the multi-modal burnout and dropout prediction system — draw this diagram]**

*Diagram description (for drawing):*

- **Left: three input sources.**
  1. Academic records (grades, submissions, login activity) feed the **GRU Academic Risk** model (IT22916426, port 5001).
  2. Weekly behavioural data feeds the **VAE Behavioural Anomaly** model (IT22215710, port 5003).
  3. Student-written text (weekly check-ins, reflections, messages) feeds the **Emotional NLP** model (IT22196392, port 5004).
- **Centre:** arrows from all three into the **Meta-FNN Integration Layer** (IT22253194, port 5002), which outputs the final burnout and dropout risk and alert level (High, Medium, Low).
- **Right:** the **Integrated Web Dashboard** used by academic advisors. It has tabs for Test Students, Model Metrics, Meta-FNN Risk, Emotional Analysis, Academic Risk, Behavioural Anomaly and others.

The overall system follows a multi-modal design: each component produces a risk signal from a different kind of evidence, and the Meta-FNN integration layer combines them into a single prediction for each student.

**[FIGURE 2: Emotional NLP component architecture — draw this diagram]**

*Diagram description (left to right):*

1. **Input:** a student text (single text or weekly text per student).
2. **Text cleaning and normalisation:** repair of broken characters, removal of links, user mentions and markup, and restoration of contractions.
3. **Tokenisation:** BERT and RoBERTa tokenisers, 256 tokens, head-and-tail truncation.
4. **Two fine-tuned models in parallel:** BERT (bert-base-uncased) and RoBERTa (roberta-base).
5. **Weighted soft voting:** RoBERTa 0.55 + BERT 0.45.
6. **Outputs:** predicted class, confidence and all 7 class probabilities.
7. **Post-processing branches:**
   - (a) **Risk mapping**, giving risk level, distress score (0–100) and advisor recommendation;
   - (b) **XAI**, giving the attention heatmap of the words that drove the prediction;
   - (c) **Weekly scoring**, giving the emotional stress score (0–4) per student per week.
8. **Delivery:** a REST API (Flask, port 5004) serves the dashboard (Emotional Analysis tab), and the weekly stress-score export feeds the Meta-FNN integration layer.

The component takes text as input and passes it through a cleaning and normalisation stage, then through two tokenisers, one for each transformer. The fine-tuned BERT and RoBERTa models each produce a probability distribution over the seven classes. These are combined by weighted soft voting into the final prediction.

The prediction is then turned into three outputs:
1. A risk level, distress score and advisor recommendation for the advisor.
2. An attention-based explanation showing which words influenced the decision.
3. A numeric weekly emotional stress score that is passed to the Meta-FNN integration layer.

All outputs are served through a REST API, which the dashboard calls in real time.

#### 2.1.2 Software Solution

The component was developed following the Software Development Life Cycle (SDLC), a structured process for designing, building and testing high-quality software. Planning ahead reduces risk and helps the result meet user expectations. In a traditional waterfall approach, each phase must be completed before the next begins, which makes it difficult to respond to new findings. Machine learning projects are experimental by nature: model results often lead to changes in data preparation or model design. The **Agile methodology** was therefore adopted. Work was organised in short iterations covering requirements, design, implementation, testing and review, so that improvements could be added continuously.

**[FIGURE 3: The Software Development Life Cycle — circular diagram with: 1 Identify and gather requirements, 2 Plan, 3 Design, 4 Implement, 5 Test, 6 Deploy, 7 Maintain]**

The main iterations were:

1. **Initial exploration:** data exploration, classical baselines, LSTM, rule-based and zero-shot methods, and first transformer models.
2. **Model improvement:** class-imbalance handling, RoBERTa fine-tuning and ensemble design.
3. **Explainability and deployment:** attention visualisation, REST API and web dashboard.
4. **Weekly cohort and integration:** student-level weekly dataset, retraining and evaluation of all models, and integration with the Meta-FNN layer and shared dashboard.

**Technology stack**

| Layer | Tools and technologies |
|---|---|
| Programming languages | Python, JavaScript |
| Machine learning and deep learning | PyTorch, Hugging Face Transformers, scikit-learn |
| Models | BERT (bert-base-uncased), RoBERTa (roberta-base), Logistic Regression, Linear SVM, Complement Naive Bayes, Random Forest, LSTM |
| Text processing | Hugging Face tokenisers, TF-IDF, ftfy (text repair), regular expressions |
| Data handling | pandas, NumPy, Apache Parquet |
| Visualisation | Matplotlib, HTML/CSS/JavaScript dashboard components |
| Backend | Flask REST API with CORS (port 5004) |
| Frontend | Integrated web dashboard (HTML, CSS, JavaScript) and a standalone Next.js / React interface |
| Hardware | NVIDIA RTX 4060 Laptop GPU (8 GB) for training; standard CPU for serving |
| Version control | Git and GitHub, with Git LFS for large model files |

#### 2.1.3 Requirements Gathering

Requirements for the component were identified through:
- a review of published research on mental-health text classification, transformer models and explainable AI;
- an analysis of how early-warning systems in education are used by academic advisors and counsellors;
- discussions with the project supervisors;
- integration meetings with the other members of the research group.

**[IF YOU CONDUCTED A SURVEY OR INTERVIEWS WITH STUDENTS, ADVISORS OR COUNSELLORS, ADD ONE OR TWO SENTENCES HERE DESCRIBING WHO, HOW MANY AND WHAT YOU LEARNED. IF NOT, KEEP THE PARAGRAPH AS IT IS.]**

**User requirements.** The primary users are academic advisors and student counsellors. They need to know which students are struggling emotionally, how serious the situation is, and why the system reached its conclusion. They also need to follow a student's emotional state over the semester rather than see a single snapshot, and to receive clear guidance on the next step, such as routine monitoring, a check-in or immediate intervention.

**Functional requirements**

- Accept free-text input from a student, either a single text or one text per semester week.
- Validate the input and reject empty or too-short text, with a clear message.
- Clean and normalise text from real sources, which may contain broken characters, links, user mentions, emojis and formatting symbols.
- Classify each text into one of seven categories: Normal, Stress, Anxiety, Depression, Bipolar, Personality disorder or Suicidal.
- Return the confidence of the prediction and the probability of every class.
- Map each prediction to a risk level (Low, Medium, High or Critical), a 0–100 distress score and an advisor recommendation.
- Highlight the words that most influenced each prediction (attention heatmap).
- Track students across weeks 2, 4, 8 and 12, showing the latest and the most severe result.
- Produce a 0–4 weekly emotional stress score for every student, for use by the Meta-FNN integration layer.
- Expose all functions through a REST API with health and model-information endpoints.

**Non-functional requirements**

- **Accuracy:** high overall accuracy and strong performance on rare but serious categories, measured by macro-F1.
- **Explainability:** every prediction must be accompanied by a human-readable explanation.
- **Performance:** a response time suitable for interactive use (under one second per text) on standard hardware without a GPU.
- **Reliability:** consistent results between offline evaluation and the live service.
- **Usability:** outputs understandable by non-technical advisors.
- **Interoperability:** standard JSON and CSV outputs that the other components can consume.
- **Privacy and ethics:** students identified only by pseudonymous IDs; the system supports, and does not replace, professional judgement.
- **Maintainability:** modular, documented scripts for preprocessing, training, evaluation, serving and export, so that the pipeline can be re-run end to end.

**Feasibility study**

- **Technical feasibility.** The component uses proven, open-source technologies: pre-trained transformer models, PyTorch, Hugging Face Transformers and Flask. Fine-tuning a base-size transformer on about 37,000 texts takes about 18–20 minutes on a laptop GPU, and inference runs on a standard CPU in under 0.3 seconds per text, so the solution is technically feasible with modest hardware.

- **Operational feasibility.** The component is delivered as a web dashboard and a REST API, which fit into existing university workflows. Advisors can analyse a text in one step, review a student's weekly progression, and receive a clear recommendation without any technical knowledge. The API design allows the component to work alongside the other modules of the system.

- **Economic feasibility.** All software used is open source, and the models can be served on ordinary computers without specialised hardware. The solution can reduce the time staff spend reviewing student text and help direct limited counselling resources to the students who need them most.

- **Schedule feasibility.** The Agile, iterative approach allowed the component to be delivered in working increments (baseline, transformer models, ensemble, explainability, deployment and integration) within the project timeline.

- **Data feasibility.** Large, labelled mental-health text datasets are publicly available for research use. The weekly cohort used in this study provides more than 52,000 labelled texts across seven categories, which is enough to fine-tune and evaluate transformer models reliably.

- **Ethical and legal feasibility.** Mental-health information is sensitive. The system uses pseudonymous student IDs, is intended for use by trained staff, and presents its output as decision support rather than diagnosis. A real deployment would require informed consent and compliance with data-protection law, such as Sri Lanka's Personal Data Protection Act No. 9 of 2022.

**Implementation (Development)**

The component was implemented as a set of modular Python scripts plus a web front end. The main stages are described below.

**1. Dataset**

The study uses a public mental-health text corpus [24] containing posts and statements labelled with seven mental-health categories. The corpus is organised as a weekly semester cohort: each text is linked to a pseudonymous student ID and one of four semester checkpoints (weeks 2, 4, 8 and 12). This gives 13,171 students and 52,681 texts, and allows the component to be developed and evaluated in a week-by-week monitoring setting. After cleaning, 52,582 texts remained (Table 1).

| Class | Texts | Share | Risk level |
|---|---|---|---|
| Normal | 16,329 | 31.1% | Low |
| Depression | 15,362 | 29.2% | High |
| Suicidal | 10,625 | 20.2% | Critical |
| Anxiety | 3,829 | 7.3% | Medium |
| Bipolar | 2,777 | 5.3% | High |
| Stress | 2,583 | 4.9% | Medium |
| Personality disorder | 1,077 | 2.0% | High |
| **Total** | **52,582** | **100%** | |

*Table 1 – Class distribution of the dataset after cleaning*

The dataset is strongly imbalanced: *Normal* has about 15 times more texts than *Personality disorder*. The share of each category also changes across the semester:
- *Normal* texts fall from 37.0% at week 2 to 28.1% at week 12.
- *Stress* rises from 2.9% to a peak of 6.9% at week 8.
- *Anxiety* rises from 4.8% to a peak of 9.8% at week 8.

This pattern matches the mid-semester assessment period.

**[FIGURE 4: results/figures/weekly_label_distribution.png — "Label distribution by semester week and by data split"]**

**2. Data cleaning and preprocessing**

Real-world text contains noise that does not carry emotional meaning. A transformer-friendly cleaning pipeline was developed. Unlike traditional cleaning, it keeps case, punctuation, apostrophes, digits and emojis, because transformer models use them to understand meaning; for example, "can't" differs from "cant", and "!!!" or 😭 carry emotion. The pipeline:

1. Repairs broken character encodings (for example "donâ€™t" becomes "don't", and "ðŸ˜‚" becomes 😂) using the ftfy library, and removes fragments that cannot be repaired.
2. Decodes HTML codes (for example "&amp;" becomes "&") and removes invisible zero-width characters.
3. Removes web links and user mentions (@username), and keeps hashtag words without the "#" symbol.
4. Removes markdown formatting symbols (such as \*\*, \_\_ and "---" lines).
5. Shortens exaggerated repeated characters ("wtffffff" becomes "wtfff"; "!!!!!!" becomes "!!!") without changing numbers.
6. Restores contractions that had lost their apostrophes ("don t" becomes "don't", "i m" becomes "i'm").
7. Normalises whitespace.

The original text was kept alongside the cleaned text for traceability. Cleaning changed 16,563 texts. Records that could not be used were removed (Table 3).

| Reason | Records removed |
|---|---|
| Spreadsheet error value instead of text (e.g. "#NAME?") | 17 |
| No readable words after cleaning | 11 |
| Identical text with conflicting labels | 71 |
| **Total** | **99** |

*Table 3 – Records removed during cleaning*

**3. Label encoding**

The seven categories were mapped to fixed numeric IDs: Anxiety 0, Normal 1, Depression 2, Suicidal 3, Stress 4, Bipolar 5, Personality disorder 6. These IDs are shared by all models, the API and the dashboard, so their outputs are always consistent.

**4. Student-level data split**

To estimate how the model will perform on *new students*, the data was split at the student level rather than the text level. All four weekly texts of a student are always in the same split. Students who share an identical text were also grouped together, so that the same text never appears in two splits.

The split used stratified group k-fold partitioning with 20 folds: 14 folds for training, 3 for validation and 3 for testing. This keeps the class balance almost identical in every split. Automated checks confirmed zero overlap of students and of texts between the training, validation and test sets.

| Split | Texts | Students | Share |
|---|---|---|---|
| Training | 36,803 | 9,220 | 70.0% |
| Validation | 7,892 | 1,977 | 15.0% |
| Test | 7,887 | 1,974 | 15.0% |

*Table 2 – Student-level data split*

**5. Tokenisation and input length**

Texts were tokenised with the original tokenisers of each model: WordPiece for BERT and byte-level BPE for RoBERTa. The token length distribution was analysed to choose the maximum input length (Figure 5):
- The median text is about 75 tokens.
- With a 128-token limit only 66–67% of texts fit completely.
- With 256 tokens, 85–86% fit.

A **256-token limit** was therefore selected. For longer texts, **head-and-tail truncation** [14] keeps the first 128 tokens and the last 126 tokens instead of simply cutting off the end. The beginning of a post usually states the topic, and the end often contains the most important sentence (for example "…I want to give up" or "please help").

**[FIGURE 5: results/figures/token_length_distribution.png — "Token length distribution by class (training split)"]**

**6. Handling class imbalance**

Class weights were computed from the training split only and applied in the cross-entropy loss. Square-root balanced weights (w_c = √(N / (K·n_c)), rescaled so that the average weight per training text equals 1) were used. They give rare classes more influence without over-correcting:

- Personality disorder 2.91
- Stress 1.87
- Bipolar 1.80
- Anxiety 1.54
- Suicidal 0.92
- Depression 0.77
- Normal 0.74

Macro-F1, which treats every class equally, was used as the main metric for selecting models.

**7. Classical baseline models**

Classical baselines were trained on TF-IDF features combining word unigrams and bigrams with character 3–5-grams (214,910 features). Stop words were kept on purpose, because words such as "not", "no" and "never" are important in mental-health text. Four models were trained with class weighting, and their main settings were tuned on the validation set:
- Logistic Regression (C = 4)
- Linear SVM (C = 0.3)
- Complement Naive Bayes (α = 0.1)
- Random Forest (300 trees)

A majority-class predictor was included as a minimum reference.

**8. Transformer fine-tuning**

BERT (bert-base-uncased) and RoBERTa (roberta-base) were fine-tuned for seven-class classification using the configuration in Table 4. To reduce training time, texts of similar length were grouped into the same batch (length-bucketed batching), which avoids unnecessary padding. After each epoch the model was evaluated on the validation set, and the checkpoint with the highest validation macro-F1 was kept. The test set was used only once, for the final evaluation.

| Setting | Value |
|---|---|
| Base models | bert-base-uncased, roberta-base |
| Maximum length | 256 tokens (head 128 + tail 126) |
| Optimiser | AdamW [28], learning rate 2 × 10⁻⁵, weight decay 0.01 |
| Learning-rate schedule | Linear warm-up (6% of steps), then linear decay |
| Batch size | 16 |
| Epochs | 3 (best epoch chosen by validation macro-F1) |
| Loss | Cross-entropy with square-root balanced class weights |
| Precision | Mixed precision (bfloat16) |
| Gradient clipping | 1.0 |
| Random seed | 42 |
| Hardware | NVIDIA RTX 4060 Laptop GPU (8 GB) |
| Training time | About 6 minutes per epoch per model |

*Table 4 – Transformer training configuration*

**9. Weighted ensemble**

The ensemble combines the class probabilities of the two models by weighted soft voting: P = w · P_RoBERTa + (1 − w) · P_BERT. The weight w was searched from 0 to 1 in steps of 0.05 on the validation set. The best value was **w = 0.55**, giving RoBERTa 0.55 and BERT 0.45, with a validation macro-F1 of 0.835.

**10. Explainability (XAI)**

For each prediction, attention weights were extracted from all 12 layers and 12 attention heads of the BERT model and averaged. The attention row of the classification ([CLS]) token was taken as the importance of each word, the special tokens were removed, and the weights were normalised to sum to one. Sub-word pieces are displayed as readable words. In the dashboard, each word is coloured on a five-level scale from low (grey) to high (red) attention, and the exact percentage appears when the user hovers over a word. For long texts, a "…" marker shows where the middle part was skipped by head-and-tail truncation.

**11. Risk mapping, distress score and advisor recommendations**

Each predicted class is mapped to an actionable output (Table 5).

| Class | Risk level | Distress weight | Emotional stress score (0–4) | Advisor recommendation (summary) |
|---|---|---|---|---|
| Normal | Low | 0 | 0 | Emotionally stable; continue routine monitoring |
| Stress | Medium | 0.50 | 1 | Stress indicators; recommend a check-in with an advisor |
| Anxiety | Medium | 0.60 | 2 | Anxiety indicators; recommend a counselling session |
| Personality disorder | High | 0.70 | 3 | Recommend professional evaluation |
| Bipolar | High | 0.75 | 3 | Recommend psychiatric evaluation |
| Depression | High | 0.80 | 3 | Immediate counselling recommended |
| Suicidal | Critical | 1.00 | 4 | Immediate intervention required |

*Table 5 – Risk level, distress weight and emotional stress score per class*

- **Distress score (0–100):** the sum of each class probability multiplied by its distress weight, times 100. This gives advisors a single number instead of seven probabilities.
- **Emotional stress score (0–4):** the score used by the Meta-FNN integration layer.

**12. Deployment as a real-time service**

The ensemble is served by a Flask REST API (port 5004) that loads both models once at start-up and keeps them in memory. The API applies exactly the same cleaning and 256-token head-and-tail encoding as training, so live predictions match the evaluated model. It provides three endpoints:
- `POST /api/analyze` returns the prediction, confidence, risk level, distress score, emotional stress score, all class probabilities, the attention explanation, and each model's individual prediction.
- `GET /api/health` returns the service status, model version and ensemble settings.
- `GET /api/model-info` returns the measured test performance, read directly from the evaluation results so the dashboard always shows real numbers.

**13. Web dashboard**

The Emotional Analysis tab of the integrated dashboard provides three views:
- **Analysis:** the full result for a selected student.
- **Weekly Progression:** the student's emotional state and risk level at each week.
- **Quick Analysis:** analyses any text instantly.

A student list on the left shows each student's latest prediction and risk colour, with a filter by risk level and a summary count of Critical, High, Medium and Low students. Advisors can add a new student by entering a text for each semester checkpoint in the dashboard (weeks 2, 4, 8 and 18). The system then records the latest result and the most severe result across the semester.

The student list also contains 40 students from the held-out test set, identified by their student IDs (for example STU01957). Their four weekly texts (weeks 2, 4, 8 and 12) were scored with the deployed ensemble. The dashboard shows the dataset label next to each weekly prediction, and a summary of how many weeks match it. Across these 160 texts the model agreed with the dataset label 85.0% of the time, consistent with the full test-set accuracy. A filter lets the advisor show all students, only the test-set students, or only the demo and newly added students. A model performance panel displays the accuracy, F1 score, dataset size and number of classes.

**[FIGURE 9: Screenshot — Emotional Analysis → Quick Analysis, showing prediction, confidence, stress score, risk level, advisor recommendation, attention heatmap and class probabilities]**
**[FIGURE 10: Screenshot — Emotional Analysis → student list and Weekly Progression]**

**14. Integration with the Meta-FNN layer**

The trained ensemble was applied to every student-week, and the results were exported as two files:
- a weekly file with one row per student per week;
- a per-student file containing the mean emotional stress score (0–4), the score for each week, the peak score, the trend from the first to the last week, the latest prediction and its confidence.

Student IDs are written in the numeric format used by the integration layer, which allows 2,542 of the 2,543 students in the integration layer to receive an emotional score from this component. The Meta-FNN layer combines this score with the GRU academic risk features and the VAE behavioural features. It also computes a transparent weighted formula: 0.4 × academic risk + 0.3 × behavioural risk + 0.3 × emotional stress (normalised to 0–1).

**[FIGURE 11: Screenshot — Meta-FNN Risk tab showing the emotional stress score column]**

**Testing**

The component was tested at several levels to confirm accuracy, reliability and usability.

- **Unit testing:** each function was tested on its own. This covered the text-cleaning function on difficult examples (broken characters, HTML codes, links, mentions, markdown, repeated characters), the label mapping, the head-and-tail encoding (checked against the official tokeniser output) and the risk and stress-score mappings.
- **Data validation testing:** automated checks confirmed that there are no missing values, that every class and every week appears in each split, and that there is zero overlap of students or texts between splits.
- **Model testing:** every model was evaluated on the held-out test students using accuracy, macro-F1, weighted F1, per-class precision, recall and F1, and confusion matrices. 95% confidence intervals were calculated with a student-level bootstrap (1,000 resamples) [23], and models were compared with a paired bootstrap test.
- **Integration testing:** the live API was tested end to end. Its predictions on 300 randomly selected test texts were compared with the offline evaluation results, and all 300 matched. The export to the Meta-FNN layer was verified by checking that the integration layer receives identical scores for all matched students.
- **System testing:** the four APIs of the full system (ports 5001–5004) and the dashboard were run together, and all endpoints responded correctly.
- **User acceptance testing:** the dashboard was demonstrated **[TO THE SUPERVISORS / TO PEERS — CONFIRM]**, who confirmed that the predictions, risk levels, explanations and recommendations were clear and understandable.

#### 2.1.4 Commercialization Aspects of the Product

The emotional burnout detection component, as part of the integrated early-warning system, has strong commercial potential in the higher-education sector. Universities and colleges in Sri Lanka and the wider region are paying increasing attention to student well-being and retention, but most lack tools that turn students' own words into timely, explainable risk information. Dropout is costly for students, families and institutions, so an early-warning tool that helps staff intervene earlier offers clear value.

**Target customers**
- State and private universities
- Higher-education institutes and colleges
- Student counselling and well-being units
- Academic advising offices
- Online learning providers

**Product model.** The system will be offered as a software-as-a-service (SaaS) platform that can connect to existing learning management systems (such as Moodle) and student portals. A tiered subscription model will encourage adoption:
- **Basic tier:** single-text emotional analysis with risk levels and advisor recommendations, suitable for small counselling units and pilot programmes.
- **Professional tier:** weekly student monitoring, attention-based explanations, progression charts, cohort summaries and downloadable reports.
- **Enterprise tier:** full multi-modal integration with academic and behavioural signals, the Meta-FNN risk score, API access for integration with institutional systems, and on-premise deployment for institutions with strict data-privacy requirements.

**Revenue model**
- The main revenue will come from annual institutional licences priced by the number of enrolled students.
- Additional revenue can come from implementation and integration services, staff training workshops, and customised reporting.

**Partnerships and marketing**
- Pilot projects with university faculties will be used to demonstrate value and refine the product.
- Partnerships will be sought with student counselling services, mental-health organisations and education technology providers.
- Marketing activities will include demonstrations at academic conferences, workshops for student-support staff, and case studies from pilot institutions.

**Ethics and trust.** Because the product handles sensitive information, privacy and responsible use are central to its commercial value:
- Student data will be pseudonymised.
- Access will be restricted to authorised staff.
- The system will comply with data-protection law, such as Sri Lanka's Personal Data Protection Act No. 9 of 2022.
- Predictions will always be presented as decision support for trained professionals.

In summary, commercialization of the component as part of the integrated system can provide financial sustainability while helping institutions support students earlier and more effectively.

#### 2.1.5 Testing and Implementation

**Test Cases**

---
**Test case ID:** T01 · **Test title:** Student Text Input Validation · **Test priority:** High · **Module name:** Input Validation (API)

**Description:** Verify that the system rejects empty or too-short text and accepts valid student text for analysis.
**Pre-conditions:** Emotional NLP API (port 5004) is running.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T01 | 1. Submit the text "hi there" to the analysis endpoint. 2. Submit a full sentence. 3. Check the responses. | Short text is rejected with a clear message; a valid sentence is accepted and analysed. | "hi there" was rejected (HTTP 400, "Please enter at least a few words"); the full sentence was analysed successfully. | Pass |

*Table 6 – Test case 01*

---
**Test case ID:** T02 · **Test title:** Text Cleaning and Normalisation · **Test priority:** High · **Module name:** Preprocessing

**Description:** Verify that broken characters, HTML codes, links, mentions and formatting symbols are cleaned while meaningful content is kept.
**Pre-conditions:** Preprocessing module available.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T02 | 1. Input "I donâ€™t know ðŸ˜‚ &amp; stuff". 2. Input text with a link and "@username". 3. Run the cleaning function. | Text is repaired and normalised; links and mentions are removed; words, punctuation and emojis are kept. | Output "I don't know 😂 & stuff"; the link and mention were removed and the remaining text was unchanged. | Pass |

*Table 7 – Test case 02*

---
**Test case ID:** T03 · **Test title:** Emotional State Classification · **Test priority:** High · **Module name:** Ensemble Classifier

**Description:** Verify that the ensemble classifies student text into the correct mental-health category.
**Pre-conditions:** Both models loaded; API online.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T03 | 1. Submit "I am so stressed about my exams, I have three deadlines this week and I cannot sleep at night". 2. Submit "The semester started well. I enjoyed the lectures today and had lunch with my friends." 3. Check predictions. | First text classified as Stress; second as Normal. | First text: Stress (92.7% confidence). Second text: Normal (99.9% confidence). | Pass |

*Table 8 – Test case 03*

---
**Test case ID:** T04 · **Test title:** Class Probabilities and Model Agreement · **Test priority:** Medium · **Module name:** Ensemble Classifier

**Description:** Verify that the system returns all seven class probabilities and each model's individual prediction, and that the ensemble resolves disagreement between the models.
**Pre-conditions:** API online.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T04 | 1. Submit "I keep worrying about my presentation next week. My heart races and my hands shake whenever I think about it." 2. Check the probabilities and model breakdown. | Seven probabilities summing to 1; individual BERT and RoBERTa predictions shown; final ensemble prediction returned. | All seven probabilities returned. BERT predicted Anxiety and RoBERTa predicted Stress; the ensemble returned Anxiety (57.5%). | Pass |

*Table 9 – Test case 04*

---
**Test case ID:** T05 · **Test title:** Risk Level and Advisor Recommendation · **Test priority:** High · **Module name:** Risk Mapping

**Description:** Verify that each prediction is mapped to the correct risk level and advisor recommendation.
**Pre-conditions:** Classification completed.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T05 | 1. Submit a text expressing hopelessness and thoughts of ending one's life. 2. Check the risk level and recommendation. | Suicidal prediction mapped to Critical risk with an immediate-intervention recommendation. | Predicted Suicidal (69.5%), risk level Critical, recommendation "Immediate intervention required". | Pass |

*Table 10 – Test case 05*

---
**Test case ID:** T06 · **Test title:** Distress Score and Emotional Stress Score · **Test priority:** High · **Module name:** Scoring

**Description:** Verify that the 0–100 distress score and the 0–4 emotional stress score are computed correctly.
**Pre-conditions:** Classification completed.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T06 | 1. Analyse a Normal, a Stress and a Suicidal text. 2. Compare the scores with the defined mappings. | Scores increase with severity; emotional stress score matches the class mapping (Normal 0, Stress 1, Suicidal 4). | Normal: 0.1/100 and 0. Stress: 49.6/100 and 1. Suicidal: 93.8/100 and 4. | Pass |

*Table 11 – Test case 06*

---
**Test case ID:** T07 · **Test title:** Attention-Based Explanation (XAI) · **Test priority:** High · **Module name:** Explainability

**Description:** Verify that the system highlights the words that most influenced the prediction.
**Pre-conditions:** Classification completed.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T07 | 1. Analyse the exam-stress text. 2. Inspect the attention weights and heatmap. | Emotionally relevant words receive the highest attention and are highlighted in the heatmap. | Highest-attention words included "stressed", "cannot" and "exams"; the heatmap displayed correctly in the dashboard. | Pass |

*Table 12 – Test case 07*

---
**Test case ID:** T08 · **Test title:** Long Text Handling · **Test priority:** Medium · **Module name:** Tokenisation

**Description:** Verify that texts longer than the 256-token limit are processed without error, keeping the beginning and end of the text.
**Pre-conditions:** API online.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T08 | 1. Submit a text of more than 500 words. 2. Check the prediction and the attention output. | Text is truncated to 256 tokens using head-and-tail truncation; the skipped middle part is marked; a prediction is returned. | Text was truncated (head + tail) and a gap marker "…" appeared in the explanation; prediction returned (Depression). | Pass |

*Table 13 – Test case 08*

---
**Test case ID:** T09 · **Test title:** Service Health and Model Information · **Test priority:** Medium · **Module name:** REST API

**Description:** Verify that the API reports its status, configuration and measured performance.
**Pre-conditions:** API started.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T09 | 1. Call /api/health. 2. Call /api/model-info. 3. Check the values. | Status "online" with the ensemble settings; model information shows the measured test results. | Status online, weights BERT 0.45 / RoBERTa 0.55, maximum length 256; model information shows accuracy 85.25%, macro-F1 84.29%, 7,887 test samples. | Pass |

*Table 14 – Test case 09*

---
**Test case ID:** T10 · **Test title:** Weekly Student Tracking · **Test priority:** High · **Module name:** Dashboard – Weekly Progression

**Description:** Verify that an advisor can add a student with weekly texts and view the weekly progression, the latest result and the most severe result.
**Pre-conditions:** Dashboard open; API online.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T10 | 1. Click "Add New Student". 2. Enter texts for weeks 2, 4, 8 and 18. 3. Open the student's Weekly Progression view. | Each week is analysed; progression chart shows risk per week; student list shows the latest prediction and risk colour. | **[RUN THIS IN THE DASHBOARD AND WRITE WHAT YOU SEE — e.g., "All four weeks analysed; progression and latest/worst results displayed correctly"]** | **[Pass/Fail]** |

*Table 15 – Test case 10*

---
**Test case ID:** T11 · **Test title:** Consistency Between Live Service and Evaluation · **Test priority:** High · **Module name:** Deployment

**Description:** Verify that the deployed API produces the same predictions as the evaluated model.
**Pre-conditions:** Evaluation results and live API available.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T11 | 1. Select 300 random test texts. 2. Analyse them through the live service code. 3. Compare with the offline evaluation results. | Predictions match the evaluated model. | 300 of 300 predictions matched; average processing time 0.25 seconds per text on CPU. | Pass |

*Table 16 – Test case 11*

---
**Test case ID:** T12 · **Test title:** Integration with Meta-FNN Layer · **Test priority:** High · **Module name:** System Integration

**Description:** Verify that weekly emotional stress scores are exported, matched to the students in the integration layer, and served by the Meta-FNN API.
**Pre-conditions:** Weekly scores exported; all four APIs running.

| Test ID | Test Steps | Expected Output | Actual Output | Result |
|---|---|---|---|---|
| T12 | 1. Export weekly emotional stress scores. 2. Run the Meta-FNN integration. 3. Request the student list from the Meta-FNN API and compare the emotional scores with the exported file. | Emotional scores reach the integration layer unchanged; the dashboard shows integrated risk. | 2,542 of 2,543 students matched; scores identical to the exported file; Meta-FNN results displayed in the dashboard. | Pass |

*Table 17 – Test case 12*

---

## 3 RESULTS & DISCUSSION

### 3.1 Results

This section presents the results of the emotional NLP component. It first summarises the initial model exploration, then the final evaluation of all models on the student-level weekly dataset, followed by the weekly monitoring, explainability, deployment and integration results.

**Initial model exploration**

In the first iteration, a wide range of approaches was compared on the base corpus (51,068 cleaned texts, stratified 70/15/15 split) to select the most promising model family (Table 18).

| Model | Type | Accuracy | F1 (weighted) |
|---|---|---|---|
| VADER | Rule-based lexicon, no training | 25.02% | 22.79% |
| Zero-shot NLI (BART-large-MNLI) | Pre-trained, no fine-tuning | 32.80% | 34.39% |
| Random Forest (TF-IDF) | Classical ML | 71.87% | 69.94% |
| Linear SVM (TF-IDF) | Classical ML | 75.49% | 74.98% |
| Logistic Regression (TF-IDF) | Classical ML | 76.69% | 75.95% |
| LSTM | Deep learning | 76.84% | 76.62% |
| Emotion-DistilRoBERTa (fine-tuned) | Transformer | 80.37% | — |
| RoBERTa (fine-tuned, weighted loss) | Transformer | 82.30% | 82.31% |
| BERT + RoBERTa ensemble (equal weights) | Transformer ensemble | 82.32% | 82.32% |

*Table 18 – Initial model exploration results*

Methods without task-specific training (VADER and zero-shot) performed poorly, which shows that fine-tuning on labelled data is essential. Fine-tuned transformers clearly outperformed classical and LSTM models. Based on these results, BERT and RoBERTa were selected for the final weekly evaluation.

**Final evaluation — classical baselines**

All final results below are measured on the held-out test split: 7,887 texts from 1,974 students not seen during training.

| Model | Validation macro-F1 | Test accuracy | Test macro-F1 | Test weighted F1 |
|---|---|---|---|---|
| Majority class (reference) | 0.068 | 31.08% | 0.068 | 0.147 |
| Random Forest | 0.602 | 72.75% | 0.626 | 0.714 |
| Complement Naive Bayes | 0.648 | 72.42% | 0.673 | 0.721 |
| Linear SVM | 0.752 | 79.65% | 0.769 | 0.795 |
| **Logistic Regression** | **0.754** | **79.85%** | **0.774** | **0.798** |

*Table 19 – Classical baseline results (student-level test set)*

Logistic Regression was the strongest classical model. Combining word and character n-grams and keeping stop words raised its accuracy from 76.69% in the initial exploration to 79.85%.

**[FIGURE 6: results/figures/weekly_v2_baseline_comparison.png — "Classical baseline comparison on the test set"]**

**Final evaluation — transformer models and ensemble**

Training converged smoothly for both transformers:
- **BERT:** validation macro-F1 0.776, 0.804 and 0.820 after epochs 1, 2 and 3.
- **RoBERTa:** validation macro-F1 0.801, 0.823 and 0.832.

The final test results are shown in Table 20 (95% confidence intervals from a student-level bootstrap).

| Model | Accuracy | Macro-F1 (95% CI) | Weighted F1 | Risk-level accuracy | At-risk recall |
|---|---|---|---|---|---|
| Logistic Regression (TF-IDF) | 79.85% | 0.774 (0.759–0.786) | 0.798 | 81.65% | 96.19% |
| BERT (fine-tuned) | 84.16% | 0.830 (0.818–0.840) | 0.843 | 85.48% | 98.23% |
| RoBERTa (fine-tuned) | 85.15% | 0.840 (0.827–0.852) | 0.852 | 86.32% | 98.69% |
| **BERT + RoBERTa ensemble** | **85.25%** | **0.843 (0.831–0.854)** | **0.853** | **86.38%** | **98.55%** |

*Table 20 – Final model results (student-level test set)*

- *Risk-level accuracy* measures how often the predicted risk tier (Low, Medium, High or Critical) is correct.
- *At-risk recall* measures the share of non-Normal texts that were flagged as at risk.

The ensemble achieved the highest accuracy, macro-F1, weighted F1 and risk-level accuracy. At-risk precision was 98.06%, so almost all flagged texts were genuinely at risk.

**[FIGURE 7: results/figures/weekly_v2_all_models_comparison.png — "Comparison of all final models on the test set and by semester week"]**

| Class | Precision | Recall | F1 | Test support |
|---|---|---|---|---|
| Normal | 0.967 | 0.957 | 0.962 | 2,451 |
| Anxiety | 0.880 | 0.908 | 0.894 | 573 |
| Bipolar | 0.887 | 0.866 | 0.876 | 417 |
| Personality disorder | 0.819 | 0.809 | 0.814 | 162 |
| Depression | 0.849 | 0.751 | 0.797 | 2,304 |
| Stress | 0.733 | 0.863 | 0.792 | 387 |
| Suicidal | 0.719 | 0.817 | 0.765 | 1,593 |
| **Macro average** | — | — | **0.843** | 7,887 |

*Table 21 – Per-class results of the ensemble*

All seven classes reached an F1 score of at least 0.76, including the rare classes. Compared with the Logistic Regression baseline:
- Personality disorder improved from 0.710 to 0.814.
- Stress improved from 0.690 to 0.792.

The main remaining confusion is between Depression and Suicidal: 21% of Depression texts were predicted as Suicidal, and 16% of Suicidal texts as Depression. These two categories share much of their vocabulary.

**[FIGURE 8: results/figures/weekly_v2_ensemble_confusion.png — "Confusion matrix of the BERT + RoBERTa ensemble (test set)"]**

**Weekly results**

| Model | Week 2 | Week 4 | Week 8 | Week 12 |
|---|---|---|---|---|
| Logistic Regression | 0.769 | 0.760 | 0.787 | 0.768 |
| BERT | 0.826 | 0.819 | 0.840 | 0.828 |
| RoBERTa | 0.825 | 0.836 | 0.834 | 0.855 |
| **Ensemble** | **0.833** | **0.836** | **0.840** | **0.856** |

*Table 22 – Macro-F1 by semester week*

The ensemble was the best or joint-best model at every checkpoint. Its performance stayed between 0.833 and 0.856 throughout the semester, including at week 2, the earliest checkpoint.

**Statistical comparison**

| Comparison | Macro-F1 difference | 95% CI | p-value (one-sided) |
|---|---|---|---|
| RoBERTa vs Logistic Regression | +0.066 | 0.054 to 0.078 | < 0.001 |
| RoBERTa vs BERT | +0.010 | 0.000 to 0.019 | 0.023 |
| Ensemble vs RoBERTa | +0.003 | −0.002 to 0.009 | 0.11 |

*Table 23 – Statistical comparison of models (paired student-level bootstrap, 1,000 resamples)*

The transformer models are significantly better than the classical baseline, and RoBERTa is significantly better than BERT. The ensemble gives a further small improvement over RoBERTa and is used as the deployed model because it gives the best and most stable results across weeks.

**Weekly emotional trend of the cohort**

Applying the ensemble to the test-split students week by week shows how the cohort's emotional state changed across the semester (Table 24).

| Measure | Week 2 | Week 4 | Week 8 | Week 12 |
|---|---|---|---|---|
| Mean emotional stress score (0–4) | 2.00 | 2.05 | 2.20 | 2.21 |
| Predicted Normal | 36.7% | 32.7% | 26.1% | 27.5% |
| Predicted Stress | 3.8% | 6.4% | 7.4% | 5.5% |
| Predicted Anxiety | 5.6% | 6.1% | 9.6% | 8.7% |

*Table 24 – Emotional stress trend of test-split students across the semester*

- The average emotional stress score rose steadily from week 2 to week 12.
- The share of Normal predictions fell by about nine percentage points.
- Stress and anxiety peaked at week 8, around the mid-semester assessment period.
- At the student level, 34.8% of students had a higher stress score at week 12 than at week 2, and 28.8% had a lower score.

**Explainability results**

The attention explanations highlighted emotionally meaningful words:
- For "I am so stressed about my exams, I have three deadlines this week and I cannot sleep at night", the most-attended words included "stressed", "cannot" and "exams".
- For a text expressing thoughts of ending one's life, the word "ending" was among the most-attended words.

In the dashboard, these explanations appear as a colour-coded heatmap together with the class probabilities and advisor recommendation (Figure 9).

**Deployment results**

- The REST API responds in about **0.13 seconds** for a typical short student message, and averaged **0.25 seconds** per text across 300 test texts of all lengths, on a standard CPU. This is suitable for interactive use.
- In a consistency test, the live service reproduced the evaluated model's predictions on **300 of 300** test texts.
- All four APIs of the integrated system and the dashboard run together and respond correctly.

**Integration results**

| Measure | Value |
|---|---|
| Students in the integration layer | 2,543 |
| Students receiving an emotional score from this component | 2,542 |
| Meta-FNN test accuracy | 78.53% |
| Meta-FNN test F1 | 0.696 |
| Meta-FNN AUC-ROC | 0.856 |
| Meta-FNN precision / recall | 0.627 / 0.783 |
| Final alerts (High / Medium / Low) | 744 / 382 / 1,417 |

*Table 25 – Meta-FNN integration results*

The emotional NLP component now supplies a model-generated emotional stress score for 2,542 of the 2,543 students in the integration layer. The Meta-FNN combines it with the academic and behavioural signals into a final burnout and dropout risk and alert level, which is shown in the shared dashboard.

---

### 3.2 Research Findings

The results lead to several findings that go beyond raw accuracy figures.

- **Fine-tuning on labelled data is essential.** The rule-based lexicon method (VADER) and the zero-shot model reached only 25–33% accuracy, while fine-tuned transformers reached 82–85%. Mental-health categories are expressed in subtle and overlapping language that general-purpose sentiment tools and zero-shot models cannot separate reliably. A model must learn these distinctions from labelled examples.

- **Transformers clearly outperform classical models.** RoBERTa improved macro-F1 by 6.6 points over the best classical model, and the difference is statistically significant (p < 0.001). Contextual understanding matters: transformers recognise negation ("I don't feel okay"), intensity and the way feelings are described across a whole sentence, which word-count features cannot capture.

- **RoBERTa is the strongest single model.** RoBERTa outperformed BERT by about one macro-F1 point (p = 0.023). Its larger and more varied pre-training data, which includes news and general web text, matches informal writing better. Its case-sensitive tokeniser also keeps emphasis such as capital letters and emojis.

- **A weighted ensemble gives the best and most stable results.** Combining RoBERTa (0.55) and BERT (0.45) produced the highest accuracy, macro-F1 and risk-level accuracy, and the best or equal-best score at every semester checkpoint.
  - When the two models agree (91.8% of test texts), the prediction is correct 88.1% of the time.
  - On the 8.2% of texts where they disagree, the ensemble was right more often (52.9%) than either RoBERTa (51.7%) or BERT (39.6%) alone.
  - Choosing the weights on validation data, instead of using equal weights, avoids over-relying on the weaker model.

- **Careful imbalance handling protects the rare, serious categories.** With square-root class weights and macro-F1 model selection, every class reached an F1 of at least 0.76, and the rarest class, Personality disorder (2% of the data), reached 0.81. In a well-being setting, missing a rare but serious category is costly, so balanced performance across classes matters more than overall accuracy alone.

- **Keeping both the beginning and the end of long texts improves coverage.** Increasing the input length from 128 to 256 tokens raised the share of texts read in full from about two-thirds to about 85%. Head-and-tail truncation keeps the opening topic and the closing sentence of longer posts, where students often state their most important feeling.

- **Student-level evaluation gives realistic results.** By holding out entire students and reporting confidence intervals, the evaluation measures how the model performs on students it has never seen, which is the situation in real use.

- **Emotional risk can be tracked reliably from early in the semester.** The ensemble's performance was stable from week 2 to week 12 (macro-F1 0.833–0.856), so the same model can be used at every checkpoint, including the earliest. In the test cohort, the average emotional stress score rose from 2.00 to 2.21 across the semester, with stress and anxiety peaking at week 8. This shows how weekly monitoring can reveal both cohort-level pressure points and individual students whose emotional state is getting worse.

- **Very few at-risk texts are missed.** The ensemble flagged 98.6% of at-risk texts with 98.1% precision, which suits an early-warning tool: almost every student who needs attention is flagged, and few are flagged unnecessarily.

- **Explanations make predictions usable.** Attention heatmaps consistently highlighted emotionally meaningful words (such as "stressed", "exams", "cannot" and "ending"), so an advisor can see the evidence behind each prediction and decide how to respond.

- **Text-based emotional signals can be integrated into a multi-modal system.** The weekly emotional stress score was delivered to the Meta-FNN layer for 2,542 of 2,543 students and combined with academic and behavioural risk, which shows that the NLP component works as one part of a larger early-warning system.

---

### 3.3 Discussion

The results show that transformer-based NLP can turn student-written text into an accurate, explainable and timely emotional signal, and that this signal can be integrated into a wider early-warning system.

**Model performance.** The deployed ensemble correctly classified 85.25% of texts from students it had never seen, with a macro-F1 of 0.843. This performance is balanced across all seven categories rather than driven by the most common classes. Normal texts are recognised almost perfectly (F1 0.96), which keeps false alarms low. Serious categories such as Depression and Suicidal are detected with high recall (0.75 and 0.82). The most common confusion is between Depression and Suicidal. Both are High or Critical risk categories, so in practice such a confusion still leads to a serious-risk alert and an intervention recommendation. This is reflected in the high risk-level accuracy (86.38%) and the very high at-risk recall (98.6%).

**The value of explainability.** In student well-being, a prediction is only useful if a professional can understand and trust it. The attention-based heatmap shows advisors which words influenced each decision, the probability bars show how confident the model is and which alternatives it considered, and the advisor recommendation translates the result into a clear next step. Together, these turn a black-box classifier into a decision-support tool. The system is designed to support advisors and counsellors, not to replace their professional judgement.

**Weekly monitoring.** Because the model's accuracy is stable across all four checkpoints, advisors can compare a student's results from week to week with confidence. The weekly progression view and the latest and most-severe results in the student list help advisors see whether a student is improving or getting worse, and cohort-level trends, such as the rise in stress and anxiety at week 8, can help institutions plan support around high-pressure periods.

**Practical deployment.** The component meets its real-time requirement: each text is analysed in under 0.3 seconds on an ordinary CPU, so it can be deployed without expensive hardware. The live service applies exactly the same cleaning and encoding as the training pipeline, and a consistency test confirmed that the deployed model reproduces the evaluated results. Exposing the model through a REST API made it easy to connect both the dashboard and the integration layer.

**Role in the multi-modal system.** The emotional NLP component supplies the text-based view of each student, complementing the academic (GRU) and behavioural (VAE) components. Its weekly 0–4 emotional stress score feeds the Meta-FNN layer and the transparent weighted risk formula, so that the final burnout and dropout risk reflects what students say about how they feel, not only how they perform.

**Limitations and future work.**
- The texts come from a public social-media mental-health corpus organised as a weekly semester cohort. A natural next step is to validate the component on consented writing from university students, including the short messages typical of student check-ins.
- The emotional and academic records used in the integration come from different sources and are aligned through student IDs. Collecting all three types of data for the same students would allow the contribution of each signal to be measured more precisely.
- Future work also includes domain-adapted models such as MentalBERT [13], complementary explanation methods such as SHAP and LIME [21], [22], and support for Sinhala and Tamil text for Sri Lankan student populations.

---

### 3.4 Summary of Each Student's Contribution

The integrated system was developed collaboratively, with each member responsible for one component.

- **Induwara K.P.Y. (IT22196392) — Emotional Burnout Detection from Student Text (this component).** Responsible for the complete emotional NLP component:
  - dataset preparation, cleaning and student-level splitting;
  - classical baselines;
  - fine-tuning of BERT and RoBERTa with class-weighted loss and head-and-tail truncation;
  - the weighted ensemble;
  - attention-based explainability;
  - risk mapping, distress and emotional stress scoring, and advisor recommendations;
  - weekly student monitoring;
  - the Flask REST API (port 5004);
  - the Emotional Analysis dashboard;
  - export of weekly emotional stress scores to the Meta-FNN integration layer.

- **Mahavitha S.M. (IT22916426) — Academic Risk Prediction (GRU).** Responsible for the GRU-based model that predicts academic risk from academic and learning-activity data across multiple weeks, and for the academic risk API and dashboard views.

- **Karunarathne D.C. (IT22215710) — Behavioural Anomaly Detection (VAE).** Responsible for the Variational Autoencoder-based model that detects anomalies in students' weekly behavioural patterns, and for the behavioural anomaly API and dashboard view.

- **[NAME] (IT22253194) — Meta-Integration Layer (Meta-FNN).** Responsible for the Meta-FNN integration layer that combines the academic, behavioural and emotional signals into a final burnout and dropout risk and alert level, and for the integration API and Meta-FNN dashboard view.

**[CHECK THE WORDING OF EACH TEAMMATE'S DESCRIPTION WITH THEM.]**

---

## 4 CONCLUSION

This research developed an explainable, transformer-based NLP component that detects emotional burnout signals in student-generated text, as part of an explainable multi-modal system for early academic burnout and dropout prediction. The component addresses a clear gap: students' own words carry early signs of emotional strain, yet this information is rarely analysed systematically, explained to the people who must act on it, or combined with academic and behavioural data.

Using a weekly cohort of 13,171 students and 52,582 cleaned texts, the study prepared a student-level dataset, established classical baselines, and fine-tuned BERT and RoBERTa models with class-weighted loss and a 256-token head-and-tail input window. A weighted soft-voting ensemble of the two transformers was the best model:
- 85.25% accuracy and 0.843 macro-F1 on students it had never seen;
- an F1 of at least 0.76 in every one of the seven categories;
- 98.6% of at-risk texts detected;
- stable performance from the first checkpoint (week 2) to the last (week 12).

Transformers outperformed the strongest classical model by 6.6 macro-F1 points, a statistically significant improvement.

Beyond accuracy, the component turns each prediction into information an advisor can act on: a risk level, a distress score, a clear recommendation, and an attention-based explanation of the words behind the decision. It runs as a real-time REST API that analyses a text in under 0.3 seconds on a standard CPU, and it supports week-by-week monitoring through the Emotional Analysis dashboard. Its weekly emotional stress score is integrated into the Meta-FNN layer for 2,542 of 2,543 students, where it is combined with academic and behavioural risk into a final burnout and dropout alert.

Overall, the component shows that explainable, transformer-based text analysis can provide a reliable and practical emotional signal for early-warning systems in higher education. With further validation on consented student writing and support for local languages, it has strong potential to help universities identify struggling students earlier and direct support where it is needed most.

---

## 5 REFERENCES

[1] C. Maslach and M. P. Leiter, "Understanding the burnout experience: Recent research and its implications for psychiatry," *World Psychiatry*, vol. 15, no. 2, pp. 103–111, 2016.

[2] W. B. Schaufeli, I. M. Martínez, A. M. Pinto, M. Salanova, and A. B. Bakker, "Burnout and engagement in university students: A cross-national study," *Journal of Cross-Cultural Psychology*, vol. 33, no. 5, pp. 464–481, 2002.

[3] M. De Choudhury, M. Gamon, S. Counts, and E. Horvitz, "Predicting depression via social media," in *Proc. 7th Int. AAAI Conf. Weblogs and Social Media (ICWSM)*, 2013.

[4] G. Coppersmith, M. Dredze, and C. Harman, "Quantifying mental health signals in Twitter," in *Proc. Workshop on Computational Linguistics and Clinical Psychology (CLPsych)*, ACL, 2014, pp. 51–60.

[5] E. Turcan and K. McKeown, "Dreaddit: A Reddit dataset for stress analysis in social media," in *Proc. 10th Int. Workshop on Health Text Mining and Information Analysis (LOUHI)*, 2019.

[6] D. E. Losada and F. Crestani, "A test collection for research on depression and language use," in *Proc. Conf. and Labs of the Evaluation Forum (CLEF)*, LNCS vol. 9822, Springer, 2016.

[7] M. Garg, "Mental health analysis in social media posts: A survey," *Archives of Computational Methods in Engineering*, vol. 30, pp. 1819–1842, 2023.

[8] C. J. Hutto and E. Gilbert, "VADER: A parsimonious rule-based model for sentiment analysis of social media text," in *Proc. 8th Int. AAAI Conf. Weblogs and Social Media (ICWSM)*, 2014.

[9] S. Hochreiter and J. Schmidhuber, "Long short-term memory," *Neural Computation*, vol. 9, no. 8, pp. 1735–1780, 1997.

[10] A. Vaswani *et al.*, "Attention is all you need," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[11] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in *Proc. NAACL-HLT*, 2019, pp. 4171–4186.

[12] Y. Liu *et al.*, "RoBERTa: A robustly optimized BERT pretraining approach," arXiv:1907.11692, 2019.

[13] S. Ji, T. Zhang, L. Ansari, J. Fu, P. Tiwari, and E. Cambria, "MentalBERT: Publicly available pretrained language models for mental healthcare," in *Proc. 13th Language Resources and Evaluation Conf. (LREC)*, 2022, pp. 7184–7190.

[14] C. Sun, X. Qiu, Y. Xu, and X. Huang, "How to fine-tune BERT for text classification?" in *Proc. China National Conf. Chinese Computational Linguistics (CCL)*, LNCS vol. 11856, 2019, pp. 194–206.

[15] W. Yin, J. Hay, and D. Roth, "Benchmarking zero-shot text classification: Datasets, evaluation and entailment approach," in *Proc. EMNLP-IJCNLP*, 2019.

[16] M. Lewis *et al.*, "BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension," in *Proc. ACL*, 2020.

[17] J. Hartmann, M. Heitmann, C. Siebert, and C. Schamp, "More than a feeling: Accuracy and application of sentiment analysis," *International Journal of Research in Marketing*, vol. 40, no. 1, pp. 75–87, 2023.

[18] T. G. Dietterich, "Ensemble methods in machine learning," in *Proc. Int. Workshop on Multiple Classifier Systems (MCS)*, LNCS vol. 1857, 2000, pp. 1–15.

[19] S. Jain and B. C. Wallace, "Attention is not explanation," in *Proc. NAACL-HLT*, 2019.

[20] S. Wiegreffe and Y. Pinter, "Attention is not not explanation," in *Proc. EMNLP-IJCNLP*, 2019.

[21] M. T. Ribeiro, S. Singh, and C. Guestrin, "'Why should I trust you?': Explaining the predictions of any classifier," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016.

[22] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2017.

[23] B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993.

[24] S. Sarkar, "Sentiment analysis for mental health," Kaggle dataset, 2024. [Online]. Available: https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

[25] T. Wolf *et al.*, "Transformers: State-of-the-art natural language processing," in *Proc. EMNLP: System Demonstrations*, 2020, pp. 38–45.

[26] A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.

[27] F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[28] I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.

*(Cite [25]–[27] in the technology stack paragraph: "implemented with Hugging Face Transformers [25], PyTorch [26] and scikit-learn [27]".)*

---

## 6 APPENDICES

**Appendix A: Plagiarism Report**

**[INSERT TURNITIN SCREENSHOT AFTER SUBMISSION]**

**Appendix B: Sample API Request and Response**

Request — `POST http://localhost:5004/api/analyze`

```json
{ "text": "I am so stressed about my exams, I have three deadlines this week and I cannot sleep at night" }
```

Response (shortened)

```json
{
  "prediction": "Stress",
  "confidence": 0.9273,
  "risk_level": "Medium",
  "stress_score": 49.6,
  "emotional_stress_score": 1,
  "probabilities": { "Stress": 0.9273, "Anxiety": "...", "Normal": "...", "...": "..." },
  "attention": [ { "token": "stressed", "weight": "..." }, { "token": "exams", "weight": "..." } ],
  "models": { "bert": { "prediction": "Stress" }, "roberta": { "prediction": "Stress" }, "agree": true },
  "model": "BERT + RoBERTa ensemble (soft voting, weekly v2)"
}
```

**Appendix C: Training Environment and Hyperparameters**

- **Software:** Python 3.14, PyTorch 2.13 (CUDA 13.0), Hugging Face Transformers 5.16, scikit-learn 1.9, pandas 3.0
- **Hardware:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) for training; standard laptop CPU for serving
- **Hyperparameters:** as in Table 4. Ensemble weight search from 0 to 1 in steps of 0.05; best weight 0.55 for RoBERTa (validation macro-F1 0.835).
- **Evaluation:** accuracy, macro-F1, weighted F1, per-class precision, recall and F1, confusion matrix, student-level bootstrap 95% CI (1,000 resamples), and paired bootstrap significance tests.

**Appendix D: System Screenshots**

**[INSERT SCREENSHOTS: (1) Emotional Analysis — Analysis view, (2) Weekly Progression view, (3) Quick Analysis with attention heatmap, (4) Meta-FNN Risk tab, (5) all four API terminals running]**

---

## PART C — CHECKLIST FOR THE STUDENT (not part of the report)

**Fill in the placeholders:**
- Your specialization.
- The month and year.
- The co-supervisor's title.
- The name of IT22253194.
- Signatures.
- UAT details in 2.1.3.
- Test case T10.
- The Turnitin screenshot.

**Images already in your project** (folder `emotional-nlp-fresh/emotional_burnout_nlp/results/figures/`):
- `weekly_label_distribution.png` → Figure 4
- `token_length_distribution.png` → Figure 5
- `weekly_v2_baseline_comparison.png` → Figure 6
- `weekly_v2_all_models_comparison.png` → Figure 7
- `weekly_v2_ensemble_confusion.png` → Figure 8

**Screenshots to take from the dashboard** (start the APIs with Ctrl+Shift+B in VS Code):
- Quick Analysis with an example text → Figure 9
- Student list + Weekly Progression → Figure 10
- Meta-FNN Risk tab → Figure 11

**Diagrams to draw** (draw.io, PowerPoint, or ask for them to be drawn from the descriptions in 2.1.1):
- Figure 1 (whole system)
- Figure 2 (your component)
- Figure 3 (SDLC circle; the same style as the reference report is fine)

**Before submitting:** check every reference in Google Scholar, especially [7], [17] and [24].

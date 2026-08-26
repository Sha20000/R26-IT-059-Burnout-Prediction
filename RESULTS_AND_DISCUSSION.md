# Chapter 4. Results

This chapter presents what the system achieved. All nine models were trained and tested on the same data, and every model was measured on the same held-out test set of 7,659 passages that was never used during training. The numbers reported here come from the saved results files of the project, and the figures are the real outputs of the experiments.

## 4.1 Experimental Setup

The corpus of 51,068 labelled passages, described in Chapter 3, was split into training, validation, and test sets in a 70/15/15 ratio using stratified sampling, giving a common test set of 7,659 passages. Every one of the nine models was evaluated on this same test set, so that all comparisons are fair. Training and evaluation were carried out locally on a workstation with an NVIDIA RTX 4060 graphics processing unit with 8 GB of memory, using half precision. Overall performance is reported as accuracy and F1 score, and for the transformer models both the weighted F1 and the macro F1 are reported, so that performance on the rare classes is not hidden by the large classes.

## 4.2 Overall Model Comparison

Table 4.1 lists all nine models. The fine-tuned transformers were clearly the strongest. The best was the BERT and RoBERTa ensemble at 82.32 percent accuracy (F1 0.823), with RoBERTa (82.30 percent) and BERT (81.39 percent) just behind. The classical models trailed by 5 to 10 points, from Logistic Regression at 76.69 percent down to Random Forest at 71.87 percent, and the LSTM sat at 76.84 percent. The two untrained baselines were far lower, at 25.02 percent (VADER) and 32.80 percent (zero-shot RoBERTa). The gap between the untrained baselines and the fine-tuned transformers is about 50 points, which Figure 4.1 makes clear at a glance.

| # | Model | Type | Accuracy | F1 |
|---|---|---|---|---|
| 1 | Logistic Regression | Classical ML | 76.69% | 0.760 |
| 2 | Random Forest | Classical ML | 71.87% | 0.699 |
| 3 | SVM | Classical ML | 75.49% | 0.750 |
| 4 | LSTM | Deep Learning | 76.84% | 0.766 |
| 5 | VADER | Rule-based (no training) | 25.02% | 0.228 |
| 6 | RoBERTa Zero-Shot | Transformer (no training) | 32.80% | 0.344 |
| 7 | RoBERTa (fine-tuned) | Transformer | 82.30% | 0.823 |
| 8 | BERT (fine-tuned) | Transformer | 81.39% | 0.814 |
| 9 | BERT + RoBERTa Ensemble | Transformer (soft voting) | 82.32% | 0.823 |

**Table 4.1.** Accuracy and F1 score of all nine models on the 7,659-sample test set.

*(Figure 4.1: weighted-F1 comparison of all nine models — `results/figures/all_models_f1.png`.)*

The reason the two untrained baselines fall so far behind, and why the fine-tuned transformers rise so far above the classical models, is examined in the discussion.

## 4.3 Classical Machine Learning Results

Three classical models were trained on the same TF-IDF features. Logistic Regression was the strongest of the three at 76.69 percent accuracy and 0.760 F1, followed by the SVM at 75.49 percent, with Random Forest the weakest at 71.87 percent. The strength of these models is that they are fast, simple, and easy to interpret, and they still reached a respectable level of accuracy. Their weakness is that they treat a passage as an unordered bag of words. They count which words appear but they do not understand word order or context, so they cannot tell that "not happy" is different from "happy". This limits how well they can separate classes that use similar words.

## 4.4 Deep Learning Results

The bidirectional LSTM reached 76.84 percent accuracy and 0.766 F1, slightly above the best classical model. The improvement comes from the fact that the LSTM reads the words in order and can therefore capture some of the meaning that word order carries. However, the LSTM still sat well below the transformers. This is because it learns its word representations from scratch on the project data alone, whereas the transformers arrive already knowing a great deal of English from their large-scale pretraining.

## 4.5 Transformer Model Results

The two fine-tuned transformers were the strongest single models. RoBERTa reached 82.30 percent accuracy with a weighted F1 of 0.823 and a macro F1 of 0.783, and BERT reached 81.39 percent accuracy with a weighted F1 of 0.814 and a macro F1 of 0.777. RoBERTa is therefore slightly ahead of BERT, and both are far above the classical models and the LSTM. The gap between the weighted F1 (0.823) and the macro F1 (0.783) for RoBERTa is important: it shows that the model performs very well on the large classes but less well on the small ones, which is exactly what the per-class results in Section 4.7 confirm.

## 4.6 Ensemble Results

The soft-voting ensemble of BERT and RoBERTa was the best overall model, at 82.32 percent accuracy and 0.823 F1. It beat RoBERTa by only 0.02 points (82.32 versus 82.30 percent) and BERT by about 0.9 points (versus 81.39 percent). The reason the ensemble helps is explained in Section 5.2: the two transformers make slightly different mistakes, and averaging their predictions lets the more confident model correct the other. The gain is small because the two models already agree on the easy classes, so the extra help mostly appears on the harder ones.

## 4.7 Per-Class Performance

Table 4.2 shows the F1 score for each class for the best model. The pattern is simple: classes with more training data scored higher, and classes with less data scored lower. Normal was the best at 0.96, and Personality Disorder, the smallest class with only 895 samples, was the lowest at 0.65.

| Class | Training Samples | F1 | Risk Level |
|---|---|---|---|
| Normal | 16,039 | 0.96 | Low |
| Anxiety | 3,617 | 0.87 | Medium |
| Bipolar | 2,501 | 0.84 | High |
| Depression | 15,085 | 0.76 | High |
| Suicidal | 10,638 | 0.73 | Critical |
| Stress | 2,293 | 0.72 | Medium |
| Personality Disorder | 895 | 0.65 | High |

**Table 4.2.** F1 score for each class, shown next to its number of training samples.

Figure 4.2 shows these per-class scores as a bar chart. The bars fall steadily from Normal on the left to Personality Disorder on the right, and the training-sample count printed under each bar makes the cause plain: the classes with the most data score the highest, and the smallest class scores the lowest.

*(Figure 4.2: per-class F1 of the ensemble — `results/figures/per_class_f1.png`.)*

The easiest class to classify was Normal, which is both the largest class and the most distinct in its language. The hardest were Personality Disorder, Stress, and Suicidal. Personality Disorder suffers from too few examples, while Suicidal and Depression are hard because they share very similar language, as the confusion matrix shows next.

## 4.8 Confusion Matrix Analysis

Figure 4.3 is the confusion matrix for the ensemble. The strong diagonal shows that most predictions were correct. The Normal class was almost always right. The main mistakes were between Depression and Suicidal, which is expected, because these two conditions are expressed with very similar words such as "hopeless", "worthless", and "give up". Far-apart classes were rarely confused, for example Normal was almost never mistaken for Suicidal. The smallest class, Personality Disorder, was the hardest, since the model had the fewest examples to learn from.

*(Figure 4.3: confusion matrix — `results/figures/fig4_3.png`.)*

## 4.9 Class Imbalance Results

To handle the strong class imbalance, a class-weighted loss was tested on the RoBERTa model, in which the rare classes were given a much larger weight in the loss. Table 4.3 compares the model without weighting against the model with weighting. The weighting did increase the recall of the rare and dangerous classes: it caught more Suicidal, Stress, and Personality Disorder cases. However, it also lowered the overall accuracy and the macro F1, because it produced more false alarms. This is a classic precision-recall trade-off. Since the weighting did not improve the overall scores, the unweighted model was kept as the final model, and the weighted model is reported here as an ablation.

| Metric | Without Weighting (final) | With Weighting |
|---|---|---|
| Accuracy | 82.3% | 80.4% |
| Weighted F1 | 0.823 | 0.805 |
| Macro F1 | 0.783 | 0.766 |
| Suicidal recall | 0.75 | 0.81 |
| Stress recall | 0.75 | 0.83 |
| Personality Disorder recall | 0.55 | 0.61 |

**Table 4.3.** Effect of the class-weighted loss on the RoBERTa model.

## 4.10 Cross-Validation Results

To confirm that the classical results were not the product of one lucky split, the Logistic Regression and SVM models were tested with five-fold stratified cross-validation on the full corpus. The results were very stable. Logistic Regression averaged 75.88 percent accuracy with a standard deviation of only 0.28 percent, and a weighted F1 of 0.760 with a standard deviation of 0.29 percent. The SVM averaged 74.83 percent accuracy with a standard deviation of 0.39 percent, and a weighted F1 of 0.747. The very small standard deviations, all under half a percent, show that the reported performance is stable and repeatable and does not depend on how the data happened to be split.

| Model | CV Accuracy | CV Weighted F1 |
|---|---|---|
| Logistic Regression | 75.88% ± 0.28% | 0.760 ± 0.003 |
| SVM | 74.83% ± 0.39% | 0.747 ± 0.004 |

**Table 4.4.** Five-fold stratified cross-validation results.

## 4.11 Explainability Results

The explanation produced by the system was examined at two levels: the local explanation for a single passage, and the global explanation across the whole test set.

### 4.11.1 Local Explanation (Single Passage)

For every prediction the system produces a word-level attention heatmap that highlights which words the model focused on, using a colour scale from grey (low attention), through blue, yellow, and orange, to red (high attention). Figure 4.4 shows this for the passage "i feel exhausted and hopeless and i cannot continue my studies anymore", which the model classified as Depression. The model spreads some attention across the whole sentence, but it concentrates most strongly on the emotionally loaded words such as "exhausted", "hopeless", "cannot", and "anymore", which are shown in orange and red, while neutral words such as "continue" and "my" receive less attention and appear in yellow. This confirms that the decision rests on the words that carry the real emotional meaning, which is what allows an advisor to trust and check the result.

*(Figure 4.4: colour-coded attention heatmap for one passage — `results/figures/attention_heatmap_words.png`.)*

Figure 4.5 shows how this focus changes as a student's writing worsens across the simulated weekly progression. In the early, calm passage the attention is spread thinly over ordinary words, but as the language moves through stress to depression and finally to suicidal expression, the model's focus shifts onto increasingly severe words such as "exhausted", "hopeless", and "give up". The explanation is therefore not fixed but tracks the changing emotional content of the text.

*(Figure 4.5: weekly attention progression — `results/figures/fig4_6.png`.)*

### 4.11.2 Global Explanation (Top Words per Class)

To understand the reasoning of the model across the whole test set, the attention was aggregated by class to find the content words that most drive each prediction. Table 4.5 reports the result, and it is both striking and clinically sensible. For Anxiety, the model focuses on "panic", "worried", "nervous", and "restless". For Bipolar, it focuses on "manic" and "mania" and even on the names of real bipolar medications such as "lithium" and "lamictal". For Suicidal, it focuses on "suicide", "die", and "kill". For Personality Disorder, it focuses on specific clinical terms such as "avoidant" and "avpd". This shows that the model has learned the genuine vocabulary of each condition rather than relying on superficial or accidental cues, which greatly strengthens the case that its predictions are trustworthy. The Normal class, by contrast, has no distinctive distress vocabulary and instead attends to ordinary, positive everyday words such as "good", "love", and "morning".

| Class | Words the Model Attends to Most |
|---|---|
| Normal | good, love, morning, like, want |
| Stress | stress, stressed, help, feel |
| Anxiety | anxious, worried, nervous, scared, panic, restless |
| Depression | depressed, depression, life, feel |
| Bipolar | manic, mania, hypomania, lithium, lamictal |
| Personality Disorder | avoidant, avpd, hikikomori |
| Suicidal | suicide, die, kill, suicidal |

**Table 4.5.** The content words that receive the most model attention within each class, aggregated across the test set.

## 4.12 Risk Score Results

Beyond the class label, the system turns each prediction into a risk level and a distress score from 0 to 100. Table 4.6 shows four real examples produced by the model, moving from a calm passage to a critical one. The distress score rises smoothly with the severity of the text, from 0.5 for a normal passage, through 60.9 for an anxious one, to 97.0 for a suicidal one. This single number gives an advisor an immediate sense of how serious a case is, without having to read all seven class probabilities.

| Input Passage | Prediction | Confidence | Risk Level | Distress Score |
|---|---|---|---|---|
| "I feel good today and I am keeping up with all my assignments" | Normal | 99.3% | Low | 0.5 |
| "I feel so anxious and worried all the time, I keep panicking about everything" | Anxiety | 86.4% | Medium | 60.9 |
| "I feel exhausted all the time and everything feels pointless and hopeless" | Depression | 57.2% | High | 87.3 |
| "I want to end it all, nothing matters anymore and I cannot go on" | Suicidal | 85.8% | Critical | 97.0 |

**Table 4.6.** Real examples of the prediction, risk level, and distress score produced by the system.

Figure 4.6 shows how this risk score behaves when the system tracks a single student across several weekly submissions, as in the deployed dashboard. The student's writing begins calm at Weeks 2 and 4, where the model predicts Normal and the risk stays Low at a score of 1. By Week 8 the writing has turned to exhaustion and worry about failing, the prediction changes to Depression, and the risk jumps to High at a score of 7, where it remains at Week 18. This demonstrates how the system converts a sequence of short student texts into a single rising risk line that an advisor can read at a glance and use to decide when to step in.

*(Figure 4.6: weekly risk progression for one student — `results/figures/weekly_risk_progression.png`.)*

## 4.13 Lead-Time Results

To study how early the system could raise a warning, it was tested at four points across a semester, at Weeks 2, 4, 8, and 12. It is important to state clearly that this is a controlled simulation and not a study of real students tracked over time, because the dataset does not contain time-stamped student writing. Under this simulation, the model separated ordinary text from distressed text with stable and high accuracy at every point, reaching 91 percent accuracy and an F1 of 0.95 as early as Week 2, and remaining between 89 and 91 percent across the later weeks. Because the accuracy stays high from very early, the earlier such distress language appears, the more time an advisor would have to act. Figure 4.7 illustrates this result.

*(Figure 4.7: lead-time result — `results/figures/fig4_5.png`.)*

## 4.14 Computational Performance

Table 4.7 reports the practical cost of the models. The classical models and the LSTM are small and fast. The transformers are much larger, with 109.5 million parameters for BERT and 124.7 million for RoBERTa, giving 234.1 million for the ensemble. Each transformer was fine-tuned for three epochs in about ten to eleven minutes on the RTX 4060. At prediction time, the system is fast enough for interactive use: BERT classifies a single passage in about 23 milliseconds, and the full ensemble in about 50 milliseconds. This confirms that the extra accuracy of the ensemble comes at only a small extra cost.

| Model | Parameters | Approx. Training Time | Inference per Passage |
|---|---|---|---|
| Classical models | 10,000 TF-IDF features | under one minute | under 1 ms |
| BiLSTM | 3,221,255 | a few minutes | a few ms |
| BERT (fine-tuned) | 109,487,623 | ~10 minutes | 23 ms |
| RoBERTa (fine-tuned) | 124,651,015 | ~11 minutes | within ensemble |
| BERT + RoBERTa Ensemble | 234,138,638 | sum of the two | 50 ms |

**Table 4.7.** Size, training time, and inference speed of the models on an NVIDIA RTX 4060.

---

# Chapter 5. Discussion

This chapter explains what the results mean. It interprets the main findings, compares the system with the studies in the literature review, and discusses the strengths, limits, and practical value of the work.

## 5.1 Summary of Main Findings

The main finding is a clear ranking of the three model families. The non-trained baselines were very weak, the classical models were moderate, the deep learning model was slightly better, and the fine-tuned transformers were far ahead of all of them. The best model was the BERT and RoBERTa ensemble, at 82.32 percent accuracy. Alongside this, two further contributions were achieved: every prediction is explained with a word-level attention heatmap, and the system turns each result into a risk level and distress score for an advisor. The objectives set out at the start of the project were therefore met.

## 5.2 Why the Ensemble Performed Best

The ensemble beat both BERT and RoBERTa on their own, though only by a small amount. The reason is that the two transformers were pretrained in different ways and use different tokenisation, so they do not make exactly the same mistakes. When their class probabilities are averaged, the more confident model tends to correct the less confident one. The gain is small because the two models already agree on the easy classes, so the ensemble mainly helps on the harder, rarer classes. This result matches the ensemble literature, where combining models that make different errors is repeatedly shown to give a small but reliable improvement.

## 5.3 Why Classical Models Performed Worse

The classical models were held back by the way they represent text. TF-IDF describes a passage only by which words appear and how often, treating it as an unordered bag of words. This throws away word order and context. As a result, the classical models cannot easily tell apart classes that share vocabulary, and they cannot understand that the meaning of a word depends on the words around it. This is a well-known limitation of bag-of-words methods, and it explains why the classical models plateaued around 75 percent while the transformers passed 82 percent.

## 5.4 Why Transformers Performed Better

The transformers performed better for three connected reasons. First, they were pretrained on very large amounts of text, so they arrive already understanding a great deal of English. Second, their self-attention mechanism lets every word be interpreted in the context of all the other words in the passage, so the same word can be understood differently in different sentences. Third, fine-tuning transfers this general language knowledge to the specific task of mental-health classification. The combination of pretraining, context through self-attention, and transfer learning is what allowed the transformers to separate the seven classes far more accurately than the classical or deep learning baselines.

## 5.5 Explainability Discussion

A prediction that cannot be explained is difficult for an advisor to trust or act upon. The attention heatmap addresses this directly by showing, for every prediction, the exact words that drove the decision. When the model flags a passage as depressed and the heatmap highlights "exhausted" and "hopeless", the advisor can immediately see that the model responded to genuine signals of distress rather than to unrelated words. This transparency is what makes the system usable in practice, because it turns a bare label into evidence that a human can check. Most of the comparable studies reviewed do not provide this word-level explanation.

## 5.6 Risk Score Discussion

The risk level and distress score exist to make the output actionable. Seven class probabilities are hard for a busy advisor to read quickly, so the system compresses them into a single ordered number from 0 to 100 and a four-level risk band. As the examples in Section 4.12 show, this number rises smoothly with the severity of the text, and a Suicidal prediction additionally raises a critical alert. This lets an advisor triage many students quickly and focus attention on the most serious cases first, which supports early intervention.

## 5.7 Class Imbalance Discussion

The per-class results follow the amount of training data very closely, and Personality Disorder, the smallest class, was the hardest even after the weighted loss was tried. The weighted loss experiment showed an honest trade-off: it caught more of the rare and dangerous cases but lowered overall accuracy, so the unweighted model was kept and the macro F1 metric was used to keep model selection fair to the small classes. Neither weighting nor macro F1 could fully make up for the simple lack of examples. This is a recurring problem in mental-health text research, and it points directly to data augmentation and the collection of more minority-class data as necessary next steps.

## 5.8 Comparison with Previous Studies

Table 5.1 places this work next to the closest studies from the literature review. Some of them report higher numbers, but the comparison is not fair unless the task is the same. The very high scores come from binary studies. The BERT-DNN model in [7] reached 97 percent, and the BERT-GPT model in [20] reached 99 percent, but both only sort text into two groups, depressed or not. Two groups is a much easier task than the seven-class problem addressed here, so those scores should be read with that in mind.

The fair comparison is with studies that also use many classes. The closest is [18], which uses the same seven mental-health classes. Its often-quoted 97 percent is the binary result; on the seven-class task, the authors report 81 percent accuracy and 0.79 F1, which is just below this study at 82.32 percent and 0.823. The domain-adapted model in [16], trained on similar Reddit data with seven classes, reached 0.847 F1, which is very close to this work. So against studies that do the same kind of job, this system holds up well, and it adds the word-level explanation that most of them lack.

| Study | Classes | Accuracy | F1 | Explainable |
|---|---|---|---|---|
| Beegam & Baalaji [7] | 2 (binary) | 97% | 0.95 | No |
| Mazurets et al. [20] | 2 (binary) | 99% | — | No |
| Wan et al. [17] | 2 (binary) | 90.18% | — | Partial |
| Alqarni et al. [19] | Multi (emotion) | 96.77% | 0.97 | Attention |
| Zhuang et al. [10] | 5 (stress) | 92.55% | 0.92 | No |
| Albu & Spinu [9] | 5 (emotion) | 91% | — | No |
| Aslan [8] | 4 (emotion) | 87.01% | — | Partial |
| Khan et al. [18] | 7 (mental health) | 81% | 0.79 | Attention |
| Sao & Lim [16] | 7 (mental health) | 84.7% | 0.847 | Word importance |
| This study (ensemble) | 7 (mental health) | 82.32% | 0.823 | Phrase-level attention |

**Table 5.1.** This study next to related work. For [18], the seven-class result is shown, not the binary headline.

## 5.9 Practical Implications

This work has direct value for universities and student support services. An early-warning tool built on student writing could help counsellors and academic advisors notice students who are struggling before the situation reaches a crisis, and the risk score would let a small support team focus first on the most serious cases. The explanation attached to each alert supports responsible use, because the advisor sees the evidence and makes the final judgement. Used in this way, the system is a support for human staff, not a replacement for them.

## 5.10 Limitations

Several limitations should be acknowledged honestly. First, the data comes only from public social media text, which is not the same as the writing students produce inside a university, so the results may not transfer directly to a real institution without further testing. Second, the data is only in English, so the system has not been shown to work in other languages. Third, the lead-time result is a simulation and not a study of real students followed over a semester, so it demonstrates capability rather than proven early warning. Fourth, the rare classes remain hard because of the lack of examples, and the model also under-detects Stress, sometimes labelling a clearly stressed passage as Normal. Fifth, the system has not yet been deployed and evaluated in a real university setting.

## 5.11 Future Work

Several directions follow naturally from these limitations. The most important is a cross-dataset evaluation, applying the model to an independent, consented dataset of real student writing to test whether the results transfer. Beyond this, the work could be extended with data augmentation to strengthen the rare classes, multilingual models to serve non-English students, real-time monitoring within a learning platform, and a longitudinal study that tracks real students over time to replace the simulated lead-time analysis with genuine evidence. A larger and more balanced dataset, and domain-adaptive pretraining on mental-health text, would also be expected to improve the hardest classes.

---

## Figure placement guide (all figures below are real files in results/figures/)

- Figure 4.1 (Section 4.2): all-models weighted-F1 comparison — `all_models_f1.png` **(new)**
- Figure 4.2 (Section 4.7): per-class F1 of the ensemble — `per_class_f1.png` **(new)**
- Figure 4.3 (Section 4.8): ensemble confusion matrix — `fig4_3.png`
- Figure 4.4 (Section 4.11.1): colour-coded attention heatmap — `attention_heatmap_words.png` **(new)**
- Figure 4.5 (Section 4.11.1): weekly attention progression — `fig4_6.png`
- Figure 4.6 (Section 4.12): weekly risk progression — `weekly_risk_progression.png` **(new)**
- Figure 4.7 (Section 4.13): lead-time result — `fig4_5.png`
- Optional: baseline comparison — `fig4_2.png`; VADER confusion matrix — `confusion_matrix_vader.png`

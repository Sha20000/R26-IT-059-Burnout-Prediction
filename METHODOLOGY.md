# 3. METHODOLOGY

This section describes how the proposed emotional burnout detection system was designed, implemented, and evaluated. It sets out the overall architecture, the research design, the collection and preparation of the data, the models and their mathematical basis, the strategies used to handle class imbalance and to make the system explainable, the risk scoring and lead-time analysis, the evaluation protocol, the implementation and deployment, and the ethical considerations and limitations of the work. The methodology describes how the system was built and assessed. The empirical outcomes of that assessment are reported in the Results section that follows.

## 3.1 Overall System Architecture

The proposed emotional burnout detection system is organised as an end-to-end pipeline that transforms a single passage of student authored text into an interpretable, risk-graded prediction. Figure 3.1 presents this architecture at a high level. The pipeline proceeds through a sequence of stages, beginning with the raw text input and advancing through cleaning, feature representation, model training and selection, risk scoring, explanation, and the final presentation of results in an advisor facing dashboard. This overview is presented before the individual components so that the role of each stage may be understood in relation to the system as a whole. Each stage is then examined in detail in the subsections that follow.

```
                 Student Text (input)
                          |
                          v
              Text Cleaning and Preprocessing
                          |
                          v
              Feature Representation
        (TF-IDF for classical models;
         subword tokenisation for transformers)
                          |
                          v
        Model Training and Evaluation
          (nine models, three families)
                          |
                          v
                    Best Model
        (BERT + RoBERTa soft-voting ensemble)
                          |
                          v
                   Risk Scoring
          (risk level and distress score)
                          |
                          v
            Attention-based Explanation
                (word-level XAI heatmap)
                          |
                          v
                Advisor Dashboard
```
**Figure 3.1.** High-level architecture of the proposed emotional burnout detection system.

The pipeline begins with a passage of student text, which is standardised by the cleaning procedure described in Section 3.4. The cleaned text is then converted into one of two feature representations, selected according to the family of model that will consume it. Nine models spanning classical machine learning, deep learning, and fine-tuned transformers are trained and evaluated on a common held-out test set, and the strongest of these, a soft voting ensemble of the fine-tuned BERT and RoBERTa models, is selected as the operational model. For each prediction, the system computes an ordinal risk level and a continuous distress score, and it extracts a token level attention explanation that identifies the words most responsible for the decision. The prediction, the risk indicators, and the explanation are finally rendered in an advisor facing dashboard, which serves as the point of contact for the end user.

## 3.2 Research Design

This study adopts a quantitative, experimental research design grounded in supervised machine learning and framed within the design science research paradigm, in which knowledge is produced through the construction and rigorous evaluation of a purposeful artefact [x]. The artefact developed here is the complete emotional burnout detection system introduced in Section 3.1, comprising nine trained or configured classification models, an attention based explainability module, a risk scoring layer, and an advisor facing dashboard. The primary objective is to develop and evaluate a system that classifies student authored text into seven mental-health categories and that explains each decision in human readable form, so that an academic advisor may identify and understand emerging emotional distress at an early stage.

The research follows the structured experimental pipeline set out in Figure 3.1, proceeding from data acquisition and text preprocessing, through feature representation and model construction, to training, evaluation against a common held-out test set, and explainability analysis. The design is guided by three methodological principles. First, a graded progression of model families is constructed, advancing from classical machine learning baselines through a deep learning sequence model to fine-tuned transformer models, so that the contribution of each successive increase in modelling capacity can be measured rather than assumed. Second, two non-trained baselines are included so that the specific contribution of task adaptation, that is, fine-tuning, can be isolated and quantified. Third, explainability is embedded directly in the final model through attention attribution, ensuring that predictions function as transparent and actionable evidence rather than as opaque outputs.

## 3.3 Data Collection

### 3.3.1 Data Sources

The data used in this study is a combined corpus of publicly available mental-health related text, obtained from the "Sentiment Analysis for Mental Health" dataset compiled by Sarkar and hosted on the Kaggle platform [x]. The dataset is released with its database structure under the Open Database Licence (ODbL v1.0) and its contents under the Database Contents Licence (DbCL v1.0), and it is accessible at https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health. The dataset aggregates several pre-existing, publicly shared mental-health datasets, each contributing text associated with one of the mental-health categories examined in this study. All contributing text was drawn from public online sources, including social media platforms, on which individuals expressed their thoughts and feelings in their own words. Because the data is publicly available under an open licence and contains no personally identifiable information, its use for academic research is permitted without further restriction, as discussed in the ethical considerations of Section 3.18.

### 3.3.2 Dataset Description

The corpus consists of short passages of self-expressed, natural language text, each labelled with a single mental-health category. Every record contains the original passage, a cleaned version of that passage produced by the preprocessing described in Section 3.4, and the corresponding category label together with its integer identifier. The passages vary considerably in length, from single sentences to extended personal accounts, and are written in the informal English typical of online self-expression. After cleaning and the removal of null and duplicate entries, the working corpus comprises 51,068 labelled passages distributed across the seven categories defined in Section 3.3.3.

### 3.3.3 Label Definitions

Each passage is assigned to exactly one of seven mental-health categories. The categories, and the states they represent, are defined in Table 3.1. These seven categories were chosen because together they span the full spectrum of emotional states relevant to academic burnout, from ordinary functioning, through the intermediate conditions of stress and anxiety, to the severe and clinically critical conditions of depression, bipolar disorder, personality disorder, and suicidal ideation. This range allows the system to distinguish not merely whether a student is distressed, but how severe the distress is, which is precisely the information an advisor needs in order to respond proportionately. The definitions are applied consistently throughout the study, and the ordinal risk level associated with each category is used later by the risk scoring layer of Section 3.12.

| Category | Definition |
|---|---|
| Normal | Text expressing ordinary, emotionally stable states with no indication of distress. |
| Stress | Text expressing feeling under pressure, overwhelmed, or unable to cope with demands. |
| Anxiety | Text expressing excessive worry, fear, nervousness, or panic. |
| Depression | Text expressing persistent sadness, hopelessness, loss of interest, or low mood. |
| Bipolar | Text reflecting marked mood instability, alternating between elevated and depressed states. |
| Personality Disorder | Text reflecting persistent difficulties with identity, self-image, or patterns of thinking and relating to others. |
| Suicidal | Text expressing thoughts of self-harm, a wish to die, or an intent to end one's life. |

**Table 3.1.** Definitions of the seven mental-health categories.

### 3.3.4 Dataset Statistics

The principal statistics of the corpus are summarised in Table 3.2, and the distribution of passages across the seven categories is reported in Table 3.3 and illustrated in Figure 3.2. The dataset comprises 51,068 labelled passages with a combined vocabulary of approximately 72,850 unique words. The passages have a mean length of 112 words and a median length of 61 words, reflecting a right-skewed distribution in which most passages are short but a minority are very long. The corpus is strongly imbalanced: the Normal category contains 16,039 passages whereas the Personality Disorder category contains only 895, a ratio of approximately eighteen to one. A further notable characteristic is that passage length varies systematically with category. Passages labelled Normal are very short, averaging around seventeen words, whereas passages describing more severe conditions are substantially longer, averaging between 140 and 177 words. This pattern indicates that individuals experiencing greater distress tend to express themselves at greater length, a tendency that itself provides a weak signal to the models. This pronounced imbalance directly motivates the imbalance handling strategies described in Section 3.10.

| Statistic | Value |
|---|---|
| Total labelled passages | 51,068 |
| Number of categories | 7 |
| Vocabulary size (unique words) | ~72,850 |
| Mean passage length | 112 words |
| Median passage length | 61 words |
| Shortest / longest passage | 1 / 5,410 words |
| Largest class (Normal) | 16,039 |
| Smallest class (Personality Disorder) | 895 |
| Imbalance ratio | approximately 18 : 1 |

**Table 3.2.** Summary statistics of the corpus.

| Class | Samples | Proportion |
|---|---|---|
| Normal | 16,039 | 31.4% |
| Depression | 15,085 | 29.5% |
| Suicidal | 10,638 | 20.8% |
| Anxiety | 3,617 | 7.1% |
| Bipolar | 2,501 | 4.9% |
| Stress | 2,293 | 4.5% |
| Personality Disorder | 895 | 1.8% |
| **Total** | **51,068** | **100%** |

**Table 3.3.** Distribution of passages across the seven categories.

## 3.4 Data Preprocessing

Before any feature representation was applied, every passage was standardised through a deterministic cleaning procedure. Each passage was converted to lower case, and uniform resource locators, user mentions, and hashtags were removed. All non-alphabetic characters were then discarded, so that punctuation and numerals did not enter the vocabulary, and repeated whitespace was collapsed into single spaces. After cleaning, records containing null text and exact duplicate passages were removed, so that the models could not be rewarded for memorising repeated content and so that no duplicated passage could appear in more than one data partition. Finally, the seven categorical labels were encoded as integer identifiers from 0 to 6 to provide the numerical targets required for supervised training. The cleaned text was retained alongside the original passage in the working dataset, and it is this cleaned text that all subsequent feature representations operate upon.

The corpus was partitioned into training, validation, and test subsets in a 70/15/15 ratio using stratified sampling, which preserves the class proportions of Table 3.3 within each subset and guarantees that even the rarest categories are represented in every partition. A fixed random seed was applied so that the partition is identical for every model, which allows all nine models to be compared on exactly the same held-out data. The partitioning produced 35,738 training passages, 7,658 validation passages, and 7,659 test passages.

## 3.5 Feature Engineering

Because the study evaluates three families of model with different input requirements, the cleaned text was converted into several distinct numerical representations. The classical machine learning models operate on term frequency inverse document frequency vectors, the deep learning model operates on learned word embeddings built over a custom vocabulary, and the transformer models operate on subword token sequences with accompanying attention masks. Each representation is described in turn below.

### 3.5.1 TF-IDF Features

For the classical machine learning baselines, each passage was represented as a term frequency inverse document frequency vector. This scheme weights each term by how frequently it occurs within a passage and how rarely it occurs across the corpus as a whole, so that terms which are distinctive of a passage receive the greatest weight while terms common to all passages receive little. The present study retained the 10,000 most informative features, considered both single words and adjacent word pairs, and removed common English stop words. The result is a sparse, fixed-length weighted vector for each passage, suitable for the linear and tree-based classifiers.

### 3.5.2 Vocabulary Construction

For the deep learning model, a custom vocabulary was constructed from the training partition alone, comprising the 20,000 most frequent tokens. Two special entries were reserved: a padding token, used to bring all sequences to a common length, and an unknown token, to which any word absent from the vocabulary is mapped at inference time. Each passage was then represented as a sequence of the integer indices of its words, truncated or padded to a fixed length of 150 tokens. Constructing the vocabulary from the training data only ensures that no information from the validation or test partitions can influence the model. The transformer models, by contrast, do not require a constructed vocabulary, since each uses the fixed vocabulary learned during its own large-scale pretraining.

### 3.5.3 Word Embeddings

The deep learning model converts its integer word indices into dense vectors through a learned embedding layer. This layer maps each of the 20,000 vocabulary indices to a continuous 128-dimensional vector, with the padding index fixed to a zero vector so that padding contributes nothing to the computation. Unlike sparse term frequency vectors, these embeddings are dense and are optimised jointly with the rest of the network during training, allowing words that occur in similar contexts to acquire similar representations. The embeddings are learned from the present corpus rather than initialised from an external pretrained set.

### 3.5.4 Transformer Tokenization

The transformer models use the subword tokenizer associated with each pretrained architecture, namely WordPiece for the BERT family and byte level byte pair encoding for the RoBERTa family. Rather than treating each word as an indivisible unit, these tokenizers decompose words into frequently occurring subword fragments drawn from the model's fixed pretrained vocabulary, which allows rare or previously unseen words to be represented as combinations of known fragments and thereby avoids the unknown-word problem that affects fixed word vocabularies. Each passage was tokenized in this way, truncated to a maximum length of 128 tokens, and converted into the sequence of integer input identifiers that the encoder expects.

### 3.5.5 Attention Masks

Alongside the input identifiers, the tokenizer produces an attention mask for each passage. Because passages differ in length, shorter sequences are padded to the common length of their batch, and the attention mask is a binary vector that marks which positions correspond to real tokens and which correspond to padding. During self-attention, the model consults this mask so that padded positions are ignored and do not influence the representation of the genuine tokens. Padding was applied dynamically within each batch, so that sequences were padded only to the length of the longest passage in that batch rather than to a fixed maximum, which reduces unnecessary computation.

## 3.6 Proposed Emotional Burnout Detection Framework

This study proposes an explainable, risk-graded emotional burnout detection framework that operates directly on student authored text. The framework is designed to satisfy three requirements simultaneously, which the reviewed literature was found to address only in isolation, namely accurate multi-class classification of mental-health state, transparent explanation of each decision, and the translation of that decision into an actionable risk indicator for a human advisor. The overall flow of the framework was presented in Figure 3.1. The present section describes the framework as a set of cooperating functional layers and explains the rationale for its structure, while the detailed realisation of each layer is given in the sections that follow.

The framework comprises four functional layers. The classification layer receives a cleaned passage of text and assigns it to one of the seven mental-health categories. This layer is not a single fixed model but the outcome of a controlled comparison of nine candidate models, from which the strongest, a soft voting ensemble of two fine-tuned transformers, is selected as the operational classifier. The risk scoring layer then converts the probabilistic output of the classifier into an ordinal risk level and a continuous distress score, so that the categorical prediction is expressed in terms that an advisor can act upon. The explainability layer extracts, for every prediction, the words that most influenced the decision, so that the reasoning behind an alert is made visible rather than left hidden. Finally, the presentation layer delivers the prediction, the risk indicators, and the explanation to the user through an advisor facing dashboard.

The structure of the framework follows directly from its intended use. Because the system is intended to support, and not to replace, the judgement of an academic advisor, accuracy alone is insufficient, since an advisor must also be able to see why a student has been flagged and how serious the indication is. The framework therefore treats classification, risk interpretation, and explanation as equally necessary components rather than as optional additions to a classifier. This design realises the research objectives stated in Section 3.2, and each of its layers is examined in detail in the sections that follow, namely the candidate model architectures in Section 3.7, the class imbalance handling in Section 3.10, the explainability framework in Section 3.11, and the risk scoring framework in Section 3.12.

## 3.7 Model Architectures

This section describes the architecture of each of the nine models evaluated within the framework. The models were selected to form a graded progression across three families, advancing from classical machine learning, through a deep learning sequence model, to fine-tuned transformer models, with two non-trained baselines included to isolate the contribution of task adaptation. This progression allows the benefit of each successive increase in modelling capacity to be measured rather than assumed. Table 3.4 summarises the complete set, and each model is then described in turn.

| # | Model | Family | Key Configuration | Parameters |
|---|---|---|---|---|
| 1 | Logistic Regression | Classical ML (TF-IDF) | balanced class weights, 1,000 iterations | 10k features |
| 2 | Random Forest | Classical ML (TF-IDF) | 200 trees, balanced class weights | 10k features |
| 3 | Support Vector Machine | Classical ML (TF-IDF) | linear kernel, balanced class weights | 10k features |
| 4 | BiLSTM | Deep Learning (RNN) | 2 bidirectional layers, 128 hidden units | 3,221,255 |
| 5 | VADER | Rule-based (no training) | lexicon, threshold mapping | none |
| 6 | Zero-Shot RoBERTa | Transformer (no training) | BART-large-MNLI, hypothesis labels | none trained |
| 7 | BERT (fine-tuned) | Transformer | bert-base-uncased, 7-way head | 109,487,623 |
| 8 | RoBERTa (fine-tuned) | Transformer | roberta-base, 7-way head | 124,651,015 |
| 9 | BERT + RoBERTa Ensemble | Transformer (soft voting) | equal-weight probability averaging | 234,138,638 |

**Table 3.4.** Overview of the nine models evaluated in the present study.

### 3.7.1 Logistic Regression

Logistic Regression is a linear classifier that estimates, for each category, the probability that a passage belongs to that category as a function of a weighted sum of its term frequency inverse document frequency features. It was configured for multi-class classification across the seven categories, with balanced class weights and up to one thousand optimisation iterations. It provides a simple and interpretable linear reference against which the more complex models are judged.

### 3.7.2 Random Forest

The Random Forest classifier is an ensemble of two hundred decision trees, each trained on a random subset of the data and features, whose individual predictions are combined by majority vote. Unlike the linear models, it is able to capture non-linear interactions between features. It was trained on the same term frequency inverse document frequency representation with balanced class weights.

### 3.7.3 Support Vector Machine

The Support Vector Machine seeks the boundary that maximises the margin between categories in the high-dimensional feature space. A linear kernel was used, operating on the term frequency inverse document frequency features with balanced class weights. It constitutes a strong and widely used classical baseline for text classification.

### 3.7.4 BiLSTM

The bidirectional Long Short Term Memory network is the deep learning model of the study, comprising 3,221,255 trainable parameters. Each word index is mapped to a 128-dimensional embedding, and the sequence, fixed to a length of 150 tokens, is processed by a two layer bidirectional recurrent encoder of 128 hidden units per direction, which reads the passage in both directions so that each word is interpreted in the context of those that precede and follow it. The final hidden states of the two directions are concatenated into a 256-dimensional summary, to which dropout is applied before a fully connected layer produces the seven category scores. Unlike the classical models, this architecture explicitly models the order of words.

### 3.7.5 VADER

VADER is a rule based sentiment analyser that scores a passage using a fixed lexicon of sentiment-bearing words together with grammatical heuristics, without any learning from the training data. Its compound sentiment score was mapped onto the seven categories through a set of thresholds, and a supplementary rule assigns any passage exhibiting a high proportion of strongly negative words to the Suicidal category. It represents the level of performance attainable by lexical rules alone.

### 3.7.6 Zero-Shot RoBERTa

The zero-shot classifier applies a large transformer trained for natural language inference, specifically the BART large model fine-tuned on the multi-genre natural language inference corpus, without any adaptation to the present task. Each of the seven categories is expressed as a short descriptive hypothesis, and the model scores how strongly each hypothesis is entailed by the passage, the most strongly entailed category being taken as the prediction. This baseline measures what a large pretrained model can achieve before any task specific fine-tuning.

### 3.7.7 BERT

BERT is a bidirectional transformer encoder comprising twelve layers, twelve self-attention heads per layer, and a hidden dimension of 768, pretrained on large general corpora. The base uncased variant was used, extended with a classification head that maps the representation of the leading classification token onto the seven categories, giving 109,487,623 trainable parameters. All parameters were updated during fine-tuning so that the general language representation is specialised to the mental-health classification task.

### 3.7.8 RoBERTa

RoBERTa shares the twelve layer, twelve head, 768-dimensional encoder structure of BERT, but it is pretrained under a more robust and extended regime with a larger corpus and a byte level byte pair encoding vocabulary. The base variant was used with an equivalent classification head, giving 124,651,015 trainable parameters, and it was fine-tuned under the same configuration as BERT so that the two architectures may be compared directly.

### 3.7.9 Ensemble

The ensemble is the final and best performing model. For each passage, the fine-tuned BERT and RoBERTa models each produce a seven-dimensional vector of class probabilities, the two vectors are averaged with equal weight, and the category with the highest averaged probability is returned. Because the two transformers differ in their pretraining and tokenisation, they make partially independent errors, and averaging their probabilities allows the more confident model to compensate where the other is uncertain, which yields a modest but reliable improvement over either model used alone.

## 3.8 Mathematical Formulation

This section defines the mathematical operations that are specific to the proposed framework or central to its training. Standard operations that are used without modification, namely the term frequency inverse document frequency weighting of Section 3.5.1, the transformer self-attention mechanism, and the AdamW optimiser, are applied as defined in their original sources and are cited there rather than reproduced here.

**(a) Softmax.** Given the vector of raw output scores z = (z_1, ..., z_C) produced for the C = 7 categories, the predicted probability of category i is

&nbsp;&nbsp;&nbsp;&nbsp;p_i = exp(z_i) / Sum_j exp(z_j)&nbsp;&nbsp;&nbsp;&nbsp;(3.1)

which converts the scores into positive values that sum to one. These probabilities are the input to both the risk score and the ensemble defined below.

**(b) Cross-Entropy Loss.** For a passage whose true category is y, with predicted probabilities p, and a training batch of B passages, the objective minimised by the deep learning and transformer models is

&nbsp;&nbsp;&nbsp;&nbsp;L = - (1/B) Sum_b log(p_{b, y_b})&nbsp;&nbsp;&nbsp;&nbsp;(3.2)

In the class-weighted experiment of Section 3.10, each term is additionally scaled by the class weight w_y, giving L = - (1/B) Sum_b w_{y_b} log(p_{b, y_b}), so that errors on rare categories contribute more heavily to the loss.

**(c) Risk Score.** The continuous distress score is computed from the predicted class probabilities as

&nbsp;&nbsp;&nbsp;&nbsp;Risk = 100 x ( 1.00 p_Suicidal + 0.80 p_Depression + 0.75 p_Bipolar + 0.70 p_Personality + 0.60 p_Anxiety + 0.50 p_Stress + 0.00 p_Normal )&nbsp;&nbsp;&nbsp;&nbsp;(3.3)

so that categories of greater clinical severity contribute more heavily and the score lies on a scale from 0 to 100.

**(d) Weighted Ensemble.** For the final model, the probability assigned to category c is the equally weighted average of the two fine-tuned transformer outputs,

&nbsp;&nbsp;&nbsp;&nbsp;p_c^ens = w_B p_c^BERT + w_R p_c^RoBERTa,&nbsp;&nbsp;&nbsp;&nbsp;w_B = w_R = 0.5&nbsp;&nbsp;&nbsp;&nbsp;(3.4)

and the predicted category is the one with the highest averaged probability.

## 3.9 Training Configuration and Hyperparameter Selection

The two fine-tuned transformer models, which constitute the components of the final ensemble, were trained under a single common configuration so that any difference in their performance reflects the architectures themselves rather than the training regime. Optimisation used the AdamW optimiser at a learning rate of 2e-5 with a weight decay of 0.01, a batch size of 32, and half precision arithmetic, for three epochs. The models were evaluated on the validation set at the end of each epoch, and the checkpoint achieving the highest macro F1 score was retained as the final model. The values of these hyperparameters, and the justification for each, are recorded in Table 3.5.

| Hyperparameter | Value | Justification |
|---|---|---|
| Optimiser | AdamW | The standard optimiser for transformer fine-tuning, combining adaptive learning rates with decoupled weight decay. |
| Learning rate | 2e-5 (linearly decayed) | Within the narrow range recommended for fine-tuning pretrained transformers; small enough to adapt the model without erasing its pretrained knowledge. |
| Maximum sequence length | 128 tokens | The median passage is 61 words and most passages fit within 128 subword tokens, so 128 preserves the great majority of content while limiting memory and computation. |
| Batch size | 32 | The largest batch that fits the 8 GB memory of the RTX 4060 under half precision while keeping training stable. |
| Training epochs | 3 | Pretrained transformers adapt quickly; three epochs were sufficient, and further epochs mainly increase the risk of overfitting, which is guarded against by validation-based checkpoint selection. |
| Weight decay | 0.01 | Standard regularisation that discourages large weights and improves generalisation. |
| Mixed precision | fp16 | Halves memory use and accelerates training on the GPU with negligible effect on accuracy. |
| Model selection metric | Macro F1 | Averages all seven categories equally, so the retained checkpoint is the one that is fairest to the rare categories under the strong class imbalance. |

**Table 3.5.** Training configuration and the justification for each hyperparameter of the fine-tuned transformer models.

Four choices merit specific comment. The learning rate of 2e-5 lies within the narrow band recommended for fine-tuning BERT and RoBERTa, since a larger rate risks destabilising the pretrained weights while a much smaller rate slows convergence without benefit. The maximum sequence length of 128 tokens is justified directly by the data, because the median passage contains only 61 words and the great majority fall within 128 subword tokens, so little content is lost while memory and computation are kept manageable. The batch size of 32 is the largest that the 8 GB memory of the RTX 4060 accommodates under half precision arithmetic, and it provides stable gradient estimates. Finally, three epochs were used because a model that already possesses a strong pretrained representation adapts to the target task quickly; validation performance was monitored at each epoch and the best checkpoint was retained, so that additional epochs, which would chiefly increase the risk of overfitting, were unnecessary. The deep learning model was trained separately with the Adam optimiser at a learning rate of 0.001 for five epochs, and the classical baselines used the standard settings of their respective algorithms.

## 3.10 Class Imbalance Handling

The corpus is strongly imbalanced, as reported in Section 3.3.4, with the largest category exceeding the smallest by a ratio of approximately eighteen to one. If left unaddressed, such imbalance would encourage a model to favour the majority categories at the expense of the rare ones. The present study addressed this imbalance through two measures applied to the final models, and evaluated a third as a controlled experiment.

For the classical machine learning baselines, balanced class weights were applied during fitting. Under this scheme the contribution of each category to the training objective is scaled inversely to its frequency, so that the rare categories exert an influence comparable to that of the common ones.

For the fine-tuned transformer models, which form the final ensemble, the imbalance was addressed at the point of model selection. Rather than retaining the checkpoint with the highest overall accuracy, which would reward a model that performs well only on the majority categories, the checkpoint with the highest macro F1 score on the validation set was retained. Because macro F1 averages the seven categories with equal weight, this criterion selects the model that is fairest across all categories, including the rare ones.

In addition, an explicit class-weighted loss was evaluated as a controlled experiment on the RoBERTa model. A custom training routine was implemented by overriding the loss computation of the standard trainer, so that the cross-entropy loss was weighted by the inverse frequency of each class, as given in Equation (3.2). The weights, computed from the training distribution by the balanced heuristic and reported in Table 3.6, ranged from 0.46 for the most frequent category to 8.16 for the rarest, so that an error on the Personality Disorder category contributed almost eighteen times more to the loss than an error on the Normal category. This weighted variant was trained under the configuration of Section 3.9 and compared against the unweighted model. As reported in the Results, the weighting increased the recall of the minority categories but reduced overall accuracy and macro F1, which is a precision-recall trade-off, and the unweighted model was therefore retained as the final model. The weighted variant is reported as an ablation.

| Class | Training Samples | Inverse-Frequency Weight |
|---|---|---|
| Personality Disorder | 626 | 8.16 |
| Stress | 1,605 | 3.18 |
| Bipolar | 1,751 | 2.92 |
| Anxiety | 2,532 | 2.02 |
| Suicidal | 7,444 | 0.69 |
| Depression | 10,559 | 0.48 |
| Normal | 11,221 | 0.46 |

**Table 3.6.** Inverse-frequency class weights used in the weighted-loss experiment, computed from the training partition.

## 3.11 Explainable AI Framework

A central design requirement of the framework is that every prediction should be accompanied by a human readable explanation, so that an advisor can see the specific evidence on which a classification was based. The framework provides two complementary views of the model's reasoning: a local view, which explains a single prediction, and a global view, which summarises the model's behaviour across the whole test set. Both are derived from the self-attention mechanism of the fine-tuned transformer.

### 3.11.1 Local Attention Extraction

For an individual passage, the explanation is derived from the encoder's self-attention. During inference the encoder is configured to return its attention tensors, which are stacked across all twelve encoder layers and averaged across both the layers and the twelve attention heads to yield a single aggregated attention map. The row of this map corresponding to the leading classification token is taken as the importance assigned to each input token, since that token aggregates information from the whole sequence in order to form the classification. The special classification and separator tokens are removed, subword fragments are merged back into whole words with their attention summed, and the remaining weights are normalised to sum to one, so that the importance values form a proper distribution over the words of the passage.

### 3.11.2 Attention Visualisation

The normalised local weights are rendered as a colour-coded heatmap in which more heavily attended words appear in warmer colours. For a representative input such as "I feel exhausted and hopeless", the words carrying the clearest emotional signal receive the greatest attention, which confirms that the model based its decision on relevant evidence rather than on incidental words. This heatmap accompanies every classification in the advisor dashboard and is the explanation an advisor sees for each individual student. An example is shown in Figure 3.3.

### 3.11.3 Global Class-Level Attention Analysis

To move beyond single predictions and understand the behaviour of the model as a whole, the attention was also aggregated across the entire test set. For every passage the word-level attention was computed as in Section 3.11.1, function words were removed, and the attention mass of each remaining content word was accumulated separately for each of the seven classes. Ranking the words by their total accumulated attention within each class yields, for that class, the vocabulary that most consistently drives the model towards it. This provides a population-level, global explanation that complements the single-passage view, and it allows the reasoning of the model to be inspected and validated against clinical expectation rather than trusted blindly. The result of this analysis is reported in Section 4.11.

## 3.12 Risk Scoring Framework

Beyond the categorical prediction, the framework translates each result into an interpretable risk summary for the advisor. Each of the seven categories is mapped to one of four ordinal risk levels, from Low for the Normal category to Critical for the Suicidal category, as set out in Table 3.7. In addition, a single continuous distress score on a scale of 0 to 100 is computed as the weighted sum of the predicted class probabilities defined in Equation (3.3), in which the weight assigned to each category reflects its clinical severity. The Suicidal probability carries the full weight of one, the Depression probability a weight of 0.8, and the remaining categories progressively smaller weights, with the Normal category contributing nothing. This score condenses the full probability distribution into a single ordered quantity that an advisor can monitor at a glance, and a prediction of the Suicidal category additionally raises an explicit critical alert within the dashboard.

| Category | Risk Level | Severity Weight |
|---|---|---|
| Normal | Low | 0.00 |
| Stress | Medium | 0.50 |
| Anxiety | Medium | 0.60 |
| Personality Disorder | High | 0.70 |
| Bipolar | High | 0.75 |
| Depression | High | 0.80 |
| Suicidal | Critical | 1.00 |

**Table 3.7.** Mapping of categories to ordinal risk levels and to the severity weights used in the distress score.

## 3.13 Lead-Time Analysis

To investigate how early a reliable warning could be raised, the present study conducted a lead-time analysis structured around four notional points across a semester, corresponding to Weeks 2, 4, 8, and 12. The nature of this analysis must be stated precisely. The corpus does not contain time-stamped writing from individual students tracked over a semester, so the analysis is a controlled simulation rather than a longitudinal study. For each of the four points, an independent random sample of Normal and Depression texts was drawn from the corpus and classified by the fine-tuned BERT model, and the classification performance was recorded. The weeks remaining values attached to each point express the size of the intervention window that early detection would afford, and they function as a framing device rather than as a measured longitudinal outcome. The analysis therefore demonstrates that the model separates ordinary from distressed language with stable and high accuracy at any point at which such language appears, from which the practical value of early detection is inferred. A qualitative companion to this analysis presents four illustrative sentences, authored to depict a plausible progression from stability to crisis, together with the prediction and attention heatmap produced for each. These sentences are illustrative and are not drawn from real students.

## 3.14 Evaluation Metrics and Experimental Protocol

All models were trained and evaluated under a common protocol so that their comparison is fair. The models were trained in order of increasing complexity, from the classical baselines, through the deep learning model, to the fine-tuned transformers, and every model used the identical stratified partition described in Section 3.4, with the same fixed random seed, so that no model enjoyed an easier split than another. The transformer models were validated at the end of each training epoch, and the checkpoint with the highest validation macro F1 was selected as the final model, which serves as an implicit early-stopping criterion that guards against overfitting.

All models were then evaluated on the common held-out test set of 7,659 passages. Overall performance is reported as classification accuracy and as the F1 score, and for the transformer models both the weighted F1 and the macro F1 are recorded, since the weighted F1 reflects performance across the corpus as a whole while the macro F1 averages the seven categories equally and therefore exposes weakness on the rare categories that the weighted measure can conceal. Per-class precision, recall, and F1 are reported for the best performing model so that its behaviour on each category can be examined individually, and a confusion matrix is produced to reveal the categories between which the model most often confuses its predictions. To confirm that the reported baseline performance is not an artefact of a single fortunate partition, the Logistic Regression and Support Vector Machine baselines were additionally subjected to five-fold stratified cross validation, in which the feature extraction and the classifier were enclosed within a single pipeline and refitted on each fold to prevent information leakage. The use of an identical test partition for every model, together with this cross validation of the baselines, ensures that all reported comparisons are made on equal and stable terms.

## 3.15 Implementation Environment and Reproducibility

The system was implemented in Python. The classical baselines and the evaluation metrics were built with the scikit-learn library, the deep learning and transformer models were implemented in PyTorch, and the pretrained models, tokenizers, and training utilities were provided by the Hugging Face Transformers and Datasets libraries. The VADER baseline used the vaderSentiment package, and all figures were produced with Matplotlib and Seaborn. Model training and evaluation were performed locally on a workstation equipped with an NVIDIA RTX 4060 graphics processing unit with 8 GB of memory. To support reproducibility, a fixed random seed was used for every data partition and stochastic operation, so that the training, validation, and test splits are identical on each run. The trained model weights, the preprocessing and training code, the notebooks, and the evaluation scripts are all maintained under version control with large file storage in the project repository. Given the same data, code, and seed, the reported results can be reproduced.

## 3.16 Computational Considerations

The computational cost of each model was recorded so that the practical feasibility of the framework can be assessed. Table 3.8 summarises the principal figures. The classical baselines and the deep learning model are comparatively small, the latter comprising 3.2 million parameters, whereas the fine-tuned transformers are substantially larger, with 109.5 million parameters for BERT and 124.7 million for RoBERTa, and 234.1 million for the ensemble that combines them. Each transformer was fine-tuned for three epochs in approximately ten to eleven minutes on the RTX 4060 under half precision, and the batch size of 32 was chosen as the largest that fits within the 8 GB of available memory at this precision. At inference, a single passage is classified by the BERT model in approximately 23 milliseconds and by the full ensemble in approximately 50 milliseconds on the same hardware, which is well within the requirements of an interactive advisor tool and confirms that the additional cost of the ensemble over a single model is modest in absolute terms.

| Model | Parameters | Approximate Training Time | Inference per Passage |
|---|---|---|---|
| Logistic Regression, Random Forest, SVM | 10,000 TF-IDF features | under one minute | under 1 ms |
| BiLSTM | 3,221,255 | a few minutes (5 epochs) | a few ms |
| BERT (fine-tuned) | 109,487,623 | approximately 10 minutes (3 epochs) | 23 ms |
| RoBERTa (fine-tuned) | 124,651,015 | approximately 11 minutes (3 epochs) | within ensemble |
| BERT + RoBERTa Ensemble | 234,138,638 | sum of the two transformers | 50 ms |

**Table 3.8.** Parameter count, training time, and inference latency of the models, measured on an NVIDIA RTX 4060.

## 3.17 Deployment Architecture

To demonstrate the practical use of the framework, the best model was deployed as an interactive advisor tool. A lightweight Python inference script loads the trained model and, given a passage of text, returns a structured response containing the predicted category, its confidence, the associated risk level, the distress score, the full class probability distribution, and the token level attention. This script is exposed through a web application that serves as the advisor facing dashboard, in which the prediction, the colour-coded risk indicators, the probability distribution, and the attention heatmap are rendered for the user, and in which a Critical prediction raises a visible alert. A second, simpler interface was also provided for the direct testing of individual passages. The deployment confirms that the framework operates end to end, from raw text to explained, risk-graded output, in a form usable by a non-technical advisor.

## 3.18 Ethical Considerations

The data used in this study are drawn from a publicly available, openly licensed dataset in which contributors wrote without any expectation of individual identification, and no personally identifiable information is retained in the working corpus. The subject matter is nonetheless sensitive, and the framework is treated accordingly. It is designed strictly as a decision-support tool for academic advisors and qualified staff. It is not a diagnostic instrument, and it does not replace professional clinical judgement. Every alert generated by the system, and in particular any classification into the Suicidal category, requires human review before any action is taken, and no automated consequence of any kind is triggered by a prediction. The explainability framework of Section 3.11 reinforces this principle by ensuring that every prediction is accompanied by a human readable reason, which supports informed and accountable decisions rather than deference to an opaque output. A clear disclaimer to this effect accompanies the system, and all model weights and code are maintained under version control to support transparency.

## 3.19 Limitations of the Methodology

Several methodological limitations are acknowledged. First, the corpus is drawn entirely from public social media text, which differs in register and context from the writing that students produce within an academic institution, so the transfer of performance to a genuine institutional setting cannot be assumed and requires validation on an independent, consented dataset. Such a cross-dataset evaluation, in which the trained models would be applied without further adaptation to a separate corpus collected under different conditions, is identified as the principal direction for future work. Second, the data is exclusively in English, and the framework has not been evaluated on other languages. Third, the lead-time analysis of Section 3.13 is a controlled simulation rather than a study of real students tracked over time, since no time-stamped longitudinal student data was available, and its conclusions should therefore be read as a demonstration of detection capability rather than as validated longitudinal evidence. Fourth, despite the imbalance handling of Section 3.10, the rarest categories remain the hardest to classify, reflecting the limited number of examples available for them. These limitations do not undermine the contributions of the framework, but they define the boundary within which its results should be interpreted and the work that remains before real deployment.

---

## References to insert (replace each [x] with your numbered citation)

- 3.2 Design Science Research: Hevner, March, Park and Ram (2004)
- 3.3.1 Dataset: Sarkar, S. *Sentiment Analysis for Mental Health* [Dataset]. Kaggle. https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health
- 3.7.4 LSTM: Hochreiter and Schmidhuber (1997)
- 3.7.5 VADER: Hutto and Gilbert (2014)
- 3.7.6 Zero-shot NLI / BART: Lewis et al. (2020); Yin, Hay and Roth (2019)
- 3.7.7 BERT: Devlin, Chang, Lee and Toutanova (2019)
- 3.7.8 RoBERTa: Liu et al. (2019)
- 3.8 / 3.9 Self-attention and AdamW: Vaswani et al. (2017); Loshchilov and Hutter (2019)
- 3.15 scikit-learn: Pedregosa et al. (2011); PyTorch: Paszke et al. (2019); Transformers: Wolf et al. (2020)

## Figure placement guide (add your result figures here)

- Figure 3.1 (Section 3.1): architecture diagram. Redraw the pipeline above neatly in Word SmartArt or draw.io.
- Figure 3.2 (Section 3.3.4): class distribution. Use `results/figures/fig3_2.png`.
- Figure 3.3 (Section 3.11.2): attention heatmap example. Use `results/figures/attention_heatmap_words.png` (the same colour-coded figure used in Results; show it once, wherever you prefer).
- Optional in Section 3.13: `results/figures/fig4_5.png` and `results/figures/fig4_6.png`.
- The confusion matrices and the all-models comparison chart belong in your Results section, not here.

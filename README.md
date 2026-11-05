# ML Assignment - Early Stage Diabetes Classification

## Problem Statement

Early detection of Type 2 Diabetes Mellitus is critical for initiating timely
clinical intervention and preventing complications such as retinopathy,
nephropathy, neuropathy, and cardiovascular disease.

The objective of this project is to design, implement, evaluate, and compare
five supervised machine learning classification models (Logistic Regression,
Decision Tree Classifier, K-Nearest Neighbor Classifier, Naive Bayes Classifier,
and Support Vector Machine) to predict whether a patient is at risk for
early-stage diabetes (`Positive` vs. `Negative`) from demographic and clinical
symptom indicators.

## Dataset Description

-   **Dataset Name**: Early Stage Diabetes Risk Prediction
-   **Dataset Link**:
    [UCI ML Repository — Early Stage Diabetes Risk Prediction](https://archive.ics.uci.edu/dataset/529/early+stage+diabetes+risk+prediction+dataset)
-   **Dataset Dimensions**: `520` instances and `17` attributes (`16` input
    features + `1` output target class)
-   **Missing Values**: **0 missing values** across all 17 columns
-   **Train / Test Split**: 80% / 20% split between training set and test set
    respectively
-   **Target Class Distribution**:
    -   `Positive` (Diabetic Risk, encoded as `1`): **320 instances (61.54%)** —
        256 in training set, 64 in test set
    -   `Negative` (Healthy Control, encoded as `0`): **200 instances (38.46%)**
        — 160 in training set, 40 in test set

## Github Repository Link

[https://github.com/anandshah05/aiml-ml-assignment2](https://github.com/anandshah05/aiml-ml-assignment2)

### How to Run Locally

```bash
# 1. Install dependencies
pip3 install -r requirements.txt

# 2. Train models and regenerate evaluation artifacts
python3 -P train_models.py

# 3. Launch the interactive Streamlit web application
python3 -m streamlit run app.py
```

## Models used

All five classification models were trained on the exact same stratified
training split ($n_{\text{train}} = 416$) and evaluated on the held-out test
split ($n_{\text{test}} = 104$: 64 Positive, 40 Negative):

1.  **Logistic Regression**: L2-regularized binary logistic classifier (`C=1.0`,
    `max_iter=1000`, `random_state=42`).
2.  **Decision Tree Classifier**: CART decision tree using Gini impurity
    splitting criterion (`criterion='gini'`, `random_state=42`).
3.  **K-Nearest Neighbor Classifier (KNN)**: Distance-based neighborhood voting
    classifier (`n_neighbors=5`, `Euclidean L2 metric`).
4.  **Naive Bayes Classifier**: Gaussian Naive Bayes (`GaussianNB`) modeling
    class-conditional feature likelihoods (also benchmarked against
    `MultinomialNB` on unscaled non-negative counts).
5.  **Support Vector Machine (SVM)**: Radial Basis Function (RBF) kernel
    classifier with Platt probability calibration (`kernel='rbf'`, `C=1.0`,
    `gamma='scale'`, `probability=True`, `random_state=42`).

### Model Evaluation Metrics Comparison Table

ML Model Name          | Accuracy   | AUC        | Precision  | Recall     | F1         | MCC
---------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ---
Logistic Regression    | 0.9423     | 0.9902     | 0.9833     | 0.9219     | 0.9516     | 0.8832
Decision Tree          | **0.9904** | 0.9922     | **1.0000** | **0.9844** | **0.9921** | **0.9800**
KNN                    | 0.9327     | 0.9697     | 0.9831     | 0.9062     | 0.9431     | 0.8653
Naive Bayes            | 0.9423     | 0.9863     | 0.9677     | 0.9375     | 0.9524     | 0.8800
Support Vector Machine | 0.9808     | **0.9973** | 0.9844     | **0.9844** | 0.9844     | 0.9594

### Observations

| ML Model Name                     | Observation about model performance      |
| --------------------------------- | ---------------------------------------- |
| Logistic Regression               | Delivers strong results with `94.23%`    |
:                                   : accuracy and high                        :
:                                   : precision.<br>Identifies clear symptoms  :
:                                   : well, but misses a few cases with        :
:                                   : complex symptom combinations.            :
| Decision Tree                     | Best performing model with `99.04%`      |
:                                   : accuracy and zero false                  :
:                                   : positives.<br>Simple if-else decision    :
:                                   : rules match the yes/no symptom questions :
:                                   : perfectly.                               :
| KNN                               | Achieves `93.27%` accuracy, ranking lowest |
:                                   : among the five models.<br>Treating all   :
:                                   : symptoms equally allows less important   :
:                                   : symptoms to reduce accuracy.             :
| Naive Bayes                       | Delivers fast and reliable results with  |
:                                   : `94.23%` accuracy and good                 :
:                                   : recall.<br>Assuming symptoms are         :
:                                   : independent leads to a few false alarms  :
:                                   : when symptoms overlap.                   :
| Support Vector Machine            | Outstanding performance with `98.08%`    |
:                                   : accuracy and the highest AUC score of    :
:                                   : `0.9973`.<br>Accurately separates positive :
:                                   : and negative cases even with complex     :
:                                   : symptom patterns.                        :
| Overall, Winner for your dataset? | Decision Tree is the overall winner with |
:                                   : `99.04%` accuracy and near-perfect       :
:                                   : predictions.<br>Support Vector Machine   :
:                                   : is a close runner-up with the best       :
:                                   : probability ranking.                     :

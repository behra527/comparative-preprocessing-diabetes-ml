# Comparative Evaluation of Preprocessing Choices

A controlled machine learning experiment investigating how different preprocessing and class-imbalance strategies affect diabetes classification performance using the **Pima Indians Diabetes Dataset**.

The project compares multiple preprocessing configurations using the same **Logistic Regression** classifier, fixed train-test split, and consistent evaluation methodology.

---

## Project Objective

The primary objective is to understand how preprocessing decisions influence downstream machine learning performance, particularly the detection of the minority **diabetic class**.

The experiment evaluates:

* Invalid-value treatment and median imputation
* Feature scaling
* SMOTE-based oversampling
* Class weighting
* Classification threshold adjustment
* Clinical feature influence

The project emphasizes **recall, F1-score, and false-negative reduction** rather than relying only on accuracy.

---

## Dataset

**Dataset:** Pima Indians Diabetes Dataset

The dataset contains clinical measurements used to predict whether a patient has diabetes.

### Features

| Feature                  | Description                                      |
| ------------------------ | ------------------------------------------------ |
| Pregnancies              | Number of pregnancies                            |
| Glucose                  | Plasma glucose concentration                     |
| BloodPressure            | Diastolic blood pressure                         |
| SkinThickness            | Triceps skin fold thickness                      |
| Insulin                  | Serum insulin level                              |
| BMI                      | Body mass index                                  |
| DiabetesPedigreeFunction | Diabetes pedigree/family-history-related measure |
| Age                      | Patient age                                      |
| Outcome                  | Diabetes classification target                   |

### Target

`Outcome`

* `0` → Non-Diabetic
* `1` → Diabetic

### Dataset Size

* **768 observations**
* **8 predictor variables**
* **1 target variable**

---

## Problem Statement

Clinical machine learning datasets can contain invalid values, missing information, features with different numerical scales, and imbalanced target classes.

These characteristics can affect model behavior.

For this dataset, zero values in selected clinical variables are treated as invalid or missing observations where clinically appropriate. The project then investigates whether different preprocessing and imbalance-handling strategies improve diabetes classification performance.

---

## Experimental Design

To ensure a fair comparison, all experiments use the same:

* Dataset
* Logistic Regression classifier
* 80/20 train-test split
* Stratified sampling
* `random_state=42`
* Evaluation metrics
* Test set

Only the preprocessing or imbalance-handling strategy changes between experiments.

### Experimental Configurations

| Experiment        | Imputation | Scaling        | Imbalance Handling |
| ----------------- | ---------- | -------------- | ------------------ |
| E0  Baseline      | Median     | No             | None               |
| E1  Scaling       | Median     | StandardScaler | None               |
| E2 SMOTE          | Median     | StandardScaler | SMOTE              |
| E3  Class Weight  | Median     | StandardScaler | Class Weight       |

All preprocessing operations are implemented using pipelines to reduce the risk of data leakage.

---

## Evaluation Metrics

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix
* False-Positive Rate
* False-Negative Rate
* ROC-AUC
* PR-AUC

Because this is a clinical classification problem, particular attention is given to:

**Diabetic-class Recall → F1-score → False Negatives**

---

## Results

### Overall Performance

| Experiment        |   Accuracy |  Precision |     Recall |         F1 | ROC-AUC |     PR-AUC |
| ----------------- | ---------: | ---------: | ---------: | ---------: | ------: | ---------: |
| E0   Baseline     |     70.13% |     58.70% |     50.00% |     54.00% |  0.8130 |     0.6723 |
| E1   Scaling      |     70.78% |     60.00% |     50.00% |     54.55% |  0.8130 |     0.6733 |
| E2   SMOTE        |     70.78% |     57.38% |     64.81% |     60.87% |  0.8113 | **0.6758** |
| E3   Class Weight | **73.38%** | **60.32%** | **70.37%** | **64.96%** |  0.8126 |     0.6727 |

---

## Key Findings

### 1. Scaling Alone

Adding `StandardScaler` produced only a small improvement.

* Accuracy: **70.13% → 70.78%**
* Recall remained at **50.00%**
* F1 increased slightly from **54.00% → 54.55%**

This suggests that scaling alone did not substantially improve minority-class detection in this experiment.

### 2. SMOTE

SMOTE produced a substantial improvement in diabetic-class recall.

* Recall: **50.00% → 64.81%**
* F1: **54.00% → 60.87%**
* False negatives: **27 → 19**

However, precision decreased to **57.38%**, demonstrating the trade-off between detecting more diabetic cases and generating additional false positives.

### 3. Class Weighting

Class weighting produced the strongest overall result among the tested configurations.

* Accuracy: **73.38%**
* Diabetic precision: **60.32%**
* Diabetic recall: **70.37%**
* Diabetic F1: **64.96%**
* False negatives: **16**
* False-negative rate: **29.63%**

Compared with the baseline, diabetic recall improved by **20.37 percentage points**.

---

## Confusion Matrix Comparison

| Experiment        | TN | FP | FN | TP |
| ----------------- | -: | -: | -: | -: |
| E0   Baseline     | 81 | 19 | 27 | 27 |
| E1   Scaling      | 82 | 18 | 27 | 27 |
| E2   SMOTE        | 74 | 26 | 19 | 35 |
| E3   Class Weight | 75 | 25 | 16 | 38 |

The E3 configuration identifies **38 of 54 diabetic cases**, compared with only **27 of 54** detected by the baseline.

---

## Threshold Analysis

The E3 model was further evaluated using different classification thresholds.

| Threshold | Precision |     Recall |         F1 |
| --------: | --------: | ---------: | ---------: |
|      0.30 |    54.55% | **88.89%** |     67.61% |
|      0.40 |    56.96% |     83.33% | **67.67%** |
|      0.50 |    60.32% |     70.37% |     64.96% |
|      0.60 |    59.62% |     57.41% |     58.49% |

Lowering the threshold increases diabetic-case recall but also increases false-positive predictions.

Among the tested thresholds, **0.40 produced the highest F1-score**, while **0.30 produced the highest recall**.

These threshold results are based on the fixed evaluation set and should not be interpreted as clinical validation.

---

## Advanced Evaluation

ROC-AUC values were very similar across all configurations:

* E0: **0.8130**
* E1: **0.8130**
* E2: **0.8113**
* E3: **0.8126**

This indicates that preprocessing choices did not substantially change the model's overall ranking/discrimination ability.

PR-AUC showed a similarly small difference, with E2 achieving the highest value:

**E2 PR-AUC = 0.6758**

Therefore, the largest practical differences between the configurations were observed in **threshold-dependent metrics**, particularly diabetic recall, F1-score, and false-negative reduction.

---

## Clinical Feature Interpretation

The selected E3 Logistic Regression model was analyzed using its standardized coefficients.

Coefficient analysis provides an indication of the relative predictive influence of the clinical variables within the fitted model.

Important interpretation principles:

* Positive coefficient  increases model tendency toward the diabetic class.
* Negative coefficient  decreases model tendency toward the diabetic class.
* Larger absolute coefficient  stronger model influence.
* Coefficients represent predictive associations, not causal relationships.

The analysis focuses on clinically relevant variables such as:

* Glucose
* BMI
* Age
* DiabetesPedigreeFunction
* BloodPressure
* Pregnancies
* SkinThickness
* Insulin

---

## Why Categorical Encoding Was Not Used

The original task description included categorical encoding as a possible preprocessing technique.

However, the predictor variables used in this dataset are numerical. Therefore, categorical encoding was **not required** for the implemented experiments.

Adding unnecessary encoding would not provide a meaningful preprocessing comparison for this dataset.

---

## Limitations

This project has several limitations:

1. Evaluation is based on a single fixed train-test split.
2. Model performance may vary across different data splits.
3. The Pima dataset may not represent all clinical populations.
4. Logistic Regression assumes a linear relationship between predictors and the log-odds of the outcome.
5. Correlated clinical variables can share predictive information.
6. Feature coefficients represent model associations rather than causal relationships.
7. Threshold performance was evaluated on the fixed test set.
8. The results do not constitute clinical validation.
9. External validation and calibration would be required before real-world deployment.

---

## Technologies Used

* Python
* Google Colab
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Imbalanced-learn
* Logistic Regression
* SMOTE
* StandardScaler
* Pipeline

---

## Project Structure

```text
comparative-preprocessing-diabetes-ml/
│
├── README.md
├── diabetes.csv
└── comparative_preprocessing.ipynb
```

> Dataset files and notebook names may vary depending on the final repository structure.

---

## Conclusion

This experiment demonstrates that preprocessing decisions can have different effects on machine learning performance.

Feature scaling alone produced only a minor improvement, while both SMOTE and class weighting substantially improved diabetic-class recall.

Among the tested configurations, **E3 — Class Weight** provided the strongest overall balance:

* **Accuracy:** 73.38%
* **Diabetic Recall:** 70.37%
* **Diabetic F1:** 64.96%
* **ROC-AUC:** 0.8126
* **False Negatives:** 16

The experiment also demonstrates that preprocessing and classification threshold selection influence model behavior differently.

Overall, the project builds practical intuition for selecting preprocessing strategies in clinical machine learning while emphasizing that benchmark performance should not be confused with clinical readiness.

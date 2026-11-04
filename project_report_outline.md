# Early Prediction of Drug-Induced Autoimmune Reactions Using AI
### Final Year Project Report — Outline

---

## Abstract
Drug-induced autoimmune reactions (e.g., drug-induced lupus, autoimmune
hepatitis) are rare but serious adverse events that are difficult to
anticipate using traditional clinical judgment alone. This project
proposes a machine-learning-based system that predicts a patient's
risk of developing such a reaction before it occurs, using patient
demographics, drug exposure details, genetic/phenotypic risk markers,
and baseline laboratory values. The system further incorporates
explainable AI (SHAP) to provide per-patient reasoning, and is
demonstrated through an interactive web dashboard.

## 1. Introduction
- Motivation: adverse drug reactions (ADRs) are a leading cause of
  hospitalization; autoimmune-type reactions are under-studied and
  hard to predict.
- Problem statement: early, individualized risk prediction to support
  (not replace) clinical decision-making.
- Objectives:
  1. Build a predictive model for drug-induced autoimmune reaction risk.
  2. Handle rare-event class imbalance appropriately.
  3. Provide explainability for clinical trust.
  4. Deliver an interactive demo dashboard.

## 2. Literature Survey
(Fill in with 8–10 papers you read — suggested search terms:
"drug-induced lupus prediction machine learning", "adverse drug
reaction prediction deep learning", "FAERS machine learning",
"pharmacovigilance AI", "HLA drug hypersensitivity prediction",
"explainable AI healthcare SHAP")

| # | Paper / Source | Method Used | Key Finding | Gap Addressed by This Project |
|---|-----------------|-------------|-------------|-------------------------------|
| 1 |  |  |  |  |
| 2 |  |  |  |  |
| 3 |  |  |  |  |

## 3. Dataset
- **Type:** Literature-informed synthetic dataset (8,000 patient-drug
  records), generated via `src/generate_dataset.py`.
- **Rationale for synthetic data:** real linked patient-level data
  (MIMIC-III/IV, FAERS with lab linkage) requires credentialed access
  / data-use agreements not obtainable within project timelines.
  Feature relationships are grounded in published drug-induced
  autoimmunity risk factors.
- **Features (16 total):**
  - Demographics: age, sex, BMI, smoking status
  - Drug details: drug name, drug class, dosage (mg/day), duration of use
  - Genetic/phenotypic markers: slow acetylator (NAT2) status, HLA risk allele
  - Clinical history: prior autoimmune history, family autoimmune history
  - Concurrent medication count
  - Baseline labs: ANA titer, CRP, WBC count
- **Target:** binary label — developed a drug-induced autoimmune
  reaction (1) or not (0)
- **Class balance:** ~15% positive class (rare-event setting)

## 4. Methodology

### 4.1 Preprocessing
- Numeric features standardized (StandardScaler)
- Categorical features (sex, drug name, drug class) one-hot encoded
- Binary/flag features passed through unchanged
- 80/20 stratified train-test split

### 4.2 Handling Class Imbalance
- Class-weighted loss (`class_weight="balanced"` for Logistic
  Regression / Random Forest; `scale_pos_weight` for XGBoost)
- Evaluation prioritizes Precision/Recall/F1/PR-AUC over accuracy,
  since accuracy is misleading for rare positive classes.

### 4.3 Models Trained
1. Logistic Regression (interpretable baseline)
2. Random Forest (non-linear, robust to feature scaling)
3. XGBoost (gradient boosting, often best for tabular data)

Model selection criterion: **PR-AUC** (Precision-Recall AUC), the most
appropriate metric for imbalanced/rare-event classification, since
ROC-AUC can look optimistic when negatives vastly outnumber positives.

### 4.4 Explainability
- SHAP (SHapley Additive exPlanations) TreeExplainer applied to the
  selected model.
- Global explanation: `outputs/shap_summary.png`, `outputs/shap_bar.png`
- Local (per-patient) explanation: generated live in the dashboard
  for each prediction, showing which factors pushed risk up/down.

### 4.5 System Architecture
```
[Patient/Drug Data Input]
        │
        ▼
[Preprocessing Pipeline: scaling + encoding]
        │
        ▼
[Trained ML Model (Random Forest / XGBoost)]
        │
        ├──► [Risk Probability Score]
        │
        └──► [SHAP Explainer] ──► [Per-Patient Explanation]
        │
        ▼
[Streamlit Dashboard: risk score + explanation for clinician/researcher]
```

## 5. Results

*(Fill in with your actual run's numbers from `models/metrics.json`
— example from a sample run below; regenerate before submission.)*

| Model | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| Logistic Regression | 0.257 | 0.519 | 0.344 | 0.673 | 0.263 |
| Random Forest        | 0.327 | 0.312 | 0.320 | 0.676 | 0.276 |
| XGBoost               | 0.238 | 0.300 | 0.265 | 0.634 | 0.228 |

**Best model:** Random Forest (selected by PR-AUC)

- ROC and Precision-Recall curves: `outputs/roc_pr_curves.png`
- Confusion matrix: `outputs/confusion_matrix.png`
- Feature importance: `outputs/feature_importance.png`
- SHAP global summary: `outputs/shap_summary.png`

**Key insight from feature importance / SHAP:** drug identity (esp.
hydralazine/procainamide/isoniazid), slow acetylator status, prior
autoimmune history, and elevated baseline ANA titer are the strongest
predictors — consistent with known drug-induced lupus risk factors in
the clinical literature, which validates the model's learned behavior
against domain knowledge.

## 6. Discussion
- The model demonstrates that a multi-factor combination (drug +
  genetic phenotype + clinical history + labs) is more informative
  than drug identity alone.
- Explainability (SHAP) is essential in healthcare AI — a black-box
  score alone would not be trusted or actionable by clinicians.
- Performance is currently moderate (PR-AUC ~0.28), which is expected
  and realistic for a rare-event problem on synthetic data — real
  clinical data with more granular genetic and temporal features
  (e.g., time-series drug exposure) would likely improve this further.

## 7. Limitations
- Trained on synthetic, not real patient data — real-world
  confounders, data entry noise, and unmeasured variables are not
  captured.
- Rare-event prediction is inherently hard; precision at usable
  thresholds is limited — this is a decision-support tool, not a
  diagnostic tool.
- No temporal/sequential modeling of drug exposure history (a future
  extension could use LSTM/Transformer architectures on longitudinal
  EHR data).
- Not validated against real clinical outcomes or by domain experts.

## 8. Ethical Considerations
- Patient data privacy: any real-data extension must follow HIPAA/
  applicable data protection regulations and use de-identified data
  under IRB/ethics approval.
- Not for direct clinical deployment: outputs are a research/decision-
  support aid; over-reliance without clinician oversight is discouraged.
- Bias: synthetic/limited data may not represent all populations
  equally (e.g., underrepresented ethnicities, pediatric patients) —
  real deployment would require fairness auditing across subgroups.

## 9. Conclusion & Future Work
- Successfully built an end-to-end explainable ML pipeline for early
  drug-induced autoimmune reaction risk prediction.
- Future work: validate on real FAERS/MIMIC data, add time-series
  modeling of cumulative drug exposure, incorporate genomic data (HLA
  typing) more granularly, and pilot with clinician feedback.

## 10. References
(Add your literature survey citations here in your institution's
required format — IEEE/APA etc.)

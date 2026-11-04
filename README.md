# Early Prediction of Drug-Induced Autoimmune Reactions Using AI

Final Year AIML Project — end-to-end pipeline: dataset generation →
preprocessing → model training/evaluation → SHAP explainability →
interactive Streamlit dashboard.

⚠️ **Disclaimer:** This project uses a literature-informed **synthetic
dataset** for demonstration/educational purposes. It is **not** a
clinical-grade tool and must not be used for real patient care
decisions without proper validation, regulatory approval, and real
clinical data under an IRB/ethics-approved study.

---

## 1. Project Structure

```
drug_autoimmune_project/
├── data/
│   └── patient_drug_dataset.csv       # generated synthetic dataset
├── models/
│   ├── final_model.joblib             # trained pipeline (best model)
│   ├── meta.joblib                    # which model was selected
│   ├── metrics.json                   # evaluation metrics, all models
│   ├── X_test.csv / y_test.csv        # held-out test split
├── outputs/
│   ├── roc_pr_curves.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│   └── shap_bar.png
├── src/
│   ├── generate_dataset.py            # step 1: build synthetic dataset
│   ├── preprocessing.py               # shared feature pipeline
│   ├── train.py                       # step 2: train & evaluate models
│   ├── explain.py                     # step 3: SHAP explainability plots
│   └── app.py                         # step 4: Streamlit dashboard
├── report/
│   └── project_report_outline.md      # ready-to-use report skeleton
├── requirements.txt
└── README.md
```

## 2. Setup

```bash
python3 -m venv venv
source venv/bin/activate            # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. How to Run (in order)

```bash
cd src

# 1. Generate the synthetic dataset
python3 generate_dataset.py

# 2. Train & evaluate models (Logistic Regression, Random Forest, XGBoost)
python3 train.py

# 3. Generate SHAP explainability plots
python3 explain.py

# 4. Launch the interactive dashboard
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` — enter patient/drug
details and get a live risk score + SHAP-based explanation of why.

## 4. What Makes This Project Stand Out

1. **Handles class imbalance properly** — autoimmune reactions are a
   rare event, so the pipeline uses class-weighting and evaluates with
   Precision/Recall/F1/ROC-AUC/**PR-AUC** instead of misleading plain
   accuracy.
2. **Explainable AI (SHAP)** — instead of a black-box score, the system
   shows exactly which factors (drug, acetylator status, prior
   history, labs, etc.) drove a given prediction — this is what
   clinicians and evaluators care about most in healthcare AI.
3. **Clinically-informed synthetic data** — feature relationships are
   grounded in real drug-induced autoimmunity literature (e.g.,
   hydralazine/procainamide/isoniazid + slow acetylator phenotype →
   drug-induced lupus risk), rather than arbitrary random data.
4. **End-to-end deployable demo** — a working Streamlit app a
   non-technical evaluator/doctor can interact with live.

## 5. Extending This to Real Data

For a stronger project or publication, replace `generate_dataset.py`'s
output with real-world sources such as:
- **FAERS** (FDA Adverse Event Reporting System) — free, public
- **SIDER** / **OFFSIDES / TWOSIDES** — drug side-effect resources
- **MIMIC-III / MIMIC-IV** — de-identified ICU EHR data (requires
  credentialed access + CITI training)

Keep the same `preprocessing.py` / `train.py` / `explain.py` structure
— only the data-loading step changes.

## 6. Suggested Report Sections

See `report/project_report_outline.md` for a ready-to-fill skeleton
covering Abstract, Literature Survey, Methodology, Results, Limitations,
and Future Work.

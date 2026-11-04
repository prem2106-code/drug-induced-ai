"""
train.py
--------
Drug-Induced Autoimmunity Prediction

Models:
1. Logistic Regression
2. Random Forest
3. XGBoost

Evaluation:
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Feature Importance
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier

from preprocessing import (
    load_raw_dataset,
    build_preprocessor,
    get_feature_names,
    ALL_FEATURES,
    TARGET
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"
OUT_DIR = PROJECT_DIR / "outputs"

DATA_PATH = DATA_DIR / "patient_drug_dataset.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DISPLAY PATHS
# ============================================================

print("=" * 70)
print("DRUG-INDUCED AUTOIMMUNITY PREDICTION")
print("=" * 70)

print("Project Directory :", PROJECT_DIR)
print("Dataset Path      :", DATA_PATH)
print("Models Directory  :", MODEL_DIR)
print("Outputs Directory :", OUT_DIR)

print("=" * 70)


# ============================================================
# CHECK DATASET
# ============================================================

if not DATA_PATH.exists():

    print("\nERROR: Dataset file not found!")
    print("\nExpected file:")
    print(DATA_PATH)

    print("\nPlease check that this file exists:")
    print("data/patient_drug_dataset.csv")

    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate(name, y_true, y_pred, y_proba):

    metrics = {
        "precision": round(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            4
        ),

        "recall": round(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            4
        ),

        "f1": round(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),
            4
        ),

        "roc_auc": round(
            roc_auc_score(
                y_true,
                y_proba
            ),
            4
        ),

        "pr_auc": round(
            average_precision_score(
                y_true,
                y_proba
            ),
            4
        )
    }

    print("\n" + "=" * 60)
    print("MODEL:", name)
    print("=" * 60)

    for key, value in metrics.items():
        print(f"{key:>12}: {value}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            zero_division=0
        )
    )

    return metrics


# ============================================================
# MAIN FUNCTION
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    print("\n[1/8] Loading dataset...")

    df = load_raw_dataset(DATA_PATH)

    print("Dataset shape:", df.shape)

    print("\nTarget distribution:")
    print(df[TARGET].value_counts())


    # --------------------------------------------------------
    # FEATURES AND TARGET
    # --------------------------------------------------------

    print("\n[2/8] Preparing features...")

    X = df[ALL_FEATURES]
    y = df[TARGET]

    print("Number of features:", len(ALL_FEATURES))
    print("Target:", TARGET)


    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    print("\n[3/8] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples :", len(X_test))


    # --------------------------------------------------------
    # PREPROCESSOR
    # --------------------------------------------------------

    print("\n[4/8] Building preprocessing pipeline...")

    preprocessor = build_preprocessor()


    # --------------------------------------------------------
    # CLASS IMBALANCE
    # --------------------------------------------------------

    positive_count = (y_train == 1).sum()
    negative_count = (y_train == 0).sum()

    scale_pos_weight = (
        negative_count /
        max(positive_count, 1)
    )

    print("Positive samples:", positive_count)
    print("Negative samples:", negative_count)
    print("XGBoost scale_pos_weight:", scale_pos_weight)


    # --------------------------------------------------------
    # MODELS
    # --------------------------------------------------------

    print("\n[5/8] Creating models...")

    models = {

        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1
        )
    }


    # --------------------------------------------------------
    # STORAGE
    # --------------------------------------------------------

    all_metrics = {}
    fitted_pipelines = {}
    proba_store = {}
    prediction_store = {}


    # --------------------------------------------------------
    # TRAIN MODELS
    # --------------------------------------------------------

    print("\n[6/8] Training models...")

    for name, clf in models.items():

        print("\n" + "-" * 60)
        print("Training:", name)
        print("-" * 60)

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("clf", clf)
            ]
        )

        pipe.fit(
            X_train,
            y_train
        )

        y_proba = pipe.predict_proba(
            X_test
        )[:, 1]

        y_pred = (
            y_proba >= 0.5
        ).astype(int)

        metrics = evaluate(
            name,
            y_test,
            y_pred,
            y_proba
        )

        all_metrics[name] = metrics
        fitted_pipelines[name] = pipe
        proba_store[name] = y_proba
        prediction_store[name] = y_pred


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    for name, metrics in all_metrics.items():

        print(
            f"{name:25s} "
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"PR-AUC={metrics['pr_auc']:.4f} | "
            f"F1={metrics['f1']:.4f}"
        )


    # --------------------------------------------------------
    # SELECT BEST MODEL
    # --------------------------------------------------------

    best_name = max(
        all_metrics,
        key=lambda name:
        all_metrics[name]["pr_auc"]
    )

    best_pipe = fitted_pipelines[best_name]

    print("\n" + "=" * 70)
    print("BEST MODEL:", best_name)
    print(
        "Best PR-AUC:",
        all_metrics[best_name]["pr_auc"]
    )
    print("=" * 70)


    # ========================================================
    # SAVE MODEL
    # ========================================================

    final_model_path = (
        MODEL_DIR /
        "final_model.joblib"
    )

    meta_path = (
        MODEL_DIR /
        "meta.joblib"
    )

    metrics_path = (
        MODEL_DIR /
        "metrics.json"
    )

    joblib.dump(
        best_pipe,
        final_model_path
    )

    joblib.dump(
        {
            "best_model_name": best_name
        },
        meta_path
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            {
                "best_model": best_name,
                "all_metrics": all_metrics
            },
            f,
            indent=4
        )

    print("\nModel saved successfully:")
    print(final_model_path)


    # ========================================================
    # ROC CURVE
    # ========================================================

    print("\nCreating ROC Curve...")

    plt.figure(
        figsize=(8, 6)
    )

    for name, proba in proba_store.items():

        fpr, tpr, _ = roc_curve(
            y_test,
            proba
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{name} "
                f"(AUC={all_metrics[name]['roc_auc']:.3f})"
            )
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title(
        "ROC Curve - Drug-Induced Autoimmunity Prediction"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    roc_path = (
        OUT_DIR /
        "roc_curve.png"
    )

    plt.savefig(
        roc_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", roc_path)


    # ========================================================
    # PRECISION RECALL CURVE
    # ========================================================

    print("\nCreating Precision-Recall Curve...")

    plt.figure(
        figsize=(8, 6)
    )

    for name, proba in proba_store.items():

        precision, recall, _ = (
            precision_recall_curve(
                y_test,
                proba
            )
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{name} "
                f"(AP={all_metrics[name]['pr_auc']:.3f})"
            )
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(
        "Precision-Recall Curve - Drug-Induced Autoimmunity"
    )

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    pr_path = (
        OUT_DIR /
        "precision_recall_curve.png"
    )

    plt.savefig(
        pr_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", pr_path)


    # ========================================================
    # COMBINED ROC + PR CURVES
    # ========================================================

    print("\nCreating combined ROC + PR curves...")

    plt.figure(
        figsize=(14, 6)
    )


    # ROC

    plt.subplot(
        1,
        2,
        1
    )

    for name, proba in proba_store.items():

        fpr, tpr, _ = roc_curve(
            y_test,
            proba
        )

        plt.plot(
            fpr,
            tpr,
            label=(
                f"{name} "
                f"(AUC={all_metrics[name]['roc_auc']:.3f})"
            )
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")

    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)


    # Precision Recall

    plt.subplot(
        1,
        2,
        2
    )

    for name, proba in proba_store.items():

        precision, recall, _ = (
            precision_recall_curve(
                y_test,
                proba
            )
        )

        plt.plot(
            recall,
            precision,
            label=(
                f"{name} "
                f"(AP={all_metrics[name]['pr_auc']:.3f})"
            )
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")

    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)

    plt.tight_layout()

    combined_path = (
        OUT_DIR /
        "roc_pr_curves.png"
    )

    plt.savefig(
        combined_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", combined_path)


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\nCreating Confusion Matrix...")

    y_pred_best = prediction_store[
        best_name
    ]

    cm = confusion_matrix(
        y_test,
        y_pred_best
    )

    plt.figure(
        figsize=(7, 6)
    )

    plt.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues"
    )

    plt.title(
        f"Confusion Matrix - {best_name}"
    )

    plt.colorbar()

    plt.xticks(
        [0, 1],
        [
            "No Reaction",
            "Reaction"
        ]
    )

    plt.yticks(
        [0, 1],
        [
            "No Reaction",
            "Reaction"
        ]
    )

    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=15
            )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    cm_path = (
        OUT_DIR /
        "confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Saved:", cm_path)


    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    if best_name in (
        "random_forest",
        "xgboost"
    ):

        print("\nCreating Feature Importance...")

        clf = (
            best_pipe
            .named_steps["clf"]
        )

        preproc = (
            best_pipe
            .named_steps["preprocess"]
        )

        feat_names = get_feature_names(
            preproc
        )

        importances = (
            clf.feature_importances_
        )

        idx = np.argsort(
            importances
        )[::-1][:15]

        plt.figure(
            figsize=(10, 7)
        )

        plt.barh(
            [
                feat_names[i]
                for i in idx
            ][::-1],
            importances[idx][::-1]
        )

        plt.xlabel(
            "Feature Importance"
        )

        plt.ylabel(
            "Features"
        )

        plt.title(
            f"Top 15 Feature Importances - {best_name}"
        )

        plt.tight_layout()

        feature_path = (
            OUT_DIR /
            "feature_importance.png"
        )

        plt.savefig(
            feature_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved:", feature_path)


    # ========================================================
    # SAVE TEST DATA
    # ========================================================

    print("\nSaving test data...")

    X_test.to_csv(
        MODEL_DIR / "X_test.csv",
        index=False
    )

    y_test.to_csv(
        MODEL_DIR / "y_test.csv",
        index=False
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        "\nBest Model:",
        best_name
    )

    print(
        "ROC-AUC:",
        all_metrics[best_name]["roc_auc"]
    )

    print(
        "PR-AUC:",
        all_metrics[best_name]["pr_auc"]
    )

    print(
        "Precision:",
        all_metrics[best_name]["precision"]
    )

    print(
        "Recall:",
        all_metrics[best_name]["recall"]
    )

    print(
        "F1 Score:",
        all_metrics[best_name]["f1"]
    )

    print("\nGenerated files:")

    print("1.", roc_path)
    print("2.", pr_path)
    print("3.", combined_path)
    print("4.", cm_path)

    if best_name in (
        "random_forest",
        "xgboost"
    ):
        print("5.", feature_path)

    print("\n" + "=" * 70)


# ============================================================
# START PROGRAM
# ============================================================

if __name__ == "__main__":
    main()
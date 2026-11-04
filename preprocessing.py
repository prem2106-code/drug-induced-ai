"""
preprocessing.py
-----------------
Shared preprocessing pipeline used by both training and inference
(dashboard) so that train/serve skew is avoided.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age",
    "dosage_mg",
    "duration_days",
    "concurrent_meds_count",
    "baseline_ana_titer",
    "baseline_crp",
    "baseline_wbc",
    "bmi",
]

BINARY_FEATURES = [
    "slow_acetylator",
    "hla_risk_allele",
    "prior_autoimmune_history",
    "family_autoimmune_history",
    "smoker",
]

CATEGORICAL_FEATURES = [
    "sex",
    "drug_name",
    "drug_class",
]

ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
TARGET = "label"


def load_raw_dataset(path):
    df = pd.read_csv(path)
    return df


def build_preprocessor():
    """Returns an sklearn ColumnTransformer for feature preprocessing."""
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def get_feature_names(preprocessor):
    """Extract readable feature names after ColumnTransformer.fit()."""
    num_names = NUMERIC_FEATURES
    bin_names = BINARY_FEATURES
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    return num_names + bin_names + cat_names

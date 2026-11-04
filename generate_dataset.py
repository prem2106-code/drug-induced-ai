"""
generate_dataset.py
--------------------
Generates a realistic SYNTHETIC dataset for the project:
"Early Prediction of Drug-Induced Autoimmune Reactions Using AI"

WHY SYNTHETIC DATA?
Real patient-level data linking specific drugs to autoimmune reactions
(e.g., MIMIC-III/IV, FAERS with linked labs) requires data-use
agreements / credentialing that a student project usually cannot obtain
in time. This script builds a CLINICALLY-INFORMED synthetic dataset:
feature distributions and risk relationships are based on published
drug-induced autoimmunity literature (e.g., higher risk with
hydralazine, procainamide, isoniazid, minocycline, TNF-inhibitors;
HLA associations; slow acetylator phenotype; female sex bias in
drug-induced lupus, etc.), with added random noise so the task is not
trivially separable.

This lets you build and demonstrate the FULL ML pipeline end-to-end.
For your report, clearly state this is synthetic/semi-synthetic data
inspired by literature, and that a natural next step is validating on
real EHR/FAERS data under proper data governance.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_SAMPLES = 8000

# ---------------------------------------------------------------------
# 1. Drug catalogue with literature-informed base autoimmune risk
#    (loosely calibrated, NOT clinical-grade — for project use only)
# ---------------------------------------------------------------------
DRUGS = {
    "Hydralazine":        {"class": "Antihypertensive",     "base_risk": 0.20},
    "Procainamide":       {"class": "Antiarrhythmic",       "base_risk": 0.22},
    "Isoniazid":          {"class": "Antitubercular",       "base_risk": 0.10},
    "Minocycline":        {"class": "Antibiotic",           "base_risk": 0.08},
    "Methyldopa":         {"class": "Antihypertensive",     "base_risk": 0.07},
    "Infliximab":         {"class": "TNF-inhibitor",        "base_risk": 0.09},
    "Etanercept":         {"class": "TNF-inhibitor",        "base_risk": 0.06},
    "Adalimumab":         {"class": "TNF-inhibitor",        "base_risk": 0.06},
    "Quinidine":          {"class": "Antiarrhythmic",       "base_risk": 0.05},
    "Chlorpromazine":     {"class": "Antipsychotic",        "base_risk": 0.04},
    "Carbamazepine":      {"class": "Anticonvulsant",       "base_risk": 0.03},
    "Phenytoin":          {"class": "Anticonvulsant",       "base_risk": 0.03},
    "Sulfasalazine":      {"class": "Anti-inflammatory",    "base_risk": 0.05},
    "Statins (Atorvastatin)": {"class": "Lipid-lowering",   "base_risk": 0.015},
    "Metformin":          {"class": "Antidiabetic",         "base_risk": 0.005},
    "Amoxicillin":        {"class": "Antibiotic",           "base_risk": 0.004},
    "Ibuprofen":          {"class": "NSAID",                "base_risk": 0.006},
    "Paracetamol":        {"class": "Analgesic",            "base_risk": 0.002},
    "Levothyroxine":      {"class": "Hormone",              "base_risk": 0.002},
    "Omeprazole":         {"class": "PPI",                  "base_risk": 0.003},
}

drug_names = list(DRUGS.keys())


def sample_patients(n):
    age = RNG.normal(48, 16, n).clip(5, 95).round(0)
    sex = RNG.choice(["F", "M"], size=n, p=[0.55, 0.45])

    # Slow acetylator phenotype (NAT2) — strongly linked to hydralazine/
    # procainamide/isoniazid-induced lupus. ~50-60% of population.
    slow_acetylator = RNG.choice([1, 0], size=n, p=[0.55, 0.45])

    # HLA risk allele presence (simplified single flag)
    hla_risk_allele = RNG.choice([1, 0], size=n, p=[0.15, 0.85])

    prior_autoimmune_history = RNG.choice([1, 0], size=n, p=[0.08, 0.92])
    family_autoimmune_history = RNG.choice([1, 0], size=n, p=[0.12, 0.88])

    dosage_mg = RNG.gamma(shape=3, scale=60, size=n).clip(5, 1000).round(0)
    duration_days = RNG.gamma(shape=2, scale=90, size=n).clip(1, 3650).round(0)

    concurrent_meds = RNG.poisson(2.2, size=n).clip(0, 12)

    # Baseline labs
    ana_titer = RNG.lognormal(mean=1.0, sigma=0.6, size=n).clip(1, 640).round(1)
    crp = RNG.gamma(shape=2, scale=2.5, size=n).clip(0.1, 60).round(2)
    wbc = RNG.normal(7.0, 2.0, n).clip(1.5, 20).round(1)

    smoker = RNG.choice([1, 0], size=n, p=[0.22, 0.78])
    bmi = RNG.normal(26, 5, n).clip(14, 55).round(1)

    drug = RNG.choice(drug_names, size=n)
    drug_class = np.array([DRUGS[d]["class"] for d in drug])
    base_risk = np.array([DRUGS[d]["base_risk"] for d in drug])

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "drug_name": drug,
        "drug_class": drug_class,
        "dosage_mg": dosage_mg,
        "duration_days": duration_days,
        "slow_acetylator": slow_acetylator,
        "hla_risk_allele": hla_risk_allele,
        "prior_autoimmune_history": prior_autoimmune_history,
        "family_autoimmune_history": family_autoimmune_history,
        "concurrent_meds_count": concurrent_meds,
        "baseline_ana_titer": ana_titer,
        "baseline_crp": crp,
        "baseline_wbc": wbc,
        "smoker": smoker,
        "bmi": bmi,
        "_base_risk": base_risk,
    })
    return df


def compute_label(df):
    """
    Combine base drug risk with clinically-motivated modifiers to get a
    probability of developing a drug-induced autoimmune reaction, then
    sample a binary label. Modifiers are directionally realistic:
      - slow acetylators + hydralazine/procainamide/isoniazid -> higher risk
      - HLA risk allele -> higher risk
      - female sex -> higher risk (drug-induced lupus literature)
      - longer duration & higher cumulative dose -> higher risk
      - prior/family autoimmune history -> higher risk
      - elevated baseline ANA/CRP -> higher risk
      - age (middle-aged skew for drug-induced lupus) -> mild effect
    """
    p = df["_base_risk"].copy()

    high_risk_drugs = {"Hydralazine", "Procainamide", "Isoniazid"}
    is_high_risk_drug = df["drug_name"].isin(high_risk_drugs)

    p = p + is_high_risk_drug * df["slow_acetylator"] * 0.18
    p = p + df["hla_risk_allele"] * 0.05
    p = p + (df["sex"] == "F") * 0.03
    p = p + (df["duration_days"] > 180) * 0.04
    p = p + (df["dosage_mg"] > 300) * 0.02
    p = p + df["prior_autoimmune_history"] * 0.15
    p = p + df["family_autoimmune_history"] * 0.05
    p = p + (df["baseline_ana_titer"] > 80) * 0.10
    p = p + (df["baseline_crp"] > 10) * 0.03
    p = p + df["smoker"] * 0.015
    p = p + ((df["age"] >= 30) & (df["age"] <= 60)) * 0.01

    # small random noise so the problem isn't perfectly separable
    noise = RNG.normal(0, 0.03, len(df))
    p = (p + noise).clip(0.001, 0.95)

    label = RNG.binomial(1, p)
    df["autoimmune_reaction_risk_prob"] = p.round(4)
    df["label"] = label
    return df


def main():
    df = sample_patients(N_SAMPLES)
    df = compute_label(df)
    df = df.drop(columns=["_base_risk"])

    out_path = "/home/claude/drug_autoimmune_project/data/patient_drug_dataset.csv"
    df.to_csv(out_path, index=False)

    print("Saved dataset:", out_path)
    print("Shape:", df.shape)
    print("Positive class rate: {:.2%}".format(df["label"].mean()))
    print(df.head())


if __name__ == "__main__":
    main()

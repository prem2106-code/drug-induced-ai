import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import html

from preprocessing import get_feature_names
from generate_dataset import DRUGS


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "../models"

st.set_page_config(
    page_title="Drug-Induced Autoimmune Safety AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    pipe = joblib.load(f"{MODEL_DIR}/final_model.joblib")
    meta = joblib.load(f"{MODEL_DIR}/meta.joblib")
    return pipe, meta


pipe, meta = load_model()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False

if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None

if "shap_contrib" not in st.session_state:
    st.session_state.shap_contrib = None

if "lab_result" not in st.session_state:
    st.session_state.lab_result = None


# ============================================================
# HEADER
# ============================================================

st.title("🧬 Drug-Induced Autoimmune Safety AI")

st.caption(
    "Explainable AI framework for preliminary assessment of "
    "potential drug-induced autoimmune and immune-related safety signals."
)


# ============================================================
# NAVIGATION BAR
# ============================================================

page = st.radio(
    "Navigation",
    [
        "🏠 Prediction",
        "🧪 Pre-Market Lab",
        "📄 Risk Report",
        "ℹ️ About",
    ],
    horizontal=True,
)

st.divider()


# ============================================================
# HELPER FUNCTION
# ============================================================

def create_input_dataframe(
    age,
    sex,
    bmi,
    smoker,
    prior_autoimmune_history,
    family_autoimmune_history,
    slow_acetylator,
    hla_risk_allele,
    drug_name,
    dosage_mg,
    duration_days,
    concurrent_meds_count,
    baseline_ana_titer,
    baseline_crp,
    baseline_wbc,
):

    drug_class = DRUGS[drug_name]["class"]

    return pd.DataFrame([{
        "age": age,
        "sex": sex,
        "drug_name": drug_name,
        "drug_class": drug_class,
        "dosage_mg": dosage_mg,
        "duration_days": duration_days,
        "slow_acetylator": int(slow_acetylator),
        "hla_risk_allele": int(hla_risk_allele),
        "prior_autoimmune_history": int(
            prior_autoimmune_history
        ),
        "family_autoimmune_history": int(
            family_autoimmune_history
        ),
        "concurrent_meds_count": concurrent_meds_count,
        "baseline_ana_titer": baseline_ana_titer,
        "baseline_crp": baseline_crp,
        "baseline_wbc": baseline_wbc,
        "smoker": int(smoker),
        "bmi": bmi,
    }])


# ============================================================
# SHAP FUNCTION
# ============================================================

def calculate_shap(input_df):

    preproc = pipe.named_steps["preprocess"]
    clf = pipe.named_steps["clf"]

    feat_names = get_feature_names(preproc)

    X_input_transformed = preproc.transform(input_df)

    if hasattr(X_input_transformed, "toarray"):
        X_input_transformed = X_input_transformed.toarray()

    explainer = shap.TreeExplainer(clf)

    shap_values = explainer.shap_values(
        X_input_transformed
    )

    # --------------------------------------------------------
    # Handle different SHAP versions
    # --------------------------------------------------------

    if isinstance(shap_values, list):

        sv = np.asarray(shap_values[1])

        if sv.ndim == 2:
            sv = sv[0]

    else:

        sv = np.asarray(shap_values)

        # Shape: (1, features, classes)
        if sv.ndim == 3:
            sv = sv[0, :, 1]

        # Shape: (features, classes)
        elif sv.ndim == 2:

            if sv.shape[1] == 2:
                sv = sv[:, 1]

            elif sv.shape[0] == 2:
                sv = sv[1]

            else:
                sv = sv[0]

        # Shape: (features,)
        elif sv.ndim == 1:
            sv = sv

    sv = np.asarray(sv).reshape(-1)

    if len(sv) != len(feat_names):
        return None

    contrib = (
        pd.Series(
            sv,
            index=feat_names
        )
        .sort_values(
            key=abs,
            ascending=False
        )
        .head(10)
    )

    return contrib


# ============================================================
# PAGE 1 - PREDICTION
# ============================================================

if page == "🏠 Prediction":

    st.header("🏠 Patient & Drug Risk Prediction")

    st.info(
        "Enter patient and drug information to estimate a "
        "preliminary probability of a drug-induced autoimmune reaction."
    )

    col1, col2 = st.columns(2)

    # --------------------------------------------------------
    # PATIENT
    # --------------------------------------------------------

    with col1:

        st.subheader("👤 Patient Details")

        age = st.slider(
            "Age",
            5,
            95,
            45
        )

        sex = st.selectbox(
            "Sex",
            ["F", "M"]
        )

        bmi = st.slider(
            "BMI",
            14.0,
            55.0,
            25.0
        )

        smoker = st.checkbox(
            "Smoker"
        )

        prior_autoimmune_history = st.checkbox(
            "Prior autoimmune disease history"
        )

        family_autoimmune_history = st.checkbox(
            "Family history of autoimmune disease"
        )

        slow_acetylator = st.checkbox(
            "Known/suspected slow acetylator (NAT2)"
        )

        hla_risk_allele = st.checkbox(
            "HLA risk allele present (if genotyped)"
        )

    # --------------------------------------------------------
    # DRUG
    # --------------------------------------------------------

    with col2:

        st.subheader("💊 Drug & Laboratory Details")

        drug_name = st.selectbox(
            "Drug",
            sorted(DRUGS.keys())
        )

        dosage_mg = st.slider(
            "Dosage (mg/day)",
            5,
            1000,
            100
        )

        duration_days = st.slider(
            "Duration of use (days)",
            1,
            3650,
            90
        )

        concurrent_meds_count = st.slider(
            "Number of concurrent medications",
            0,
            12,
            2
        )

        baseline_ana_titer = st.slider(
            "Baseline ANA titer",
            1.0,
            640.0,
            20.0
        )

        baseline_crp = st.slider(
            "Baseline CRP (mg/L)",
            0.1,
            60.0,
            3.0
        )

        baseline_wbc = st.slider(
            "Baseline WBC (x10^9/L)",
            1.5,
            20.0,
            7.0
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔍 Predict Risk",
        type="primary",
        use_container_width=True
    ):

        input_df = create_input_dataframe(
            age,
            sex,
            bmi,
            smoker,
            prior_autoimmune_history,
            family_autoimmune_history,
            slow_acetylator,
            hla_risk_allele,
            drug_name,
            dosage_mg,
            duration_days,
            concurrent_meds_count,
            baseline_ana_titer,
            baseline_crp,
            baseline_wbc,
        )

        proba = pipe.predict_proba(
            input_df
        )[0, 1]

        # Risk classification
        if proba < 0.15:

            risk_label = "LOW RISK"
            risk_icon = "🟢"

        elif proba < 0.40:

            risk_label = "MODERATE RISK"
            risk_icon = "🟠"

        else:

            risk_label = "HIGH RISK"
            risk_icon = "🔴"

        # Save prediction
        st.session_state.prediction_done = True

        st.session_state.prediction_data = {
            "drug_name": drug_name,
            "drug_class": DRUGS[drug_name]["class"],
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "dosage_mg": dosage_mg,
            "duration_days": duration_days,
            "concurrent_meds_count": concurrent_meds_count,
            "baseline_ana_titer": baseline_ana_titer,
            "baseline_crp": baseline_crp,
            "baseline_wbc": baseline_wbc,
            "risk_probability": float(proba),
            "risk_label": risk_label,
        }

        # SHAP
        contrib = calculate_shap(input_df)

        st.session_state.shap_contrib = contrib

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.success("Prediction completed successfully.")

        st.subheader("🎯 Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Predicted Risk",
                f"{proba:.1%}"
            )

        with result_col2:

            st.metric(
                "Risk Category",
                f"{risk_icon} {risk_label}"
            )

        with result_col3:

            st.metric(
                "Drug",
                drug_name
            )

        st.progress(
            min(float(proba), 1.0)
        )

        # ----------------------------------------------------
        # SHAP
        # ----------------------------------------------------

        st.subheader(
            "🔎 Why did the model predict this?"
        )

        if contrib is not None:

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            colors = [
                "#d62728" if value > 0
                else "#2ca02c"
                for value in contrib.values[::-1]
            ]

            ax.barh(
                contrib.index[::-1],
                contrib.values[::-1],
                color=colors
            )

            ax.axvline(
                0,
                linewidth=1
            )

            ax.set_xlabel(
                "SHAP value "
                "(→ increases predicted risk | "
                "← decreases predicted risk)"
            )

            ax.set_title(
                "Top Factors Influencing This Prediction"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.caption(
                "🔴 Red = pushes predicted risk upward · "
                "🟢 Green = pushes predicted risk downward."
            )

        else:

            st.warning(
                "SHAP explanation could not be generated."
            )


# ============================================================
# PAGE 2 - PRE-MARKET LAB
# ============================================================

elif page == "🧪 Pre-Market Lab":

    st.header("🧪 Pre-Market Immune Safety Lab")

    st.info(
        "This module performs an AI-assisted preliminary screening "
        "of potential immune-related safety signals before considering "
        "a drug for wider clinical/market exposure."
    )

    st.warning(
        "⚠️ This is a research simulation using the project's "
        "synthetic-data model. It is NOT a real laboratory test, "
        "regulatory assessment, or clinical safety decision."
    )

    st.divider()

    lab_drug = st.selectbox(
        "Select Drug for Pre-Market Screening",
        sorted(DRUGS.keys()),
        key="lab_drug"
    )

    drug_class = DRUGS[lab_drug]["class"]

    st.markdown(
        f"""
        ### 💊 Selected Drug

        **Drug:** {lab_drug}

        **Drug Class:** {drug_class}
        """
    )

    st.divider()

    st.subheader(
        "🔬 Standardized AI Screening"
    )

    st.write(
        "The system evaluates the selected drug under a "
        "standardized reference profile so that the drug-level "
        "screening can be demonstrated consistently."
    )

    if st.button(
        "🧪 Run Pre-Market Lab Screening",
        type="primary",
        use_container_width=True
    ):

        # ----------------------------------------------------
        # STANDARDIZED REFERENCE PROFILE
        # ----------------------------------------------------

        lab_input = create_input_dataframe(
            age=45,
            sex="F",
            bmi=25.0,
            smoker=False,
            prior_autoimmune_history=False,
            family_autoimmune_history=False,
            slow_acetylator=False,
            hla_risk_allele=False,
            drug_name=lab_drug,
            dosage_mg=100,
            duration_days=90,
            concurrent_meds_count=0,
            baseline_ana_titer=20.0,
            baseline_crp=3.0,
            baseline_wbc=7.0,
        )

        lab_probability = pipe.predict_proba(
            lab_input
        )[0, 1]

        # ----------------------------------------------------
        # LAB CLASSIFICATION
        # ----------------------------------------------------

        if lab_probability < 0.15:

            concern = "LOW CONCERN"
            concern_icon = "🟢"

        elif lab_probability < 0.40:

            concern = "MODERATE CONCERN"
            concern_icon = "🟠"

        else:

            concern = "HIGH CONCERN"
            concern_icon = "🔴"

        # ----------------------------------------------------
        # SAVE LAB RESULT
        # ----------------------------------------------------

        st.session_state.lab_result = {
            "drug_name": lab_drug,
            "drug_class": drug_class,
            "probability": float(lab_probability),
            "concern": concern,
        }

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "🧬 Preliminary Immune Safety Assessment"
        )

        lab_col1, lab_col2 = st.columns(2)

        with lab_col1:

            st.metric(
                "Potential Risk Signal",
                f"{lab_probability:.1%}"
            )

        with lab_col2:

            st.metric(
                "Screening Status",
                f"{concern_icon} {concern}"
            )

        st.progress(
            min(float(lab_probability), 1.0)
        )

        st.divider()

        # ----------------------------------------------------
        # IMMUNE IMPACT
        # ----------------------------------------------------

        st.subheader(
            "🛡️ Potential Immune-System Impact"
        )

        if lab_probability < 0.15:

            st.success(
                "🟢 LOW CONCERN — The model did not identify "
                "a strong immune-related risk signal under the "
                "standardized screening profile."
            )

        elif lab_probability < 0.40:

            st.warning(
                "🟠 MODERATE CONCERN — The model identified "
                "a potential immune-related safety signal that "
                "may warrant additional investigation."
            )

        else:

            st.error(
                "🔴 HIGH CONCERN — The model identified a "
                "strong preliminary immune-related safety signal "
                "under the standardized screening profile."
            )

        st.divider()

        # ----------------------------------------------------
        # SCREENING AREAS
        # ----------------------------------------------------

        st.subheader(
            "🔬 Suggested Preclinical Evaluation Areas"
        )

        if lab_probability >= 0.40:

            tests = [
                "Immune-related safety assessment",
                "Autoimmune marker / autoantibody evaluation",
                "Inflammatory and cytokine-related assessment",
                "Immune-cell profiling where scientifically justified",
                "General toxicity and histopathological evaluation",
                "Expert review of the overall nonclinical evidence",
            ]

        elif lab_probability >= 0.15:

            tests = [
                "Additional immune-related safety assessment",
                "Autoimmune marker evaluation",
                "Inflammatory marker assessment",
                "General toxicity evaluation",
                "Review of available nonclinical evidence",
            ]

        else:

            tests = [
                "Standard preclinical safety evaluation",
                "General toxicity assessment",
                "Routine monitoring of relevant safety endpoints",
            ]

        for test in tests:

            st.markdown(
                f"✓ {test}"
            )

        st.divider()

        st.caption(
            "Important: These are suggested evaluation areas for "
            "a student research demonstration. Actual drug-development "
            "testing must be designed and interpreted by qualified "
            "scientists, toxicologists, clinicians, and regulatory experts."
        )


# ============================================================
# PAGE 3 - RISK REPORT
# ============================================================

elif page == "📄 Risk Report":

    st.header("📄 Drug Immune Safety Risk Report")

    if not st.session_state.prediction_done:

        st.info(
            "No prediction report is available yet."
        )

        st.write(
            "Go to **🏠 Prediction**, enter the patient and drug "
            "details, and click **Predict Risk** first."
        )

    else:

        data = st.session_state.prediction_data
        contrib = st.session_state.shap_contrib
        lab = st.session_state.lab_result

        # ----------------------------------------------------
        # REPORT HEADER
        # ----------------------------------------------------

        st.markdown(
            """
            <div style="
                border:2px solid #333;
                border-radius:12px;
                padding:25px;
                text-align:center;
                margin-bottom:20px;
            ">
                <h1>🧬 DRUG IMMUNE SAFETY REPORT</h1>
                <p>
                Explainable AI-Based Preliminary Risk Assessment
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        st.subheader("🎯 Risk Summary")

        report_col1, report_col2, report_col3 = st.columns(3)

        with report_col1:

            st.metric(
                "Drug",
                data["drug_name"]
            )

        with report_col2:

            st.metric(
                "Predicted Risk",
                f"{data['risk_probability']:.1%}"
            )

        with report_col3:

            st.metric(
                "Risk Category",
                data["risk_label"]
            )

        st.divider()

        # ----------------------------------------------------
        # PATIENT PROFILE
        # ----------------------------------------------------

        st.subheader("👤 Patient Profile")

        patient_df = pd.DataFrame({
            "Parameter": [
                "Age",
                "Sex",
                "BMI",
                "Dosage",
                "Duration",
                "Concurrent Medications",
                "Baseline ANA",
                "Baseline CRP",
                "Baseline WBC",
            ],
            "Value": [
                data["age"],
                data["sex"],
                data["bmi"],
                f"{data['dosage_mg']} mg/day",
                f"{data['duration_days']} days",
                data["concurrent_meds_count"],
                data["baseline_ana_titer"],
                f"{data['baseline_crp']} mg/L",
                data["baseline_wbc"],
            ]
        })

        st.table(patient_df)

        st.divider()

        # ----------------------------------------------------
        # DRUG INFORMATION
        # ----------------------------------------------------

        st.subheader("💊 Drug Information")

        drug_df = pd.DataFrame({
            "Parameter": [
                "Drug Name",
                "Drug Class",
            ],
            "Value": [
                data["drug_name"],
                data["drug_class"],
            ]
        })

        st.table(drug_df)

        st.divider()

        # ----------------------------------------------------
        # SHAP FACTORS
        # ----------------------------------------------------

        st.subheader(
            "🔎 Model Explanation"
        )

        if contrib is not None:

            factor_df = pd.DataFrame({
                "Factor": contrib.index,
                "SHAP Contribution": [
                    round(float(v), 4)
                    for v in contrib.values
                ],
                "Direction": [
                    "↑ Increases risk"
                    if v > 0
                    else "↓ Decreases risk"
                    for v in contrib.values
                ]
            })

            st.dataframe(
                factor_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "SHAP factor information is unavailable."
            )

        st.divider()

        # ----------------------------------------------------
        # IMMUNE ASSESSMENT
        # ----------------------------------------------------

        st.subheader(
            "🛡️ Immune-System Safety Assessment"
        )

        if data["risk_probability"] < 0.15:

            immune_status = "LOW CONCERN"

            immune_text = (
                "The model did not identify a strong "
                "immune-related safety signal for this "
                "input profile."
            )

        elif data["risk_probability"] < 0.40:

            immune_status = "MODERATE CONCERN"

            immune_text = (
                "The model identified a potential "
                "immune-related safety signal that may "
                "warrant additional investigation."
            )

        else:

            immune_status = "HIGH CONCERN"

            immune_text = (
                "The model identified a strong preliminary "
                "immune-related safety signal for this "
                "input profile."
            )

        st.markdown(
            f"""
            ### {immune_status}

            {immune_text}
            """
        )

        st.divider()

        # ----------------------------------------------------
        # LAB RESULT
        # ----------------------------------------------------

        st.subheader(
            "🧪 Pre-Market Lab Screening"
        )

        if lab is not None:

            st.write(
                f"**Drug:** {lab['drug_name']}"
            )

            st.write(
                f"**Screening Score:** "
                f"{lab['probability']:.1%}"
            )

            st.write(
                f"**Status:** {lab['concern']}"
            )

            st.caption(
                "The pre-market screening is an AI research "
                "simulation based on the project's synthetic-data "
                "model and should not be interpreted as an actual "
                "laboratory or regulatory result."
            )

        else:

            st.info(
                "Pre-Market Lab screening has not been performed "
                "for this report."
            )

        st.divider()

        # ----------------------------------------------------
        # RECOMMENDATION
        # ----------------------------------------------------

        st.subheader(
            "📌 Preliminary Recommendation"
        )

        if data["risk_probability"] >= 0.40:

            recommendation = (
                "The model indicates a strong preliminary "
                "immune-related safety signal. Additional "
                "nonclinical evaluation and expert review "
                "should be considered before progression."
            )

        elif data["risk_probability"] >= 0.15:

            recommendation = (
                "The model indicates a potential immune-related "
                "safety signal. Additional investigation and "
                "appropriate preclinical evaluation may be "
                "considered."
            )

        else:

            recommendation = (
                "The model indicates a low preliminary "
                "immune-related risk signal under the evaluated "
                "profile. Standard safety evaluation remains "
                "necessary."
            )

        st.info(
            recommendation
        )

        st.divider()

        # ----------------------------------------------------
        # PRINT REPORT
        # ----------------------------------------------------

        st.subheader(
            "🖨️ Print / Save Report"
        )

        st.write(
            "Click the button below to open the browser print "
            "dialog. You can select **Save as PDF**."
        )

        # Build printable HTML
        factors_html = ""

        if contrib is not None:

            for feature, value in contrib.items():

                direction = (
                    "Increases risk"
                    if value > 0
                    else "Decreases risk"
                )

                factors_html += f"""
                <tr>
                    <td>{html.escape(str(feature))}</td>
                    <td>{float(value):.4f}</td>
                    <td>{direction}</td>
                </tr>
                """

        printable_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Drug Immune Safety Report</title>

            <style>

                body {{
                    font-family: Arial, sans-serif;
                    padding: 30px;
                    color: #111;
                }}

                h1 {{
                    text-align: center;
                }}

                h2 {{
                    border-bottom: 1px solid #999;
                    padding-bottom: 5px;
                }}

                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}

                th, td {{
                    border: 1px solid #999;
                    padding: 8px;
                    text-align: left;
                }}

                th {{
                    background: #eee;
                }}

                .risk {{
                    font-size: 22px;
                    font-weight: bold;
                    text-align: center;
                    padding: 15px;
                    border: 2px solid #333;
                }}

                .warning {{
                    border: 1px solid #999;
                    padding: 15px;
                    margin-top: 20px;
                }}

                @media print {{
                    .print-button {{
                        display: none;
                    }}
                }}

            </style>
        </head>

        <body>

            <button
                class="print-button"
                onclick="window.print()"
                style="
                    padding:10px 20px;
                    font-size:16px;
                    margin-bottom:20px;
                    cursor:pointer;
                "
            >
                🖨️ Print / Save as PDF
            </button>

            <h1>🧬 DRUG IMMUNE SAFETY REPORT</h1>

            <p style="text-align:center;">
                Explainable AI-Based Preliminary Risk Assessment
            </p>

            <div class="risk">

                Drug:
                {html.escape(str(data["drug_name"]))}

                <br>

                Predicted Risk:
                {data["risk_probability"]:.1%}

                <br>

                Category:
                {html.escape(str(data["risk_label"]))}

            </div>

            <h2>Drug Information</h2>

            <table>

                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>

                <tr>
                    <td>Drug Name</td>
                    <td>{html.escape(str(data["drug_name"]))}</td>
                </tr>

                <tr>
                    <td>Drug Class</td>
                    <td>{html.escape(str(data["drug_class"]))}</td>
                </tr>

            </table>


            <h2>Patient Profile</h2>

            <table>

                <tr>
                    <th>Parameter</th>
                    <th>Value</th>
                </tr>

                <tr>
                    <td>Age</td>
                    <td>{data["age"]}</td>
                </tr>

                <tr>
                    <td>Sex</td>
                    <td>{data["sex"]}</td>
                </tr>

                <tr>
                    <td>BMI</td>
                    <td>{data["bmi"]}</td>
                </tr>

                <tr>
                    <td>Dosage</td>
                    <td>{data["dosage_mg"]} mg/day</td>
                </tr>

                <tr>
                    <td>Duration</td>
                    <td>{data["duration_days"]} days</td>
                </tr>

                <tr>
                    <td>Concurrent Medications</td>
                    <td>{data["concurrent_meds_count"]}</td>
                </tr>

                <tr>
                    <td>Baseline ANA</td>
                    <td>{data["baseline_ana_titer"]}</td>
                </tr>

                <tr>
                    <td>Baseline CRP</td>
                    <td>{data["baseline_crp"]} mg/L</td>
                </tr>

                <tr>
                    <td>Baseline WBC</td>
                    <td>{data["baseline_wbc"]}</td>
                </tr>

            </table>


            <h2>Model Explanation</h2>

            <table>

                <tr>
                    <th>Factor</th>
                    <th>SHAP Value</th>
                    <th>Direction</th>
                </tr>

                {factors_html}

            </table>


            <h2>Immune-System Assessment</h2>

            <p>
                <strong>{immune_status}</strong>
            </p>

            <p>
                {html.escape(immune_text)}
            </p>


            <h2>Pre-Market Screening</h2>

            <p>
                This AI-assisted screening provides a preliminary
                assessment of potential immune-related safety signals.
            </p>

            <p>
                It is not a real laboratory result and must not be
                used as a clinical or regulatory decision.
            </p>


            <h2>Recommendation</h2>

            <p>
                {html.escape(recommendation)}
            </p>


            <div class="warning">

                <strong>⚠️ Important Disclaimer</strong>

                <p>
                This system is a research and educational prototype
                trained using synthetic data. It has not been
                clinically validated and does not replace laboratory
                testing, clinical judgment, toxicology studies,
                regulatory assessment, or medical advice.
                </p>

            </div>

            <br>

            <p>
                Generated by Drug-Induced Autoimmune Safety AI
            </p>

        </body>
        </html>
        """

        components.html(
            printable_html,
            height=80,
            scrolling=False
        )


# ============================================================
# PAGE 4 - ABOUT
# ============================================================

elif page == "ℹ️ About":

    st.header(
        "ℹ️ About the Project"
    )

    st.markdown(
        """
        ## 🧬 Early Prediction of Drug-Induced Autoimmune Reactions Using AI

        This project proposes an explainable machine-learning framework
        for the preliminary prediction of potential drug-induced
        autoimmune reaction risk.

        ### 🔬 Main Components

        **1. Patient & Drug Data**

        The system considers demographic, clinical, laboratory,
        genetic-risk and drug-related features.

        **2. Machine Learning**

        A trained classification model estimates the probability
        of a drug-induced autoimmune reaction.

        **3. Explainable AI**

        SHAP is used to identify which input features contributed
        most strongly to an individual prediction.

        **4. Pre-Market AI Screening**

        The proposed laboratory module demonstrates how a drug-level
        preliminary safety signal could be screened before wider
        clinical/market exposure.

        **5. Automated Risk Report**

        The system generates a structured report containing:

        - Patient information
        - Drug information
        - Predicted risk
        - Risk category
        - SHAP-based explanation
        - Preliminary immune-system assessment
        - Pre-market screening result
        - Recommendation
        - Safety disclaimer

        ### ⚠️ Important Limitation

        The current student prototype uses a literature-informed
        synthetic dataset. Therefore, its predictions should be
        considered demonstrations of the machine-learning framework,
        not real-world clinical evidence.

        ### 🎓 Project Purpose

        The goal is to demonstrate how **AI + Explainable AI +
        Drug Safety Screening** can be combined into an interpretable
        decision-support prototype.
        """
    )

    st.divider()

    st.info(
        f"Model currently in use: {meta['best_model_name']}"
    )

    st.warning(
        "Research / Educational Demonstration Only — "
        "Not for clinical use."
    )
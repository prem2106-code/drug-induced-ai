import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st
import streamlit.components.v1 as components
import html
import textwrap

from pathlib import Path

from preprocessing import get_feature_names
from generate_dataset import DRUGS
from chatbox import build_local_chat_response


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Drug-Induced Autoimmune Safety AI",
    page_icon=":material/biotech:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PREMIUM LIGHT MEDICAL AI UI
# ============================================================

st.markdown(
    """
<style>

/* ==========================================================
   GOOGLE FONTS
   ========================================================== */

@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');


/* ==========================================================
   GLOBAL PAGE
   ========================================================== */

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    letter-spacing: 0.01em;
}

:root {
    --brand-teal: #087f8c;
    --brand-cyan: #16b8b2;
    --brand-ink: #163b4d;
    --brand-mist: #e9fbf8;
}

.stApp {

    min-height: 100vh;

    background:
        linear-gradient(
            rgba(245, 252, 255, 0.91),
            rgba(240, 248, 255, 0.94)
        ),

        url("https://images.unsplash.com/photo-1530026405186-ed1f139313f8?auto=format&fit=crop&w=2200&q=85");

    background-size: cover;
    background-position: center;
    background-attachment: fixed;

    color: #183b56;
}


/* ==========================================================
   ANIMATED BACKGROUND ORBS
   ========================================================== */

.stApp::before {

    content: "";

    position: fixed;

    width: 550px;
    height: 550px;

    top: -220px;
    left: -180px;

    background:
        radial-gradient(
            circle,
            rgba(0, 196, 255, 0.20),
            transparent 70%
        );

    filter: blur(35px);

    animation:
        lightOrbOne 12s ease-in-out infinite;

    pointer-events: none;

    z-index: 0;
}


.stApp::after {

    content: "";

    position: fixed;

    width: 500px;
    height: 500px;

    right: -180px;
    bottom: -180px;

    background:
        radial-gradient(
            circle,
            rgba(124, 77, 255, 0.16),
            transparent 70%
        );

    filter: blur(40px);

    animation:
        lightOrbTwo 15s ease-in-out infinite;

    pointer-events: none;

    z-index: 0;
}


@keyframes lightOrbOne {

    0% {
        transform:
            translate(0, 0)
            scale(1);
    }

    50% {
        transform:
            translate(150px, 100px)
            scale(1.25);
    }

    100% {
        transform:
            translate(0, 0)
            scale(1);
    }
}


@keyframes lightOrbTwo {

    0% {
        transform:
            translate(0, 0)
            scale(1);
    }

    50% {
        transform:
            translate(-130px, -100px)
            scale(1.2);
    }

    100% {
        transform:
            translate(0, 0)
            scale(1);
    }
}


/* ==========================================================
   MAIN CONTENT
   ========================================================== */

.block-container {

    max-width: 1450px;

    padding-top: 1.5rem;
    padding-bottom: 4rem;

    position: relative;

    z-index: 1;
}


/* ==========================================================
   STREAMLIT HEADER TRANSPARENT
   ========================================================== */

header[data-testid="stHeader"] {

    background:
        transparent !important;
}


/* ==========================================================
   HERO HEADER
   ========================================================== */

.hero-container {

    position: relative;

    padding: 55px 35px;

    margin-bottom: 30px;

    text-align: center;

    border-radius: 32px;

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.88),
            rgba(235,250,255,0.78),
            rgba(247,242,255,0.82)
        );

    border:
        1px solid
        rgba(0, 170, 220, 0.25);

    backdrop-filter:
        blur(22px);

    -webkit-backdrop-filter:
        blur(22px);

    box-shadow:
        0 20px 70px
        rgba(41, 113, 145, 0.15),

        inset 0 1px 0
        rgba(255,255,255,0.95);

    overflow: hidden;

    animation:
        heroEntrance 1s ease-out;
}

.report-hero {
    border: 1px solid rgba(22,184,178,0.30);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    margin-bottom: 20px;
    background: linear-gradient(135deg, rgba(22,184,178,0.10), rgba(124,58,237,0.09));
    box-shadow: 0 14px 38px rgba(22,104,115,0.10);
    animation: reportReveal 0.7s ease both;
    transition: transform 0.35s ease, box-shadow 0.35s ease;
}

.report-hero:hover {
    transform: scale(1.012) translateY(-3px);
    box-shadow: 0 20px 48px rgba(22,104,115,0.16);
}

@keyframes reportReveal {
    from { opacity: 0; transform: translateY(14px) scale(0.98); }
    to { opacity: 1; transform: translateY(0) scale(1); }
}

.stMarkdown h2,
.stMarkdown h3,
[data-testid="stHeading"] {
    transition: transform 0.25s ease, color 0.25s ease;
    transform-origin: left center;
}

.stMarkdown h2:hover,
.stMarkdown h3:hover,
[data-testid="stHeading"]:hover {
    transform: scale(1.025);
    color: #087f8c !important;
}


@keyframes heroEntrance {

    from {

        opacity: 0;

        transform:
            translateY(-25px)
            scale(0.97);
    }

    to {

        opacity: 1;

        transform:
            translateY(0)
            scale(1);
    }
}


/* ==========================================================
   HERO LIGHT SWEEP ANIMATION
   ========================================================== */

.hero-container::before {

    content: "";

    position: absolute;

    top: -100px;

    left: -30%;

    width: 35%;

    height: 320px;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(0, 210, 255, 0.20),
            transparent
        );

    transform:
        rotate(20deg);

    animation:
        heroSweep 6s ease-in-out infinite;
}


@keyframes heroSweep {

    0% {
        left: -30%;
    }

    50% {
        left: 100%;
    }

    100% {
        left: 120%;
    }
}


/* ==========================================================
   DNA ICON
   ========================================================== */

.dna-glow {

    width: 86px;
    height: 86px;
    margin: 0 auto 15px;
    display: grid;
    place-items: center;
    border: 1px solid rgba(8,127,140,0.25);
    border-radius: 28px;
    background: linear-gradient(145deg, rgba(255,255,255,0.92), rgba(220,250,246,0.8));
    color: var(--brand-teal);
    font-size: 48px;

    margin-bottom: 15px;

    filter:
        drop-shadow(
            0 0 8px
            rgba(0, 190, 255, 0.65)
        )

        drop-shadow(
            0 0 25px
            rgba(80, 80, 255, 0.30)
        );

    animation:
        dnaFloat 3s ease-in-out infinite;
}

.dna-glow svg {
    width: 52px;
    height: 52px;
    animation: dnaSpin 12s linear infinite;
}

@keyframes dnaSpin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}


@keyframes dnaFloat {

    0%, 100% {

        transform:
            translateY(0)
            rotate(0deg);
    }

    50% {

        transform:
            translateY(-12px)
            rotate(5deg);
    }
}


/* ==========================================================
   TITLE
   ========================================================== */

.hero-title {

    position: relative;

    font-family:
        'Space Grotesk',
        sans-serif;

    font-size:
        clamp(30px, 5vw, 58px);

    font-weight:
        800;

    letter-spacing:
        3px;

    background:
        linear-gradient(
            90deg,
            #0077b6,
            #00b4d8,
            #6c63ff,
            #b5179e,
            #0077b6
        );

    background-size:
        300% auto;

    -webkit-background-clip:
        text;

    -webkit-text-fill-color:
        transparent;

    animation:
        titleGradient 5s linear infinite;

    display: block;
}


@keyframes titleGradient {

    0% {
        background-position:
            0% center;
    }

    50% {
        background-position:
            100% center;
    }

    100% {
        background-position:
            0% center;
    }
}


/* ==========================================================
   TITLE UNDERLINE
   ========================================================== */

.hero-title::after {

    content: "";

    display: block;

    width: 100px;

    height: 4px;

    margin:
        15px auto 0;

    border-radius:
        20px;

    background:
        linear-gradient(
            90deg,
            #00b4d8,
            #6c63ff,
            #b5179e
        );

    box-shadow:
        0 0 12px
        rgba(0,180,216,0.5);

    animation:
        linePulse 2s ease-in-out infinite;
}


@keyframes linePulse {

    0%, 100% {

        width: 80px;

        opacity:
            0.6;
    }

    50% {

        width: 150px;

        opacity:
            1;
    }
}


/* ==========================================================
   SUBTITLE
   ========================================================== */

.hero-subtitle {

    margin-top:
        18px;

    color:
        #486581;

    font-size:
        16px;

    font-weight:
        500;

    letter-spacing:
        0.5px;
}


/* ==========================================================
   BADGES
   ========================================================== */

.hero-badges {

    display:
        flex;

    justify-content:
        center;

    flex-wrap:
        wrap;

    gap:
        10px;

    margin-top:
        25px;
}


.hero-badge {

    padding:
        9px 16px;

    border-radius:
        50px;

    font-size:
        12px;

    font-weight:
        700;

    color:
        #155e75;

    background:
        rgba(255,255,255,0.78);

    border:
        1px solid
        rgba(0,180,216,0.25);

    box-shadow:
        0 5px 18px
        rgba(0,120,170,0.08);

    transition:
        all 0.3s ease;
}

.badge-icon {
    display: inline-grid;
    place-items: center;
    width: 20px;
    height: 20px;
    margin-right: 5px;
    border-radius: 7px;
    color: #087f8c;
    background: #d8f6f1;
    font-size: 17px;
    line-height: 1;
    vertical-align: -2px;
    transition: transform 0.3s ease;
}

.hero-badge:hover .badge-icon {
    transform: rotate(12deg) scale(1.15);
}


.hero-badge:hover {

    transform:
        translateY(-5px)
        scale(1.04);

    background:
        rgba(255,255,255,0.98);

    border-color:
        #00b4d8;

    box-shadow:
        0 8px 25px
        rgba(0,180,216,0.20);
}


/* ==========================================================
   HEADINGS
   ========================================================== */

h1 {

    font-family:
        'Space Grotesk',
        sans-serif !important;

    color:
        #123b5d !important;
}


h2, h3 {

    font-family: 'Space Grotesk', sans-serif !important;

    color:
        #164e63 !important;

    font-weight:
        700 !important;
}


/* ==========================================================
   NAVIGATION
   ========================================================== */

div[role="radiogroup"] {

    background:
        rgba(255,255,255,0.78);

    border:
        1px solid
        rgba(0,180,216,0.22);

    padding:
        8px 12px;

    border-radius:
        18px;

    backdrop-filter:
        blur(15px);

    box-shadow:
        0 10px 35px
        rgba(41,113,145,0.12);

}


div[role="radiogroup"] label {

    color:
        #24556f !important;

    font-weight:
        600 !important;
}

/* Purple selected circle for navigation radio options */
div[role="radiogroup"] label > div,
div[role="radiogroup"] label[data-selected="true"] > div > div {
    background-color: transparent !important;
    box-shadow: none !important;
}

div[role="radiogroup"] label[data-selected="true"] > div > div > div {
    background-color: #7c3aed !important;
    border-color: #7c3aed !important;
    box-shadow: none !important;
}

div[role="radiogroup"] label[data-selected="true"]:hover > div > div > div,
div[role="radiogroup"] label[data-selected="true"]:focus-within > div > div > div {
    background-color: #7c3aed !important;
    border-color: #7c3aed !important;
}


/* ==========================================================
   BUTTONS
   ========================================================== */

.stButton > button {

    border:
        1px solid
        rgba(0,180,216,0.35);

    border-radius:
        14px;

    background:
        linear-gradient(
            135deg,
            #ffffff,
            #eafaff
        );

    color:
        #075985;

    font-weight:
        700;

    min-height:
        48px;

    box-shadow:
        0 8px 25px
        rgba(0,120,170,0.10);

    transition:
        all 0.25s ease;

    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.02em;
}


.stButton > button:hover {

    transform:
        translateY(-3px)
        scale(1.01);

    border-color:
        #00b4d8;

    box-shadow:
        0 12px 30px
        rgba(0,180,216,0.20);
}


.stButton > button:active {

    transform:
        scale(0.97);
}

.stButton > button:focus,
.stButton > button:focus-visible,
div[data-baseweb="select"] > div:focus-within,
input:focus,
textarea:focus {
    border-color: var(--brand-cyan) !important;
    box-shadow: 0 0 0 3px rgba(22,184,178,0.18) !important;
    outline: none !important;
}


/* ==========================================================
   INPUT BOXES
   ========================================================== */

div[data-baseweb="select"] > div {

    background:
        rgba(255,255,255,0.85)
        !important;

    border:
        1px solid
        rgba(0,150,190,0.22)
        !important;

    border-radius:
        12px
        !important;
}


input {

    background:
        rgba(255,255,255,0.85)
        !important;

    color:
        #123b5d
        !important;

    border-radius:
        10px
        !important;
}

/* ==========================================================
    SLIDER - LIGHT PURPLE TRACK + DARK PURPLE KNOB
   ========================================================== */

/* ==========================================================
   SLIDER - LIGHT PURPLE TRACK + PURPLE ACTIVE LINE + KNOB
   ========================================================== */

/* Slider text */
div[data-testid="stSlider"] label,
div[data-testid="stSlider"] p {
    color: #111827 !important;
}

/* Full slider track - LIGHT PURPLE */
div[data-testid="stSlider"]
[data-baseweb="slider"]
[role="slider"] {
    background: #7c3aed !important;
    border: 3px solid #6d28d9 !important;
    box-shadow: 0 0 12px rgba(109, 40, 217, 0.35) !important;
}

/* Slider track */
div[data-testid="stSlider"]
[data-baseweb="slider"] > div > div {
    background-color: #ddd6fe !important;
}

/* ACTIVE / FILLED LINE - LIGHT PURPLE */
div[data-testid="stSlider"]
[data-baseweb="slider"]
> div > div > div {
    background-color: #8b5cf6 !important;
}

/* FORCE ACTIVE SLIDER LINE */
div[data-testid="stSlider"] [data-baseweb="slider"] div[style] {
    background: #8b5cf6 !important;
    background-color: #8b5cf6 !important;
}

/* Current Streamlit knob wrapper */
div[data-testid="stSlider"] [data-rac][style*="position: absolute"] {
    background: #7c3aed !important;
    background-color: #7c3aed !important;
    border: 3px solid #6d28d9 !important;
    border-radius: 50% !important;
    box-shadow: 0 0 12px rgba(109, 40, 217, 0.35) !important;
}

/* Knob */
div[data-testid="stSlider"] [role="slider"] {
    background-color: #7c3aed !important;
    border: 3px solid #6d28d9 !important;
    box-shadow: 0 0 12px rgba(109, 40, 217, 0.35) !important;
}

/* Knob hover */
div[data-testid="stSlider"] [role="slider"]:hover {
    background-color: #6d28d9 !important;
    border-color: #5b21b6 !important;
    box-shadow: 0 0 18px rgba(109, 40, 217, 0.45) !important;
}

/* Range input */
div[data-testid="stSlider"] input[type="range"] {
    accent-color: #8b5cf6 !important;
}

/* Chrome / Edge */
div[data-testid="stSlider"]
input[type="range"]::-webkit-slider-runnable-track {
    background: #ddd6fe !important;
}

div[data-testid="stSlider"]
input[type="range"]::-webkit-slider-thumb {
    background: #7c3aed !important;
    border: 2px solid #6d28d9 !important;
}

/* Firefox */
div[data-testid="stSlider"]
input[type="range"]::-moz-range-track {
    background: #ddd6fe !important;
}

div[data-testid="stSlider"]
input[type="range"]::-moz-range-progress {
    background: #8b5cf6 !important;
}

div[data-testid="stSlider"]
input[type="range"]::-moz-range-thumb {
    background: #7c3aed !important;
    border: 2px solid #6d28d9 !important;
}
/* ==========================================================
   METRIC CARDS
   ========================================================== */

div[data-testid="stMetric"] {

    background:
        linear-gradient(
            135deg,
            rgba(255,255,255,0.86),
            rgba(238,250,255,0.78)
        );

    border:
        1px solid
        rgba(0,180,216,0.20);

    padding:
        20px;

    border-radius:
        18px;

    box-shadow:
        0 10px 30px
        rgba(41,113,145,0.10);

    transition:
        all 0.3s ease;
}


div[data-testid="stMetric"]:hover {

    transform:
        translateY(-6px);

    box-shadow:
        0 15px 35px
        rgba(0,180,216,0.16);

    transform: translateY(-6px) scale(1.025);
}


div[data-testid="stMetricLabel"] {

    color:
        #627d98 !important;
}


div[data-testid="stMetricValue"] {

    color:
        #0086ad !important;

    font-family:
        'Space Grotesk',
        sans-serif;
}


/* ==========================================================
   ALERT BOXES
   ========================================================== */

div[data-testid="stAlert"] {

    border-radius:
        16px !important;

    border:
        1px solid
        rgba(0,180,216,0.18);

    backdrop-filter:
        blur(10px);
}


/* ==========================================================
   CHAT
   ========================================================== */

div[data-testid="stChatMessage"] {

    border-radius:
        18px;

    margin-bottom:
        12px;

    border:
        1px solid
        rgba(0,180,216,0.15);

    background:
        rgba(255,255,255,0.75);

    backdrop-filter:
        blur(15px);

    animation:
        chatAppear 0.35s ease;
}


@keyframes chatAppear {

    from {

        opacity:
            0;

        transform:
            translateY(12px);
    }

    to {

        opacity:
            1;

        transform:
            translateY(0);
    }
}


div[data-testid="stChatInput"] {

    border-radius:
        18px !important;
}


div[data-testid="stChatInput"] textarea {

    background:
        rgba(255,255,255,0.90)
        !important;

    color:
        #123b5d
        !important;
}


/* ==========================================================
   TABLES
   ========================================================== */

table {

    border-radius:
        12px;

    overflow:
        hidden;
}


thead tr th {

    background:
        linear-gradient(
            90deg,
            rgba(0,180,216,0.15),
            rgba(108,99,255,0.12)
        )
        !important;

    color:
        #164e63 !important;
}


tbody tr:hover {

    background:
        rgba(0,180,216,0.05)
        !important;

        transform: scale(1.006);
        transition: transform 0.2s ease, background 0.2s ease;
}


/* ==========================================================
   DATAFRAME
   ========================================================== */

div[data-testid="stDataFrame"] {

    border-radius:
        16px;

    overflow:
        hidden;

    border:
        1px solid
        rgba(0,180,216,0.15);

    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

div[data-testid="stDataFrame"]:hover {
    transform: scale(1.008);
    box-shadow: 0 14px 34px rgba(8,127,140,0.12);
}


/* ==========================================================
   DIVIDER
   ========================================================== */

hr {

    border:
        none !important;

    height:
        1px !important;

    background:
        linear-gradient(
            90deg,
            transparent,
            rgba(0,180,216,0.45),
            rgba(108,99,255,0.40),
            transparent
        )
        !important;
}


/* ==========================================================
   SCROLLBAR
   ========================================================== */

::-webkit-scrollbar {

    width:
        8px;
}


::-webkit-scrollbar-track {

    background:
        #eef8fc;
}


::-webkit-scrollbar-thumb {

    background:
        linear-gradient(
            #00b4d8,
            #6c63ff
        );

    border-radius:
        10px;
}


/* ==========================================================
   TEXT SELECTION
   ========================================================== */

::selection {

    background:
        rgba(0,180,216,0.22);

    color:
        #123b5d;
}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (max-width: 768px) {

    .hero-title {

        font-size:
            28px;
    }

    .block-container {

        padding-left:
            1rem;

        padding-right:
            1rem;
    }

    .hero-container {

        padding:
            35px 15px;
    }

}
/* ===== FORCE STREAMLIT SLIDER PURPLE ===== */

div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #7c3aed !important;
    border: 3px solid #5b21b6 !important;
    box-shadow: 0 0 10px rgba(124, 58, 237, 0.4) !important;
}

/* Full track */
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
    background-color: #ddd6fe !important;
}

/* Active / filled line */
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {
    background-color: #8b5cf6 !important;
}

/* Force every slider child background */
div[data-testid="stSlider"] [data-baseweb="slider"] div {
    --slider-color: #8b5cf6 !important;
}

/* Knob hover */
div[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"]:hover {
    background-color: #6d28d9 !important;
    border-color: #4c1d95 !important;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# OLLAMA CONFIGURATION
# ============================================================

OLLAMA_MODEL = "gemma4:e2b"


def get_ollama_response(messages):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": False,
        },
        timeout=10,
    )
    response.raise_for_status()
    result = response.json()
    message = result.get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama returned an empty or invalid chat response.")
    return content


# ============================================================
# RISK BAND
# ============================================================

def get_risk_band(probability):

    if probability < 0.15:

        return {
            "label": "LOW RISK",
            "icon": "LOW",
            "level": "low",

            "summary": (
                "The model detected a relatively low preliminary "
                "drug-induced autoimmune risk signal for the entered profile."
            ),

            "why": (
                "The predicted probability is below 15%, which falls within "
                "the project's LOW RISK range. Based on the current patient, "
                "drug, treatment, and laboratory inputs, the model found fewer "
                "strong factors pushing the prediction toward an autoimmune "
                "safety concern."
            ),

            "interpretation": [
                "The combined input profile does not show a strong risk pattern.",
                "The strongest model factors are not heavily pushing the prediction upward.",
                "Some factors may still contribute to risk, but their combined effect remains lower.",
                "LOW RISK does not mean zero risk or guaranteed clinical safety."
            ],

            "recommendation": (
                "Continue appropriate safety evaluation and monitoring. "
                "This AI result should be treated as a preliminary research signal."
            )
        }

    elif probability < 0.40:

        return {
            "label": "MODERATE RISK",
            "icon": "MODERATE",
            "level": "moderate",

            "summary": (
                "The model detected a moderate preliminary immune-related "
                "risk signal for the current patient and drug profile."
            ),

            "why": (
                "The predicted probability falls between 15% and 40%, placing "
                "the result in the project's MODERATE RISK range. This means "
                "that one or more patient characteristics, drug-related factors, "
                "laboratory values, treatment exposure, or genetic-risk indicators "
                "are contributing enough to move the prediction above the low-risk range."
            ),

            "interpretation": [
                "Some important factors are pushing the predicted risk upward.",
                "The combined patient and drug profile requires closer interpretation.",
                "The risk signal is not strong enough to enter the HIGH RISK category.",
                "The SHAP explanation should be reviewed to identify the strongest contributors."
            ],

            "recommendation": (
                "Review the major risk-driving factors and consider additional "
                "immune-related evaluation, monitoring, and scientific investigation."
            )
        }

    else:

        return {
            "label": "HIGH RISK",
            "icon": "HIGH",
            "level": "high",

            "summary": (
                "The model detected a strong preliminary immune-related "
                "risk signal for the current patient and drug profile."
            ),

            "why": (
                "The predicted probability is 40% or higher, placing the result "
                "in the project's HIGH RISK range. Multiple patient, drug, "
                "laboratory, treatment, or genetic-risk factors may be combining "
                "to push the model toward a stronger immune-related safety signal."
            ),

            "interpretation": [
                "Several factors may be contributing positively to the predicted risk.",
                "The combined profile deserves additional investigation.",
                "SHAP analysis should be used to identify the strongest risk-driving factors.",
                "HIGH RISK is a model-generated research signal and does not prove that a drug causes autoimmune disease."
            ],

            "recommendation": (
                "Consider additional nonclinical evaluation, immune-related safety "
                "assessment, monitoring, and expert review."
            )
        }


# ============================================================
# DRUG DETAIL
# ============================================================

def build_drug_detail_paragraph(data):

    risk = get_risk_band(
        float(data["risk_probability"])
    )

    return (
        f"The predicted {risk['label'].lower()} for "
        f"{data['drug_name']} is based on a combination of patient "
        f"and drug factors. This medication belongs to the "
        f"{data['drug_class']} class, with a dosage of "
        f"{data['dosage_mg']} mg/day over "
        f"{data['duration_days']} days. The model also considered "
        f"concurrent medications, baseline ANA "
        f"{data['baseline_ana_titer']}, CRP "
        f"{data['baseline_crp']} mg/L, WBC "
        f"{data['baseline_wbc']}, and patient characteristics such "
        f"as age, sex, BMI, and autoimmune history. The current "
        f"risk level is {risk['label']} "
        f"({data['risk_probability']:.1%}). "
        f"The final interpretation is: {risk['summary']}"
    )


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR.parent / "models"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    pipe = joblib.load(
        MODEL_DIR / "final_model.joblib"
    )

    meta = joblib.load(
        MODEL_DIR / "meta.joblib"
    )

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

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ============================================================
# HERO HEADER
# ============================================================

st.html(
    textwrap.dedent(
        """
    <div class="hero-container">

        <div class="dna-glow">
            <svg viewBox="0 0 64 64" aria-hidden="true">
                <path d="M20 8c20 8 24 16 24 24S40 48 20 56M44 8C24 16 20 24 20 32s4 16 24 24M24 15h16M20 24h24M20 40h24M24 49h16" fill="none" stroke="currentColor" stroke-linecap="round" stroke-width="3"/>
            </svg>
        </div>

        <h1 class="hero-title">
            AUTOIMMUNE SAFETY AI
        </h1>

        <p class="hero-subtitle">
            Explainable Artificial Intelligence Framework
            for Preliminary Drug Safety Risk Assessment
        </p>

        <div class="hero-badges">

            <div class="hero-badge">
                <span class="badge-icon">⌁</span> MACHINE LEARNING
            </div>

            <div class="hero-badge">
                <span class="badge-icon">⌕</span> SHAP EXPLAINABLE AI
            </div>

            <div class="hero-badge">
                <span class="badge-icon">◈</span> SAFETY SCREENING
            </div>

            <div class="hero-badge">
                <span class="badge-icon">◌</span> LOCAL OLLAMA AI
            </div>

        </div>

    </div>
        """
    )
)

# ============================================================
# NAVIGATION
# ============================================================

page = st.radio(
    "Navigation",
    [
        "Prediction",
        "Pre-Market Lab",
        "Risk Report",
        "AI Chat",
        "About",
    ],
    horizontal=True,
)

st.divider()


# ============================================================
# CREATE INPUT DATAFRAME
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

    return pd.DataFrame([
        {
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
        }
    ])


# ============================================================
# SHAP
# ============================================================

def calculate_shap(input_df):

    try:

        preproc = pipe.named_steps["preprocess"]

        clf = pipe.named_steps["clf"]

        feat_names = get_feature_names(
            preproc
        )

        X_input_transformed = preproc.transform(
            input_df
        )

        if hasattr(
            X_input_transformed,
            "toarray"
        ):

            X_input_transformed = (
                X_input_transformed.toarray()
            )

        explainer = shap.TreeExplainer(
            clf
        )

        shap_values = explainer.shap_values(
            X_input_transformed
        )

        if isinstance(
            shap_values,
            list
        ):

            sv = np.asarray(
                shap_values[1]
            )

            if sv.ndim == 2:
                sv = sv[0]

        else:

            sv = np.asarray(
                shap_values
            )

            if sv.ndim == 3:

                sv = sv[0, :, 1]

            elif sv.ndim == 2:

                if sv.shape[1] == 2:

                    sv = sv[:, 1]

                elif sv.shape[0] == 2:

                    sv = sv[1]

                else:

                    sv = sv[0]

        sv = np.asarray(
            sv
        ).reshape(-1)

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

    except Exception:

        return None


# ============================================================
# SHAP HELPERS
# ============================================================

def clean_feature_name(feature):

    feature = str(feature)

    feature = feature.replace(
        "num__",
        ""
    )

    feature = feature.replace(
        "cat__",
        ""
    )

    feature = feature.replace(
        "remainder__",
        ""
    )

    feature = feature.replace(
        "_",
        " "
    )

    feature_lower = feature.lower()

    replacements = {

        "baseline ana titer":
            "Baseline ANA Titer",

        "baseline crp":
            "Baseline CRP Level",

        "baseline wbc":
            "Baseline White Blood Cell Count",

        "prior autoimmune history":
            "Previous Autoimmune Disease History",

        "family autoimmune history":
            "Family History of Autoimmune Disease",

        "slow acetylator":
            "Slow Acetylator Status",

        "hla risk allele":
            "HLA Risk Allele",

        "concurrent meds count":
            "Number of Concurrent Medications",

        "dosage mg":
            "Drug Dosage",

        "duration days":
            "Duration of Drug Exposure",

        "drug name":
            "Selected Drug",

        "drug class":
            "Drug Class",
    }

    for key, value in replacements.items():

        if key in feature_lower:

            return value

    return feature.title()


def explain_factor(
    feature,
    shap_value
):

    name = clean_feature_name(
        feature
    )

    if shap_value > 0:

        return (
            f"The factor '{name}' pushed the AI prediction "
            f"toward a higher autoimmune risk."
        )

    return (
        f"The factor '{name}' pushed the AI prediction "
        f"toward a lower autoimmune risk."
    )


def get_detailed_factor_explanation(contrib):

    if contrib is None or len(contrib) == 0:

        return {
            "increases": [],
            "decreases": []
        }

    increases = []

    decreases = []

    for feature, value in contrib.items():

        item = {

            "factor":
                clean_feature_name(feature),

            "value":
                float(value),

            "explanation":
                explain_factor(
                    feature,
                    float(value)
                )
        }

        if value > 0:

            increases.append(item)

        else:

            decreases.append(item)

    increases = sorted(
        increases,
        key=lambda x: abs(x["value"]),
        reverse=True
    )

    decreases = sorted(
        decreases,
        key=lambda x: abs(x["value"]),
        reverse=True
    )

    return {
        "increases": increases,
        "decreases": decreases
    }


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

if page == "Prediction":

    st.header(
        "Patient & Drug Risk Prediction"
    )

    st.info(
        "Enter patient and drug information to estimate a "
        "preliminary probability of a drug-induced autoimmune reaction."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Patient Details"
        )

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

    with col2:

        st.subheader(
            "Drug & Laboratory Details"
        )

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

    if st.button(
        "Predict Risk",
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

        risk_band = get_risk_band(
            float(proba)
        )

        st.session_state.prediction_done = True

        st.session_state.prediction_data = {

            "drug_name":
                drug_name,

            "drug_class":
                DRUGS[drug_name]["class"],

            "age":
                age,

            "sex":
                sex,

            "bmi":
                bmi,

            "dosage_mg":
                dosage_mg,

            "duration_days":
                duration_days,

            "concurrent_meds_count":
                concurrent_meds_count,

            "baseline_ana_titer":
                baseline_ana_titer,

            "baseline_crp":
                baseline_crp,

            "baseline_wbc":
                baseline_wbc,

            "risk_probability":
                float(proba),

            "risk_label":
                risk_band["label"],
        }

        contrib = calculate_shap(
            input_df
        )

        st.session_state.shap_contrib = contrib

        st.success(
            "Prediction completed successfully."
        )

        st.subheader(
            "Prediction Result"
        )

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:

            st.metric(
                "Predicted Risk",
                f"{proba:.1%}"
            )

        with result_col2:

            st.metric(
                "Risk Category",
                f"{risk_band['icon']} {risk_band['label']}"
            )

        with result_col3:

            st.metric(
                "Drug",
                drug_name
            )

        st.progress(
            min(float(proba), 1.0)
        )

        st.markdown(
            f"""
## Risk Interpretation

{risk_band['summary']}

### {risk_band['icon']} Why is this classified as {risk_band['label']}?

{risk_band['why']}
"""
        )

        st.markdown(
            "### Detailed Interpretation"
        )

        for point in risk_band["interpretation"]:

            st.markdown(
                f"- {point}"
            )

        st.markdown(
            "### Preliminary Recommendation"
        )

        st.info(
            risk_band["recommendation"]
        )

        st.markdown(
            "### Drug Details\n"
            f"- Drug: **{drug_name}**\n"
            f"- Drug Class: **{DRUGS[drug_name]['class']}**\n"
            f"- Dosage: **{dosage_mg} mg/day**\n"
            f"- Duration: **{duration_days} days**\n"
            f"- Concurrent Medications: **{concurrent_meds_count}**\n"
            f"- Baseline ANA: **{baseline_ana_titer}**\n"
            f"- Baseline CRP: **{baseline_crp} mg/L**\n"
            f"- Baseline WBC: **{baseline_wbc} x10^9/L**"
        )

        st.markdown(
            "### Summary Paragraph\n"
            f"{build_drug_detail_paragraph(st.session_state.prediction_data)}"
        )

        st.subheader(
            "Why did the model predict this?"
        )

        if contrib is not None:

            fig, ax = plt.subplots(
                figsize=(9, 6)
            )

            colors = [
                "#d97706"
                if value > 0
                else "#087f8c"
                for value in contrib.values[::-1]
            ]

            ax.barh(
                [
                    clean_feature_name(x)
                    for x in contrib.index[::-1]
                ],
                contrib.values[::-1],
                color=colors
            )

            ax.axvline(
                0,
                linewidth=1
            )

            ax.set_xlabel(
                "SHAP Contribution"
            )

            ax.set_title(
                "Top Factors Influencing This Prediction"
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            st.caption(
                "Warm bars increase predicted risk | Cool bars decrease predicted risk."
            )

        else:

            st.info(
                "SHAP explanation could not be generated."
            )

    # ========================================================
    # SAVED FACTOR ANALYSIS
    # ========================================================

    saved_contrib = (
        st.session_state.shap_contrib
    )

    if (
        st.session_state.prediction_done
        and saved_contrib is not None
    ):

        factor_info = (
            get_detailed_factor_explanation(
                saved_contrib
            )
        )

        st.divider()

        st.subheader(
            "Detailed Factor Analysis"
        )

        increase_col, decrease_col = st.columns(2)

        with increase_col:

            st.markdown(
                "### Factors Increasing Predicted Risk"
            )

            if factor_info["increases"]:

                for item in factor_info["increases"]:

                    st.markdown(
                        f"""
### {item['factor']}

**SHAP Contribution:** `{item['value']:+.4f}`

{item['explanation']}
"""
                    )

            else:

                st.success(
                    "No major factors were identified as increasing "
                    "the predicted risk."
                )

        with decrease_col:

            st.markdown(
                "### Factors Decreasing Predicted Risk"
            )

            if factor_info["decreases"]:

                for item in factor_info["decreases"]:

                    st.markdown(
                        f"""
### {item['factor']}

**SHAP Contribution:** `{item['value']:+.4f}`

{item['explanation']}
"""
                    )

            else:

                st.info(
                    "No major factors were identified as decreasing "
                    "the predicted risk."
                )


# ============================================================
# PAGE 2 — PRE-MARKET LAB
# ============================================================

elif page == "Pre-Market Lab":

    st.header(
        "Pre-Market Immune Safety Lab"
    )

    st.info(
        "This module performs an AI-assisted preliminary screening "
        "of potential immune-related safety signals before considering "
        "a drug for wider clinical or market exposure."
    )

    st.warning(
        "Note: This is a research simulation using the project's "
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
### Selected Drug

**Drug:** {lab_drug}

**Drug Class:** {drug_class}
"""
    )

    st.divider()

    st.subheader(
        "Standardized AI Screening"
    )

    st.write(
        "The system evaluates the selected drug under a standardized "
        "reference profile so that the drug-level screening can be "
        "demonstrated consistently."
    )

    if st.button(
        "Run Pre-Market Lab Screening",
        type="primary",
        use_container_width=True
    ):

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

        lab_risk = get_risk_band(
            float(lab_probability)
        )

        concern = (
            lab_risk["label"]
            .replace(
                "RISK",
                "CONCERN"
            )
        )

        st.session_state.lab_result = {

            "drug_name":
                lab_drug,

            "drug_class":
                drug_class,

            "probability":
                float(lab_probability),

            "concern":
                concern,
        }

        st.divider()

        st.subheader(
            "Preliminary Immune Safety Assessment"
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
                f"{lab_risk['icon']} {concern}"
            )

        st.progress(
            min(
                float(lab_probability),
                1.0
            )
        )

        st.markdown(
            f"""
## Detailed Lab Explanation

The selected drug **{lab_drug}** belongs to the
**{drug_class}** class.

The standardized AI screening produced a potential
risk signal of **{lab_probability:.1%}**.

### {lab_risk['icon']} Why is the screening status {concern}?

{lab_risk['why']}

### Overall Screening Summary

{lab_risk['summary']}
"""
        )

        st.markdown(
            "### What does this Pre-Market result mean?"
        )

        for point in lab_risk["interpretation"]:

            st.markdown(
                f"- {point}"
            )

        st.markdown(
            "### Suggested Next Step"
        )

        st.info(
            lab_risk["recommendation"]
        )

        st.divider()

        st.subheader(
            "Potential Immune-System Impact"
        )

        if lab_probability < 0.15:

            st.success(
                "LOW CONCERN — The model did not identify "
                "a strong immune-related risk signal under the "
                "standardized screening profile."
            )

        elif lab_probability < 0.40:

            st.warning(
                "MODERATE CONCERN — The model identified "
                "a potential immune-related safety signal that "
                "may warrant additional investigation."
            )

        else:

            st.error(
                "HIGH CONCERN — The model identified a "
                "strong preliminary immune-related safety signal "
                "under the standardized screening profile."
            )

        st.divider()

        st.subheader(
            "Suggested Preclinical Evaluation Areas"
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

        st.caption(
            "Important: These are suggested evaluation areas for "
            "a student research demonstration. Actual drug-development "
            "testing must be designed and interpreted by qualified "
            "scientists, toxicologists, clinicians, and regulatory experts."
        )


# ============================================================
# PAGE 3 — RISK REPORT
# ============================================================

elif page == "Risk Report":

    st.header(
        "Drug Immune Safety Risk Report"
    )

    if not st.session_state.prediction_done:

        st.info(
            "No prediction report is available yet."
        )

        st.write(
            "Go to **Prediction**, enter the patient and drug "
            "details, and click **Predict Risk** first."
        )

    else:

        data = (
            st.session_state.prediction_data
        )

        contrib = (
            st.session_state.shap_contrib
        )

        lab = (
            st.session_state.lab_result
        )

        st.markdown(
            """
<div class="report-hero">

<h1>DRUG IMMUNE SAFETY REPORT</h1>

<p>
Explainable AI-Based Preliminary Risk Assessment
</p>

</div>
""",
            unsafe_allow_html=True
        )

        st.subheader(
            "Risk Summary"
        )

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

        risk_band = get_risk_band(
            float(
                data["risk_probability"]
            )
        )

        st.divider()

        st.markdown(
            f"""
## Risk Interpretation

{risk_band['summary']}

### {risk_band['icon']} Why was this classified as {risk_band['label']}?

{risk_band['why']}
"""
        )

        st.markdown(
            "### Detailed Prediction Explanation"
        )

        st.write(
            build_drug_detail_paragraph(
                data
            )
        )

        st.markdown(
            "### Detailed Interpretation"
        )

        for point in risk_band["interpretation"]:

            st.markdown(
                f"- {point}"
            )

        st.markdown(
            "### Preliminary Recommendation"
        )

        st.info(
            risk_band["recommendation"]
        )

        st.divider()

        st.subheader(
            "Patient Profile"
        )

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

        st.table(
            patient_df
        )

        st.divider()

        st.subheader(
            "Drug Information"
        )

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

        st.table(
            drug_df
        )

        st.divider()

        st.subheader(
            "Model Explanation"
        )

        if contrib is not None:

            factor_df = pd.DataFrame({

                "Factor": [
                    clean_feature_name(x)
                    for x in contrib.index
                ],

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

        st.subheader(
            "Immune-System Safety Assessment"
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

        st.subheader(
            "Pre-Market Lab Screening"
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

        else:

            st.info(
                "Pre-Market Lab screening has not been performed "
                "for this report."
            )

        st.caption(
            "The pre-market screening is an AI research "
            "simulation based on the project's synthetic-data "
            "model and should not be interpreted as an actual "
            "laboratory or regulatory result."
        )

        st.divider()

        st.subheader(
            "Preliminary Recommendation"
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

        st.subheader(
            "Print / Save Report"
        )

        st.write(
            "Click below to open the browser print dialog. "
            "You can select **Save as PDF**."
        )

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
<td>{html.escape(clean_feature_name(feature))}</td>
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

</style>

</head>

<body>

<button
    onclick="window.print()"
    style="
        padding:10px 20px;
        font-size:16px;
        margin-bottom:20px;
        cursor:pointer;
    "
>
Print / Save as PDF
</button>

<h1>
DRUG IMMUNE SAFETY REPORT
</h1>

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

<h2>
Drug Information
</h2>

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

<h2>
Patient Profile
</h2>

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

<h2>
Model Explanation
</h2>

<table>

<tr>
<th>Factor</th>
<th>SHAP Value</th>
<th>Direction</th>
</tr>

{factors_html}

</table>

<h2>
Immune-System Assessment
</h2>

<p>
<strong>{immune_status}</strong>
</p>

<p>
{html.escape(immune_text)}
</p>

<h2>
Recommendation
</h2>

<p>
{html.escape(recommendation)}
</p>

<div class="warning">

<strong>Important Disclaimer</strong>

<p>
This system is a research and educational prototype
trained using synthetic data. It has not been clinically
validated and does not replace laboratory testing,
clinical judgment, toxicology studies, regulatory
assessment, or medical advice.
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
            height=100,
            scrolling=True
        )


# ============================================================
# PAGE 4 — OLLAMA AI CHAT
# ============================================================

elif page == "AI Chat":

    st.header(
        "Drug Safety AI Assistant"
    )

    st.info(
        "Ask questions about the selected drug, autoimmune "
        "risk factors, SHAP explanation, prediction result, "
        "and the project's AI-based drug safety assessment."
    )

    st.info(
        "Offline chat answers are available. Ollama is optional for general questions."
    )

    # --------------------------------------------------------
    # PREVIOUS MESSAGES
    # --------------------------------------------------------

    for message in st.session_state.chat_messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

    user_prompt = st.chat_input(
        "Ask about the drug, risk prediction, immune impact..."
    )

    if user_prompt:

        st.session_state.chat_messages.append(
            {
                "role": "user",
                "content": user_prompt
            }
        )

        with st.chat_message(
            "user"
        ):

            st.markdown(
                user_prompt
            )

        # ----------------------------------------------------
        # PROJECT CONTEXT
        # ----------------------------------------------------

        project_context = """
You are the AI assistant inside a student research project called:

Drug-Induced Autoimmune Safety AI

The project uses machine learning and Explainable AI (SHAP)
to estimate a preliminary drug-induced autoimmune risk signal.

IMPORTANT:

This is a research and educational prototype using synthetic data.

It is NOT:

- A clinical diagnosis
- Medical advice
- A regulatory assessment
- An actual laboratory result
- Proof that a drug causes autoimmune disease

Always distinguish between:

1. AI model prediction
2. Scientific interpretation
3. Actual clinical evidence

Do not claim that the model proves that a drug causes
autoimmune disease.

Answer clearly and in simple language when possible.

If discussing medical decisions, remind the user that
qualified healthcare professionals should make clinical decisions.
"""

        # ----------------------------------------------------
        # CURRENT PREDICTION
        # ----------------------------------------------------

        if st.session_state.prediction_done:

            data = (
                st.session_state.prediction_data
            )

            project_context += f"""

CURRENT PREDICTION:

Drug:
{data["drug_name"]}

Drug Class:
{data["drug_class"]}

Patient Age:
{data["age"]}

Sex:
{data["sex"]}

BMI:
{data["bmi"]}

Dosage:
{data["dosage_mg"]} mg/day

Duration:
{data["duration_days"]} days

Concurrent Medications:
{data["concurrent_meds_count"]}

Baseline ANA:
{data["baseline_ana_titer"]}

Baseline CRP:
{data["baseline_crp"]} mg/L

Baseline WBC:
{data["baseline_wbc"]}

Predicted Risk:
{data["risk_probability"]:.2%}

Risk Category:
{data["risk_label"]}
"""

            # ------------------------------------------------
            # SHAP
            # ------------------------------------------------

            contrib = (
                st.session_state.shap_contrib
            )

            if contrib is not None:

                project_context += (
                    "\nSHAP CONTRIBUTIONS:\n"
                )

                for feature, value in contrib.items():

                    direction = (
                        "increases risk"
                        if value > 0
                        else "decreases risk"
                    )

                    project_context += (
                        f"- "
                        f"{clean_feature_name(feature)}: "
                        f"{float(value):+.4f} "
                        f"({direction})\n"
                    )

        # ----------------------------------------------------
        # AI INSTRUCTIONS
        # ----------------------------------------------------

        system_message = project_context + """

Answer the user's question in a helpful,
clear, and research-oriented way.

When explaining a prediction:

1. Start with the predicted probability if a prediction exists.

2. Explain why the result belongs to:

   - LOW RISK
   - MODERATE RISK
   - HIGH RISK

3. Use these project thresholds:

   Below 15% = LOW RISK

   15% to below 40% = MODERATE RISK

   40% and above = HIGH RISK

4. Use SHAP contributions when available.

5. Explain positive SHAP values as factors that push
the model prediction toward higher risk.

6. Explain negative SHAP values as factors that push
the model prediction toward lower risk.

7. Always distinguish:

   AI model prediction
   Scientific interpretation
   Actual clinical evidence

8. Never claim that the model proves that a drug causes
autoimmune disease.

9. Remember that this project uses synthetic data.

10. Do not invent patient information or prediction values
that are not provided in the project context.

11. If the user asks a general scientific question,
explain it clearly even when there is no current prediction.

12. Keep answers understandable for a student project presentation.
"""

        # ----------------------------------------------------
        # OLLAMA CALL
        # ----------------------------------------------------

        current_prediction = (
            st.session_state.prediction_data
            if st.session_state.prediction_done
            else None
        )
        answer = build_local_chat_response(
            user_prompt,
            prediction=current_prediction,
            drug_catalogue=DRUGS,
        )
        ollama_failed = False

        with st.chat_message("assistant"):
            with st.spinner("Preparing a helpful response..."):
                if answer is None:
                    messages = [
                        {
                            "role": "system",
                            "content": system_message
                        },
                        *st.session_state.chat_messages,
                    ]
                    try:
                        answer = get_ollama_response(messages)
                    except (
                        requests.exceptions.RequestException,
                        ValueError,
                        KeyError,
                    ):
                        ollama_failed = True
                        answer = build_local_chat_response(
                            user_prompt,
                            prediction=current_prediction,
                            drug_catalogue=DRUGS,
                            ollama_unavailable=True,
                        )

                st.markdown(answer)
                if ollama_failed:
                    st.caption(
                        "Ollama is unavailable; this response came from the "
                        "built-in offline guide."
                    )

        st.session_state.chat_messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    # --------------------------------------------------------
    # CLEAR CHAT
    # --------------------------------------------------------

    st.divider()

    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.chat_messages = []

        st.rerun()


# ============================================================
# PAGE 5 — ABOUT
# ============================================================

elif page == "About":

    st.header(
        "About the Project"
    )

    st.markdown(
        """
## Early Prediction of Drug-Induced Autoimmune Reactions Using AI

This project proposes an explainable machine-learning framework
for the preliminary prediction of potential drug-induced
autoimmune reaction risk.

### Main Components

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
clinical or market exposure.

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

### Important Limitation

The current student prototype uses a literature-informed
synthetic dataset. Therefore, its predictions should be
considered demonstrations of the machine-learning framework,
not real-world clinical evidence.

### Project Purpose

The goal is to demonstrate how:

**AI + Explainable AI + Drug Safety Screening**

can be combined into an interpretable decision-support prototype.
"""
    )

    st.divider()

    st.info(
        f"Model currently in use: "
        f"{meta.get('best_model_name', 'Unknown')}"
    )

    st.warning(
        "Research / Educational Demonstration Only — "
        "Not for clinical use."
    )

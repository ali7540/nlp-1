"""
streamlit_app.py – NLP Quality Feedback Analyzer Dashboard
Week 4 Deliverable — Faiz Ali Internship Project

Awwwards-level dark glassmorphism UI redesign.

Run:
    streamlit run app/streamlit_app.py
"""

import os
import sys

# Allow imports relative to the project root when run from the app/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io

from app.utils import analyse_review, load_dataset, TOPIC_LABELS

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Quality Feedback Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Plotly Dark Theme Template
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_DARK_TEMPLATE = dict(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#c4c9d4", size=12),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
        colorway=["#7c3aed", "#06b6d4", "#f472b6", "#34d399", "#fbbf24", "#f87171"],
    )
)

# Accent color palette
ACCENT = {
    "violet":  "#7c3aed",
    "cyan":    "#06b6d4",
    "pink":    "#f472b6",
    "emerald": "#34d399",
    "amber":   "#fbbf24",
    "rose":    "#f87171",
}

# ─────────────────────────────────────────────────────────────────────────────
# Master CSS — Dark Glassmorphism + Animations
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">

<style>
/* ═══════════════════════════════════════════════════════════════════════════
   GLOBAL RESET & BACKGROUND
   ═══════════════════════════════════════════════════════════════════════════ */
@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-6px); }
}

@keyframes pulse-glow {
    0%, 100% { box-shadow: 0 0 20px rgba(124,58,237,0.15); }
    50%  { box-shadow: 0 0 35px rgba(124,58,237,0.3); }
}

@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

@keyframes borderGlow {
    0%, 100% { border-color: rgba(124,58,237,0.3); }
    50%      { border-color: rgba(6,182,212,0.5); }
}

/* Root variables */
:root {
    --bg-primary:   #0a0a1a;
    --bg-secondary: #111127;
    --bg-card:      rgba(17, 17, 39, 0.65);
    --bg-glass:     rgba(255,255,255,0.03);
    --border-glass: rgba(255,255,255,0.08);
    --text-primary: #e8eaf0;
    --text-secondary: #8b92a5;
    --text-muted:   #5a6178;
    --accent-violet: #7c3aed;
    --accent-cyan:   #06b6d4;
    --accent-pink:   #f472b6;
    --accent-emerald:#34d399;
    --accent-amber:  #fbbf24;
    --accent-rose:   #f87171;
    --radius-lg:   16px;
    --radius-md:   12px;
    --radius-sm:   8px;
    --radius-pill: 100px;
}

/* Streamlit app background */
.stApp, [data-testid="stAppViewContainer"],
.main .block-container {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Animated mesh gradient overlay */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 20%, rgba(124,58,237,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(6,182,212,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 50% 60% at 50% 50%, rgba(244,114,182,0.05) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
    animation: gradientShift 20s ease infinite;
    background-size: 200% 200%;
}

.stApp > * { position: relative; z-index: 1; }

/* ═══════════════════════════════════════════════════════════════════════════
   SCROLLBAR
   ═══════════════════════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--accent-violet), var(--accent-cyan));
    border-radius: 100px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   HEADER
   ═══════════════════════════════════════════════════════════════════════════ */
.hero-header {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg,
        rgba(124,58,237,0.25) 0%,
        rgba(6,182,212,0.15) 50%,
        rgba(244,114,182,0.1) 100%);
    backdrop-filter: blur(20px) saturate(1.4);
    -webkit-backdrop-filter: blur(20px) saturate(1.4);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: var(--radius-lg);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    animation: fadeInUp 0.8s ease-out;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: linear-gradient(45deg,
        transparent 30%,
        rgba(124,58,237,0.06) 50%,
        transparent 70%);
    animation: shimmer 6s linear infinite;
}
.hero-header .hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.3);
    color: #c4b5fd;
    padding: 4px 14px;
    border-radius: var(--radius-pill);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero-header h1 {
    margin: 0;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #e8eaf0 0%, #c4b5fd 40%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
}
.hero-header .hero-sub {
    margin: 0.6rem 0 0;
    color: var(--text-secondary);
    font-size: 0.95rem;
    font-weight: 400;
    line-height: 1.5;
}
.hero-header .hero-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 1.2rem;
}
.hero-header .hero-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: var(--radius-pill);
    padding: 5px 14px;
    font-size: 0.75rem;
    color: var(--text-secondary);
    font-weight: 500;
    transition: all 0.3s ease;
}
.hero-header .hero-chip:hover {
    background: rgba(124,58,237,0.15);
    border-color: rgba(124,58,237,0.3);
    color: #c4b5fd;
    transform: translateY(-1px);
}

/* ═══════════════════════════════════════════════════════════════════════════
   TABS
   ═══════════════════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-md) !important;
    padding: 5px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 24px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(124,58,237,0.1) !important;
    color: var(--text-primary) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.2), rgba(6,182,212,0.15)) !important;
    color: #e8eaf0 !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.15) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ═══════════════════════════════════════════════════════════════════════════
   GLASS CARDS — Section wrappers
   ═══════════════════════════════════════════════════════════════════════════ */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(16px) saturate(1.3);
    -webkit-backdrop-filter: blur(16px) saturate(1.3);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-lg);
    padding: 1.8rem;
    margin-bottom: 1.2rem;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    animation: fadeInUp 0.6s ease-out both;
}
.glass-card:hover {
    border-color: rgba(124,58,237,0.2);
    box-shadow: 0 8px 40px rgba(124,58,237,0.08);
    transform: translateY(-2px);
}

/* Section titles */
.section-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600;
    font-size: 1.1rem;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
    padding-bottom: 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.section-title .title-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-violet);
    box-shadow: 0 0 10px var(--accent-violet);
}

/* ═══════════════════════════════════════════════════════════════════════════
   KPI / METRIC CARDS
   ═══════════════════════════════════════════════════════════════════════════ */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 1.5rem;
}
.kpi-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-md);
    padding: 1.4rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: all 0.35s ease;
    animation: fadeInUp 0.6s ease-out both;
}
.kpi-card:nth-child(1) { animation-delay: 0.05s; }
.kpi-card:nth-child(2) { animation-delay: 0.1s; }
.kpi-card:nth-child(3) { animation-delay: 0.15s; }
.kpi-card:nth-child(4) { animation-delay: 0.2s; }
.kpi-card:hover {
    border-color: rgba(124,58,237,0.25);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(0,0,0,0.3);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 3px 3px 0 0;
}
.kpi-card:nth-child(1)::before { background: linear-gradient(90deg, var(--accent-violet), var(--accent-cyan)); }
.kpi-card:nth-child(2)::before { background: linear-gradient(90deg, var(--accent-amber), var(--accent-pink)); }
.kpi-card:nth-child(3)::before { background: linear-gradient(90deg, var(--accent-emerald), var(--accent-cyan)); }
.kpi-card:nth-child(4)::before { background: linear-gradient(90deg, var(--accent-rose), var(--accent-pink)); }
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}
.kpi-value .kpi-unit {
    font-size: 0.9rem;
    font-weight: 400;
    color: var(--text-secondary);
    margin-left: 4px;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SENTIMENT BADGES
   ═══════════════════════════════════════════════════════════════════════════ */
.sentiment-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 16px;
    border-radius: var(--radius-pill);
    font-weight: 600;
    font-size: 0.82rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    transition: all 0.3s ease;
}
.sentiment-badge.positive {
    background: rgba(52,211,153,0.12);
    color: #34d399;
    border: 1px solid rgba(52,211,153,0.25);
}
.sentiment-badge.negative {
    background: rgba(248,113,113,0.12);
    color: #f87171;
    border: 1px solid rgba(248,113,113,0.25);
}
.sentiment-badge.neutral {
    background: rgba(251,191,36,0.12);
    color: #fbbf24;
    border: 1px solid rgba(251,191,36,0.25);
}

/* ═══════════════════════════════════════════════════════════════════════════
   ENTITY TAGS
   ═══════════════════════════════════════════════════════════════════════════ */
.entity-cloud {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}
.entity-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(124,58,237,0.08);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: var(--radius-pill);
    padding: 6px 16px;
    font-size: 0.82rem;
    font-weight: 500;
    color: #c4b5fd;
    transition: all 0.3s ease;
    animation: fadeInUp 0.5s ease-out both;
}
.entity-tag:hover {
    background: rgba(124,58,237,0.18);
    border-color: rgba(124,58,237,0.4);
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(124,58,237,0.15);
}

/* ═══════════════════════════════════════════════════════════════════════════
   ANALYZER RESULT CARDS
   ═══════════════════════════════════════════════════════════════════════════ */
.result-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass);
    border-radius: var(--radius-md);
    padding: 1.5rem;
    transition: all 0.35s ease;
    animation: fadeInUp 0.5s ease-out both;
}
.result-card:hover {
    border-color: rgba(124,58,237,0.2);
    box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}
.result-card .card-icon {
    font-size: 2rem;
    margin-bottom: 0.6rem;
}
.result-card .card-label {
    font-size: 0.7rem;
    font-weight: 600;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 4px;
}
.result-card .card-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-primary);
}
.result-card .card-sub {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 0.3rem;
}

/* ═══════════════════════════════════════════════════════════════════════════
   GAUGE  (circular progress for confidence)
   ═══════════════════════════════════════════════════════════════════════════ */
.gauge-wrapper {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
}
.gauge-ring {
    width: 100px; height: 100px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
}
.gauge-ring .gauge-inner {
    width: 76px; height: 76px;
    border-radius: 50%;
    background: var(--bg-secondary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-primary);
}
.gauge-ring-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ═══════════════════════════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg,
        rgba(10,10,26,0.97) 0%,
        rgba(17,17,39,0.97) 100%) !important;
    border-right: 1px solid var(--border-glass) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}
section[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 1.1rem !important;
    background: linear-gradient(90deg, var(--accent-violet), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ═══════════════════════════════════════════════════════════════════════════
   STREAMLIT OVERRIDES — Inputs / Buttons / Metrics
   ═══════════════════════════════════════════════════════════════════════════ */
/* Text area */
.stTextArea textarea {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    transition: all 0.3s ease !important;
    padding: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent-violet) !important;
    box-shadow: 0 0 0 2px rgba(124,58,237,0.2) !important;
}
.stTextArea textarea::placeholder { color: var(--text-muted) !important; }

/* Primary button */
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--accent-violet) 0%, #9333ea 50%, var(--accent-cyan) 100%) !important;
    background-size: 200% auto !important;
    border: none !important;
    border-radius: var(--radius-pill) !important;
    color: white !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 0.7rem 2.5rem !important;
    transition: all 0.4s ease !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
    letter-spacing: 0.02em !important;
}
.stButton > button[kind="primary"]:hover,
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background-position: right center !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.45) !important;
}

/* Secondary buttons */
.stButton > button[kind="secondary"],
.stButton > button[data-testid="stBaseButton-secondary"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-secondary) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    transition: all 0.3s ease !important;
}
.stButton > button[kind="secondary"]:hover,
.stButton > button[data-testid="stBaseButton-secondary"]:hover {
    background: rgba(124,58,237,0.1) !important;
    border-color: rgba(124,58,237,0.3) !important;
    color: #c4b5fd !important;
}

/* Metrics */
div[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-md) !important;
    padding: 1rem 1.2rem !important;
}
div[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-md) !important;
}

/* Dividers / captions */
hr { border-color: rgba(255,255,255,0.06) !important; }
.stCaption, .stMarkdown small { color: var(--text-muted) !important; }

/* Info/warning/spinner */
div[data-testid="stNotification"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-secondary) !important;
}

/* Spinner */
.stSpinner > div { color: var(--accent-violet) !important; }

/* Multi-select */
.stMultiSelect [data-baseweb="select"] {
    background: var(--bg-glass) !important;
    border: 1px solid var(--border-glass) !important;
    border-radius: var(--radius-sm) !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background: rgba(124,58,237,0.15) !important;
    border: 1px solid rgba(124,58,237,0.3) !important;
    border-radius: var(--radius-pill) !important;
}

/* Slider */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent-violet) !important;
}

/* Word cloud images */
.stImage {
    border-radius: var(--radius-md) !important;
    overflow: hidden;
}

/* ═══════════════════════════════════════════════════════════════════════════
   FOOTER
   ═══════════════════════════════════════════════════════════════════════════ */
.footer-section {
    text-align: center;
    padding: 2rem 0 1rem;
    border-top: 1px solid rgba(255,255,255,0.04);
    margin-top: 2rem;
}
.footer-section .footer-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    background: linear-gradient(90deg, var(--accent-violet), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.footer-section .footer-sub {
    color: var(--text-muted);
    font-size: 0.72rem;
    margin-top: 0.3rem;
}
.footer-stack {
    display: flex;
    justify-content: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 0.8rem;
}
.footer-stack .f-chip {
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    padding: 3px 12px;
    border-radius: var(--radius-pill);
    font-size: 0.68rem;
    color: var(--text-muted);
    font-weight: 500;
}

/* ═══════════════════════════════════════════════════════════════════════════
   RESPONSIVE
   ═══════════════════════════════════════════════════════════════════════════ */
@media (max-width: 768px) {
    .hero-header { padding: 1.5rem; }
    .hero-header h1 { font-size: 1.6rem; }
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: Plotly dark layout
# ─────────────────────────────────────────────────────────────────────────────
def _dark_layout(fig, height=320, **kwargs):
    """Apply dark glassmorphism-consistent theme to a Plotly figure."""
    # Build axis defaults, then merge any caller overrides on top
    default_axis = dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.04)")
    xaxis_kw = {**default_axis, **(kwargs.pop("xaxis", {}) if "xaxis" in kwargs else {})}
    yaxis_kw = {**default_axis, **(kwargs.pop("yaxis", {}) if "yaxis" in kwargs else {})}

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#c4c9d4", size=12),
        margin=dict(l=0, r=0, t=10, b=0),
        height=height,
        xaxis=xaxis_kw,
        yaxis=yaxis_kw,
        **kwargs,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">⚡ NLP-Powered Analysis Engine</div>
    <h1>Quality Feedback Analyzer</h1>
    <p class="hero-sub">
        Decode customer sentiment, discover hidden topics, and extract actionable entities
        from Amazon product reviews — powered by state-of-the-art NLP models.
    </p>
    <div class="hero-chips">
        <span class="hero-chip">🧠 VADER</span>
        <span class="hero-chip">🤖 DistilBERT</span>
        <span class="hero-chip">📊 LDA Topics</span>
        <span class="hero-chip">🏷️ spaCy NER</span>
        <span class="hero-chip">📦 10K Reviews</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["⚡  Live Analyzer", "📊  Aggregate Insights"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Analyzer
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title"><span class="title-dot"></span>Real-Time Review Analysis</div>
        <p style="color:var(--text-secondary); font-size:0.88rem; margin-top:-0.5rem;">
            Paste any product review below and watch the NLP pipeline dissect it in real time.
        </p>
    </div>
    """, unsafe_allow_html=True)

    SAMPLE_REVIEWS = [
        "Absolutely love this coffee! The aroma is incredible and it brews perfectly every time. Will definitely buy again.",
        "The packaging was completely destroyed when it arrived. The product inside was also stale and smelled terrible. Total waste of money.",
        "Decent dog food but my dog seems to only like it sometimes. Nothing special about it, pretty average for the price.",
        "This tea is so refreshing and organic! Perfect for mornings. I love the delicate flavor — favorite purchase this year.",
        "I ordered the chocolate bars but received something completely different. Customer service was slow and unhelpful. Very disappointed.",
    ]

    # Initialise session state key once so st.text_area is the sole owner
    if "review_input" not in st.session_state:
        st.session_state["review_input"] = ""

    col_input, col_sample = st.columns([3, 1])
    with col_sample:
        st.markdown("**💡 Try a sample:**")
        for i, s in enumerate(SAMPLE_REVIEWS):
            if st.button(f"Sample {i+1}", key=f"sample_{i}", width="stretch"):
                st.session_state["review_input"] = s
                st.rerun()

    with col_input:
        # Do NOT pass value= when key= is already bound to session_state;
        # session_state is the single source of truth.
        review_text = st.text_area(
            "Review Text",
            height=160,
            placeholder="Paste a product review here and click Analyze…",
            key="review_input",
            label_visibility="collapsed",
        )

    analyze_btn = st.button("⚡ Analyze Review", type="primary")

    if analyze_btn:
        text = st.session_state.get("review_input", "").strip()
        if not text:
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Running analysis pipeline…"):
                try:
                    results = analyse_review(text)
                except Exception as _err:
                    st.error(
                        f"⚠️ Analysis failed: {_err}\n\n"
                        "Please try again or contact the project maintainer."
                    )
                    st.stop()

            # ── Sentiment Results ────────────────────────────────────────
            vader = results["vader"]
            dbert = results["distilbert"]
            agree = vader["label"] == dbert["label"]

            # Build gauge conic gradient for DistilBERT confidence
            conf = dbert["confidence"]
            conf_color = ACCENT["emerald"] if conf >= 70 else ACCENT["amber"] if conf >= 50 else ACCENT["rose"]

            st.markdown(f"""
            <div class="glass-card" style="animation-delay:0.1s;">
                <div class="section-title"><span class="title-dot"></span>Sentiment Analysis</div>
                <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px;">
                    <div class="result-card">
                        <div class="card-icon">🎯</div>
                        <div class="card-label">VADER Compound</div>
                        <div class="card-value">{vader['compound']:+.4f}</div>
                        <div style="margin-top:0.7rem;">
                            <span class="sentiment-badge {vader['label']}">{vader['label'].upper()}</span>
                        </div>
                        <div class="card-sub" style="margin-top:0.5rem;">
                            pos {vader['pos']}  ·  neg {vader['neg']}  ·  neu {vader['neu']}
                        </div>
                    </div>
                    <div class="result-card" style="animation-delay:0.15s;">
                        <div class="card-icon">🤖</div>
                        <div class="card-label">DistilBERT</div>
                        <div class="card-value" style="margin-bottom:0.6rem;">
                            <span class="sentiment-badge {dbert['label']}">{dbert['label'].upper()}</span>
                        </div>
                        <div class="gauge-wrapper">
                            <div class="gauge-ring" style="background:conic-gradient({conf_color} {conf * 3.6}deg, rgba(255,255,255,0.05) 0deg);">
                                <div class="gauge-inner">{conf}%</div>
                            </div>
                            <span class="gauge-ring-label">Confidence</span>
                        </div>
                    </div>
                    <div class="result-card" style="animation-delay:0.2s;">
                        <div class="card-icon">{'🤝' if agree else '⚔️'}</div>
                        <div class="card-label">Model Agreement</div>
                        <div class="card-value" style="color:{'var(--accent-emerald)' if agree else 'var(--accent-rose)'};">
                            {'Consensus' if agree else 'Divergent'}
                        </div>
                        <div class="card-sub">
                            {'Both models agree on sentiment direction.' if agree else f'VADER → {vader["label"]}  ·  DistilBERT → {dbert["label"]}'}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Topic Prediction ─────────────────────────────────────────
            topic = results["topic"]

            st.markdown(f"""
            <div class="glass-card" style="animation-delay:0.25s;">
                <div class="section-title"><span class="title-dot" style="background:var(--accent-cyan);box-shadow:0 0 10px var(--accent-cyan);"></span>Topic Prediction · LDA</div>
            </div>
            """, unsafe_allow_html=True)

            tc1, tc2 = st.columns([1, 2])
            with tc1:
                topic_conf = topic["confidence"]
                topic_color = ACCENT["cyan"] if topic_conf >= 30 else ACCENT["amber"]
                st.markdown(f"""
                <div class="result-card">
                    <div class="card-icon">📊</div>
                    <div class="card-label">Dominant Topic</div>
                    <div class="card-value" style="font-size:1.2rem; color:var(--accent-cyan);">
                        {topic['label'].replace('_', ' ').title()}
                    </div>
                    <div style="margin-top:1rem;">
                        <div class="gauge-wrapper">
                            <div class="gauge-ring" style="background:conic-gradient({topic_color} {topic_conf * 3.6}deg, rgba(255,255,255,0.05) 0deg);">
                                <div class="gauge-inner">{topic_conf}%</div>
                            </div>
                            <span class="gauge-ring-label">Topic Confidence</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with tc2:
                probs_df = pd.DataFrame(
                    {"Topic": list(topic["all_probs"].keys()),
                     "Probability (%)": list(topic["all_probs"].values())}
                ).sort_values("Probability (%)", ascending=True)
                # Replace underscores for display
                probs_df["Topic"] = probs_df["Topic"].str.replace("_", " ").str.title()
                fig_t = px.bar(
                    probs_df, x="Probability (%)", y="Topic", orientation="h",
                    color="Probability (%)",
                    color_continuous_scale=["#1e1b4b", "#7c3aed", "#06b6d4"],
                    height=280,
                )
                _dark_layout(fig_t, height=280, coloraxis_showscale=False)
                fig_t.update_traces(marker_line_width=0, marker_cornerradius=4)
                st.plotly_chart(fig_t, width="stretch")

            # ── Entities ─────────────────────────────────────────────────
            entities = results["entities"]
            if entities == ["none"]:
                ent_html = '<p style="color:var(--text-muted); font-size:0.88rem;">No entities or quality keywords detected in this review.</p>'
            else:
                tags = "".join(
                    f'<span class="entity-tag" style="animation-delay:{i*0.06}s;">🏷️ {e}</span>'
                    for i, e in enumerate(entities)
                )
                ent_html = f'<div class="entity-cloud">{tags}</div>'

            st.markdown(f"""
            <div class="glass-card" style="animation-delay:0.3s;">
                <div class="section-title">
                    <span class="title-dot" style="background:var(--accent-pink);box-shadow:0 0 10px var(--accent-pink);"></span>
                    Extracted Entities & Quality Keywords
                </div>
                {ent_html}
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Aggregate Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    with st.spinner("Loading dataset…"):
        df = load_dataset()

    # ── Sidebar filters ──────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0;">
            <div style="font-family:'Space Grotesk',sans-serif; font-size:0.7rem; font-weight:600;
                        color:var(--text-muted); text-transform:uppercase; letter-spacing:0.12em;
                        margin-bottom:0.5rem;">Dashboard</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("## 🎛️ Filters")

        all_topics = sorted(df["topic_label"].unique())
        sel_topics = st.multiselect("Topic", all_topics, default=all_topics)

        all_sentiments = sorted(df["vader_label"].unique())
        sel_sentiments = st.multiselect("VADER Sentiment", all_sentiments, default=all_sentiments)

        rating_range = st.slider("Rating Range", 1, 5, (1, 5))

        st.markdown("---")
        st.markdown(f"""
        <div style="background:rgba(124,58,237,0.08); border:1px solid rgba(124,58,237,0.2);
                    border-radius:8px; padding:12px 16px; text-align:center;">
            <div style="font-size:0.68rem; color:var(--text-muted); text-transform:uppercase;
                        letter-spacing:0.1em; font-weight:600;">Dataset Size</div>
            <div style="font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:700;
                        color:var(--accent-violet); margin-top:4px;">{len(df):,}</div>
            <div style="font-size:0.72rem; color:var(--text-muted);">total reviews</div>
        </div>
        """, unsafe_allow_html=True)

    # Apply filters
    mask = (
        df["topic_label"].isin(sel_topics) &
        df["vader_label"].isin(sel_sentiments) &
        df["rating"].between(*rating_range)
    )
    df_f = df[mask].copy()

    # ── KPI Row (custom HTML) ─────────────────────────────────────────────
    pos_pct = (df_f["vader_label"] == "positive").mean() * 100 if len(df_f) > 0 else 0
    neg_pct = (df_f["vader_label"] == "negative").mean() * 100 if len(df_f) > 0 else 0
    avg_r = df_f["rating"].mean() if len(df_f) > 0 else 0

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Filtered Reviews</div>
            <div class="kpi-value">{len(df_f):,}<span class="kpi-unit">of {len(df):,}</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Avg. Rating</div>
            <div class="kpi-value">{avg_r:.2f}<span class="kpi-unit">⭐</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Positive (VADER)</div>
            <div class="kpi-value" style="color:var(--accent-emerald);">{pos_pct:.1f}<span class="kpi-unit">%</span></div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Negative (VADER)</div>
            <div class="kpi-value" style="color:var(--accent-rose);">{neg_pct:.1f}<span class="kpi-unit">%</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Row A: Sentiment + Topic Freq ──────────────────────────────────────
    ca1, ca2 = st.columns(2)

    with ca1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title"><span class="title-dot" style="background:var(--accent-emerald);box-shadow:0 0 10px var(--accent-emerald);"></span>Sentiment Distribution</div>
        </div>
        """, unsafe_allow_html=True)
        sent_counts = df_f["vader_label"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        color_map = {
            "positive": ACCENT["emerald"],
            "negative": ACCENT["rose"],
            "neutral":  ACCENT["amber"],
        }
        fig_sent = px.pie(sent_counts, values="Count", names="Sentiment",
                          color="Sentiment", color_discrete_map=color_map,
                          hole=0.55, height=340)
        fig_sent.update_traces(
            textinfo="percent+label",
            textfont=dict(size=13, family="Inter"),
            marker=dict(line=dict(color="rgba(0,0,0,0.3)", width=2)),
        )
        _dark_layout(fig_sent, height=340,
                     legend=dict(orientation="h", y=-0.15,
                                 font=dict(color="#8b92a5")),
                     showlegend=True)
        st.plotly_chart(fig_sent, width="stretch")

    with ca2:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title"><span class="title-dot" style="background:var(--accent-cyan);box-shadow:0 0 10px var(--accent-cyan);"></span>Topic Frequency</div>
        </div>
        """, unsafe_allow_html=True)
        topic_counts = (df_f["topic_label"].value_counts()
                        .reset_index()
                        .rename(columns={"topic_label": "Topic", "count": "Count"}))
        topic_counts["Topic"] = topic_counts["Topic"].str.replace("_", " ").str.title()
        fig_topic = px.bar(topic_counts, x="Count", y="Topic", orientation="h",
                           color="Count",
                           color_continuous_scale=["#164e63", "#06b6d4", "#67e8f9"],
                           height=340)
        _dark_layout(fig_topic, height=340, coloraxis_showscale=False,
                     yaxis=dict(categoryorder="total ascending",
                                gridcolor="rgba(255,255,255,0.04)"))
        fig_topic.update_traces(marker_line_width=0, marker_cornerradius=4)
        st.plotly_chart(fig_topic, width="stretch")

    # ── Row B: Complaint Topics + Avg Rating ──────────────────────────────
    cb1, cb2 = st.columns(2)

    with cb1:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title"><span class="title-dot" style="background:var(--accent-rose);box-shadow:0 0 10px var(--accent-rose);"></span>Top Complaint Topics</div>
        </div>
        """, unsafe_allow_html=True)
        neg_df = df_f[df_f["vader_label"] == "negative"]
        if len(neg_df) > 0:
            complaint_counts = (neg_df["topic_label"].value_counts()
                                .reset_index()
                                .rename(columns={"topic_label": "Topic", "count": "Count"}))
            complaint_counts["Topic"] = complaint_counts["Topic"].str.replace("_", " ").str.title()
            fig_comp = px.bar(complaint_counts, x="Count", y="Topic", orientation="h",
                              color="Count",
                              color_continuous_scale=["#450a0a", "#f87171", "#fca5a5"],
                              height=320)
            _dark_layout(fig_comp, height=320, coloraxis_showscale=False,
                         yaxis=dict(categoryorder="total ascending",
                                    gridcolor="rgba(255,255,255,0.04)"))
            fig_comp.update_traces(marker_line_width=0, marker_cornerradius=4)
            st.plotly_chart(fig_comp, width="stretch")
        else:
            st.info("No negative reviews match the current filters.")

    with cb2:
        st.markdown("""
        <div class="glass-card">
            <div class="section-title"><span class="title-dot" style="background:var(--accent-amber);box-shadow:0 0 10px var(--accent-amber);"></span>Avg. Rating by Topic</div>
        </div>
        """, unsafe_allow_html=True)
        avg_rating = (df_f.groupby("topic_label")["rating"].mean()
                      .reset_index()
                      .rename(columns={"topic_label": "Topic", "rating": "Avg Rating"})
                      .sort_values("Avg Rating"))
        avg_rating["Topic"] = avg_rating["Topic"].str.replace("_", " ").str.title()
        fig_avg = px.bar(avg_rating, x="Avg Rating", y="Topic", orientation="h",
                         color="Avg Rating",
                         color_continuous_scale=[ACCENT["rose"], ACCENT["amber"], ACCENT["emerald"]],
                         range_color=[1, 5], height=320)
        _dark_layout(fig_avg, height=320, coloraxis_showscale=False,
                     yaxis=dict(categoryorder="total ascending",
                                gridcolor="rgba(255,255,255,0.04)"))
        fig_avg.add_vline(x=3, line_dash="dash", line_color="rgba(255,255,255,0.15)",
                          annotation_text="neutral",
                          annotation_font=dict(color="#8b92a5", size=11))
        fig_avg.update_traces(marker_line_width=0, marker_cornerradius=4)
        st.plotly_chart(fig_avg, width="stretch")

    # ── Row C: Word Clouds ────────────────────────────────────────────────
    st.markdown("""
    <div class="glass-card">
        <div class="section-title"><span class="title-dot" style="background:var(--accent-pink);box-shadow:0 0 10px var(--accent-pink);"></span>Keyword Landscape — Positive vs Negative</div>
    </div>
    """, unsafe_allow_html=True)

    wc1, wc2 = st.columns(2)

    def make_wordcloud(texts, colormap, bg_color="#0a0a1a"):
        corpus = " ".join(texts.dropna().astype(str).tolist())
        if not corpus.strip():
            return None
        wc = WordCloud(
            width=700, height=320,
            background_color=bg_color,
            colormap=colormap,
            max_words=90,
            prefer_horizontal=0.8,
            min_font_size=10,
            contour_width=0,
        ).generate(corpus)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        fig.patch.set_facecolor(bg_color)
        ax.set_facecolor(bg_color)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                    facecolor=bg_color, edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return buf

    pos_texts = df_f[df_f["vader_label"] == "positive"]["review text"]
    neg_texts = df_f[df_f["vader_label"] == "negative"]["review text"]

    with wc1:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="background:rgba(52,211,153,0.15);color:#34d399;padding:4px 12px;
                         border-radius:100px;font-size:0.75rem;font-weight:600;border:1px solid rgba(52,211,153,0.25);">
                ☀️ Positive
            </span>
        </div>
        """, unsafe_allow_html=True)
        wc_pos = make_wordcloud(pos_texts, "winter")
        if wc_pos:
            st.image(wc_pos, width="stretch")
        else:
            st.info("Not enough data.")

    with wc2:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
            <span style="background:rgba(248,113,113,0.15);color:#f87171;padding:4px 12px;
                         border-radius:100px;font-size:0.75rem;font-weight:600;border:1px solid rgba(248,113,113,0.25);">
                🌧️ Negative
            </span>
        </div>
        """, unsafe_allow_html=True)
        wc_neg = make_wordcloud(neg_texts, "hot")
        if wc_neg:
            st.image(wc_neg, width="stretch")
        else:
            st.info("Not enough data.")

    # ── Row D: Agreement Breakdown ────────────────────────────────────────
    st.markdown("""
    <div class="glass-card">
        <div class="section-title"><span class="title-dot"></span>Sentiment Model Agreement Breakdown</div>
    </div>
    """, unsafe_allow_html=True)

    agree_df = df_f.copy()
    agree_df["agreement"] = agree_df.apply(
        lambda r: "Agree" if r["vader_label"] == r["distilbert_label"] else "Disagree", axis=1)
    agree_summary = agree_df.groupby(["topic_label", "agreement"]).size().reset_index(name="Count")
    agree_summary["topic_label"] = agree_summary["topic_label"].str.replace("_", " ").str.title()
    fig_agree = px.bar(agree_summary, x="topic_label", y="Count", color="agreement",
                       barmode="stack",
                       color_discrete_map={
                           "Agree": ACCENT["emerald"],
                           "Disagree": ACCENT["rose"],
                       },
                       labels={"topic_label": "Topic", "Count": "Reviews"},
                       height=340)
    _dark_layout(fig_agree, height=340,
                 xaxis_tickangle=-20,
                 legend_title="Model Agreement",
                 legend=dict(font=dict(color="#8b92a5")))
    fig_agree.update_traces(marker_line_width=0, marker_cornerradius=4)
    st.plotly_chart(fig_agree, width="stretch")

    # ── Row E: Filtered data table ────────────────────────────────────────
    with st.expander("📋  View Filtered Data Table"):
        st.dataframe(
            df_f[["review text", "rating", "vader_label", "distilbert_label",
                  "distilbert_confidence", "topic_label", "entities"]],
            width="stretch",
            height=400,
        )

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer-section">
        <div class="footer-brand">Quality Feedback Analyzer</div>
        <div class="footer-sub">NLP Internship Project · Faiz Ali · 2024–25</div>
        <div class="footer-stack">
            <span class="f-chip">Streamlit</span>
            <span class="f-chip">HuggingFace</span>
            <span class="f-chip">NLTK</span>
            <span class="f-chip">spaCy</span>
            <span class="f-chip">scikit-learn</span>
            <span class="f-chip">Plotly</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

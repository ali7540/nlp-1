"""
utils.py – Model & Data loading helpers for the NLP Quality Feedback Analyzer.
All heavy assets are cached so they load only once per Streamlit session.
"""

import os
import re
import ssl
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths (resolved relative to this file so the app works from any cwd) ─────
ROOT = Path(__file__).parent.parent
DATA_PATH   = ROOT / "outputs" / "extracted_topics.csv"
LDA_PATH    = ROOT / "models" / "lda_model" / "lda_model.pkl"
VECTR_PATH  = ROOT / "models" / "lda_model" / "count_vectorizer.pkl"

TOPIC_LABELS = {
    0: "nutritional_profile",
    1: "beverages_and_liquids",
    2: "chips_and_snacks",
    3: "order_and_delivery",
    4: "pet_food",
    5: "coffee_and_tea_pods",
}

FEEDBACK_KEYWORDS = {
    # complaints
    "broken", "damaged", "leaked", "moldy", "stale", "expired",
    "slow", "missing", "worst", "waste",
    # positive features
    "fresh", "delicious", "organic", "sweet", "perfect", "favorite", "excellent",
}

# ── SSL fix for NLTK downloads on macOS ──────────────────────────────────────
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# ── Cached loaders ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Loading LDA model…")
def load_lda_artifacts():
    """Returns (lda_model, count_vectorizer)."""
    lda   = joblib.load(LDA_PATH)
    vectr = joblib.load(VECTR_PATH)
    return lda, vectr


@st.cache_resource(show_spinner="Loading DistilBERT pipeline…")
def load_distilbert():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )


@st.cache_resource(show_spinner="Loading spaCy model…")
def load_spacy():
    import spacy
    return spacy.load("en_core_web_sm")


@st.cache_resource(show_spinner="Loading VADER…")
def load_vader():
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


# ── Analysis helpers ──────────────────────────────────────────────────────────

def analyse_review(text: str) -> dict:
    """
    Run the full inference pipeline on a single raw review text.
    Returns a dict with vader, distilbert, topic, and entities results.
    """
    results = {}

    # ── VADER ─────────────────────────────────────────────────────────────
    sia = load_vader()
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.05:
        vader_label = "positive"
    elif compound <= -0.05:
        vader_label = "negative"
    else:
        vader_label = "neutral"
    results["vader"] = {
        "label": vader_label,
        "compound": round(compound, 4),
        "pos": round(scores["pos"], 4),
        "neg": round(scores["neg"], 4),
        "neu": round(scores["neu"], 4),
    }

    # ── DistilBERT ────────────────────────────────────────────────────────
    clf = load_distilbert()
    truncated = " ".join(text.split()[:400])
    db_result = clf(truncated)[0]
    results["distilbert"] = {
        "label": db_result["label"].lower(),
        "confidence": round(db_result["score"] * 100, 1),
    }

    # ── LDA Topic ─────────────────────────────────────────────────────────
    lda, vectr = load_lda_artifacts()

    # Minimal clean for the vectorizer (lowercase, remove punct/special chars)
    cleaned = re.sub(r"[^a-z\s]", " ", text.lower())
    dtm_row  = vectr.transform([cleaned])
    lda_probs = lda.transform(dtm_row)[0]
    dom_idx   = int(np.argmax(lda_probs))
    results["topic"] = {
        "label": TOPIC_LABELS.get(dom_idx, f"topic_{dom_idx}"),
        "confidence": round(float(lda_probs[dom_idx]) * 100, 1),
        "all_probs": {TOPIC_LABELS.get(i, f"topic_{i}"): round(float(p) * 100, 1)
                      for i, p in enumerate(lda_probs)},
    }

    # ── spaCy NER + custom keywords ───────────────────────────────────────
    nlp = load_spacy()
    doc = nlp(text)
    extracted = set()
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT"):
            clean_ent = re.sub(r"^\W+|\W+$", "", ent.text.strip().lower())
            if len(clean_ent) > 1:
                extracted.add(clean_ent)
    words = [w.strip(".,!?\"'()").lower() for w in text.split()]
    for w in words:
        if w in FEEDBACK_KEYWORDS:
            extracted.add(w)
    results["entities"] = sorted(extracted) if extracted else ["none"]

    return results

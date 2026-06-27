"""
utils.py – Model & Data loading helpers for the NLP Quality Feedback Analyzer.
All heavy assets are cached so they load only once per Streamlit session.

Key design decisions:
- LDA model is ALWAYS retrained from the CSV at first run (cached to disk),
  so it is never subject to numpy/pickle version mismatches.
- DistilBERT and spaCy are loaded lazily (on first Analyze click, not at page load).
- TRANSFORMERS_NO_ADVISORY_WARNINGS suppresses torchvision noise.
"""

import os
import re
import ssl
import json
import logging
from pathlib import Path

# Suppress transformers' advisory warnings about torchvision / image models
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ── Paths (resolved relative to this file so the app works from any cwd) ──────
ROOT       = Path(__file__).parent.parent
DATA_PATH  = ROOT / "outputs" / "extracted_topics.csv"
MODEL_DIR  = ROOT / "models" / "lda_model"
LDA_PATH   = MODEL_DIR / "lda_model.pkl"
VECTR_PATH = MODEL_DIR / "count_vectorizer.pkl"

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

# ── SSL fix for NLTK downloads on macOS ───────────────────────────────────────
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass


# ── Internal: retrain LDA from CSV ────────────────────────────────────────────

def _retrain_lda():
    """
    Train a fresh LDA + CountVectorizer from the labelled CSV and save to disk.
    Called automatically when the saved artefacts can't be loaded.
    Returns (lda, vectorizer).
    """
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation

    logger.info("Training topic model from dataset...")
    df = pd.read_csv(DATA_PATH)
    text_col = "review text"

    def _clean(text: str) -> str:
        return re.sub(r"[^a-z\s]", " ", str(text).lower())

    corpus = df[text_col].fillna("").apply(_clean).tolist()

    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words="english",
        max_features=10_000,
    )
    dtm = vectorizer.fit_transform(corpus)

    lda = LatentDirichletAllocation(
        n_components=len(TOPIC_LABELS),
        random_state=42,
        max_iter=15,
        learning_method="online",
        n_jobs=1,
    )
    lda.fit(dtm)

    # Strip random_state before saving — it holds numpy internal objects that
    # cause BitGenerator pickle errors across numpy versions.
    lda.random_state = None

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(lda,        LDA_PATH,   compress=3)
    joblib.dump(vectorizer, VECTR_PATH, compress=3)

    return lda, vectorizer


# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Loading topic model…")
def load_lda_artifacts():
    """
    Load (lda_model, count_vectorizer) from disk.
    If the files are missing or the pickle is incompatible with the current
    environment, automatically retrains from the CSV and saves fresh artefacts.
    """
    try:
        lda   = joblib.load(LDA_PATH)
        vectr = joblib.load(VECTR_PATH)
        # Quick smoke-test: make sure transform() works
        test_vec = vectr.transform(["test coffee flavor"])
        lda.transform(test_vec)
        return lda, vectr
    except Exception as exc:
        logger.warning("LDA artefacts failed to load (%s) — retraining…", exc)
        return _retrain_lda()


@st.cache_resource(show_spinner="Loading sentiment model…")
def load_distilbert():
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )


@st.cache_resource(show_spinner="Loading NLP model…")
def load_spacy():
    import spacy
    return spacy.load("en_core_web_sm")


@st.cache_resource(show_spinner="Loading VADER…")
def load_vader():
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


# ── Main analysis pipeline ────────────────────────────────────────────────────

def analyse_review(text: str) -> dict:
    """
    Run the full inference pipeline on a single raw review text.
    Returns a dict with vader, distilbert, topic, and entities results.
    Raises StreamlitError-friendly exceptions on failure.
    """
    results = {}

    # ── VADER ──────────────────────────────────────────────────────────────
    sia    = load_vader()
    scores = sia.polarity_scores(text)
    compound = scores["compound"]
    vader_label = (
        "positive" if compound >= 0.05
        else "negative" if compound <= -0.05
        else "neutral"
    )
    results["vader"] = {
        "label":    vader_label,
        "compound": round(compound, 4),
        "pos":      round(scores["pos"], 4),
        "neg":      round(scores["neg"], 4),
        "neu":      round(scores["neu"], 4),
    }

    # ── DistilBERT ─────────────────────────────────────────────────────────
    clf       = load_distilbert()
    truncated = " ".join(text.split()[:400])
    db_result = clf(truncated)[0]
    results["distilbert"] = {
        "label":      db_result["label"].lower(),
        "confidence": round(db_result["score"] * 100, 1),
    }

    # ── LDA Topic ──────────────────────────────────────────────────────────
    lda, vectr = load_lda_artifacts()
    cleaned    = re.sub(r"[^a-z\s]", " ", text.lower())
    dtm_row    = vectr.transform([cleaned])
    lda_probs  = lda.transform(dtm_row)[0]
    dom_idx    = int(np.argmax(lda_probs))
    results["topic"] = {
        "label":      TOPIC_LABELS.get(dom_idx, f"topic_{dom_idx}"),
        "confidence": round(float(lda_probs[dom_idx]) * 100, 1),
        "all_probs":  {
            TOPIC_LABELS.get(i, f"topic_{i}"): round(float(p) * 100, 1)
            for i, p in enumerate(lda_probs)
        },
    }

    # ── spaCy NER + custom keywords ────────────────────────────────────────
    nlp  = load_spacy()
    doc  = nlp(text)
    extracted = set()
    for ent in doc.ents:
        if ent.label_ in ("ORG", "PRODUCT"):
            clean_ent = re.sub(r"^\W+|\W+$", "", ent.text.strip().lower())
            if len(clean_ent) > 1:
                extracted.add(clean_ent)
    for w in (w.strip(".,!?\"'()").lower() for w in text.split()):
        if w in FEEDBACK_KEYWORDS:
            extracted.add(w)
    results["entities"] = sorted(extracted) if extracted else ["none"]

    return results

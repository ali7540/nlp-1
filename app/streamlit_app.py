"""
streamlit_app.py – NLP Quality Feedback Analyzer Dashboard
Week 4 Deliverable — Faiz Ali Internship Project

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
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global Styles
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #2a5298;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
    }
    .metric-card h4 { margin: 0 0 0.3rem; color: #555; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-card p  { margin: 0; font-size: 1.5rem; font-weight: 700; }

    /* Sentiment badge */
    .badge-positive { background: #d4edda; color: #155724; border-radius: 20px; padding: 0.2rem 0.8rem; font-weight: 600; }
    .badge-negative { background: #f8d7da; color: #721c24; border-radius: 20px; padding: 0.2rem 0.8rem; font-weight: 600; }
    .badge-neutral  { background: #fff3cd; color: #856404; border-radius: 20px; padding: 0.2rem 0.8rem; font-weight: 600; }

    /* Section title */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e3c72;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }

    /* Wider metric column */
    div[data-testid="column"] { padding: 0 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔍 NLP Quality Feedback Analyzer</h1>
    <p>Amazon Product Reviews · VADER · DistilBERT · LDA Topic Modeling · spaCy NER</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🧪 Live Analyzer", "📊 Aggregate Insights"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Analyzer
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Paste any product review for real-time NLP analysis</div>',
                unsafe_allow_html=True)

    SAMPLE_REVIEWS = [
        "Absolutely love this coffee! The aroma is incredible and it brews perfectly every time. Will definitely buy again.",
        "The packaging was completely destroyed when it arrived. The product inside was also stale and smelled terrible. Total waste of money.",
        "Decent dog food but my dog seems to only like it sometimes. Nothing special about it, pretty average for the price.",
        "This tea is so refreshing and organic! Perfect for mornings. I love the delicate flavor — favorite purchase this year.",
        "I ordered the chocolate bars but received something completely different. Customer service was slow and unhelpful. Very disappointed.",
    ]

    col_input, col_sample = st.columns([3, 1])
    with col_sample:
        st.markdown("**💡 Try a sample:**")
        for i, s in enumerate(SAMPLE_REVIEWS):
            if st.button(f"Sample {i+1}", key=f"sample_{i}", width="stretch"):
                st.session_state["review_input"] = s

    with col_input:
        review_text = st.text_area(
            "Review Text",
            value=st.session_state.get("review_input", ""),
            height=160,
            placeholder="Paste a product review here and click Analyze…",
            key="review_input",
            label_visibility="collapsed",
        )

    analyze_btn = st.button("🔍 Analyze Review", type="primary", width="content")

    if analyze_btn:
        text = st.session_state.get("review_input", "").strip()
        if not text:
            st.warning("Please enter a review first.")
        else:
            with st.spinner("Running analysis pipeline…"):
                results = analyse_review(text)

            st.markdown("---")

            # ── Row 1: Sentiment Results ────────────────────────────────────
            st.markdown('<div class="section-title">Sentiment Analysis</div>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)

            vader = results["vader"]
            dbert = results["distilbert"]
            vader_cls  = f"badge-{vader['label']}"
            dbert_cls  = f"badge-{dbert['label']}"

            with c1:
                st.metric("VADER Compound Score", f"{vader['compound']:+.4f}")
                st.markdown(f"Label: <span class='{vader_cls}'>{vader['label'].upper()}</span>",
                            unsafe_allow_html=True)
                st.caption(f"pos={vader['pos']} · neg={vader['neg']} · neu={vader['neu']}")

            with c2:
                st.metric("DistilBERT Label", dbert["label"].upper())
                st.metric("Confidence", f"{dbert['confidence']}%")

            with c3:
                # Agreement badge
                agree = vader["label"] == dbert["label"]
                st.metric("Model Agreement", "✅ Agree" if agree else "⚠️ Disagree")
                if not agree:
                    st.caption(f"VADER says **{vader['label']}**, DistilBERT says **{dbert['label']}**.")

            # ── Row 2: Topic ────────────────────────────────────────────────
            st.markdown('<div class="section-title">Topic Prediction (LDA)</div>', unsafe_allow_html=True)

            topic = results["topic"]
            tc1, tc2 = st.columns([1, 2])
            with tc1:
                st.metric("Dominant Topic", topic["label"].replace("_", " ").title())
                st.metric("Topic Confidence", f"{topic['confidence']}%")

            with tc2:
                probs_df = pd.DataFrame(
                    {"Topic": list(topic["all_probs"].keys()),
                     "Probability (%)": list(topic["all_probs"].values())}
                ).sort_values("Probability (%)", ascending=True)
                fig_t = px.bar(
                    probs_df, x="Probability (%)", y="Topic", orientation="h",
                    color="Probability (%)", color_continuous_scale="Blues",
                    height=240,
                )
                fig_t.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                    coloraxis_showscale=False)
                st.plotly_chart(fig_t, width="stretch")

            # ── Row 3: Entities ─────────────────────────────────────────────
            st.markdown('<div class="section-title">Extracted Entities & Quality Keywords</div>',
                        unsafe_allow_html=True)

            entities = results["entities"]
            if entities == ["none"]:
                st.info("No entities or quality keywords detected.")
            else:
                cols = st.columns(min(len(entities), 5))
                for i, ent in enumerate(entities):
                    cols[i % 5].markdown(
                        f"<span style='background:#e8f4fd;padding:3px 10px;"
                        f"border-radius:12px;font-size:0.9rem;'>🏷️ {ent}</span>",
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Aggregate Insights
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    df = load_dataset()

    # ── Sidebar filters ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔧 Filters")

        all_topics = sorted(df["topic_label"].unique())
        sel_topics = st.multiselect("Topic", all_topics, default=all_topics)

        all_sentiments = sorted(df["vader_label"].unique())
        sel_sentiments = st.multiselect("VADER Sentiment", all_sentiments, default=all_sentiments)

        rating_range = st.slider("Rating Range", 1, 5, (1, 5))

        st.markdown("---")
        st.caption(f"Dataset: **{len(df):,}** reviews total")

    # Apply filters
    mask = (
        df["topic_label"].isin(sel_topics) &
        df["vader_label"].isin(sel_sentiments) &
        df["rating"].between(*rating_range)
    )
    df_f = df[mask].copy()

    st.markdown(f'<div class="section-title">Showing <b>{len(df_f):,}</b> of {len(df):,} reviews</div>',
                unsafe_allow_html=True)

    # ── KPI Row ────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Reviews", f"{len(df_f):,}")
    k2.metric("Avg. Rating", f"{df_f['rating'].mean():.2f} ⭐")
    pos_pct = (df_f["vader_label"] == "positive").mean() * 100
    neg_pct = (df_f["vader_label"] == "negative").mean() * 100
    k3.metric("VADER Positive", f"{pos_pct:.1f}%")
    k4.metric("VADER Negative", f"{neg_pct:.1f}%")

    st.markdown("---")

    # ── Row A: Sentiment Distribution + Topic frequency ───────────────────
    ca1, ca2 = st.columns(2)

    with ca1:
        st.markdown('<div class="section-title">VADER Sentiment Distribution</div>',
                    unsafe_allow_html=True)
        sent_counts = df_f["vader_label"].value_counts().reset_index()
        sent_counts.columns = ["Sentiment", "Count"]
        color_map = {"positive": "#2ecc71", "negative": "#e74c3c", "neutral": "#f1c40f"}
        fig_sent = px.pie(sent_counts, values="Count", names="Sentiment",
                          color="Sentiment", color_discrete_map=color_map,
                          hole=0.45, height=320)
        fig_sent.update_traces(textinfo="percent+label")
        fig_sent.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig_sent, width="stretch")

    with ca2:
        st.markdown('<div class="section-title">Topic Frequency</div>',
                    unsafe_allow_html=True)
        topic_counts = (df_f["topic_label"].value_counts()
                        .reset_index()
                        .rename(columns={"topic_label": "Topic", "count": "Count"}))
        fig_topic = px.bar(topic_counts, x="Count", y="Topic", orientation="h",
                           color="Count", color_continuous_scale="Teal",
                           height=320)
        fig_topic.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                coloraxis_showscale=False,
                                yaxis=dict(categoryorder="total ascending"))
        st.plotly_chart(fig_topic, width="stretch")

    # ── Row B: Complaint Topics + Sentiment by Topic ───────────────────────
    cb1, cb2 = st.columns(2)

    with cb1:
        st.markdown('<div class="section-title">Top Complaint Topics (Negative Reviews)</div>',
                    unsafe_allow_html=True)
        neg_df = df_f[df_f["vader_label"] == "negative"]
        if len(neg_df) > 0:
            complaint_counts = (neg_df["topic_label"].value_counts()
                                .reset_index()
                                .rename(columns={"topic_label": "Topic", "count": "Count"}))
            fig_comp = px.bar(complaint_counts, x="Count", y="Topic", orientation="h",
                              color="Count", color_continuous_scale="Reds",
                              height=300)
            fig_comp.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                                   coloraxis_showscale=False,
                                   yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_comp, width="stretch")
        else:
            st.info("No negative reviews match the current filters.")

    with cb2:
        st.markdown('<div class="section-title">Avg. Rating by Topic</div>',
                    unsafe_allow_html=True)
        avg_rating = (df_f.groupby("topic_label")["rating"].mean()
                      .reset_index()
                      .rename(columns={"topic_label": "Topic", "rating": "Avg Rating"})
                      .sort_values("Avg Rating"))
        fig_avg = px.bar(avg_rating, x="Avg Rating", y="Topic", orientation="h",
                         color="Avg Rating",
                         color_continuous_scale=["#e74c3c", "#f1c40f", "#2ecc71"],
                         range_color=[1, 5], height=300)
        fig_avg.update_layout(margin=dict(l=0, r=0, t=10, b=0),
                              coloraxis_showscale=False,
                              yaxis=dict(categoryorder="total ascending"))
        fig_avg.add_vline(x=3, line_dash="dash", line_color="grey", annotation_text="neutral")
        st.plotly_chart(fig_avg, width="stretch")

    # ── Row C: Keyword Word Clouds ─────────────────────────────────────────
    st.markdown('<div class="section-title">Top Keywords — Positive vs Negative Reviews</div>',
                unsafe_allow_html=True)

    wc1, wc2 = st.columns(2)

    def make_wordcloud(texts, colormap):
        corpus = " ".join(texts.dropna().astype(str).tolist())
        if not corpus.strip():
            return None
        wc = WordCloud(width=600, height=280, background_color="white",
                       colormap=colormap, max_words=80).generate(corpus)
        fig, ax = plt.subplots(figsize=(6, 2.8))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    pos_texts = df_f[df_f["vader_label"] == "positive"]["review text"]
    neg_texts = df_f[df_f["vader_label"] == "negative"]["review text"]

    with wc1:
        st.caption("☀️ Positive Reviews")
        wc_pos = make_wordcloud(pos_texts, "Greens")
        if wc_pos:
            st.image(wc_pos, width="stretch")
        else:
            st.info("Not enough data.")

    with wc2:
        st.caption("🌧️ Negative Reviews")
        wc_neg = make_wordcloud(neg_texts, "Reds")
        if wc_neg:
            st.image(wc_neg, width="stretch")
        else:
            st.info("Not enough data.")

    # ── Row D: DistilBERT vs VADER agreement table ─────────────────────────
    st.markdown('<div class="section-title">Sentiment Model Agreement Breakdown</div>',
                unsafe_allow_html=True)

    agree_df = df_f.copy()
    agree_df["agreement"] = agree_df.apply(
        lambda r: "✅ Agree" if r["vader_label"] == r["distilbert_label"] else "⚠️ Disagree", axis=1)
    agree_summary = agree_df.groupby(["topic_label", "agreement"]).size().reset_index(name="Count")
    fig_agree = px.bar(agree_summary, x="topic_label", y="Count", color="agreement",
                       barmode="stack",
                       color_discrete_map={"✅ Agree": "#2ecc71", "⚠️ Disagree": "#e74c3c"},
                       labels={"topic_label": "Topic", "Count": "Reviews"},
                       height=320)
    fig_agree.update_layout(margin=dict(l=0, r=0, t=10, b=20),
                            xaxis_tickangle=-20, legend_title="Model Agreement")
    st.plotly_chart(fig_agree, width="stretch")

    # ── Row E: Filtered data table ─────────────────────────────────────────
    with st.expander("📋 View Filtered Data Table"):
        st.dataframe(
            df_f[["review text", "rating", "vader_label", "distilbert_label",
                  "distilbert_confidence", "topic_label", "entities"]],
            width="stretch",
            height=400,
        )

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("NLP Quality Feedback Analyzer · Internship Project · Built with Streamlit, HuggingFace, NLTK, spaCy, and scikit-learn.")

"""
export_lda_to_npz.py  –  Export LDA + vectorizer to numpy-native formats.

This makes the artefacts completely immune to pickle/numpy version mismatches.
The LDA model components are stored as plain numpy arrays (.npz),
the vectorizer vocabulary as a plain JSON file.

Run this once with whichever Python environment has the working packages:

    streamlit run app/streamlit_app.py     # verify app works first
    python export_lda_to_npz.py            # then export

After running, commit the new files in models/lda_model/
"""

import json
import re
from pathlib import Path

import joblib
import numpy as np

ROOT       = Path(__file__).parent
MODEL_DIR  = ROOT / "models" / "lda_model"
LDA_PATH   = MODEL_DIR / "lda_model.pkl"
VECTR_PATH = MODEL_DIR / "count_vectorizer.pkl"

print("Loading existing pickled artefacts …")
lda   = joblib.load(LDA_PATH)
vectr = joblib.load(VECTR_PATH)

# ── Export LDA components as plain numpy arrays ───────────────────────────────
# lda.components_  shape: (n_topics, n_features)  – the topic-word distributions
# lda.exp_dirichlet_component_  – normalised form used in .transform()
# We store everything needed to reconstruct transform() manually.

print("Exporting LDA components to NPZ …")
np.savez_compressed(
    MODEL_DIR / "lda_components.npz",
    components          = lda.components_,          # raw counts
    doc_topic_prior     = np.atleast_1d(np.asarray(lda.doc_topic_prior) if lda.doc_topic_prior is not None else np.array([1.0 / lda.n_components])),
    topic_word_prior    = np.atleast_1d(np.asarray(lda.topic_word_prior) if lda.topic_word_prior is not None else np.array([1.0 / lda.n_components])),
    n_components        = np.array([lda.n_components]),
)

# ── Export vectorizer vocabulary as JSON ──────────────────────────────────────
print("Exporting CountVectorizer vocabulary to JSON …")
vocab_data = {
    "vocabulary":   {k: int(v) for k, v in vectr.vocabulary_.items()},
    "stop_words":   list(vectr.stop_words_) if hasattr(vectr, "stop_words_") and vectr.stop_words_ else [],
    "max_df":       float(vectr.max_df) if isinstance(vectr.max_df, float) else int(vectr.max_df),
    "min_df":       float(vectr.min_df) if isinstance(vectr.min_df, float) else int(vectr.min_df),
}
with open(MODEL_DIR / "vectorizer_vocab.json", "w") as f:
    json.dump(vocab_data, f, separators=(",", ":"))

# ── Verify round-trip ─────────────────────────────────────────────────────────
print("\nVerifying round-trip …")
data = np.load(MODEL_DIR / "lda_components.npz")
components = data["components"]           # (n_topics, n_vocab)
# Normalise: each row = probability distribution over words
norm_comp = components + data["topic_word_prior"][0]
norm_comp = norm_comp / norm_comp.sum(axis=1, keepdims=True)

with open(MODEL_DIR / "vectorizer_vocab.json") as f:
    vocab = json.load(f)["vocabulary"]

# Build a sparse vector manually (same as vectorizer.transform does)
test = "great coffee taste delicious morning"
test_cleaned = re.sub(r"[^a-z\s]", " ", test.lower())
words = test_cleaned.split()
vec = np.zeros(len(vocab), dtype=np.float64)
for w in words:
    if w in vocab:
        vec[vocab[w]] += 1.0

# Compute topic distribution via variational inference approximation
# (Use the same approach as sklearn: E-step with norm_comp)
doc_topic = np.ones(components.shape[0]) / components.shape[0]
for _ in range(100):
    # E-step
    phi = norm_comp * vec[np.newaxis, :]     # (n_topics, n_vocab)
    phi = phi / (phi.sum(axis=0, keepdims=True) + 1e-10)
    # M-step
    doc_topic_new = phi.sum(axis=1) + data["doc_topic_prior"][0]
    doc_topic_new = doc_topic_new / doc_topic_new.sum()
    if np.abs(doc_topic_new - doc_topic).max() < 1e-6:
        break
    doc_topic = doc_topic_new

dom = int(doc_topic.argmax())
TOPIC_LABELS = {
    0: "nutritional_profile", 1: "beverages_and_liquids",
    2: "chips_and_snacks",    3: "order_and_delivery",
    4: "pet_food",            5: "coffee_and_tea_pods",
}
print(f"  Test input: '{test}'")
print(f"  Dominant topic: {dom} ({TOPIC_LABELS[dom]}) — {doc_topic[dom]*100:.1f}%")
print(f"  Probs sum: {doc_topic.sum():.4f}")

# Also verify sklearn still works with original pickle
probs_sklearn = lda.transform(vectr.transform([re.sub(r"[^a-z\s]", " ", test.lower())]))[0]
dom_sk = int(probs_sklearn.argmax())
print(f"\n  sklearn pickle result: topic {dom_sk} ({TOPIC_LABELS[dom_sk]}) — {probs_sklearn[dom_sk]*100:.1f}%")
print("\n✅ Export complete. Commit models/lda_model/lda_components.npz and vectorizer_vocab.json")

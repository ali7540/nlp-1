"""
retrain_lda.py  –  Retrain LDA + CountVectorizer from the labelled CSV
and save both artefacts using Python's built-in pickle (protocol 4) so
they are independent of NumPy's internal random-state serialisation.

Run once locally with Python 3.11 before every push that changes the
model artefacts:

    python retrain_lda.py

Requirements (already in requirements.txt):
    scikit-learn, pandas, joblib
"""

import re
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent
DATA_PATH  = ROOT / "outputs" / "extracted_topics.csv"
MODEL_DIR  = ROOT / "models" / "lda_model"
LDA_PATH   = MODEL_DIR / "lda_model.pkl"
VECTR_PATH = MODEL_DIR / "count_vectorizer.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── Topic labels (must match TOPIC_LABELS in utils.py) ────────────────────────
TOPIC_LABELS = {
    0: "nutritional_profile",
    1: "beverages_and_liquids",
    2: "chips_and_snacks",
    3: "order_and_delivery",
    4: "pet_food",
    5: "coffee_and_tea_pods",
}

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading data from {DATA_PATH} …")
df = pd.read_csv(DATA_PATH)
text_col = "review text"
assert text_col in df.columns, f"Expected column '{text_col}' not found."

def clean(text: str) -> str:
    """Minimal cleaning identical to what utils.py does at inference time."""
    return re.sub(r"[^a-z\s]", " ", str(text).lower())

corpus = df[text_col].fillna("").apply(clean).tolist()
print(f"  {len(corpus)} documents loaded.")

# ── Fit CountVectorizer ────────────────────────────────────────────────────────
print("Fitting CountVectorizer …")
vectorizer = CountVectorizer(
    max_df=0.95,
    min_df=2,
    stop_words="english",
    max_features=10_000,
)
dtm = vectorizer.fit_transform(corpus)
print(f"  DTM shape: {dtm.shape[0]} docs × {dtm.shape[1]} vocab terms")

# ── Train LDA ─────────────────────────────────────────────────────────────────
n_topics = len(TOPIC_LABELS)
print(f"Training LDA with {n_topics} topics …  (this may take ~30 s)")
lda = LatentDirichletAllocation(
    n_components=n_topics,
    random_state=42,
    max_iter=15,
    learning_method="online",
    n_jobs=1,       # deterministic
)
lda.fit(dtm)

# Print top words per topic for a sanity check
feature_names = vectorizer.get_feature_names_out()
print("\nTop words per topic:")
for idx, topic_vec in enumerate(lda.components_):
    top = [feature_names[i] for i in topic_vec.argsort()[:-11:-1]]
    print(f"  Topic {idx} ({TOPIC_LABELS[idx]}): {', '.join(top)}")

# ── Remove random_state before serialising ────────────────────────────────────
# random_state stores a NumPy RandomState / Generator object that is
# serialised with NumPy-version-specific Cython classes (MT19937 etc.).
# It is only used during .fit(); inference (.transform()) never touches it.
lda.random_state = None

# ── Save artefacts with protocol 4 (Python 3.8+, no NumPy internals) ─────────
print(f"\nSaving LDA model  → {LDA_PATH}")
joblib.dump(lda,        LDA_PATH,   compress=3)

print(f"Saving vectorizer → {VECTR_PATH}")
joblib.dump(vectorizer, VECTR_PATH, compress=3)

# ── Reload & verify ───────────────────────────────────────────────────────────
print("\nVerifying round-trip load …")
lda2   = joblib.load(LDA_PATH)
vectr2 = joblib.load(VECTR_PATH)

sample = clean("great coffee taste delicious morning")
row    = vectr2.transform([sample])
probs  = lda2.transform(row)[0]
assert abs(probs.sum() - 1.0) < 1e-5, "Topic probabilities don't sum to 1!"
dom    = int(probs.argmax())
print(f"  Sample → dominant topic {dom} ({TOPIC_LABELS[dom]}) "
      f"with {probs[dom]*100:.1f}% confidence")
print("  ✅ Artefacts saved and verified successfully.")

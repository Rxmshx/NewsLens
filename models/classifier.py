import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cleaner import clean_text

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH       = "data/ag_news.csv"
MODEL_PATH      = "results/tfidf_model.joblib"
VECTORIZER_PATH = "results/tfidf_vectorizer.joblib"
SAMPLE_SIZE     = 20000

os.makedirs("results", exist_ok=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "category"])
df = df[df["category"] != "general"]
df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

print("\n📊 Category Distribution:")
print(df["category"].value_counts())

# ── Clean Text ────────────────────────────────────────────────────────────────
print("\n🧹 Cleaning text...")
df["cleaned_text"] = df["text"].apply(clean_text)

# ── Features & Labels ─────────────────────────────────────────────────────────
X = df["cleaned_text"]
y = df["category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ── TF-IDF ────────────────────────────────────────────────────────────────────
print("⚙️  Vectorizing...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True        # ✅ log scaling — better for long docs
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# ── Train ─────────────────────────────────────────────────────────────────────
print("🚀 Training Logistic Regression...")
model = LogisticRegression(
    max_iter=500,
    class_weight="balanced",
    C=5.0,                   # ✅ stronger regularization
    solver="lbfgs",
)
model.fit(X_train_vec, y_train)

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test_vec)
print(f"\n📈 Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ── Save ──────────────────────────────────────────────────────────────────────
joblib.dump(model,      MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"💾 Model saved      → {MODEL_PATH}")
print(f"💾 Vectorizer saved → {VECTORIZER_PATH}")

# ── Inference Function (used by API) ─────────────────────────────────────────
def load_classifier():
    model      = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str) -> dict:
    """Predict category + confidence for a single text."""
    m, v    = load_classifier()
    cleaned = clean_text(text)
    vec     = v.transform([cleaned])
    label   = m.predict(vec)[0]
    probs   = m.predict_proba(vec)[0]
    classes = m.classes_

    return {
        "category":   label,
        "confidence": round(float(probs.max()) * 100, 2),
        "all_scores": {
            cls: round(float(p) * 100, 2)
            for cls, p in zip(classes, probs)
        }
    }


# ── Sample Prediction ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = "Government plans new AI regulation for tech companies"
    result = predict(sample)
    print(f"\n🧪 Sample Prediction:")
    print(f"  Text:       {sample}")
    print(f"  Category:   {result['category']}")
    print(f"  Confidence: {result['confidence']}%")
    print(f"  All scores: {result['all_scores']}")
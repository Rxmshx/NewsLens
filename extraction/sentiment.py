import sys
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cleaner import clean_for_ner

# ── Load FinBERT ──────────────────────────────────────────────────────────────
MODEL_NAME = "ProsusAI/finbert"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🤖 Loading FinBERT on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print("✅ FinBERT ready")


def get_sentiment(text: str) -> dict:
    """
    Analyze sentiment using FinBERT.
    Returns label, confidence scores for all 3 classes.
    """
    cleaned = clean_for_ner(text)
    if not cleaned or len(cleaned.strip()) < 10:
        return {
            "label":      "neutral",
            "confidence": 0.0,
            "scores":     {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
        }

    inputs = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs  = F.softmax(logits, dim=-1).cpu().numpy()[0]
    labels = ["positive", "negative", "neutral"]  # FinBERT label order

    pred_idx    = int(probs.argmax())
    pred_label  = labels[pred_idx]
    confidence  = float(probs[pred_idx])

    return {
        "label":      pred_label,
        "confidence": round(confidence * 100, 2),
        "scores": {
            "positive": round(float(probs[0]) * 100, 2),
            "negative": round(float(probs[1]) * 100, 2),
            "neutral":  round(float(probs[2]) * 100, 2),
        }
    }


def get_sentiment_label(text: str) -> str:
    """Simple wrapper — returns just the label string."""
    return get_sentiment(text)["label"]


if __name__ == "__main__":
    samples = [
        "Apple surged to record profits, beating all analyst expectations.",
        "Tesla shares plunged after reporting massive losses and layoffs.",
        "The Federal Reserve held interest rates steady in its latest meeting.",
        "AI is transforming the world in a great way.",
        "Markets crashed amid fears of a global recession and rising inflation.",
    ]

    for s in samples:
        result = get_sentiment(s)
        print(f"\nText: {s[:65]}")
        print(f"  Label:      {result['label']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Scores:     {result['scores']}")
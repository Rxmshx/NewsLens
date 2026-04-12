import torch
import sys
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use simple cleaning instead of spaCy for LSTM
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return " ".join(tokens)
    
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🤖 Loading sentiment model on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print("✅ Sentiment model ready")


def get_sentiment(text: str) -> dict:
    cleaned = clean_text(text)
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
    labels = ["negative", "neutral", "positive"]  # RoBERTa label order

    pred_idx   = int(probs.argmax())
    pred_label = labels[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "label":      pred_label,
        "confidence": round(confidence * 100, 2),
        "scores": {
            "positive": round(float(probs[2]) * 100, 2),
            "negative": round(float(probs[0]) * 100, 2),
            "neutral":  round(float(probs[1]) * 100, 2),
        }
    }


def get_sentiment_label(text: str) -> str:
    return get_sentiment(text)["label"]


if __name__ == "__main__":
    samples = [
        "Apple surged to record profits, beating all analyst expectations.",
        "Tesla shares plunged after reporting massive losses and layoffs.",
        "Manchester United won the Premier League title in a stunning comeback.",
        "The president signed a controversial immigration bill into law.",
        "Scientists discover a potential cure for Alzheimer's disease.",
        "Markets crashed amid fears of a global recession.",
    ]

    for s in samples:
        result = get_sentiment(s)
        print(f"\nText: {s[:65]}")
        print(f"  Label:      {result['label']}")
        print(f"  Confidence: {result['confidence']}%")
        print(f"  Scores:     {result['scores']}")
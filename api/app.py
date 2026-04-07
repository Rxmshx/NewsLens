import os
import json
import torch
import requests
from bs4 import BeautifulSoup
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cleaner import clean_text, clean_for_ner
from extraction.ner import extract_entities, extract_entities_flat
from extraction.keywords import extract_keywords, extract_keywords_simple
from extraction.sentiment import get_sentiment

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="NLP News Analysis API",
    description="Text classification and information extraction for news articles",
    version="2.0.0"
)

# Add this after app = FastAPI(...)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def root():
    return FileResponse("static/index.html")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_DIR      = "./results/bert_model"
LABEL_MAP_PATH = "./results/label_map.json"
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load DistilBERT Classifier ────────────────────────────────────────────────
print(f"🤖 Loading DistilBERT from {MODEL_DIR}...")
bert_tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
bert_model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
bert_model.to(DEVICE)
bert_model.eval()

with open(LABEL_MAP_PATH) as f:
    label_data = json.load(f)
id2label = {int(k): v for k, v in label_data["id2label"].items()}

print(f"✅ API ready | Device: {DEVICE}")

# ── Request Schemas ───────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

class URLRequest(BaseModel):
    url: str

# ── Core Functions ────────────────────────────────────────────────────────────
def classify_text(text: str) -> dict:
    """Run DistilBERT classification."""
    cleaned = clean_text(text)
    inputs  = bert_tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = bert_model(**inputs).logits

    probs      = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    pred_id    = int(np.argmax(probs))
    confidence = float(probs[pred_id])

    return {
        "category":   id2label[pred_id],
        "confidence": round(confidence * 100, 2),
        "all_scores": {
            id2label[i]: round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        }
    }


def extract_info(text: str) -> dict:
    """Run full information extraction pipeline."""
    return {
        "entities":  extract_entities(text),        # from ner.py
        "keywords":  extract_keywords(text, top_n=10),  # from keywords.py
        "sentiment": get_sentiment(text),           # from sentiment.py (FinBERT)
    }


def scrape_url(url: str) -> dict:
    """Scrape title and body content from a news URL."""
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

    soup  = BeautifulSoup(response.text, "html.parser")
    title = soup.find("title")
    title = title.get_text(strip=True) if title else "No title found"

    paragraphs = soup.find_all("p")
    content    = " ".join(p.get_text(strip=True) for p in paragraphs)

    if len(content.strip()) < 100:
        raise HTTPException(
            status_code=422,
            detail="Could not extract enough content from this URL."
        )

    return {"title": title, "content": content}


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":  "ok",
        "device":  str(DEVICE),
        "model":   MODEL_DIR,
        "version": "2.0.0"
    }


@app.post("/classify")
def classify(req: TextRequest):
    """Classify news text into a category using DistilBERT."""
    if len(req.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text too short. Minimum 20 characters.")
    return {
        "input_length":   len(req.text),
        "classification": classify_text(req.text)
    }


@app.post("/extract")
def extract(req: TextRequest):
    """Extract entities, keywords, and sentiment from text."""
    if len(req.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text too short.")
    return extract_info(req.text)


@app.post("/analyze")
def analyze(req: TextRequest):
    """Full pipeline: classify + extract on raw text."""
    if len(req.text.strip()) < 20:
        raise HTTPException(status_code=422, detail="Text too short.")
    return {
        "input_length":   len(req.text),
        "classification": classify_text(req.text),
        "extraction":     extract_info(req.text)
    }


@app.post("/analyze-url")
def analyze_url(req: URLRequest):
    """Full pipeline: scrape URL → classify + extract."""
    scraped = scrape_url(req.url)
    text    = scraped["title"] + " " + scraped["content"]
    return {
        "title":          scraped["title"],
        "url":            req.url,
        "classification": classify_text(text),
        "extraction":     extract_info(text)
    }
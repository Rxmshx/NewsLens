---
title: NewsLens
emoji: 📰
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: AI-powered news classification and information extraction
---

# 📰 NewsLens - AI News Analysis System

An end-to-end **AI-powered news analysis platform** that performs **news classification, sentiment analysis, entity extraction, and keyword extraction** using advanced NLP and deep learning models.

---

## 🚀 Features

* 🧠 **News Classification** using DistilBERT (92.3% accuracy)
* 💰 **Financial Sentiment Analysis** using FinBERT (94%+ confidence)
* 🏷️ **Named Entity Recognition** using spaCy
* 🔑 **Keyword Extraction** using NLP techniques
* 🌐 **Web Scraping** (BBC & Yahoo Finance)
* ⚡ **FastAPI Backend** for real-time predictions
* 🎨 **Glassmorphism Web UI**
* 🚀 **GPU-Accelerated Training & Inference**

---

## 🧠 Models Used

| Task | Model |
|---|---|
| Text Classification | DistilBERT |
| Sentiment Analysis | FinBERT |
| NER | spaCy en_core_web_sm |
| Keyword Extraction | POS-based NLP |

---

## ▶️ How to Run

```bash
git clone https://github.com/Rxmshx/NewsLens.git
cd NewsLens
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m uvicorn api.app:app --reload --port 8000
```

---

## 📊 Model Performance

* ✅ Accuracy: 92.3%
* ✅ F1 Score: 92.26%
* ✅ Balanced Precision & Recall
* 🚀 Trained on RTX 5070 Ti GPU
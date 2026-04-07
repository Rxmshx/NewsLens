# 📰 NLP News Analysis System

An end-to-end **AI-powered news analysis platform** that performs **news classification, sentiment analysis, entity extraction, and keyword extraction** using advanced NLP and deep learning models.

---

## 🚀 Features

* 🧠 **News Classification** using DistilBERT (Transformer-based model)
* 💰 **Financial Sentiment Analysis** using FinBERT
* 🏷️ **Named Entity Recognition (NER)** using spaCy
* 🔑 **Keyword Extraction** using NLP techniques
* 🌐 **Web Scraping Support** (BBC & Reuters)
* ⚡ **FastAPI Backend** for real-time predictions
* 🎨 **Web UI Integration** for user interaction
* 🚀 **GPU-Accelerated Training & Inference**

---

## 🧠 Models Used

| Task                | Model      |
| ------------------- | ---------- |
| Text Classification | DistilBERT |
| Sentiment Analysis  | FinBERT    |
| NER                 | spaCy      |
| Keyword Extraction  | RAKE / NLP |

---

## 📁 Project Structure

```
web-nlp-project/
│
├── api/                # FastAPI backend
├── data/               # Datasets (BBC, AG News, etc.)
├── extraction/         # NER, sentiment, keywords
├── models/             # ML & BERT models
├── preprocessing/      # Text cleaning
├── scraper/            # News scraping scripts
├── results/            # Saved models & outputs
├── static/             # UI (HTML, CSS)
├── ui/                 # Frontend logic
├── utils/              # Helper functions
│
├── main.py             # Pipeline runner
└── README.md
```

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/news-nlp-analysis-system.git
cd news-nlp-analysis-system
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### 🔹 Run FastAPI Server

```bash
uvicorn api.app:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Example Usage

### Input

```json
{
  "text": "Tesla shares plunged after reporting massive losses."
}
```

### Output

```json
{
  "category": "business",
  "sentiment": "negative",
  "entities": ["Tesla"],
  "keywords": ["losses", "shares", "plunged"]
}
```

---

## 📊 Model Performance

* ✅ Accuracy: ~92%
* ✅ F1 Score: ~92%
* ✅ Balanced Precision & Recall
* 🚀 Trained using GPU acceleration

---

## 🔥 Key Highlights

* Combines **multiple NLP models** in a single pipeline
* Uses **Transformer-based deep learning (BERT)**
* Handles **real-world news data**
* Designed as a **production-ready API system**

---

## 🛠️ Tech Stack

* Python
* PyTorch
* Hugging Face Transformers
* FastAPI
* spaCy
* NLTK
* BeautifulSoup (Scraping)

---

## 📌 Future Improvements

* 🌍 Real-time news streaming
* 📊 Dashboard for analytics
* 🌐 Deployment (Render / AWS)
* 📱 Mobile-friendly UI

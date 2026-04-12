import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from collections import Counter
import sys
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

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "data/ag_news.csv"
SAVE_DIR     = "results/lstm_model"
VOCAB_PATH   = "results/lstm_vocab.json"
LABEL_PATH   = "results/lstm_labels.json"
SAMPLE_SIZE  = 20000
MAX_LEN      = 128
VOCAB_SIZE   = 20000
EMBED_DIM    = 128
HIDDEN_DIM   = 256
NUM_LAYERS   = 2
DROPOUT      = 0.3
BATCH_SIZE   = 64
EPOCHS       = 10
LR           = 1e-3
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"✅ Using device: {DEVICE}")

# ── Load & Prepare Data ───────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text", "category"])
df = df[df["category"] != "general"]
df = df.sample(min(SAMPLE_SIZE, len(df)), random_state=42).reset_index(drop=True)

labels    = sorted(df["category"].unique().tolist())
label2id  = {l: i for i, l in enumerate(labels)}
id2label  = {i: l for l, i in label2id.items()}
df["label"] = df["category"].map(label2id)

print(f"📊 Classes: {labels}")
print(f"📈 Samples: {len(df)}")

# ── Clean Text ────────────────────────────────────────────────────────────────
print("🧹 Cleaning text...")
df["cleaned"] = df["text"].apply(clean_text)

# ── Build Vocabulary ──────────────────────────────────────────────────────────
print("📖 Building vocabulary...")
all_words = []
for text in df["cleaned"]:
    all_words.extend(text.split())

word_counts = Counter(all_words)
vocab = ["<PAD>", "<UNK>"] + [w for w, c in word_counts.most_common(VOCAB_SIZE - 2)]
word2id = {w: i for i, w in enumerate(vocab)}

# Save vocab and labels
with open(VOCAB_PATH, "w") as f:
    json.dump(word2id, f)
with open(LABEL_PATH, "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f)

print(f"📚 Vocabulary size: {len(word2id)}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class NewsDataset(Dataset):
    def __init__(self, texts, labels, word2id, max_len):
        self.texts  = texts
        self.labels = labels
        self.w2id   = word2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def encode(self, text):
        tokens = text.split()[:self.max_len]
        ids    = [self.w2id.get(t, 1) for t in tokens]  # 1 = <UNK>
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx):
        return self.encode(self.texts[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded  = pad_sequence(texts, batch_first=True, padding_value=0)
    return texts_padded, torch.stack(labels)


# ── Train/Test Split ──────────────────────────────────────────────────────────
train_df, test_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

train_ds = NewsDataset(train_df["cleaned"].tolist(), train_df["label"].tolist(), word2id, MAX_LEN)
test_ds  = NewsDataset(test_df["cleaned"].tolist(),  test_df["label"].tolist(),  word2id, MAX_LEN)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ── LSTM Model ────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers    = num_layers,
            batch_first   = True,
            dropout       = dropout if num_layers > 1 else 0,
            bidirectional = True         # ✅ Bidirectional LSTM
        )
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        embedded        = self.dropout(self.embedding(x))
        output, (hn, _) = self.lstm(embedded)
        # Use last hidden state from both directions
        hn_forward  = hn[-2]   # last layer, forward
        hn_backward = hn[-1]   # last layer, backward
        hidden      = torch.cat([hn_forward, hn_backward], dim=1)
        hidden      = self.dropout(hidden)
        return self.fc(hidden)


model = LSTMClassifier(
    vocab_size  = len(word2id),
    embed_dim   = EMBED_DIM,
    hidden_dim  = HIDDEN_DIM,
    num_layers  = NUM_LAYERS,
    num_classes = len(labels),
    dropout     = DROPOUT
).to(DEVICE)

print(f"\n🤖 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Training ──────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
criterion = nn.CrossEntropyLoss()

best_acc  = 0.0

print("\n🚀 Starting LSTM training...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_preds  = []
    all_labels = []

    for texts, lbls in train_loader:
        texts, lbls = texts.to(DEVICE), lbls.to(DEVICE)

        optimizer.zero_grad()
        logits = model(texts)
        loss   = criterion(logits, lbls)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(lbls.cpu().numpy())

    train_acc  = accuracy_score(all_labels, all_preds)
    train_loss = total_loss / len(train_loader)

    # ── Validation ────────────────────────────────────────────────────────────
    model.eval()
    val_preds  = []
    val_labels = []

    with torch.no_grad():
        for texts, lbls in test_loader:
            texts, lbls = texts.to(DEVICE), lbls.to(DEVICE)
            logits = model(texts)
            preds  = logits.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_labels.extend(lbls.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1  = f1_score(val_labels, val_preds, average="macro")
    scheduler.step(1 - val_acc)

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f"{SAVE_DIR}/lstm_best.pt")
        print(f"  💾 Best model saved (acc={best_acc:.4f})")

# ── Final Evaluation ──────────────────────────────────────────────────────────
print("\n📊 Final Evaluation:")
print(classification_report(val_labels, val_preds, target_names=labels, zero_division=0))

# ── Inference Function ────────────────────────────────────────────────────────
def load_lstm():
    with open(VOCAB_PATH)  as f: w2id    = json.load(f)
    with open(LABEL_PATH)  as f: ldata   = json.load(f)
    id2label = {int(k): v for k, v in ldata["id2label"].items()}
    m = LSTMClassifier(len(w2id), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, len(id2label), DROPOUT).to(DEVICE)
    m.load_state_dict(torch.load(f"{SAVE_DIR}/lstm_best.pt", map_location=DEVICE))
    m.eval()
    return m, w2id, id2label


def predict_lstm(text: str) -> dict:
    m, w2id, id2label = load_lstm()
    cleaned = clean_text(text)
    tokens  = cleaned.split()[:MAX_LEN]
    ids     = torch.tensor([[w2id.get(t, 1) for t in tokens]], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        logits = m(ids)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_id    = int(probs.argmax())
    return {
        "category":   id2label[pred_id],
        "confidence": round(float(probs[pred_id]) * 100, 2),
        "all_scores": {id2label[i]: round(float(p)*100, 2) for i, p in enumerate(probs)},
        "model":      "BiLSTM"
    }


if __name__ == "__main__":
    sample = "Government plans new AI regulation for tech companies"
    result = predict_lstm(sample)
    print(f"\n🧪 Sample: {sample}")
    print(f"   Category:   {result['category']}")
    print(f"   Confidence: {result['confidence']}%")
    print(f"   All scores: {result['all_scores']}")
import os
import json
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from datasets import Dataset

# ── Device ──────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME      = "distilbert-base-uncased"
DATA_PATH       = "data/ag_news.csv"
OUTPUT_DIR      = "./results/bert_model"
LABEL_MAP_PATH  = "./results/label_map.json"
SAMPLE_SIZE     = 20000   # increase if you want more accuracy
MAX_LENGTH      = 128
BATCH_SIZE      = 32
EPOCHS          = 4
LR              = 2e-5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("./results", exist_ok=True)

# ── Load & Prepare Data ──────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
df = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

# Label encoding
labels     = sorted(df["category"].unique().tolist())
label2id   = {label: i for i, label in enumerate(labels)}
id2label   = {i: label for label, i in label2id.items()}
df["label"] = df["category"].map(label2id)

# Save label map for API use later
with open(LABEL_MAP_PATH, "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
print(f"💾 Label map saved → {LABEL_MAP_PATH}")
print(f"📊 Classes: {labels}")

# Train / Val / Test split  (70 / 15 / 15)
train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val_df,   test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42)
print(f"📈 Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ── Tokenizer ────────────────────────────────────────────────────────────────
print("🔤 Loading tokenizer...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

def make_dataset(dataframe):
    ds = Dataset.from_pandas(dataframe[["text", "label"]].reset_index(drop=True))
    ds = ds.map(tokenize, batched=True, batch_size=256)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return ds

print("⚙️  Tokenizing datasets...")
train_dataset = make_dataset(train_df)
val_dataset   = make_dataset(val_df)
test_dataset  = make_dataset(test_df)

# ── Model ────────────────────────────────────────────────────────────────────
print("🤖 Loading model...")
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)
model.to(device)

# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels_true = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels_true, preds),
        "f1_macro":  f1_score(labels_true, preds, average="macro"),
        "precision": precision_score(labels_true, preds, average="macro", zero_division=0),
        "recall":    recall_score(labels_true, preds, average="macro", zero_division=0),
    }

# ── Training Arguments ───────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir                  = OUTPUT_DIR,
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate               = LR,
    warmup_steps                = 100,
    weight_decay                = 0.01,
    fp16                        = True,
    save_strategy               = "best",
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1_macro",
    greater_is_better           = True,
    logging_steps               = 50,
    report_to                   = "none",
    dataloader_num_workers      = 0,
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    compute_metrics = compute_metrics,
)

# ── Train ────────────────────────────────────────────────────────────────────
print("\n🚀 Starting training...\n")
trainer.train()

# ── Final Evaluation on Test Set ─────────────────────────────────────────────
print("\n📊 Evaluating on test set...")
results = trainer.evaluate(test_dataset)
print("\n── Test Results ──────────────────────────────")
for k, v in results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# ── Save Model & Tokenizer ───────────────────────────────────────────────────
print(f"\n💾 Saving model → {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅ Training complete! Model saved.")
from datasets import load_dataset
import pandas as pd

# Load dataset
dataset = load_dataset("ag_news")

# Convert to pandas
train_df = dataset["train"].to_pandas()

# Rename columns
train_df = train_df.rename(columns={
    "text": "text",
    "label": "category"
})

# Map labels to names
label_map = {
    0: "world",
    1: "sports",
    2: "business",
    3: "tech"
}

train_df["category"] = train_df["category"].map(label_map)

# Save
train_df.to_csv("data/ag_news.csv", index=False)

print("✅ AG News dataset saved!")
print(train_df.head())
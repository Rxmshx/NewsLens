import pandas as pd
from preprocessing.cleaner import clean_text
from extraction.ner import extract_entities
from extraction.sentiment import get_sentiment
from extraction.keywords import extract_keywords


# 🔹 Load datasets
bbc = pd.read_csv("data/bbc_news.csv")
reuters = pd.read_csv("data/reuters_news.csv")

# 🔹 Combine datasets
df = pd.concat([bbc, reuters], ignore_index=True)

# 🔹 Remove empty rows
df = df.dropna(subset=["text"])

print("📊 Total combined articles:", len(df))


# 🔹 Apply NLP pipeline
df["cleaned_text"] = df["text"].apply(clean_text)
df["entities"] = df["text"].apply(extract_entities)
df["sentiment"] = df["text"].apply(get_sentiment)
df["keywords"] = df["text"].apply(extract_keywords)


# 🔹 Save FINAL dataset (only one file)
df.to_csv("data/combined_news_full.csv", index=False)

print("\n✅ FINAL dataset saved: data/combined_news_full.csv")
print(df[["text", "cleaned_text", "sentiment", "keywords"]].head())
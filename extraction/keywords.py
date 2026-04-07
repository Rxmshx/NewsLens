import spacy
import sys
import os
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cleaner import clean_for_ner

nlp = spacy.load("en_core_web_sm")


def extract_keywords(text: str, top_n: int = 10) -> list:
    """
    Extract top keywords using spaCy POS tagging.
    Returns list of dicts with word + score.
    """
    cleaned = clean_for_ner(text)
    if not cleaned:
        return []

    doc = nlp(cleaned)

    # Keep only nouns, proper nouns, adjectives — skip stopwords/punct
    candidates = [
        token.lemma_.lower()
        for token in doc
        if token.pos_ in ("NOUN", "PROPN", "ADJ")
        and not token.is_stop
        and not token.is_punct
        and len(token.text.strip()) > 2
    ]

    if not candidates:
        return []

    counts = Counter(candidates)
    total  = sum(counts.values())

    keywords = [
        {
            "keyword": word,
            "score":   round(count / total, 4),
            "count":   count
        }
        for word, count in counts.most_common(top_n)
    ]

    return keywords


def extract_keywords_simple(text: str, top_n: int = 10) -> list:
    """
    Returns just a list of keyword strings.
    Used by sentiment.py and other modules.
    """
    return [k["keyword"] for k in extract_keywords(text, top_n)]


if __name__ == "__main__":
    sample = """
    Apple reported record revenue of $25 billion in Q1 2024.
    The tech giant saw strong iPhone sales despite global economic uncertainty.
    CEO Tim Cook credited AI features and emerging markets for the growth.
    """
    print("── Keywords with scores ──────────────────")
    for kw in extract_keywords(sample):
        print(f"  {kw['keyword']:20} score={kw['score']:.4f}  count={kw['count']}")

    print("\n── Simple list ───────────────────────────")
    print(extract_keywords_simple(sample))
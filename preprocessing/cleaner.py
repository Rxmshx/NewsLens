import re
import nltk
import spacy
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words("english"))


def clean_text(text: str, lemmatize: bool = True) -> str:
    """
    Clean text for classification/keyword extraction.
    Lowercases AFTER spaCy processing to preserve NER quality.
    """
    if not text or len(text.strip()) < 5:
        return ""

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    if lemmatize:
        doc = nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if token.text.lower() not in stop_words
            and not token.is_punct
            and not token.is_space
            and len(token.text.strip()) > 1
        ]
        return " ".join(tokens)
    else:
        # Light clean only — preserves case for NER
        return text.strip()


def clean_for_ner(text: str) -> str:
    """
    Minimal cleaning for NER — preserves case and entities.
    Only removes HTML and URLs.
    """
    if not text or len(text.strip()) < 5:
        return ""
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


if __name__ == "__main__":
    sample = "AI is transforming the world! Companies like Google and Microsoft are leading $4.2B investments in 2024."
    print("Original:      ", sample)
    print("Cleaned:       ", clean_text(sample))
    print("NER-preserved: ", clean_for_ner(sample))
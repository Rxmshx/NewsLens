import spacy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.cleaner import clean_for_ner

nlp = spacy.load("en_core_web_sm")

# All entity types we care about
ENTITY_TYPES = {
    "PERSON":  "People",
    "ORG":     "Organizations",
    "GPE":     "Locations",
    "LOC":     "Locations",
    "MONEY":   "Financial",
    "PERCENT": "Financial",
    "DATE":    "Dates",
    "PRODUCT": "Products",
    "EVENT":   "Events",
    "NORP":    "Groups",        # nationalities, political groups
}


def extract_entities(text: str) -> dict:
    """
    Extract named entities from text.
    Uses clean_for_ner() to preserve case for accurate NER.
    """
    cleaned = clean_for_ner(text)
    if not cleaned:
        return {}

    doc = nlp(cleaned)

    entities = {}
    for ent in doc.ents:
        if ent.label_ not in ENTITY_TYPES:
            continue

        label = ent.label_
        if label not in entities:
            entities[label] = []

        # ✅ Deduplicate
        if ent.text not in entities[label]:
            entities[label].append(ent.text)

    return entities


def extract_entities_flat(text: str) -> list:
    """
    Returns a flat list of (text, label) tuples.
    Useful for display in Streamlit.
    """
    cleaned = clean_for_ner(text)
    if not cleaned:
        return []

    doc = nlp(cleaned)
    return [
        {"text": ent.text, "label": ent.label_, "description": ENTITY_TYPES.get(ent.label_, ent.label_)}
        for ent in doc.ents
        if ent.label_ in ENTITY_TYPES
    ]


if __name__ == "__main__":
    sample = """
    Elon Musk visited India and met Prime Minister Narendra Modi at New Delhi.
    Tesla reported $25.7B in revenue for Q1 2024, up 12% from last year.
    Apple and Google are competing in the AI space with new products.
    """

    print("── Grouped Entities ──────────────────────")
    from pprint import pprint
    pprint(extract_entities(sample))

    print("\n── Flat Entities ─────────────────────────")
    for ent in extract_entities_flat(sample):
        print(f"  {ent['label']:10} | {ent['text']}")
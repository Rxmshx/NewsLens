import torch
import sys
import os
from transformers import BartForConditionalGeneration, BartTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "facebook/bart-large-cnn"

print(f"🤖 Loading summarizer ({'GPU' if torch.cuda.is_available() else 'CPU'})...")
tokenizer  = BartTokenizer.from_pretrained(MODEL_NAME)
model      = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()
print("✅ Summarizer ready")


def summarize(text: str, max_length: int = 130, min_length: int = 30) -> dict:
    if not text or len(text.strip()) < 100:
        return {
            "summary":           text.strip(),
            "original_length":   len(text.split()),
            "summary_length":    len(text.split()),
            "compression_ratio": 1.0
        }

    words     = text.split()
    truncated = " ".join(words[:800])

    input_words = len(truncated.split())
    max_length  = min(max_length, max(30, input_words // 3))
    min_length  = min(min_length, max_length - 10)

    inputs = tokenizer(
        truncated,
        return_tensors = "pt",
        truncation     = True,
        max_length     = 1024
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams  = 4,
            max_length = max_length,
            min_length = min_length,
            length_penalty    = 2.0,
            early_stopping    = True,
            no_repeat_ngram_size = 3
        )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return {
        "summary":           summary,
        "original_length":   len(words),
        "summary_length":    len(summary.split()),
        "compression_ratio": round(len(summary.split()) / len(words), 2)
    }


def summarize_headline(text: str) -> str:
    return summarize(text)["summary"]


if __name__ == "__main__":
    sample = """
    Apple Inc. reported record-breaking quarterly revenue of $119.6 billion, 
    surpassing analyst expectations by a wide margin. CEO Tim Cook attributed 
    the strong performance to robust iPhone sales, particularly in emerging markets 
    such as India and Southeast Asia. The company also saw significant growth in 
    its services division, which includes the App Store, Apple Music, and iCloud, 
    generating $23.1 billion in revenue — a 16% year-over-year increase. 
    Despite global supply chain challenges and macroeconomic uncertainty, Apple 
    maintained strong profit margins. Cook expressed optimism about the upcoming 
    product lineup, hinting at new AI-powered features across iPhone, Mac, and 
    iPad devices. The company also announced a $110 billion share buyback program, 
    the largest in its history, signaling strong confidence in its future growth prospects.
    Analysts on Wall Street responded positively, with several firms raising their 
    price targets for Apple stock following the earnings announcement.
    """

    print("── Summary ───────────────────────────────────────────")
    result = summarize(sample)
    print(f"Summary:    {result['summary']}")
    print(f"Original:   {result['original_length']} words")
    print(f"Compressed: {result['summary_length']} words")
    print(f"Ratio:      {result['compression_ratio']}")
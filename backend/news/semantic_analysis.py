from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="ProsusAI/finbert",
    tokenizer="ProsusAI/finbert",
)


def analyze_sentiment(headline) -> tuple[str, float]:
    result = classifier(headline)
    return result[0]["label"], result[0]["score"]

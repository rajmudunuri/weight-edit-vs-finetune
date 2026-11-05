"""Load the classifier and score batches of text."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_id=MODEL_ID, **kwargs):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    return tok, model.to(DEVICE).eval()


@torch.no_grad()
def predict(tok, model, texts, batch_size=128):
    """Return P(positive) for each text."""
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  return_tensors="pt").to(DEVICE)
        logits = model(**enc).logits
        probs.append(logits.softmax(-1)[:, 1].float().cpu())
    return torch.cat(probs)

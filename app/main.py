import os
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel


app = FastAPI()

# ---- Bigram text generation ----
_DEFAULT_CORPUS = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]
bigram_model = BigramModel(_DEFAULT_CORPUS)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": text}


# ---- Embedding endpoint (spaCy) ----
import spacy


def _load_spacy_model():
    model_name = os.getenv("SPACY_MODEL", "en_core_web_md")
    try:
        return spacy.load(model_name)
    except Exception:
        # Safe fallback to a blank English pipeline (no vectors)
        return spacy.blank("en")


nlp = _load_spacy_model()


class EmbedRequest(BaseModel):
    text: str
    pooling: str | None = "mean"  # "mean" or "tokens"


@app.post("/embed")
def embed(req: EmbedRequest):
    doc = nlp(req.text)
    if req.pooling == "tokens":
        vectors = [t.vector.tolist() if t.has_vector else [] for t in doc]
        dim = int(len(doc[0].vector)) if len(doc) and doc[0].has_vector else 0
        return {"tokens": [t.text for t in doc], "vectors": vectors, "dim": dim}

    # Default: sentence-level mean pooling
    vecs = [t.vector for t in doc if t.has_vector]
    if not vecs:
        return {"vector": [], "dim": 0}
    import numpy as np
    mean_vec = np.mean(vecs, axis=0)
    return {"vector": mean_vec.tolist(), "dim": int(mean_vec.shape[0])}

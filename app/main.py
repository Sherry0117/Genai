from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel

app = FastAPI()

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. \
It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(request.start_word, request.length)
    return {"generated_text": generated_text}
    
# --- Embedding endpoint (spaCy) ---
import spacy
from pydantic import BaseModel

# Just load the model once
try:
    nlp
except NameError:
    nlp = spacy.load("en_core_web_lg")

class EmbedRequest(BaseModel):
    text: str
    pooling: str | None = "mean"   # "mean" or "tokens"

@app.post("/embed")
def embed(req: EmbedRequest):
    doc = nlp(req.text)
    if req.pooling == "tokens":
        return {
            "tokens": [t.text for t in doc],
            "vectors": [t.vector.tolist() for t in doc],
            "dim": int(doc[0].vector.shape[0]) if len(doc) else 0,
        }
    # Default: sentence-level mean pooling
    vecs = [t.vector for t in doc if t.has_vector]
    if not vecs:
        return {"vector": [], "dim": 0}
    import numpy as np
    mean_vec = np.mean(vecs, axis=0)
    return {"vector": mean_vec.tolist(), "dim": int(mean_vec.shape[0])}

    



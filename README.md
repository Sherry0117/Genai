# GenAI – FastAPI (Module 3 / Assignment 1)

This repo contains:
- `POST /generate` — bigram text generator.
- `POST /embed` — spaCy word embeddings (`en_core_web_lg`) with mean pooling or per-token vectors.

## Run locally
```bash
uv sync
uv run fastapi dev app/main.py
# Open http://127.0.0.1:8000 and /docs

git remote -v
git remote set-url origin https://github.com/Sherry0117/Genai.git

git config --global credential.helper osxkeychain


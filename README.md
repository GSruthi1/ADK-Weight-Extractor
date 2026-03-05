# ADK Weight Extractor (Gemini + ADK + Flask)

## What it does
POST an image path (or upload an image) → returns JSON containing `net_weight`, `confidence`, and `status`.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
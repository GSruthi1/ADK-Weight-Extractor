# ADK Weight Extraction Agent

This project is a Proof-of-Concept (POC) for extracting **Net Weight** from warehouse label images using an AI agent.

The system uses:

- Google ADK (Agent Development Kit)
- Gemini Vision Model (gemini-2.5-flash)
- Python
- Flask API

The agent reads a label image and returns structured JSON output containing the detected **net weight**, **confidence score**, and **status**.

---

# Architecture Overview

Image Input (Local Path / Upload)  
↓  
Flask API  
↓  
ADK Agent  
↓  
Gemini Vision Model  
↓  
Confidence Guardrail  
↓  
JSON Output  

---

# Setup Instructions

## 1. Clone the repository

```bash
git clone https://github.com/GSruthi1/ADK-Weight-Extractor.git
cd ADK-Weight-Extractor
```

## 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# Configure Gemini API Key

Create a `.env` file in the project root:

```bash
touch .env
```

Add your Gemini API key:

```
GOOGLE_API_KEY=your_api_key_here
```

Generate a Gemini API key here:

https://aistudio.google.com/app/apikey

---

# Run the API

Start the Flask server:

```bash
python app.py
```

The API will start at:

```
http://127.0.0.1:5000
```

---

# Test the API

### Test using image path

```bash
curl -X POST http://127.0.0.1:5000/extract-weight \
-H "Content-Type: application/json" \
-d '{"image_path": "sample_images/Box_1.jpg"}'
```

Example response:

```
{
"net_weight": "25.00 lb",
"confidence": 0.95,
"reason": "Found NET WT on label",
"status": "accepted"
}
```

---

### Test using image upload

```bash
curl -X POST http://127.0.0.1:5000/extract-weight \
-F "file=@sample_images/Box_1.jpg"
```

---

# Confidence Guardrail

The system validates results using a confidence threshold.

```
confidence ≥ 0.90 → Accepted  
confidence < 0.90 → Manual Review Required
```

Example response when confidence is low:

```
{
"net_weight": "12.00 lb",
"confidence": 0.85,
"status": "rejected",
"message": "Confidence below 0.90 — manual review required"
}
```

---

# Supported Image Formats

- JPG
- JPEG
- PNG
- TIFF

---

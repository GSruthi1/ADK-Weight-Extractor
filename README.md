# ADK Weight Extractor (Gemini + ADK + Flask)
This project is a Proof of Concept (POC) for extracting Net Weight from warehouse label images using an AI agent.

## What it does
POST an image path (or upload an image) → returns JSON containing `net_weight`, `confidence`, and `status`.

The system uses:
Google ADK (Agent Development Kit)
Gemini Vision Model (gemini-2.5-flash)
Python
Flask API

The agent reads a label image and returns structured JSON output containing the detected net weight, confidence score, and status.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt



## Architecture Overview

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

## Features

Extract Net Weight from label images
Supports multiple image formats
Returns structured JSON output
Confidence validation to avoid incorrect readings
Flags low-confidence results for manual review
Handles foggy or blurred labels
Simple API for integration with automation workflows


## Setup Instructions
1. Clone the Repository
git clone https://github.com/GSruthi1/ADK-Weight-Extractor.git
cd ADK-Weight-Extractor
2. Create a Virtual Environment
python3 -m venv .venv
source .venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
Configure Gemini API Key

Create a .env file in the project root:

touch .env

Add your Gemini API key inside the file:

GOOGLE_API_KEY=your_api_key_here

You can generate an API key from:

https://aistudio.google.com/app/apikey

Run the API

Start the Flask server:

python app.py

The service will start at:

http://127.0.0.1:5000
Test the API
Option 1 — Test using an image path
curl -X POST http://127.0.0.1:5000/extract-weight \
-H "Content-Type: application/json" \
-d '{
"image_path": "sample_images/Box_1.jpg"
}'

Example response:

{
"net_weight": "25.00 lb",
"confidence": 0.95,
"reason": "Found NET WT on label",
"status": "accepted"
}
Option 2 — Upload an image file
curl -X POST http://127.0.0.1:5000/extract-weight \
-F "file=@sample_images/Box_1.jpg"
Confidence Guardrail

The system applies a confidence validation step to reduce incorrect extractions.

confidence ≥ 0.90  → Accepted
confidence < 0.90  → Manual Review Required

Example low-confidence response:

{
"net_weight": "12.00 lb",
"confidence": 0.85,
"status": "rejected",
"message": "Confidence below 0.90 — manual review required"
}

Supported Image Formats
The system supports common label image formats including:
JPG
JPEG
PNG
TIFF

# ADK Weight Extraction Agent

This project is a Proof-of-Concept (POC) for extracting **Net Weight** from warehouse label images using an AI agent.

The system uses:

- Google ADK (Agent Development Kit)
- Gemini Vision Model (gemini-2.5-flash)
- Python
- Flask API
- Real-ESRGAN (Image Enhancement GAN)

The agent reads a label image and returns structured JSON output containing the detected **net weight**, **confidence score**, and **status**.

---

### Architecture Overview

```
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
Conditional GAN Enhancement (Real-ESRGAN)
        ↓
Final Confidence Validation
        ↓
JSON Output
```

---

### Image Enhancement Pipeline

To improve label readability for unclear or blurry images, the system integrates **Real-ESRGAN**, a pretrained GAN-based super-resolution model.

Pipeline logic:

1. The original image is processed by the **ADK Agent + Gemini Vision model**
2. The agent extracts:
   - `net_weight`
   - `confidence`
3. A confidence validation step evaluates the result
4. If confidence is **above the threshold**, the result is returned
5. If confidence is **below the threshold**, the image is enhanced using **Real-ESRGAN**
6. The enhanced image is processed again by the agent
7. The system compares **original vs enhanced results** and returns the best output

This ensures enhancement runs **only when necessary** and avoids over-processing clear images.

---

### Setup Instructions

#### 1. Clone the repository

```bash
git clone https://github.com/GSruthi1/ADK-Weight-Extractor.git
cd ADK-Weight-Extractor
```

#### 2. Create virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### Configure Gemini API Key

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

### Run the API

Start the Flask server:

```bash
python app.py
```

The API will start at:

```
http://127.0.0.1:5000
```

---

### Test the API

#### Test using image path

```bash
curl -X POST http://127.0.0.1:5000/extract-weight \
-H "Content-Type: application/json" \
-d '{"image_path": "sample_images/Box_1.jpg"}'
```

Example response:

```json
{
  "original": {
    "net_weight": "25.00 lb",
    "confidence": 0.95
  },
  "enhanced": null,
  "status": "accepted"
}
```

---

#### Example response when enhancement is triggered

```json
{
  "original": {
    "net_weight": "12.00 lb",
    "confidence": 0.55
  },
  "enhanced": {
    "net_weight": "12.00 lb",
    "confidence": 0.82
  },
  "status": "accepted"
}
```

---

#### Test using image upload

```bash
curl -X POST http://127.0.0.1:5000/extract-weight \
-F "file=@sample_images/Box_1.jpg"
```

---

### Confidence Guardrail

The system validates results using a confidence threshold.

```
confidence ≥ 0.90 → Accepted  
confidence < 0.90 → Manual Review Required
```

Example response when confidence is low:

```json
{
  "net_weight": "12.00 lb",
  "confidence": 0.70,
  "status": "rejected",
  "message": "Confidence below 0.90 — manual review required"
}
```

---

### Supported Image Formats

- JPG
- JPEG
- PNG
- TIFF

---

### Real-ESRGAN Reference

Real-ESRGAN is a pretrained GAN model used for image super-resolution.

Official repository:

https://github.com/xinntao/Real-ESRGAN

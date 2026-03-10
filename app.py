from enhance_image import enhance_image
import asyncio
import json
import re
import mimetypes
import io
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PIL import Image

from google.genai.types import Content, Part
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService

from weight_agent.agent import root_agent

load_dotenv()

app = Flask(__name__)

APP_NAME = "adk_weight_service"
USER_ID = "local_user"

WEIGHT_RE = re.compile(r"\b(\d{2}\.\d{2})\s*lb\b", re.IGNORECASE)

session_service = InMemorySessionService()
runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=session_service)


async def run_agent_with_image(image_path: str, image_bytes: bytes):
    session_id = f"s_{asyncio.get_running_loop().time()}"
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    mime_type = mimetypes.guess_type(image_path)[0]

    if mime_type == "image/tiff":
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        image_bytes = buffer.getvalue()
        mime_type = "image/jpeg"

    if not mime_type:
        mime_type = "image/jpeg"

    user_content = Content(
        role="user",
        parts=[
            Part(text="Extract the NET WEIGHT from this label image and return JSON."),
            Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ),
        ],
    )

    final_text = None
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session_id,
        new_message=user_content,
    ):
        if getattr(event, "content", None):
            for part in getattr(event.content, "parts", []):
                if getattr(part, "text", None):
                    final_text = part.text

    return final_text


def safe_parse_agent_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except Exception:
            return None


def compute_regex_confidence(net_weight: str) -> float:
    if not net_weight:
        return 0.0
    if WEIGHT_RE.search(net_weight):
        return 0.95
    return 0.60


@app.route("/extract-weight", methods=["POST"])
def extract_weight():
    try:
        # -------- INPUT IMAGE --------
        if request.is_json:
            image_path = request.json.get("image_path")
            if not image_path:
                return jsonify({"error": "image_path is required"}), 400

            image_path = os.path.abspath(image_path)
            print("IMAGE PATH:", image_path)

            with open(image_path, "rb") as f:
                original_bytes = f.read()

        else:
            if "file" not in request.files:
                return jsonify({"error": "Upload file using form-data key 'file'"}), 400

            file = request.files["file"]
            original_bytes = file.read()
            image_path = file.filename

            with open(image_path, "wb") as f:
                f.write(original_bytes)

        # -------- RUN OCR ON ORIGINAL IMAGE --------
        original_text = asyncio.run(
            run_agent_with_image(image_path, original_bytes)
        )

        if not original_text:
            return jsonify({"error": "No response from agent (original image)"}), 500

        original_parsed = safe_parse_agent_json(original_text)

        if not original_parsed:
            return jsonify({
                "error": "Agent returned invalid JSON (original image)",
                "raw": original_text
            }), 500

        original_weight = original_parsed.get("net_weight")
        original_model_conf = float(original_parsed.get("confidence", 0.0))
        original_regex_conf = compute_regex_confidence(original_weight)
        original_final_conf = min(original_model_conf, original_regex_conf)

        # -------- DECIDE IF ESRGAN IS NEEDED --------

        ENHANCE_THRESHOLD = 0.75

        if original_final_conf >= ENHANCE_THRESHOLD:

            response = {
                "original": {
                    "net_weight": original_weight,
                    "confidence": round(original_final_conf, 2)
                },
                "enhanced": None,
                "status": "accepted",
                "message": "Image clear — enhancement skipped."
            }

            return jsonify(response)

        # -------- RUN ESRGAN ENHANCEMENT --------
        base, ext = os.path.splitext(image_path)
        enhanced_path = f"{base}_enhanced{ext}"

        print("Low confidence detected — running ESRGAN enhancement")

        enhance_image(image_path, enhanced_path)

        with open(enhanced_path, "rb") as f:
            enhanced_bytes = f.read()

        # -------- RUN ENHANCED IMAGE --------
        enhanced_text = asyncio.run(
            run_agent_with_image(enhanced_path, enhanced_bytes)
        )

        if not enhanced_text:
            return jsonify({"error": "No response from agent (enhanced image)"}), 500

        enhanced_parsed = safe_parse_agent_json(enhanced_text)

        if not enhanced_parsed:
            return jsonify({
                "error": "Agent returned invalid JSON (enhanced image)",
                "raw": enhanced_text
            }), 500

        enhanced_weight = enhanced_parsed.get("net_weight")
        enhanced_model_conf = float(enhanced_parsed.get("confidence", 0.0))
        enhanced_regex_conf = compute_regex_confidence(enhanced_weight)
        enhanced_final_conf = min(enhanced_model_conf, enhanced_regex_conf)

        # -------- COMPARISON RESULT --------
        response = {
            "original": {
                "net_weight": original_weight,
                "confidence": round(original_final_conf, 2)
            },
            "enhanced": {
                "net_weight": enhanced_weight,
                "confidence": round(enhanced_final_conf, 2)
            }
        }
 
        best = max(original_final_conf, enhanced_final_conf)

        if best < 0.90:
            response["status"] = "rejected"
            response["message"] = "Confidence below 0.90 — manual review required."
            return jsonify(response), 422

        response["status"] = "accepted"
        return jsonify(response)

    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(app.url_map)
    app.run(port=5000, debug=True, use_reloader=False)
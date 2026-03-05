import asyncio
import json
import re
import mimetypes
import io
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
        if request.is_json:
            image_path = request.json.get("image_path")
            if not image_path:
                return jsonify({"error": "image_path is required"}), 400
            with open(image_path, "rb") as f:
                image_bytes = f.read()

        else:
            if "file" not in request.files:
                return jsonify({"error": "Upload file using form-data key 'file'"}), 400
            image_bytes = request.files["file"].read()
            image_path = request.files["file"].filename

        agent_text = asyncio.run(run_agent_with_image(image_path, image_bytes))
        if not agent_text:
            return jsonify({"error": "No response from agent"}), 500

        parsed = safe_parse_agent_json(agent_text)
        if not parsed:
            return jsonify({"error": "Agent returned invalid JSON", "raw": agent_text}), 500

        net_weight = parsed.get("net_weight")
        model_conf = float(parsed.get("confidence", 0.0))
        reason = parsed.get("reason", "")

        regex_conf = compute_regex_confidence(net_weight)
        final_conf = min(model_conf, regex_conf)

        result = {
            "net_weight": net_weight,
            "confidence": round(final_conf, 2),
            "reason": reason,
        }

        if final_conf < 0.90:
            return jsonify({
                **result,
                "status": "rejected",
                "message": "Confidence below 0.90 — manual review required."
            }), 422

        return jsonify({**result, "status": "accepted"})

    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)
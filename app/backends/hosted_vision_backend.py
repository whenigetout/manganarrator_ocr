"""Hosted vision inference, keeping the existing InferImageResponse contract.

Set VISION_PROVIDER=openai or gemini and configure the corresponding API key.
No model is loaded locally for hosted inference.
"""
import base64
import os
from pathlib import Path
from PIL import Image
import httpx
from app.models.domain import InferImageResponse

PROMPT = """Read the manga/manhwa image. Extract dialogue in reading order.
Output each line as [SPEAKER | GENDER | EMOTION]: "TEXT".
SPEAKER is Speaker 1, Speaker 2, Narrator, etc.
GENDER is male, female, or unknown.
EMOTION is neutral, happy, sad, angry, excited, nervous, scared, curious, playful, serious, or calm.
Do not invent unreadable dialogue. If none, output exactly NO_DIALOGUE.
Output only lines or NO_DIALOGUE."""

class HostedVisionBackend:
    def __init__(self, config):
        self.provider = os.getenv("VISION_PROVIDER", "openai").lower()
        if self.provider not in ("openai", "gemini"):
            raise ValueError("VISION_PROVIDER must be openai or gemini")
        self.model = os.getenv("VISION_MODEL", "gpt-4.1-mini" if self.provider == "openai" else "gemini-2.5-flash")
        self.api_key = os.getenv("OPENAI_API_KEY" if self.provider == "openai" else "GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing API key for configured vision provider")
        self.media_root = Path(config["media_root"]).resolve()
        self.prompt = config.get("prompt") or PROMPT

    def infer_image(self, img, prompt=None):
        path = Path(img.resolve(self.media_root)).resolve()
        if not path.is_relative_to(self.media_root):
            raise ValueError("Image path is outside media root")
        with Image.open(path) as im:
            width, height = im.size
            import io
            buf = io.BytesIO()
            im.convert("RGB").save(buf, format="JPEG", quality=90)
        encoded = base64.b64encode(buf.getvalue()).decode("ascii")
        instruction = prompt or self.prompt
        with httpx.Client(timeout=120) as client:
            if self.provider == "openai":
                resp = client.post("https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": [{"role": "user", "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}]}],
                        "max_completion_tokens": 1200})
                resp.raise_for_status()
                data = resp.json()
                output = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {})
                in_tokens, out_tokens = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
            else:
                resp = client.post(f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                    headers={"x-goog-api-key": self.api_key},
                    json={"contents": [{"parts": [{"text": instruction},
                        {"inline_data": {"mime_type": "image/jpeg", "data": encoded}}]}],
                        "generationConfig": {"maxOutputTokens": 1200}})
                resp.raise_for_status()
                data = resp.json()
                output = "".join(part.get("text", "") for c in data.get("candidates", [])
                                 for part in c.get("content", {}).get("parts", []))
                usage = data.get("usageMetadata", {})
                in_tokens, out_tokens = usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)
        return InferImageResponse(image_ref=img, image_text=output,
            image_width=width, image_height=height, input_tokens=in_tokens,
            output_tokens=out_tokens, throughput=0.0)

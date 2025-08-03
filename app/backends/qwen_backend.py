import torch
import time
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from typing import List, Dict, Optional, Union

from app.utils import Timer
from qwen_vl_utils import process_vision_info


class QwenOCRBackend:
    def __init__(self, config: dict):
        if "qwen_model_id" not in config:
            raise ValueError("Missing required config key: 'qwen_model_id'")
        self.model_id = config["qwen_model_id"]

        self.use_flash_attn = config.get("use_flash_attn", True)
        self.prompt = config.get("prompt", self.default_prompt())

        self._load_model()

    def _load_model(self):
        print(f"\n📦 Loading Qwen2.5-VL model: {self.model_id}")
        with Timer("🔧 Load processor"):
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=True,
                USE_FLASH_ATTN=self.use_flash_attn
            )

        with Timer("⚙️ Load model"):
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.use_flash_attn else torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )

    def infer_image(self, img_path: Path, prompt: Optional[str] = None) -> str:
        image = Image.open(img_path).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt if prompt is None else prompt}
            ]
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        input_tokens = inputs["input_ids"].shape[-1]

        with Timer("🚀 Inference", use_spinner=False):
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512)

        trimmed_ids = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)]
        output_tokens = trimmed_ids[0].shape[-1]
        outputs = self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)

        return {
            "text": outputs[0],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "throughput": round((input_tokens + output_tokens) / (Timer.last_duration or 1e-5), 2)
        }

    def default_prompt(self):
        return '''The following is a manhwa panel.
Extract the dialogue lines and output them in the following format:
[SPEAKER | GENDER | EMOTION]: "TEXT"
SPEAKER = "Speaker 1", "Speaker 2", "Narrator", etc.
GENDER = "male", "female", or "unknown"
EMOTION = Pick from: neutral, happy, sad, angry, excited, nervous, aroused, scared, curious, playful, serious, calm

Preserve the original order. Output only the formatted lines.
'''

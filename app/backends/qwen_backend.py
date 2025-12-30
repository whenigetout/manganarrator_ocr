import torch
import time
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from typing import List, Dict, Optional, Union
from app.models.domain import MediaRef

from app.utils import Timer
from qwen_vl_utils import process_vision_info
import os, importlib.util, torch

def _flash_attn_available() -> bool:
    try:
        # return importlib.util.find_spec("flash_attn") is not None
        import flash_attn
        return True
    except Exception as e:
        print("❌ flash_attn import failed:", repr(e))
        return False

def _pick_attn_impl(use_flash_flag: bool) -> str:
    """Return transformers attn_implementation."""
    if use_flash_flag and _flash_attn_available() and torch.cuda.is_available():
        return "flash_attention_2"     # use FlashAttention v2
    # fallback: SDPA on CUDA, eager everywhere else
    return "sdpa" if torch.cuda.is_available() else "eager"

class QwenOCRBackend:
    def __init__(self, config: dict):
        if "qwen_model_id" not in config:
            raise ValueError("Missing required config key: 'qwen_model_id'")
        self.model_id = config["qwen_model_id"]

        self.use_flash_attn = config.get("use_flash_attn", True)
        self.prompt = config.get("prompt", self.default_prompt())

        try:
            self.custom_cache_dir = None
            path = Path(config["custom_cache_dir"]).resolve()
            self.custom_cache_dir = str(path)
        except:
            print("❌ Failed to load custom_cache_dir, proceeding with default cache")

        self._load_model()

    def _load_model(self):
        print(f"\n📦 Loading Qwen2.5-VL model: {self.model_id}")
        use_flash_flag = bool(self.use_flash_attn)
        has_cuda = torch.cuda.is_available()
        flash_ok = use_flash_flag and has_cuda and _flash_attn_available()


        with Timer("🔧 Load processor"):
            self.processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                use_fast=True,
                USE_FLASH_ATTN=flash_ok
            )

        with Timer("⚙️ Load model"):
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                llm_int8_enable_fp32_cpu_offload=True
            )

            if flash_ok:
                # === FLASH PATH: keep your existing behavior here (unchanged) ===
                # If your original code had any extra kwargs, keep them.
                print(f"Flash attn available, loading with flash attn.")
                if self.custom_cache_dir:
                    print(f"Loading model from custom cache directory...")
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                                self.model_id,
                                cache_dir=self.custom_cache_dir,
                                trust_remote_code=True,
                                quantization_config=self.bnb_config,
                                attn_implementation="flash_attention_2",
                                device_map="cuda:0"
                            )
            else:
                # === NON-FLASH PATH: force a non-flash backend so config is respected ===
                print("Flash attn NOT AVAILABLE")
                non_flash_impl = "sdpa" if has_cuda else "eager"
                try:
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16 if flash_ok else torch.float16,
                        attn_implementation=non_flash_impl,
                        device_map="auto"
                    )
                except TypeError:
                    pass

    def infer_image(self, img_path: Path, prompt: Optional[str] = None) -> dict:
        image = Image.open(img_path).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": self.prompt if prompt is None else prompt}
            ]
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, discard_this = process_vision_info(messages)

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

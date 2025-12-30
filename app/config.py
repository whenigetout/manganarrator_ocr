import yaml
from pathlib import Path

def load_config(input_config_path: str) -> dict:
    config_path = Path(input_config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Basic validation
    required_keys = ["input_root_folder", "output_root_folder", "ocr_model", "qwen_model_id"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")

    return config

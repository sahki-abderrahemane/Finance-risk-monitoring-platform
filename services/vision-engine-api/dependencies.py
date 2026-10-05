"""Vision model loading and lifecycle management."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = PROJECT_ROOT / "ml" / "models" / "cnn" / "chart_cnn_weighted_best.pt"

LABEL_MAP = {0: "DOWN", 1: "FLAT", 2: "UP"}

cnn_model = None
transform = None
model_loaded = False
device = torch.device("cpu")


def load_models() -> None:
    global cnn_model, transform, model_loaded, device

    if not CHECKPOINT_PATH.exists():
        return

    from vision_engine.model import ChartCNN, CNNConfig

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    config_data = checkpoint.get("model_config", {})
    config = CNNConfig(**config_data)

    cnn_model = ChartCNN(config).to(device)
    cnn_model.load_state_dict(checkpoint["model_state_dict"])
    cnn_model.eval()

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model_loaded = True


def decode_image(image_base64: str) -> Image.Image:
    image_bytes = base64.b64decode(image_base64)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def classify_image(image: Image.Image) -> tuple[str, float, dict[str, float]]:
    if cnn_model is None or transform is None:
        raise RuntimeError("Vision model not loaded")

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = cnn_model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze()

    predicted_idx = int(torch.argmax(probs).item())
    confidence = float(probs[predicted_idx].item())

    prob_dict = {
        "DOWN": round(float(probs[0].item()), 4),
        "FLAT": round(float(probs[1].item()), 4),
        "UP": round(float(probs[2].item()), 4),
    }

    return LABEL_MAP[predicted_idx], confidence, prob_dict

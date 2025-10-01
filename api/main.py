import io
import os

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from torchvision import transforms

from app.cnn_assignment import AssignmentCNN


app = FastAPI(title="CNN CIFAR-10 Inference API")

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

_preprocess = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2023, 0.1994, 0.2010),
    ),
])

_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_weights_path = os.getenv("WEIGHTS_PATH", "artifacts/cnn_cifar10.pt")


def load_model():
    model = AssignmentCNN(num_classes=10).to(_device)
    if not os.path.exists(_weights_path):
        raise FileNotFoundError(f"Model weights not found at '{_weights_path}'. Please run training first.")
    state_dict = torch.load(_weights_path, map_location=_device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@app.on_event("startup")
def _startup():
    global _model
    _model = load_model()


def _prepare_image(file_bytes: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    tensor = _preprocess(image).unsqueeze(0)
    return tensor.to(_device)


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=415, detail="Unsupported file type. Use PNG or JPEG.")

    file_bytes = file.file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    with torch.inference_mode():
        inputs = _prepare_image(file_bytes)
        logits = _model(inputs)
        probs = F.softmax(logits, dim=1).squeeze(0)
        prob_value, class_idx = torch.max(probs, dim=0)
        class_idx = int(class_idx.item())
        prob_value = float(prob_value.item())
        class_name = CIFAR10_CLASSES[class_idx]

    return {
        "class_index": class_idx,
        "class_name": class_name,
        "probability": prob_value,
    }


@app.get("/health")
def health():
    return {"status": "ok"}



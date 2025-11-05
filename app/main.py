import os
import torch
import torch.nn as nn
import torchvision.transforms
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel

app = FastAPI()

# ---- LSTM text generation ----
_DEFAULT_CORPUS = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective",
]


class LSTMModel(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=128):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden


def _build_vocab_from_corpus(corpus):
    # Simple whitespace tokenization; lowercase
    special_tokens = ["<PAD>", "<UNK>"]
    words = set()
    for line in corpus:
        for w in line.lower().split():
            words.add(w)
    # Reserve indices: 0 for <PAD>, 1 for <UNK>
    vocab = {tok: idx for idx, tok in enumerate(special_tokens)}
    next_idx = len(vocab)
    for w in sorted(words):
        if w not in vocab:
            vocab[w] = next_idx
            next_idx += 1
    inv_vocab = {v: k for k, v in vocab.items()}
    return vocab, inv_vocab


# Build vocabulary from default corpus
vocab, inv_vocab = _build_vocab_from_corpus(_DEFAULT_CORPUS)
vocab_size = len(vocab)

# Initialize LSTM model (optionally load weights if available)
model = LSTMModel(vocab_size=vocab_size)


def generate_text(model, seed_text, length=50, temperature=1.0):
    model.eval()
    words = seed_text.lower().split()
    # Ensure at least one token exists
    if not words:
        words = ["<UNK>"]
    input_ids = [vocab.get(w, vocab["<UNK>"]) for w in words]
    input_tensor = torch.tensor(input_ids).unsqueeze(0)
    hidden = None

    with torch.no_grad():
        for _ in range(length):
            output, hidden = model(input_tensor, hidden)
            logits = output[0, -1] / max(temperature, 1e-6)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            words.append(inv_vocab.get(next_id, "<UNK>"))
            input_ids.append(next_id)
            input_tensor = torch.tensor(input_ids).unsqueeze(0)

    return " ".join(words)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int


@app.get("/")
def read_root():
    return {"status": "ok"}


@app.post("/generate")
def generate_text_endpoint(request: TextGenerationRequest):
    text = generate_text(model, request.start_word, request.length)
    return {"generated_text": text}


@app.post("/generate_with_rnn")
def generate_with_rnn(request: TextGenerationRequest):
    generated = generate_text(model, request.start_word, request.length)
    return {"generated_text": generated}


# ========== Module 9: LLM Fine-tuning Endpoint ==========
@app.post("/generate_with_llm")
def generate_with_llm_endpoint(request: TextGenerationRequest):
    """Generate text using fine-tuned GPT-2 model (Module 9).
    
    This endpoint uses the fine-tuned GPT-2 model trained on the Nectar Q&A dataset.
    The model is capable of generating text based on prompts in a conversational format.
    
    Args:
        request: Contains start_word (prompt) and length (max generation length)
        
    Returns:
        Dictionary with generated text
    """
    try:
        from helper_lib.llm_generator import generate_text_with_llm
        
        # Load fine-tuned model and generate text
        generated_text = generate_text_with_llm(
            prompt=request.start_word,
            model_path='models/gpt2_finetuned/final_model',
            max_length=request.length,
            device=gan_device.type  # Reuse the device variable
        )
        
        return {
            "generated_text": generated_text,
            "prompt": request.start_word,
            "length": request.length
        }
    except Exception as e:
        return {
            "error": f"Failed to generate text with LLM: {str(e)}",
            "prompt": request.start_word,
            "note": "Make sure the model is trained first by running: python train_llm.py"
        }


# ---- Embedding endpoint (spaCy) ----
import spacy


def _load_spacy_model():
    model_name = os.getenv("SPACY_MODEL", "en_core_web_md")
    try:
        return spacy.load(model_name)
    except Exception:
        # Safe fallback to a blank English pipeline (no vectors)
        return spacy.blank("en")


nlp = _load_spacy_model()


class EmbedRequest(BaseModel):
    text: str
    pooling: str | None = "mean"  # "mean" or "tokens"


@app.post("/embed")
def embed(req: EmbedRequest):
    doc = nlp(req.text)
    if req.pooling == "tokens":
        vectors = [t.vector.tolist() if t.has_vector else [] for t in doc]
        dim = int(len(doc[0].vector)) if len(doc) and doc[0].has_vector else 0
        return {"tokens": [t.text for t in doc], "vectors": vectors, "dim": dim}

    # Default: sentence-level mean pooling
    vecs = [t.vector for t in doc if t.has_vector]
    if not vecs:
        return {"vector": [], "dim": 0}
    import numpy as np
    mean_vec = np.mean(vecs, axis=0)
    return {"vector": mean_vec.tolist(), "dim": int(mean_vec.shape[0])}


# ---- GAN Image Generation ----
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from helper_lib.model import GANGenerator, SpecifiedCNN
from helper_lib.energy_model import EnergyModel
from helper_lib.model import get_model, DiffusionModel
from helper_lib.generator import generate_energy_samples, generate_diffusion_samples
import base64
from io import BytesIO
from PIL import Image

# Initialize GAN Generator
gan_generator = None
gan_device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

def load_gan_generator():
    """Load pre-trained GAN generator model."""
    global gan_generator
    if gan_generator is None:
        gan_generator = GANGenerator(z_dim=100).to(gan_device)
        # Try to load pre-trained weights if available
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gan_generator.pth')
        if os.path.exists(model_path):
            gan_generator.load_state_dict(torch.load(model_path, map_location=gan_device))
            print(f"Loaded GAN generator from {model_path}")
        else:
            print("Warning: No pre-trained GAN model found. Using untrained model.")
        gan_generator.eval()
    return gan_generator


class ImageGenerationRequest(BaseModel):
    num_images: int = 1
    seed: int | None = None


@app.post("/generate_image")
def generate_image(request: ImageGenerationRequest):
    """Generate handwritten digit images using trained GAN model.
    
    Args:
        num_images: Number of images to generate (default: 1)
        seed: Random seed for reproducibility (optional)
    
    Returns:
        Dictionary with base64-encoded images
    """
    generator = load_gan_generator()
    
    # Set seed if provided
    if request.seed is not None:
        torch.manual_seed(request.seed)
    
    # Generate images
    with torch.no_grad():
        noise = torch.randn(request.num_images, 100).to(gan_device)
        generated_images = generator(noise)
    
    # Convert to base64 encoded images
    images_base64 = []
    for i in range(request.num_images):
        # Get single image and convert from [-1, 1] to [0, 255]
        img_tensor = generated_images[i].cpu()
        img_tensor = (img_tensor + 1) / 2.0  # [-1, 1] -> [0, 1]
        img_tensor = torch.clamp(img_tensor, 0, 1)
        
        # Convert to PIL Image
        img_array = (img_tensor.squeeze().numpy() * 255).astype('uint8')
        img = Image.fromarray(img_array, mode='L')
        
        # Convert to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        images_base64.append(img_base64)
    
    return {
        "num_images": request.num_images,
        "images": images_base64,
        "format": "base64_png"
    }


# ---- CNN Image Classification ----
# Initialize CNN Classifier
cnn_classifier = None
cnn_device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

def load_cnn_classifier():
    """Load pre-trained CNN classifier model."""
    global cnn_classifier
    if cnn_classifier is None:
        cnn_classifier = SpecifiedCNN(num_classes=10).to(cnn_device)
        # Try to load pre-trained weights if available
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'specified_cnn_cifar10.pth')
        if os.path.exists(model_path):
            cnn_classifier.load_state_dict(torch.load(model_path, map_location=cnn_device))
            print(f"Loaded CNN classifier from {model_path}")
        else:
            print("Warning: No pre-trained CNN model found. Using untrained model.")
        cnn_classifier.eval()
    return cnn_classifier

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

class ImageClassificationRequest(BaseModel):
    image: str  # Base64 encoded image
    top_k: int = 5  # Number of top predictions to return

@app.post("/classify_image")
def classify_image(request: ImageClassificationRequest):
    """Classify an image using the trained CNN model.
    
    Args:
        image: Base64 encoded image
        top_k: Number of top predictions to return
    
    Returns:
        Dictionary with predictions and confidence scores
    """
    classifier = load_cnn_classifier()
    
    try:
        # Decode base64 image
        image_data = base64.b64decode(request.image)
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 64x64 (SpecifiedCNN requirement)
        image = image.resize((64, 64))
        
        # Convert to tensor and normalize
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(cnn_device)
        
        # Get predictions
        with torch.no_grad():
            outputs = classifier(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, request.top_k)
        
        # Format results
        predictions = []
        for i in range(request.top_k):
            class_idx = top_indices[0][i].item()
            confidence = top_probs[0][i].item()
            predictions.append({
                "class": CIFAR10_CLASSES[class_idx],
                "confidence": confidence,
                "class_index": class_idx
            })
        
        return {
            "predictions": predictions,
            "top_k": request.top_k
        }
        
    except Exception as e:
        return {"error": f"Failed to classify image: {str(e)}"}


@app.get("/cnn_info")
def get_cnn_info():
    """Get information about the CNN model."""
    classifier = load_cnn_classifier()
    return {
        "model_name": "SpecifiedCNN",
        "architecture": "Conv2D(16) -> ReLU -> MaxPool2D -> Conv2D(32) -> ReLU -> MaxPool2D -> Linear(100) -> ReLU -> Linear(10)",
        "input_size": "64x64x3",
        "num_classes": 10,
        "classes": CIFAR10_CLASSES,
        "parameters": sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    }


# ---- Energy/Diffusion Model Endpoints (merged from api/app.py) ----
def _tensor_grid_to_base64(images: torch.Tensor, nrow: int = 4) -> str:
    from torchvision.utils import make_grid
    import torchvision.transforms as T
    grid = make_grid(images.clamp(0.0, 1.0).cpu(), nrow=nrow, padding=2)
    to_pil = T.ToPILImage(mode='RGB') if grid.shape[0] == 3 else T.ToPILImage()
    pil_img = to_pil(grid)
    bio = BytesIO()
    pil_img.save(bio, format='PNG')
    bio.seek(0)
    return base64.b64encode(bio.read()).decode('utf-8')


class GenerateRequestED(BaseModel):
    num_samples: int = 16
    seed: int | None = None


class _EDModelManager:
    def __init__(self):
        self.device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        self.models = {}

        # EnergyModel
        energy_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'energy', 'energy_final.pt')
        energy_model = EnergyModel().to(self.device)
        if os.path.exists(energy_path):
            energy_model.load_state_dict(torch.load(energy_path, map_location=self.device))
        energy_model.eval()
        self.models['energy'] = energy_model

        # DiffusionModel (prefer EMA if available)
        diffusion_model: DiffusionModel = get_model('diffusion', device=self.device, image_size=64, num_channels=3)
        diffusion_ckpt = os.path.join(os.path.dirname(__file__), '..', 'models', 'diffusion', 'diffusion_final.pth')
        if os.path.exists(diffusion_ckpt):
            ckpt = torch.load(diffusion_ckpt, map_location=self.device)
            if 'ema_model_state_dict' in ckpt:
                diffusion_model.ema_network.load_state_dict(ckpt['ema_model_state_dict'])
            if 'model_state_dict' in ckpt:
                diffusion_model.network.load_state_dict(ckpt['model_state_dict'])
        diffusion_model.eval()
        self.models['diffusion'] = diffusion_model

    def list_models(self):
        return sorted(self.models.keys())

    def _seed(self, seed: int | None):
        if seed is None:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        elif torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    def generate(self, model_type: str, num_samples: int, seed: int | None):
        mtype = (model_type or '').lower()
        if mtype not in self.models:
            raise KeyError(f"Unknown model_type: {model_type}")
        self._seed(seed)
        if mtype == 'energy':
            return generate_energy_samples(self.models['energy'], self.device, num_samples=num_samples)
        if mtype == 'diffusion':
            return generate_diffusion_samples(self.models['diffusion'], self.device, num_samples=num_samples)
        raise KeyError(f"Unsupported model_type: {model_type}")


_ed_manager = _EDModelManager()


@app.get("/health")
def health_ed():
    return {"status": "ok"}


@app.get("/models")
def list_models_ed():
    return {"models": _ed_manager.list_models()}


@app.post("/generate/{model_type}")
def generate_ed(model_type: str = Path(..., description="energy or diffusion"), body: GenerateRequestED | None = None):
    body = body or GenerateRequestED()
    try:
        images = _ed_manager.generate(model_type=model_type, num_samples=body.num_samples, seed=body.seed)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")
    b64 = _tensor_grid_to_base64(images, nrow=max(1, int(body.num_samples ** 0.5)))
    return {"image_base64": b64, "num_samples": body.num_samples, "model_type": model_type}

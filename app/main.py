import os
import torch
import torch.nn as nn
import torchvision.transforms
from fastapi import FastAPI
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
import base64
from io import BytesIO
from PIL import Image

# Initialize GAN Generator
gan_generator = None
gan_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
cnn_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

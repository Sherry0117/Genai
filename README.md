# SPS GenAI - Neural Network Projects & Text Generation

A comprehensive project combining FastAPI text generation/embeddings with a neural network helper library. This project demonstrates both natural language processing capabilities and deep learning workflows for computer vision tasks.

## Features

### FastAPI Application
- **Text Generation**: Generate text using a bigram language model trained on custom corpora
- **Text Embeddings**: Get word or sentence-level embeddings using spaCy's pre-trained models
- **RESTful API**: Clean FastAPI endpoints with automatic OpenAPI documentation
- **Docker Support**: Containerized deployment with optimized Python 3.11 image

### Neural Network Helper Library
- **Data Loading**: Support for CIFAR-10/100, MNIST, FashionMNIST with automatic preprocessing
- **Model Architectures**: FCNN, CNN, EnhancedCNN, VAE, and GAN implementations
- **Training Utilities**: Complete training loops with progress tracking and evaluation
- **Advanced Models**: Variational Autoencoders (VAE) and Wasserstein GAN implementations

## API Endpoints

### Text Generation
- `POST /generate` - Generate text using bigram model
  - **Request Body**: `{"start_word": "the", "length": 10}`
  - **Response**: `{"generated_text": "the count of monte cristo is a novel written by alexandre dumas"}`

### Text Embeddings
- `POST /embed` - Get text embeddings using spaCy
  - **Request Body**: `{"text": "This is an example.", "pooling": "mean"}` (pooling: "mean" or "tokens")
  - **Response**: `{"vector": [0.1, 0.2, ...], "dim": 300}` or `{"tokens": [...], "vectors": [...], "dim": 300}`

### GAN Image Generation (Assignment 3)
- `POST /generate_image` - Generate handwritten digit images using trained GAN model
  - **Request Body**: `{"num_images": 1, "seed": 42}`
  - **Response**: `{"num_images": 1, "images": ["base64_encoded_png..."], "format": "base64_png"}`
  - **Note**: Requires trained GAN model at `models/gan_generator.pth`

### CNN Image Classification
- `POST /classify_image` - Classify images using trained CNN model
  - **Request Body**: `{"image": "base64_encoded_image", "top_k": 5}`
  - **Response**: `{"predictions": [{"class": "cat", "confidence": 0.95, ...}], "top_k": 5}`

### Health Check
- `GET /` - Basic health check endpoint

## Project Structure

```
sps_genai/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application and endpoints
│   ├── bigram_model.py      # Bigram language model implementation
│   └── cnn_assignment.py    # CNN model for assignments
├── helper_lib/              # Neural network helper library
│   ├── __init__.py
│   ├── data_loader.py       # Data loading utilities
│   ├── trainer.py           # Training functions (including VAE, WGAN, GAN)
│   ├── evaluator.py         # Model evaluation utilities
│   ├── model.py             # Neural network model definitions
│   ├── utils.py             # Utility functions
│   └── README.md            # Helper library documentation
├── api/                     # Additional API endpoints
│   └── main.py
├── models/                  # Trained model weights and outputs
│   ├── gan_generator.pth    # Trained GAN generator
│   ├── generated_samples_epoch_20.png
│   └── training_losses.png
├── data/                    # Dataset storage
│   └── MNIST/
├── example_usage.py         # Example usage of helper library
├── train.py                 # Training script for CNN assignments
├── train_gan.py            # GAN training script
├── simple_train_gan.py      # Simplified GAN training script
├── test_gan.py             # GAN testing script
├── Dockerfile               # Container configuration
├── pyproject.toml          # Project dependencies and metadata
├── requirements.txt        # Python package requirements
└── README.md              # This file
```

## Installation & Setup

### Prerequisites
- Python 3.11 (required for PyTorch compatibility)
- uv (recommended) or pip

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd sps_genai
   ```

2. **Install dependencies**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   python -m spacy download en_core_web_md
   ```

3. **Run the application**
   ```bash
   # Using uv
   uv run fastapi dev app/main.py
   
   # Or using uvicorn directly
   uvicorn app.main:app --reload
   ```

4. **Access the API**
   - API Documentation: http://127.0.0.1:8000/docs
   - Alternative docs: http://127.0.0.1:8000/redoc
   - Health check: http://127.0.0.1:8000/

## Usage Examples

### Text Generation
```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"start_word": "the", "length": 10}'
```

### Text Embeddings (Sentence-level)
```bash
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an example sentence.", "pooling": "mean"}'
```

### Text Embeddings (Token-level)
```bash
curl -X POST http://127.0.0.1:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"text": "This is an example sentence.", "pooling": "tokens"}'
```

### GAN Image Generation
```bash
# Generate 1 image with random seed
curl -X POST http://127.0.0.1:8000/generate_image \
  -H "Content-Type: application/json" \
  -d '{"num_images": 1, "seed": 42}'

# Save generated image to file
python3 << 'EOF'
import requests
import base64
from PIL import Image
from io import BytesIO

response = requests.post(
    'http://127.0.0.1:8000/generate_image',
    json={"num_images": 1, "seed": 42}
)
data = response.json()
img_data = base64.b64decode(data['images'][0])
img = Image.open(BytesIO(img_data))
img.save('generated_digit.png')
print("Saved to generated_digit.png")
EOF
```

## Docker Deployment

### Build and Run
```bash
# Build the Docker image
docker build -t sps-genai:latest .

# Run the container
docker run --rm -p 8000:8000 sps-genai:latest
```

### Access the API
- API Documentation: http://localhost:8000/docs
- Health check: http://localhost:8000/

## Technical Details

### Bigram Model
The `BigramModel` class implements a simple but effective bigram language model:
- Uses frequency-based word transitions
- Includes fallback mechanisms to avoid getting stuck
- Supports custom training corpora
- Tokenizes text using regex (letters and apostrophes only)

### Embeddings
- Uses spaCy's `en_core_web_md` model (300-dimensional vectors)
- Supports both token-level and sentence-level (mean pooling) embeddings
- Handles out-of-vocabulary words gracefully

### Dependencies
- **FastAPI**: Modern, fast web framework for building APIs
- **spaCy**: Advanced NLP library for embeddings
- **PyTorch**: Deep learning framework for neural networks
- **Torchvision**: Computer vision utilities and datasets
- **Matplotlib**: Plotting and visualization
- **Pydantic**: Data validation and serialization
- **NumPy**: Numerical computations for vector operations
- **Uvicorn**: ASGI server for running the application

## Configuration

The application uses environment variables for configuration:
- `SPACY_MODEL`: spaCy model to use (default: "en_core_web_md")

## Neural Network Helper Library

The `helper_lib` provides a comprehensive toolkit for neural network projects:

### Available Models
- **FCNN**: Fully Connected Neural Network for image classification
- **CNN**: Convolutional Neural Network with basic architecture
- **EnhancedCNN**: CNN with Batch Normalization and advanced features
- **VAE**: Variational Autoencoder for generative modeling
- **WGAN**: Wasserstein GAN implementation
- **GANGenerator & GANDiscriminator**: GAN models for MNIST digit generation

### Training Functions
- `train_model()`: Standard neural network training
- `train_vae_model()`: VAE-specific training with KL divergence
- `train_wgan()`: Wasserstein GAN training with gradient penalty
- `train_gan()`: GAN training for MNIST digit generation

### Training the GAN Model (Assignment 3)
To train your own GAN model on MNIST:

```bash
# Train GAN model (saves to models/gan_generator.pth)
python train_gan.py

# Or use the simplified version
python simple_train_gan.py
```

The training will:
- Download MNIST dataset automatically
- Train for 20 epochs (configurable)
- Save trained generator to `models/gan_generator.pth`
- Generate sample images and loss curves in `models/` directory

### Usage Example
```python
from helper_lib import get_data_loader, get_model, train_model, evaluate_model

# Load data
train_loader = get_data_loader('data', batch_size=32, train=True)
test_loader = get_data_loader('data', batch_size=32, train=False)

# Create and train model
model = get_model('CNN', num_classes=10)
trained_model = train_model(model, train_loader, criterion, optimizer, device='cuda', epochs=10)

# Evaluate
test_loss, test_accuracy = evaluate_model(trained_model, test_loader, criterion, device='cuda')
```

## Development

### Code Structure
- `app/main.py`: FastAPI application with text generation and embedding endpoints
- `app/bigram_model.py`: Bigram language model implementation
- `helper_lib/`: Complete neural network helper library
- `train.py`: Training script for CNN assignments
- `example_usage.py`: Example usage of the helper library

### Adding New Features
1. **FastAPI**: Add new endpoints in `app/main.py`
2. **Neural Networks**: Add new models in `helper_lib/model.py`
3. **Training**: Add new training functions in `helper_lib/trainer.py`
4. Update this README with new functionality


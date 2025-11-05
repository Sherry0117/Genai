# SPS GenAI - Neural Network Projects & Text Generation

A comprehensive project combining FastAPI text generation/embeddings with a neural network helper library. This project demonstrates both natural language processing capabilities and deep learning workflows for computer vision tasks, including Energy-Based Models and Diffusion Models for CIFAR-10.

## Features

### FastAPI Application
- **Text Generation**: Generate text using a bigram language model trained on custom corpora
- **Text Embeddings**: Get word or sentence-level embeddings using spaCy's pre-trained models
- **RESTful API**: Clean FastAPI endpoints with automatic OpenAPI documentation
- **Docker Support**: Containerized deployment with optimized Python 3.11 image

### Neural Network Helper Library
- **Data Loading**: Support for CIFAR-10/100, MNIST, FashionMNIST with automatic preprocessing
- **Model Architectures**: FCNN, CNN, EnhancedCNN, VAE, GAN, Energy-Based Models, and Diffusion Models
- **Training Utilities**: Complete training loops with progress tracking and evaluation
- **Advanced Models**: Variational Autoencoders (VAE), Wasserstein GAN, Energy-Based Models, and Diffusion Models

### Energy-Based Model (EBM)
- **Energy Model**: CNN-based energy function for CIFAR-10 image generation
- **Langevin Dynamics**: Stochastic sampling using gradient-based MCMC
- **Contrastive Divergence**: Training with real vs. fake energy gap

### Diffusion Model (DDPM)
- **Denoising Diffusion Probabilistic Model**: Standard DDPM implementation for CIFAR-10
- **Cosine Schedule**: DDPM cosine noise schedule for stable training
- **EMA Network**: Exponential moving average for better sample quality

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

### Energy Model & Diffusion Model Generation
- `POST /generate_energy` - Generate images using trained Energy Model
- `POST /generate_diffusion` - Generate images using trained Diffusion Model

### Health Check
- `GET /` - Basic health check endpoint

## Project Structure

```
Genai/
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
│   ├── model.py             # Neural network model definitions (including Diffusion)
│   ├── energy_model.py      # Energy-Based Model implementation
│   ├── utils.py             # Utility functions
│   └── README.md            # Helper library documentation
├── models/                  # Trained model weights and outputs
│   ├── gan_generator.pth    # Trained GAN generator
│   ├── energy/              # Energy Model checkpoints
│   └── diffusion/           # Diffusion Model checkpoints
├── samples/                 # Generated sample images (not tracked in git)
│   ├── energy/              # Energy Model samples
│   └── diffusion/           # Diffusion Model samples
├── examples/                 # Example generated samples (tracked in git)
│   └── diffusion_samples_epoch_15.png  # Best diffusion model samples
├── data/                    # Dataset storage
│   └── CIFAR10/
├── example_usage.py         # Example usage of helper library
├── train.py                 # Training script for CNN assignments
├── train_gan.py            # GAN training script
├── train_cifar10_models.py # Energy & Diffusion Model training script
├── simple_train_gan.py     # Simplified GAN training script
├── test_gan.py             # GAN testing script
├── Dockerfile              # Container configuration
├── pyproject.toml          # Project dependencies and metadata
├── requirements.txt        # Python package requirements
└── README.md              # This file
```

## Installation & Setup

### Prerequisites
- Python 3.11+ (required for PyTorch compatibility)
- PyTorch with MPS support (for Apple Silicon) or CUDA (for NVIDIA GPUs)
- uv (recommended) or pip

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/Sherry0117/Genai.git
   cd Genai
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

## Training Models

### Training Energy Model & Diffusion Model on CIFAR-10

The project includes a comprehensive training script for both Energy-Based Models and Diffusion Models:

```bash
# Train both models (quick mode: 2 epochs, 10% data)
python train_cifar10_models.py --model both --mode quick

# Train only Energy Model (fast mode: 15 epochs, full data)
python train_cifar10_models.py --model energy --mode fast

# Train only Diffusion Model (full mode: 25 epochs, full data)
python train_cifar10_models.py --model diffusion --mode full

# Use CPU instead of MPS (for debugging)
python train_cifar10_models.py --model both --mode quick --device cpu
```

#### Training Modes
- `quick`: 2 epochs, 5000 images (for quick testing)
- `fast`: 15 epochs, full dataset (recommended for development)
- `full`: 25 epochs, full dataset (for production)

#### Model Options
- `energy`: Train only Energy Model
- `diffusion`: Train only Diffusion Model
- `both`: Train both models sequentially

#### Device Options
- `mps`: Apple Silicon (MPS) - default on Mac
- `cpu`: CPU fallback (for debugging or compatibility)

#### Training Configuration

**Energy Model:**
- Architecture: CNN (32→64→128→256 channels) + FC layers
- Energy range: [-10, 10] via tanh activation
- Training: Contrastive Divergence with Langevin dynamics
- Learning rate: 1e-5 (with cosine annealing)
- Batch size: 64
- Langevin steps: 250 (training), 200 (evaluation)
- Gradient clipping: 0.3

**Diffusion Model:**
- Architecture: UNet with time embedding
- Schedule: DDPM cosine schedule
- Training: MSE loss on predicted noise
- Learning rate: 2e-4 (with cosine annealing)
- Batch size: 64
- Diffusion steps: 1000 (standard), 500 (quick)
- Gradient clipping: 1.0

### Training GAN Model (Assignment 3)

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

### Training CNN Models

```bash
# Train CNN on CIFAR-10
python train_cnn.py

# Train CNN for assignments
python train.py
```

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

### GAN Image Generation
```bash
# Generate 1 image with random seed
curl -X POST http://127.0.0.1:8000/generate_image \
  -H "Content-Type: application/json" \
  -d '{"num_images": 1, "seed": 42}'
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

### Energy-Based Model
- **Energy Function**: CNN-based architecture computing scalar energy scores
- **Sampling**: Langevin dynamics with gradient-based MCMC
- **Training**: Contrastive Divergence loss (E(real) - E(fake))
- **Energy Bounding**: tanh activation to limit energy to [-10, 10] range

### Diffusion Model
- **Architecture**: UNet with time-step embeddings
- **Noise Schedule**: DDPM cosine schedule (satisfies noise_rate² + signal_rate² = 1.0)
- **Training**: Predict noise added to images
- **Sampling**: Reverse diffusion process with 1000 steps
- **EMA**: Exponential moving average for stable generation

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
- **tqdm**: Progress bars for training loops
- **transformers**: For LLM fine-tuning support

## Configuration

The application uses environment variables for configuration:
- `SPACY_MODEL`: spaCy model to use (default: "en_core_web_md")

## Development

### Code Structure
- `app/main.py`: FastAPI application with text generation and embedding endpoints
- `app/bigram_model.py`: Bigram language model implementation
- `helper_lib/`: Complete neural network helper library
- `train_cifar10_models.py`: Training script for Energy and Diffusion models
- `train.py`: Training script for CNN assignments
- `example_usage.py`: Example usage of the helper library

### Adding New Features
1. **FastAPI**: Add new endpoints in `app/main.py`
2. **Neural Networks**: Add new models in `helper_lib/model.py` or `helper_lib/energy_model.py`
3. **Training**: Add new training functions in `helper_lib/trainer.py`
4. Update this README with new functionality

## Troubleshooting

### Apple Silicon (MPS) Issues
- If you encounter numerical instability, use `--device cpu`
- MPS may have different behavior than CUDA in some operations
- For debugging, always start with CPU to ensure stability

### Training Issues
- **Energy Model not learning**: Try reducing learning rate or increasing Langevin steps
- **Diffusion Model generating black images**: Check noise schedule, ensure proper normalization
- **Out of memory**: Reduce batch size in training scripts

## License

This project is open source and available for educational purposes.

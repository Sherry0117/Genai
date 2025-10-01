# Helper Library for Neural Network Projects

This library provides common functionalities for data loading, model training, and evaluation to reduce duplication across various neural network projects.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from helper_lib import (
    get_data_loader, 
    get_model, 
    train_model, 
    evaluate_model,
    get_optimizer, 
    get_criterion
)

# Load data
train_loader = get_data_loader('data', batch_size=32, train=True, dataset_name='CIFAR10')
test_loader = get_data_loader('data', batch_size=32, train=False, dataset_name='CIFAR10')

# Create model
model = get_model('CNN', num_classes=10)

# Setup training
criterion = get_criterion('cross_entropy')
optimizer = get_optimizer(model, 'adam', lr=0.001)

# Train model
trained_model = train_model(
    model=model,
    data_loader=train_loader,
    criterion=criterion,
    optimizer=optimizer,
    device='cuda',
    epochs=10
)

# Evaluate model
test_loss, test_accuracy = evaluate_model(
    model=trained_model,
    data_loader=test_loader,
    criterion=criterion,
    device='cuda'
)
```

## Available Models

- **FCNN**: Fully Connected Neural Network
- **CNN**: Convolutional Neural Network
- **EnhancedCNN**: CNN with Batch Normalization

## Available Datasets

- **CIFAR10**: 10-class image classification
- **CIFAR100**: 100-class image classification
- **MNIST**: Handwritten digits
- **FashionMNIST**: Fashion items

## Available Optimizers

- **adam**: Adam optimizer
- **sgd**: Stochastic Gradient Descent
- **adamw**: Adam with weight decay

## Available Loss Functions

- **cross_entropy**: Cross-entropy loss for classification
- **mse**: Mean Squared Error for regression
- **nll**: Negative Log Likelihood

## Features

- **Data Loading**: Automatic dataset downloading and preprocessing
- **Model Training**: Complete training loop with progress tracking
- **Model Evaluation**: Comprehensive evaluation metrics
- **Utilities**: Parameter counting, plotting, model saving/loading
- **Device Support**: Automatic GPU/CPU detection and usage


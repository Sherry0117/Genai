#!/usr/bin/env python3
"""
Training script for the SpecifiedCNN on CIFAR-10 dataset.

This script trains the exact CNN architecture as specified in the requirements:
- Input: RGB image, size 64×64×3
- Conv2D: 16 filters, kernel 3×3, stride=1, padding=1
- ReLU
- MaxPooling2D: kernel 2×2, stride=2
- Conv2D: 32 filters, kernel 3×3, stride=1, padding=1
- ReLU
- MaxPooling2D: kernel 2×2, stride=2
- Flatten
- Fully connected (Linear): 100 units
- ReLU
- Fully connected (Linear): 10 units (10 classes)
"""

import torch
import os
from helper_lib import (
    get_cifar10_loader, 
    get_model, 
    train_model, 
    evaluate_model,
    get_optimizer, 
    get_criterion,
    count_parameters,
    plot_training_history
)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load CIFAR-10 data (resized to 64x64 for SpecifiedCNN)
    print("Loading CIFAR-10 dataset...")
    train_loader = get_cifar10_loader(
        data_dir='./data', 
        batch_size=32, 
        train=True, 
        download=True,
        resize_to_64=True
    )
    test_loader = get_cifar10_loader(
        data_dir='./data', 
        batch_size=32, 
        train=False, 
        download=True,
        resize_to_64=True
    )
    
    # Create SpecifiedCNN model
    print("Creating SpecifiedCNN model...")
    model = get_model('SpecifiedCNN', num_classes=10, device=device)
    print(f"Model has {count_parameters(model):,} parameters")
    
    # Setup training
    criterion = get_criterion('cross_entropy')
    optimizer = get_optimizer(model, 'adam', lr=0.001)
    
    # Train model
    print("Starting training...")
    trained_model, history = train_model(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=20,
        val_loader=test_loader,
        verbose=True
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = evaluate_model(
        model=trained_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_accuracy:.2f}%")
    
    # Save the trained model
    model_path = 'models/specified_cnn_cifar10.pth'
    torch.save(trained_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training history
    plot_training_history(history, save_path='models/training_history.png')
    
    # Print CIFAR-10 class names for reference
    cifar10_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    print("\nCIFAR-10 classes:", cifar10_classes)

if __name__ == "__main__":
    main()

"""
Example usage of the helper_lib for neural network projects.
This demonstrates how to use the helper library for a complete training pipeline.
"""

import torch
from helper_lib import (
    get_data_loader, 
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
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading CIFAR-10 dataset...")
    train_loader = get_data_loader('data', batch_size=32, train=True, dataset_name='CIFAR10')
    test_loader = get_data_loader('data', batch_size=32, train=False, dataset_name='CIFAR10')
    
    # Create model
    print("Creating CNN model...")
    model = get_model('CNN', num_classes=10)
    print(f"Model has {count_parameters(model):,} parameters")
    
    # Setup training
    criterion = get_criterion('cross_entropy')
    optimizer = get_optimizer(model, 'adam', lr=0.001)
    
    # Train model
    print("Starting training...")
    trained_model = train_model(
        model=model,
        data_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=5
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = evaluate_model(
        model=trained_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()



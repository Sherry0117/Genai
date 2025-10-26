import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_data_loader(data_dir, batch_size=32, train=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train
    )
    
    return loader

def get_optimizer(model, optimizer_name='adam', lr=0.001, **kwargs):
    """Get optimizer by name.
    
    Args:
        model: PyTorch model
        optimizer_name: Name of optimizer ('adam', 'sgd', 'rmsprop')
        lr: Learning rate
        **kwargs: Additional optimizer arguments
    
    Returns:
        PyTorch optimizer
    """
    optimizer_name = optimizer_name.lower()
    
    if optimizer_name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == 'rmsprop':
        return optim.RMSprop(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}. Choose from 'adam', 'sgd', 'rmsprop'")

def get_criterion(criterion_name='cross_entropy'):
    """Get loss function by name.
    
    Args:
        criterion_name: Name of criterion ('cross_entropy', 'mse', 'nll')
    
    Returns:
        PyTorch loss function
    """
    criterion_name = criterion_name.lower()
    
    if criterion_name == 'cross_entropy':
        return nn.CrossEntropyLoss()
    elif criterion_name == 'mse':
        return nn.MSELoss()
    elif criterion_name == 'nll':
        return nn.NLLLoss()
    else:
        raise ValueError(f"Unknown criterion: {criterion_name}. Choose from 'cross_entropy', 'mse', 'nll'")

def count_parameters(model):
    """Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def plot_training_history(history, save_path=None):
    """Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary containing 'train_loss', 'train_acc', 'val_loss', 'val_acc' lists
        save_path: Optional path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(history.get('train_loss', []), label='Train Loss')
    ax1.plot(history.get('val_loss', []), label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history.get('train_acc', []), label='Train Accuracy')
    ax2.plot(history.get('val_acc', []), label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

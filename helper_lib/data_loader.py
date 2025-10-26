import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_data_loader(data_dir, batch_size=32, train=True, num_workers=2, pin_memory=True):
    """Create a DataLoader for images organized in ImageFolder format.

    Expected directory structure:
      data_dir/
        train/ class_x/ ..., class_y/ ...
        test/  class_x/ ..., class_y/ ...

    For training, light augmentation is applied; for evaluation, only resize/normalize.
    Images are resized to 224 so they are compatible with `resnet18` if used.
    """

    split_dir = "train" if train else "test"
    dataset_root = os.path.join(data_dir, split_dir)

    if train:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = datasets.ImageFolder(root=dataset_root, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_mnist_loader(data_dir='./data', batch_size=64, train=True, download=True, num_workers=2, pin_memory=True):
    """Create a DataLoader for MNIST dataset.
    
    Args:
        data_dir: Directory to store/load MNIST data
        batch_size: Batch size for DataLoader
        train: If True, load training set; otherwise load test set
        download: If True, download MNIST if not already present
        num_workers: Number of worker processes for data loading
        pin_memory: If True, pin memory for faster GPU transfer
    
    Returns:
        DataLoader for MNIST dataset
    """
    # Transform to normalize to [-1, 1] for Tanh activation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Maps [0,1] to [-1,1]
    ])
    
    dataset = datasets.MNIST(
        root=data_dir,
        train=train,
        transform=transform,
        download=download
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_cifar10_loader(data_dir='./data', batch_size=32, train=True, download=True, num_workers=2, pin_memory=True, resize_to_64=True):
    """Create a DataLoader for CIFAR-10 dataset.
    
    Args:
        data_dir: Directory to store/load CIFAR-10 data
        batch_size: Batch size for DataLoader
        train: If True, load training set; otherwise load test set
        download: If True, download CIFAR-10 if not already present
        num_workers: Number of worker processes for data loading
        pin_memory: If True, pin memory for faster GPU transfer
        resize_to_64: If True, resize images to 64x64 (for SpecifiedCNN)
    
    Returns:
        DataLoader for CIFAR-10 dataset
    """
    if resize_to_64:
        # Transform for 64x64 input (SpecifiedCNN requirement)
        if train:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    else:
        # Standard CIFAR-10 transform (32x32)
        if train:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    dataset = datasets.CIFAR10(
        root=data_dir,
        train=train,
        transform=transform,
        download=download
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

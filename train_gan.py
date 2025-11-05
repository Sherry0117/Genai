#!/usr/bin/env python3
"""
Training script for GAN on MNIST dataset.
This script trains a GAN to generate handwritten digits.
"""

import os
import torch
import torch.optim as optim
from helper_lib.model import GANGenerator, GANDiscriminator
from helper_lib.data_loader import get_mnist_loader
from helper_lib.trainer import train_gan
import matplotlib.pyplot as plt
import numpy as np

def create_models_dir():
    """Create models directory if it doesn't exist."""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def save_generated_samples(generator, device, epoch, num_samples=16, save_dir='./models'):
    """Generate and save sample images from the generator."""
    generator.eval()
    with torch.no_grad():
        # Generate samples
        noise = torch.randn(num_samples, 100).to(device)
        generated_images = generator(noise)
        
        # Convert from [-1, 1] to [0, 1] for display
        generated_images = (generated_images + 1) / 2.0
        generated_images = torch.clamp(generated_images, 0, 1)
        
        # Create a grid of images
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(num_samples):
            row = i // 4
            col = i % 4
            img = generated_images[i].squeeze().cpu().numpy()
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'generated_samples_epoch_{epoch}.png'))
        plt.close()
    
    generator.train()

def plot_training_losses(logs, save_dir='./models'):
    """Plot and save training losses."""
    epochs = [log['epoch'] for log in logs]
    d_losses = [log['D_loss'] for log in logs]
    g_losses = [log['G_loss'] for log in logs]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, d_losses, label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Discriminator Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Generator Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_losses.png'))
    plt.close()

def main():
    """Main training function."""
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models directory
    models_dir = create_models_dir()
    
    # Hyperparameters
    batch_size = 64
    z_dim = 100
    lr = 0.0002
    beta1 = 0.5
    epochs = 20
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_loader = get_mnist_loader(
        data_dir='./data',
        batch_size=batch_size,
        train=True,
        download=True
    )
    
    # Initialize models
    print("Initializing GAN models...")
    generator = GANGenerator(z_dim=z_dim).to(device)
    discriminator = GANDiscriminator().to(device)
    
    # Print model info
    total_params_g = sum(p.numel() for p in generator.parameters())
    total_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {total_params_g:,}")
    print(f"Discriminator parameters: {total_params_d:,}")
    
    # Train GAN
    print(f"Starting GAN training for {epochs} epochs...")
    logs = train_gan(
        generator=generator,
        discriminator=discriminator,
        dataloader=train_loader,
        device=device,
        z_dim=z_dim,
        lr=lr,
        beta1=beta1,
        epochs=epochs
    )
    
    # Save trained generator
    generator_path = os.path.join(models_dir, 'gan_generator.pth')
    torch.save(generator.state_dict(), generator_path)
    print(f"Saved trained generator to {generator_path}")
    
    # Generate and save sample images
    print("Generating sample images...")
    save_generated_samples(generator, device, epoch=epochs, save_dir=models_dir)
    
    # Plot training losses
    print("Plotting training losses...")
    plot_training_losses(logs, save_dir=models_dir)
    
    print("Training completed successfully!")
    print(f"Check the '{models_dir}' directory for:")
    print("- gan_generator.pth (trained model weights)")
    print("- generated_samples_epoch_20.png (sample generated images)")
    print("- training_losses.png (training loss curves)")

if __name__ == "__main__":
    main()



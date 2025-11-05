#!/usr/bin/env python3
"""
Simplified GAN training script for MNIST.
This script trains a GAN to generate handwritten digits.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Import our GAN models
from helper_lib.model import GANGenerator, GANDiscriminator

def create_models_dir():
    """Create models directory if it doesn't exist."""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir

def get_mnist_loader(batch_size=64, train=True):
    """Create MNIST DataLoader."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Maps [0,1] to [-1,1]
    ])
    
    dataset = datasets.MNIST(
        root='./data',
        train=train,
        transform=transform,
        download=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False,
    )

def train_gan_simple(generator, discriminator, dataloader, device, epochs=5):
    """Simplified GAN training function."""
    criterion = nn.BCELoss()
    opt_gen = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    generator.train()
    discriminator.train()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        for batch_idx, (real_images, _) in enumerate(dataloader):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            opt_disc.zero_grad()
            
            # Real images
            real_output = discriminator(real_images)
            loss_real = criterion(real_output, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, 100).to(device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            loss_fake = criterion(fake_output, fake_labels)
            
            loss_disc = loss_real + loss_fake
            loss_disc.backward()
            opt_disc.step()
            
            # Train Generator
            opt_gen.zero_grad()
            
            noise = torch.randn(batch_size, 100).to(device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images)
            loss_gen = criterion(fake_output, real_labels)
            
            loss_gen.backward()
            opt_gen.step()
            
            if batch_idx % 200 == 0:
                print(f"Batch {batch_idx}, D_loss: {loss_disc.item():.4f}, G_loss: {loss_gen.item():.4f}")
    
    print("Training completed!")

def save_generated_samples(generator, device, epoch, num_samples=16, save_dir='./models'):
    """Generate and save sample images."""
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(num_samples, 100).to(device)
        generated_images = generator(noise)
        
        # Convert from [-1, 1] to [0, 1]
        generated_images = (generated_images + 1) / 2.0
        generated_images = torch.clamp(generated_images, 0, 1)
        
        # Create a grid
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

def main():
    """Main training function."""
    print("Starting GAN training for MNIST...")
    
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models directory
    models_dir = create_models_dir()
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    train_loader = get_mnist_loader(batch_size=64, train=True)
    
    # Initialize models
    print("Initializing GAN models...")
    generator = GANGenerator(z_dim=100).to(device)
    discriminator = GANDiscriminator().to(device)
    
    # Print model info
    total_params_g = sum(p.numel() for p in generator.parameters())
    total_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {total_params_g:,}")
    print(f"Discriminator parameters: {total_params_d:,}")
    
    # Train GAN
    print("Starting training...")
    train_gan_simple(generator, discriminator, train_loader, device, epochs=5)
    
    # Save trained generator
    generator_path = os.path.join(models_dir, 'gan_generator.pth')
    torch.save(generator.state_dict(), generator_path)
    print(f"Saved trained generator to {generator_path}")
    
    # Generate sample images
    print("Generating sample images...")
    save_generated_samples(generator, device, epoch=5, save_dir=models_dir)
    
    print("Training completed successfully!")
    print(f"Check the '{models_dir}' directory for the trained model and sample images.")

if __name__ == "__main__":
    main()

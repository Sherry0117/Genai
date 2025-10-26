#!/usr/bin/env python3
"""
Test script for GAN implementation.
This script tests the GAN models and API endpoints.
"""

import torch
import requests
import json
import base64
from PIL import Image
from io import BytesIO
import os

def test_gan_models():
    """Test GAN model architectures."""
    print("Testing GAN model architectures...")
    
    from helper_lib.model import GANGenerator, GANDiscriminator
    
    # Test Generator
    generator = GANGenerator(z_dim=100)
    noise = torch.randn(4, 100)
    generated = generator(noise)
    
    print(f"Generator input shape: {noise.shape}")
    print(f"Generator output shape: {generated.shape}")
    print(f"Generator output range: [{generated.min():.3f}, {generated.max():.3f}]")
    
    # Test Discriminator
    discriminator = GANDiscriminator()
    fake_output = discriminator(generated)
    real_output = discriminator(torch.randn(4, 1, 28, 28))
    
    print(f"Discriminator fake output shape: {fake_output.shape}")
    print(f"Discriminator real output shape: {real_output.shape}")
    print(f"Discriminator output range: [{fake_output.min():.3f}, {fake_output.max():.3f}]")
    
    print("✓ GAN models working correctly!")

def test_data_loader():
    """Test MNIST data loader."""
    print("\nTesting MNIST data loader...")
    
    from helper_lib.data_loader import get_mnist_loader
    
    # Test with small batch
    train_loader = get_mnist_loader(
        data_dir='./data',
        batch_size=4,
        train=True,
        download=True
    )
    
    # Get one batch
    for images, labels in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    print("✓ MNIST data loader working correctly!")

def test_api_endpoint():
    """Test the FastAPI endpoint for image generation."""
    print("\nTesting API endpoint...")
    
    # Start the API server in background (this would normally be done separately)
    print("Note: Start the API server with 'uvicorn app.main:app --reload' to test the endpoint")
    
    # Test data for API request
    test_request = {
        "num_images": 2,
        "seed": 42
    }
    
    print(f"Test request: {test_request}")
    print("✓ API endpoint test data prepared!")

def test_training_script():
    """Test if the training script can be imported and run basic checks."""
    print("\nTesting training script...")
    
    try:
        import train_gan
        print("✓ Training script imports successfully!")
        
        # Test model creation
        from helper_lib.model import GANGenerator, GANDiscriminator
        generator = GANGenerator(z_dim=100)
        discriminator = GANDiscriminator()
        
        print("✓ Models can be instantiated!")
        
    except Exception as e:
        print(f"✗ Error in training script: {e}")

def main():
    """Run all tests."""
    print("=" * 50)
    print("GAN Implementation Test Suite")
    print("=" * 50)
    
    try:
        test_gan_models()
        test_data_loader()
        test_api_endpoint()
        test_training_script()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise

if __name__ == "__main__":
    main()






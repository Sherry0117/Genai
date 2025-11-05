#!/usr/bin/env python3
"""
Training script for GPT-2 fine-tuning on Nectar Q&A dataset.

This script fine-tunes a GPT-2 model on the Nectar Question-Answer dataset
from HuggingFace. It demonstrates how to fine-tune a language model for
text generation tasks.

Usage:
    python train_llm.py

The script will:
1. Download the Nectar dataset
2. Load and prepare the data
3. Fine-tune GPT-2 on Q&A pairs
4. Save the trained model to models/gpt2_finetuned/
"""

import os
import torch
from transformers import AutoModelForCausalLM
from helper_lib.llm_data_loader import prepare_llm_data
from helper_lib.llm_trainer import train_llm_model, save_final_model


def create_models_dir():
    """Create models directory if it doesn't exist."""
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def print_model_info(model):
    """Print information about the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    print(f"- Model size: ~{total_params * 4 / 1e6:.2f} MB\n")


def main():
    """Main training function."""
    # Set device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)
    
    # Create models directory
    models_dir = create_models_dir()
    checkpoint_dir = os.path.join(models_dir, 'gpt2_finetuned')
    
    # Hyperparameters
    num_samples = 1000  # Number of Q&A pairs to use for training
    batch_size = 4  # Smaller batch size for GPT-2 (adjust based on GPU memory)
    max_length = 512  # Maximum sequence length
    learning_rate = 5e-5  # Learning rate for fine-tuning
    epochs = 3  # Number of training epochs
    gradient_accumulation_steps = 4  # Accumulate gradients for effective larger batch size
    model_name = "openai-community/gpt2"  # Base GPT-2 model
    
    print(f"\nHyperparameters:")
    print(f"- Samples: {num_samples}")
    print(f"- Batch size: {batch_size}")
    print(f"- Max length: {max_length}")
    print(f"- Learning rate: {learning_rate}")
    print(f"- Epochs: {epochs}")
    print(f"- Gradient accumulation steps: {gradient_accumulation_steps}")
    print("=" * 60)
    
    # Load and prepare data
    print("\nLoading and preparing data...")
    print("-" * 60)
    
    tokenizer, train_loader = prepare_llm_data(
        num_samples=num_samples,
        split='train',
        batch_size=batch_size,
        max_length=max_length,
        model_name=model_name
    )
    
    print(f"\n✓ Data loaded successfully!")
    print(f"- Number of batches: {len(train_loader)}")
    print(f"- Batch size: {train_loader.batch_size}")
    
    # Optional: Load validation data
    print("\nLoading validation data...")
    val_tokenizer, val_loader = prepare_llm_data(
        num_samples=200,  # Smaller validation set
        split='train',  # Using train split but with different indices
        batch_size=batch_size,
        max_length=max_length,
        model_name=model_name
    )
    
    print(f"✓ Validation data loaded: {len(val_loader)} batches")
    
    # Load pretrained GPT-2 model
    print("\nLoading pretrained GPT-2 model...")
    print("-" * 60)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    print_model_info(model)
    
    # Train the model
    print("Starting fine-tuning...")
    print("=" * 60)
    
    trained_model, history = train_llm_model(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        val_loader=val_loader,
        checkpoint_dir=checkpoint_dir,
        save_every_n_epochs=1,
        gradient_accumulation_steps=gradient_accumulation_steps
    )
    
    print("\n" + "=" * 60)
    print("Training Summary:")
    print("=" * 60)
    print(f"Training completed for {epochs} epochs")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if len(history['val_loss']) > 0:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
    print("=" * 60)
    
    # Save final model
    print("\nSaving final model...")
    final_model_path = os.path.join(checkpoint_dir, 'final_model')
    save_final_model(trained_model, tokenizer, final_model_path)
    print(f"✓ Final model saved to: {final_model_path}")
    
    # Print file locations
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nCheck the '{checkpoint_dir}' directory for:")
    print("- final_model/ (directory with model weights and tokenizer)")
    print("- checkpoint_epoch_X.pt (intermediate checkpoints)")
    print("\nTo use the trained model:")
    print("  from helper_lib.llm_generator import generate_text_with_llm")
    print("  text = generate_text_with_llm(")
    print("      prompt='Your question here',")
    print("      model_path='models/gpt2_finetuned/final_model',")
    print("      device='cuda' if torch.cuda.is_available() else 'cpu'")
    print("  )")
    print("=" * 60)


if __name__ == "__main__":
    main()


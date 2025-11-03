"""
LLM Trainer for GPT-2 Fine-tuning

This module provides training functions for fine-tuning GPT-2 on Q&A datasets.
It uses HuggingFace transformers library and PyTorch.

DO NOT modify this file unless explicitly instructed.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def train_llm_model(
    model,
    tokenizer,
    train_loader,
    device='cpu',
    epochs=3,
    learning_rate=5e-5,
    val_loader=None,
    checkpoint_dir='models/gpt2_finetuned',
    save_every_n_epochs=1,
    gradient_accumulation_steps=1
):
    """
    Fine-tune a GPT-2 model on Q&A dataset.
    
    Args:
        model: GPT-2 model (AutoModelForCausalLM instance)
        tokenizer: GPT-2 tokenizer
        train_loader: Training DataLoader
        device: Training device ('cpu' or 'cuda')
        epochs: Number of training epochs
        learning_rate: Learning rate for AdamW optimizer
        val_loader: Optional validation DataLoader
        checkpoint_dir: Directory to save checkpoints
        save_every_n_epochs: Save checkpoint every N epochs
        gradient_accumulation_steps: Accumulate gradients over N steps
        
    Returns:
        Trained model
    """
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move model to device
    model.to(device)
    model.train()
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Setup learning rate scheduler
    total_steps = len(train_loader) * epochs
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    print(f"Starting training on {device}")
    print(f"Total epochs: {epochs}, Total steps: {total_steps}")
    
    global_step = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        # Progress bar for training
        train_pbar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            ncols=120
        )
        
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # For causal LM, labels are the same as input_ids (next token prediction)
            )
            
            # Get loss
            loss = outputs.loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights (with gradient accumulation)
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update optimizer and scheduler
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            # Track loss
            epoch_train_loss += loss.item() * gradient_accumulation_steps
            num_train_batches += 1
            global_step += 1
            
            # Update progress bar
            current_lr = scheduler.get_last_lr()[0]
            train_pbar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / max(num_train_batches, 1)
        history['train_loss'].append(avg_train_loss)
        
        print(f"\nEpoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Validation phase
        if val_loader is not None:
            val_loss = evaluate_llm_model(model, val_loader, device)
            history['val_loss'].append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pt')
            save_llm_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                epoch=epoch + 1,
                train_loss=avg_train_loss,
                val_loss=val_loss if val_loader is not None else None,
                filepath=checkpoint_path
            )
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model')
    save_final_model(model, tokenizer, final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    print("Training completed!")
    return model, history


def evaluate_llm_model(model, val_loader, device='cpu'):
    """
    Evaluate GPT-2 model on validation data.
    
    Args:
        model: GPT-2 model
        val_loader: Validation DataLoader
        device: Device for evaluation
        
    Returns:
        Average validation loss
    """
    model.eval()
    total_val_loss = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        val_pbar = tqdm(val_loader, desc="Validation", ncols=120)
        
        for batch in val_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            
            # Track loss
            total_val_loss += outputs.loss.item()
            num_val_batches += 1
            
            val_pbar.set_postfix({'loss': f'{outputs.loss.item():.4f}'})
    
    avg_val_loss = total_val_loss / max(num_val_batches, 1)
    return avg_val_loss


def save_llm_checkpoint(model, tokenizer, optimizer, epoch, train_loss, val_loss=None, filepath='checkpoint.pt'):
    """
    Save model checkpoint.
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        optimizer: Optimizer
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss (optional)
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    torch.save(checkpoint, filepath)


def load_llm_checkpoint(model, optimizer, filepath='checkpoint.pt', device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        model: GPT-2 model
        optimizer: Optimizer
        filepath: Path to checkpoint file
        device: Device to load on
        
    Returns:
        Starting epoch number
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Train Loss: {checkpoint['train_loss']:.4f}")
    
    if checkpoint['val_loss'] is not None:
        print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    
    return checkpoint['epoch']


def save_final_model(model, tokenizer, save_directory):
    """
    Save the final fine-tuned model in HuggingFace format.
    
    Args:
        model: Fine-tuned GPT-2 model
        tokenizer: GPT-2 tokenizer
        save_directory: Directory to save model
    """
    os.makedirs(save_directory, exist_ok=True)
    
    # Save model
    model.save_pretrained(save_directory)
    
    # Save tokenizer
    tokenizer.save_pretrained(save_directory)
    
    print(f"Model and tokenizer saved to {save_directory}")


def load_fine_tuned_model(model_name_or_path, device='cpu'):
    """
    Load a fine-tuned GPT-2 model.
    
    Args:
        model_name_or_path: Path to fine-tuned model or model name
        device: Device to load on
        
    Returns:
        Loaded model and tokenizer
    """
    print(f"Loading model from {model_name_or_path}...")
    
    # Try to load from path first (local fine-tuned model)
    if os.path.exists(model_name_or_path):
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    else:
        # Load pretrained model
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    model.to(device)
    model.eval()
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


if __name__ == "__main__":
    # Test the trainer
    print("Testing LLM Trainer...")
    print("\nThis is a test script.")
    print("To use this module, import it in train_llm.py or your main script.")
    print("\nExample usage:")
    print("from helper_lib.llm_trainer import train_llm_model, load_fine_tuned_model")
    print("from helper_lib.llm_data_loader import prepare_llm_data")
    print("\ntokenizer, train_loader = prepare_llm_data(num_samples=1000)")
    print("model = AutoModelForCausalLM.from_pretrained('openai-community/gpt2')")
    print("trained_model, history = train_llm_model(model, tokenizer, train_loader, device='cuda')")


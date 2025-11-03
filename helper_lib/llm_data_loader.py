"""
LLM Data Loader for GPT-2 Fine-tuning

This module loads and preprocesses the Nectar Q&A dataset for fine-tuning GPT-2.
It uses HuggingFace datasets and transformers libraries.

DO NOT modify this file unless explicitly instructed.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset


class QADataset(Dataset):
    """
    Dataset class for Question-Answer pairs from the Nectar dataset.
    Converts text to tokenized sequences for GPT-2 fine-tuning.
    """
    
    def __init__(self, tokenizer, qa_pairs, max_length=512):
        """
        Initialize Q&A dataset.
        
        Args:
            tokenizer: GPT-2 tokenizer instance
            qa_pairs: List of (question, answer) tuples
            max_length: Maximum sequence length for tokenization
        """
        self.tokenizer = tokenizer
        self.qa_pairs = qa_pairs
        self.max_length = max_length
    
    def __len__(self):
        return len(self.qa_pairs)
    
    def __getitem__(self, idx):
        """Get a tokenized Q&A pair."""
        question, answer = self.qa_pairs[idx]
        
        # Format as: "Question: <q> Answer: <a>"
        text = f"Question: {question} Answer: {answer}"
        
        # Tokenize the text
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'text': text
        }


def get_llm_tokenizer(model_name="openai-community/gpt2"):
    """
    Load and configure GPT-2 tokenizer.
    
    Args:
        model_name: HuggingFace model name for GPT-2
        
    Returns:
        Configured tokenizer instance
    """
    print(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token (important for batching)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Tokenizer loaded successfully.")
    return tokenizer


def load_nectar_dataset(num_samples=None, split='train'):
    """
    Load the Nectar Q&A dataset from HuggingFace.
    
    Args:
        num_samples: Number of samples to load (None for all)
        split: Dataset split ('train' or 'test')
        
    Returns:
        List of (question, answer) tuples
    """
    print(f"Loading Nectar dataset (split: {split})...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset("berkeley-nest/Nectar", split=split)
    
    # Extract question and answer pairs
    qa_pairs = []
    for item in dataset:
        question = item.get('question', '')
        answer = item.get('answer', '')
        if question and answer:
            qa_pairs.append((question, answer))
        
        # Limit number of samples if specified
        if num_samples and len(qa_pairs) >= num_samples:
            break
    
    print(f"Loaded {len(qa_pairs)} Q&A pairs from Nectar dataset.")
    return qa_pairs


def get_llm_data_loader(tokenizer, qa_pairs, batch_size=4, shuffle=True, max_length=512, 
                       num_workers=0, pin_memory=True):
    """
    Create a PyTorch DataLoader for GPT-2 fine-tuning.
    
    Args:
        tokenizer: GPT-2 tokenizer instance
        qa_pairs: List of (question, answer) tuples
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle the data
        max_length: Maximum sequence length
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        PyTorch DataLoader instance
    """
    # Create dataset
    dataset = QADataset(tokenizer, qa_pairs, max_length=max_length)
    
    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return data_loader


def prepare_llm_data(num_samples=1000, split='train', batch_size=4, 
                    max_length=512, model_name="openai-community/gpt2"):
    """
    Complete data preparation pipeline for LLM fine-tuning.
    
    Args:
        num_samples: Number of samples to load
        split: Dataset split ('train' or 'test')
        batch_size: Batch size for DataLoader
        max_length: Maximum sequence length
        model_name: HuggingFace model name
        
    Returns:
        tuple: (tokenizer, data_loader)
    """
    # Load tokenizer
    tokenizer = get_llm_tokenizer(model_name)
    
    # Load dataset
    qa_pairs = load_nectar_dataset(num_samples=num_samples, split=split)
    
    # Create DataLoader
    data_loader = get_llm_data_loader(
        tokenizer=tokenizer,
        qa_pairs=qa_pairs,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        max_length=max_length
    )
    
    return tokenizer, data_loader


if __name__ == "__main__":
    # Test the data loader
    print("Testing LLM Data Loader...")
    
    tokenizer, data_loader = prepare_llm_data(
        num_samples=100,
        batch_size=2,
        max_length=256
    )
    
    print(f"DataLoader created with {len(data_loader)} batches")
    
    # Test loading a batch
    for batch in data_loader:
        print(f"\nBatch keys: {batch.keys()}")
        print(f"Input IDs shape: {batch['input_ids'].shape}")
        print(f"Attention mask shape: {batch['attention_mask'].shape}")
        print(f"Sample text: {batch['text'][0]}")
        break
    
    print("\nTest completed successfully!")


"""
LLM Generator for GPT-2 Text Generation

This module provides text generation functions using fine-tuned GPT-2 models.
It handles loading models and generating text based on prompts.

DO NOT modify this file unless explicitly instructed.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List


def load_llm_model_and_tokenizer(model_path: str, device: str = 'cpu'):
    """
    Load fine-tuned GPT-2 model and tokenizer from disk.
    
    Args:
        model_path: Path to the fine-tuned model directory
        device: Device to load model on ('cpu' or 'cuda')
        
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"Loading model from {model_path}...")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer


def generate_text_with_llm(
    prompt: str,
    model_path: str = 'models/gpt2_finetuned/final_model',
    max_length: int = 100,
    num_return_sequences: int = 1,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    device: str = 'cpu',
    do_sample: bool = True
) -> str:
    """
    Generate text using fine-tuned GPT-2 model.
    
    Args:
        prompt: Input text prompt
        model_path: Path to fine-tuned model directory
        max_length: Maximum length of generated text
        num_return_sequences: Number of sequences to generate
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        device: Device to run inference on ('cpu' or 'cuda')
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        
    Returns:
        Generated text as string
    """
    
    # Load model and tokenizer
    model, tokenizer = load_llm_model_and_tokenizer(model_path, device)
    
    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def generate_text_with_loaded_model(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_length: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    device: str = 'cpu'
) -> str:
    """
    Generate text using already-loaded model and tokenizer.
    More efficient when generating multiple times.
    
    Args:
        prompt: Input text prompt
        model: Loaded GPT-2 model
        tokenizer: Loaded tokenizer
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        device: Device to run inference on
        
    Returns:
        Generated text as string
    """
    
    # Tokenize input prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def generate_batch_texts(
    prompts: List[str],
    model_path: str = 'models/gpt2_finetuned/final_model',
    max_length: int = 100,
    batch_size: int = 4,
    temperature: float = 0.7,
    device: str = 'cpu'
) -> List[str]:
    """
    Generate text for multiple prompts (batch processing for efficiency).
    
    Args:
        prompts: List of input text prompts
        model_path: Path to fine-tuned model directory
        max_length: Maximum length of generated text
        batch_size: Batch size for processing
        temperature: Sampling temperature
        device: Device to run inference on
        
    Returns:
        List of generated texts
    """
    
    # Load model and tokenizer
    model, tokenizer = load_llm_model_and_tokenizer(model_path, device)
    
    generated_texts = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        batch_inputs = tokenizer(
            batch_prompts, 
            return_tensors='pt', 
            padding=True,
            truncation=True
        ).to(device)
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **batch_inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and add to results
        for j in range(len(batch_prompts)):
            generated_text = tokenizer.decode(outputs[j], skip_special_tokens=True)
            generated_texts.append(generated_text)
    
    return generated_texts


def generate_answer(question: str, model_path: str = 'models/gpt2_finetuned/final_model',
                    max_length: int = 150, device: str = 'cpu') -> str:
    """
    Generate an answer to a question using the fine-tuned model.
    
    Args:
        question: Input question string
        model_path: Path to fine-tuned model directory
        max_length: Maximum length of generated answer
        device: Device to run inference on
        
    Returns:
        Generated answer string
    """
    
    # Format prompt for Q&A task
    prompt = f"Question: {question} Answer:"
    
    # Generate using the main generation function
    generated_text = generate_text_with_llm(
        prompt=prompt,
        model_path=model_path,
        max_length=max_length,
        temperature=0.5,
        do_sample=True,
        device=device
    )
    
    # Extract answer part (everything after "Answer:")
    if "Answer:" in generated_text:
        answer = generated_text.split("Answer:")[1].strip()
    else:
        answer = generated_text
    
    return answer


def generate_with_model_instance(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 0.7,
    device: str = 'cpu'
) -> str:
    """
    Generate text using provided model and tokenizer instances.
    Useful for API endpoints where model is already loaded.
    
    Args:
        model: GPT-2 model instance
        tokenizer: Tokenizer instance
        prompt: Input text prompt
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        device: Device to run inference on
        
    Returns:
        Generated text string
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and return
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


if __name__ == "__main__":
    # Test the generator
    print("Testing LLM Generator...")
    print("\nThis is a test script.")
    print("To use this module, import it in your main script or API.")
    print("\nExample usage:")
    print("from helper_lib.llm_generator import generate_text_with_llm, generate_answer")
    print("\n# Generate text")
    print("text = generate_text_with_llm('Hello, how are you?', max_length=50)")
    print("\n# Generate answer")
    print("answer = generate_answer('What is machine learning?', max_length=100)")
    print("\nFor more details, see the function docstrings.")


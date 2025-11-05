"""
Train CIFAR-10 Energy Model and Diffusion Model
Optimized for M4 MacBook Pro (Apple Silicon MPS)

Recommendation: Use MPS for faster training, but fallback to CPU if you encounter:
- Numerical instability (NaN/Inf values)
- Mode collapse in Diffusion Model
- Inconsistent results between runs

For first-time training or debugging, start with CPU to ensure stability.
"""
import os
import argparse
from typing import Tuple
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from helper_lib.energy_model import EnergyModel
from helper_lib.model import get_model, DiffusionModel


def _ensure_dirs(model_dir: str, samples_dir: str):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(samples_dir, exist_ok=True)


def _get_mode_config(mode: str) -> Tuple[int, int]:
    mode = (mode or "fast").lower()
    if mode == "quick":
        return 2, 5000
    if mode == "full":
        return 25, -1
    return 15, -1


def _cifar10_loader(batch_size: int, train: bool, resize_to_32: bool = True, num_workers: int = 0, pin_memory: bool = False, normalize_to_minus_one: bool = False) -> DataLoader:
    """
    Create CIFAR-10 DataLoader
    For Diffusion Model: use 32x32 original size, normalize to [-1, 1]
    For Energy Model: use 32x32 original size, keep [0, 1]
    """
    if resize_to_32:
        # Ensure 32x32 (CIFAR-10 default size)
        if normalize_to_minus_one:
            # Diffusion Model: normalize to [-1, 1]
            tfm = transforms.Compose([
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            # Energy Model: keep [0, 1]
            tfm = transforms.Compose([
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ])
    else:
        if normalize_to_minus_one:
            tfm = transforms.Compose([
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            tfm = transforms.Compose([
                transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ])

    dataset = datasets.CIFAR10(root="./data", train=train, transform=tfm, download=True)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def _denormalize_tensor(images: torch.Tensor) -> torch.Tensor:
    """Convert normalized images [-1, 1] back to [0, 1] for saving"""
    return (images + 1.0) / 2.0


def _save_sample_grid(images: torch.Tensor, out_path: str, nrow: int = 4):
    """Save sample grid, images should already be in [0, 1] range"""
    images = images.clamp(0.0, 1.0)
    save_image(make_grid(images, nrow=nrow, padding=2, normalize=False), out_path)


def _check_numerical_stability(tensor: torch.Tensor, name: str, device: str) -> bool:
    """Check numerical stability"""
    if torch.isnan(tensor).any():
        print(f"⚠️  Warning [{device}]: {name} contains NaN values")
        return False
    if torch.isinf(tensor).any():
        print(f"⚠️  Warning [{device}]: {name} contains Inf values")
        return False
    return True


def train_energy_model_cifar10(config):
    """Train Energy Model - Optimized for M4 MPS"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_dir = os.path.join("./models", "energy")
    samples_dir = os.path.join("./samples", "energy")
    _ensure_dirs(model_dir, samples_dir)

    epochs, max_images = _get_mode_config(config.mode)
    batch_size = 64  # M4 memory limit, reduce batch size
    
    # Energy Model uses [0, 1] range data (keep original design)
    train_loader = _cifar10_loader(batch_size=batch_size, train=True, resize_to_32=True, normalize_to_minus_one=False)

    model = EnergyModel().to(device)
    # CRITICAL FIX: Further reduce learning rate for Energy Model stability
    # Energy Models are notoriously difficult to train, need very small learning rate
    initial_lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, betas=(0.0, 0.999))
    
    # Add learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    
    # Record training history
    train_history = {
        'real_energy': [],
        'fake_energy': [],
        'loss': [],
        'lr': []
    }

    print(f"\nStarting Energy Model training (M4 MPS optimized)")
    print(f"  Batch size: {batch_size}")
    print(f"  Initial LR: {initial_lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Langevin steps: 150 (training), 200 (evaluation)")
    print(f"  Energy range: [-10, 10] (via tanh activation)")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        epoch_real_energy = 0.0
        epoch_fake_energy = 0.0
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Energy | Epoch {epoch+1}/{epochs}", ncols=120)
        seen = 0

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device, non_blocking=False)
            seen += images.size(0)
            
            # Numerical stability check
            if not _check_numerical_stability(images, "Input images", str(device)):
                print("Numerical issues detected, skipping this batch")
                continue

            # Compute energy for real images
            real_energy = model(images).mean()
            
            fake_images = model.sample(
                num_samples=images.size(0), 
                device=device, 
                num_steps=250,  
                step_size=5 
            )
            
            # Energy Model's sample() returns [0, 1] range, consistent with training data
            # No conversion needed
            
            # Numerical stability check
            if not _check_numerical_stability(fake_images, "Sampled images", str(device)):
                print("Sampled images contain NaN/Inf, skipping this batch")
                continue
            
            fake_energy = model(fake_images).mean()
            
            # Contrastive Divergence loss
            # CRITICAL FIX: Energy values are now bounded by tanh, no need for clipping
            # But we still clip to be safe and ensure numerical stability
            real_energy_safe = torch.clamp(real_energy, min=-20.0, max=20.0)
            fake_energy_safe = torch.clamp(fake_energy, min=-20.0, max=20.0)
            
            # CD loss: E(real) - E(fake)
            # We want to minimize real energy (real images should have low energy)
            # and maximize fake energy (fake images should have high energy)
            cd_loss = real_energy_safe.mean() - fake_energy_safe.mean()
            
            # L2 regularization to prevent overfitting
            l2_reg = 2e-4 * sum(p.pow(2).sum() for p in model.parameters())
            
            # CRITICAL FIX: Add energy gap penalty to encourage separation
            # This helps the model learn to distinguish real and fake images
            energy_gap = torch.abs(real_energy_safe.mean() - fake_energy_safe.mean())
            gap_penalty = -0.1 * energy_gap  # Encourage larger gap (negative because we minimize)
            
            loss = cd_loss + l2_reg + gap_penalty
            
            # Check if loss is valid
            if not _check_numerical_stability(loss.unsqueeze(0), "Loss", str(device)):
                print("Loss contains NaN/Inf, skipping this batch")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
            
            # Check gradients
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            if total_grad_norm > 100.0:
                print(f"⚠️  Warning: Gradient norm too large ({total_grad_norm:.2f}), may be unstable")
            
            optimizer.step()

            # Record statistics
            epoch_real_energy += real_energy.item()
            epoch_fake_energy += fake_energy.item()
            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                "Real_E": f"{real_energy.item():.3f}", 
                "Fake_E": f"{fake_energy.item():.3f}", 
                "Loss": f"{loss.item():.4f}",
                "Grad": f"{total_grad_norm:.2f}"
            })

            if config.mode == "quick" and max_images > 0 and seen >= max_images:
                break

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record epoch statistics
        avg_real_energy = epoch_real_energy / max(num_batches, 1)
        avg_fake_energy = epoch_fake_energy / max(num_batches, 1)
        avg_loss = epoch_loss / max(num_batches, 1)
        
        train_history['real_energy'].append(avg_real_energy)
        train_history['fake_energy'].append(avg_fake_energy)
        train_history['loss'].append(avg_loss)
        train_history['lr'].append(current_lr)
        
        print(f"Epoch {epoch+1}/{epochs} - Real_E: {avg_real_energy:.4f}, Fake_E: {avg_fake_energy:.4f}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")

        # Save checkpoint
        model_path = os.path.join(model_dir, f"energy_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_path)

        # Generate samples (use many more steps for better quality)
        model.eval()
        with torch.no_grad():
            # CRITICAL FIX: Use significantly more steps for evaluation
            # Energy Models need many MCMC steps to generate high-quality samples
            samples = model.sample(num_samples=16, device=device, num_steps=400, step_size=8)
            # Energy Model returns [0, 1] range, save directly
        _save_sample_grid(samples, os.path.join(samples_dir, f"samples_epoch_{epoch+1}.png"), nrow=4)

    # Save final model
    final_path = os.path.join(model_dir, "energy_final.pt")
    torch.save(model.state_dict(), final_path)
    
    # Save training curves
    _plot_training_curves(train_history, os.path.join(model_dir, "energy_training_curves.png"), "Energy Model")
    
    print(f"\n✓ Energy Model training completed!")
    return model


def train_diffusion_model_cifar10(config):
    """Train Diffusion Model - Optimized for M4 MPS"""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_dir = os.path.join("./models", "diffusion")
    samples_dir = os.path.join("./samples", "diffusion")
    _ensure_dirs(model_dir, samples_dir)

    epochs, max_images = _get_mode_config(config.mode)
    batch_size = 64  # M4 memory limit
    image_size = 32  # Use original CIFAR-10 size 32x32, don't resize to 64x64
    
    # Use 32x32 original size, normalize to [-1, 1] range (Diffusion Model standard practice)
    train_loader = _cifar10_loader(batch_size=batch_size, train=True, resize_to_32=True, normalize_to_minus_one=True)

    # Create Diffusion Model, use 32x32
    diffusion_model: DiffusionModel = get_model("diffusion", device=device, image_size=image_size, num_channels=3)
    diffusion_model.train()

    # Reduce learning rate
    initial_lr = 2e-4
    optimizer = torch.optim.AdamW(diffusion_model.network.parameters(), 
                              lr=initial_lr, betas=(0.9, 0.999), weight_decay=1e-4)
    
    # Add cosine annealing learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    criterion = nn.MSELoss()
    
    # Record training history
    train_history = {
        'loss': [],
        'lr': [],
        'noise_rate_min': [],
        'noise_rate_max': [],
        'signal_rate_min': [],
        'signal_rate_max': []
    }

    print(f"\nStarting Diffusion Model training (M4 MPS optimized)")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size} (original CIFAR-10 size)")
    print(f"  Initial LR: {initial_lr}")
    print(f"  Epochs: {epochs}")
    print(f"  Sampling steps: 250 (evaluation)")
    print("=" * 60)

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Diffusion | Epoch {epoch+1}/{epochs}", ncols=120)
        seen = 0
        epoch_loss = 0.0
        num_batches = 0
        
        # For recording noise_rates and signal_rates
        noise_rate_mins = []
        noise_rate_maxs = []
        signal_rate_mins = []
        signal_rate_maxs = []

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device, non_blocking=False)
            seen += images.size(0)
            
            # Numerical stability check
            if not _check_numerical_stability(images, "Input images", str(device)):
                print("Numerical issues detected, skipping this batch")
                continue

            # Diffusion process
            noises = torch.randn_like(images)
            diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
            noise_rates, signal_rates = diffusion_model.schedule_fn(diffusion_times)
            
            # Record noise_rates and signal_rates ranges
            noise_rate_mins.append(noise_rates.min().item())
            noise_rate_maxs.append(noise_rates.max().item())
            signal_rate_mins.append(signal_rates.min().item())
            signal_rate_maxs.append(signal_rates.max().item())
            
            # Check if schedule is reasonable
            if noise_rates.min().item() < 0 or noise_rates.max().item() > 1:
                print(f"⚠️  Warning: noise_rates range abnormal [{noise_rates.min().item():.4f}, {noise_rates.max().item():.4f}]")
            if signal_rates.min().item() < 0 or signal_rates.max().item() > 1:
                print(f"⚠️  Warning: signal_rates range abnormal [{signal_rates.min().item():.4f}, {signal_rates.max().item():.4f}]")
            
            noisy_images = signal_rates * images + noise_rates * noises
            
            # Numerical stability check
            if not _check_numerical_stability(noisy_images, "Noisy images", str(device)):
                print("Noisy images contain NaN/Inf, skipping this batch")
                continue

            pred_noises = diffusion_model.network(noisy_images, noise_rates ** 2)
            loss = criterion(pred_noises, noises)
            
            # Check loss
            if not _check_numerical_stability(loss.unsqueeze(0), "Loss", str(device)):
                print("Loss contains NaN/Inf, skipping this batch")
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Add gradient clipping (previously missing)
            torch.nn.utils.clip_grad_norm_(diffusion_model.network.parameters(), max_norm=1.0)
            
            # Check gradients
            total_grad_norm = 0.0
            for p in diffusion_model.network.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm ** 0.5
            
            if total_grad_norm > 100.0:
                print(f"⚠️  Warning: Gradient norm too large ({total_grad_norm:.2f})")
            
            optimizer.step()

            # EMA update
            with torch.no_grad():
                for ema_param, param in zip(diffusion_model.ema_network.parameters(), diffusion_model.network.parameters()):
                    ema_param.copy_(diffusion_model.ema_decay * ema_param + (1. - diffusion_model.ema_decay) * param)

            # Record statistics
            epoch_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Grad": f"{total_grad_norm:.2f}",
                "NR": f"[{noise_rates.min().item():.2f},{noise_rates.max().item():.2f}]",
                "SR": f"[{signal_rates.min().item():.2f},{signal_rates.max().item():.2f}]"
            })

            if config.mode == "quick" and max_images > 0 and seen >= max_images:
                break

        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record epoch statistics
        avg_loss = epoch_loss / max(num_batches, 1)
        train_history['loss'].append(avg_loss)
        train_history['lr'].append(current_lr)
        train_history['noise_rate_min'].append(min(noise_rate_mins) if noise_rate_mins else 0.0)
        train_history['noise_rate_max'].append(max(noise_rate_maxs) if noise_rate_maxs else 1.0)
        train_history['signal_rate_min'].append(min(signal_rate_mins) if signal_rate_mins else 0.0)
        train_history['signal_rate_max'].append(max(signal_rate_maxs) if signal_rate_maxs else 1.0)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        print(f"  Noise rates: [{train_history['noise_rate_min'][-1]:.4f}, {train_history['noise_rate_max'][-1]:.4f}]")
        print(f"  Signal rates: [{train_history['signal_rate_min'][-1]:.4f}, {train_history['signal_rate_max'][-1]:.4f}]")

        # Save checkpoint
        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": diffusion_model.network.state_dict(),
            "ema_model_state_dict": diffusion_model.ema_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        torch.save(ckpt, os.path.join(model_dir, f"diffusion_epoch_{epoch+1}.pth"))

        # Generate samples (use standard DDPM steps: 1000)
        # Note: For quick testing, can reduce to 500, but 1000 is standard for best quality
        diffusion_model.eval()
        with torch.no_grad():
            # Use 1000 steps for standard DDPM sampling (can be reduced for faster testing)
            eval_steps = 1000 if config.mode != "quick" else 500
            samples = diffusion_model.generate(num_images=16, diffusion_steps=eval_steps, image_size=image_size)
            # Convert to [0, 1] for saving
            samples = _denormalize_tensor(samples)
        _save_sample_grid(samples, os.path.join(samples_dir, f"samples_epoch_{epoch+1}.png"), nrow=4)
        diffusion_model.train()

    # Save final model
    torch.save(
        {
            "model_state_dict": diffusion_model.network.state_dict(),
            "ema_model_state_dict": diffusion_model.ema_network.state_dict(),
        },
        os.path.join(model_dir, "diffusion_final.pth")
    )
    
    # Save training curves
    _plot_training_curves(train_history, os.path.join(model_dir, "diffusion_training_curves.png"), "Diffusion Model")
    
    print(f"\n✓ Diffusion Model training completed!")
    return diffusion_model


def _plot_training_curves(history: dict, save_path: str, model_name: str):
    """Plot and save training curves"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{model_name} - Training Curves', fontsize=14)
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss curve
    axes[0, 0].plot(epochs, history['loss'], 'b-', label='Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].grid(True)
    axes[0, 0].legend()
    
    # Learning rate curve
    axes[0, 1].plot(epochs, history['lr'], 'r-', label='Learning Rate')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Learning Rate')
    axes[0, 1].set_title('Learning Rate Schedule')
    axes[0, 1].set_yscale('log')
    axes[0, 1].grid(True)
    axes[0, 1].legend()
    
    # Energy Model specific: Real Energy and Fake Energy
    if 'real_energy' in history:
        axes[1, 0].plot(epochs, history['real_energy'], 'g-', label='Real Energy')
        axes[1, 0].plot(epochs, history['fake_energy'], 'orange', label='Fake Energy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Energy')
        axes[1, 0].set_title('Energy Values')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
    else:
        # Diffusion Model: show noise/signal rates
        axes[1, 0].plot(epochs, history['noise_rate_min'], 'b--', label='Noise Rate Min', alpha=0.5)
        axes[1, 0].plot(epochs, history['noise_rate_max'], 'b-', label='Noise Rate Max')
        axes[1, 0].plot(epochs, history['signal_rate_min'], 'g--', label='Signal Rate Min', alpha=0.5)
        axes[1, 0].plot(epochs, history['signal_rate_max'], 'g-', label='Signal Rate Max')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Rate')
        axes[1, 0].set_title('Noise/Signal Rates')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
    
    # Blank or additional info
    axes[1, 1].axis('off')
    info_text = f"Model: {model_name}\n"
    info_text += f"Total Epochs: {len(history['loss'])}\n"
    info_text += f"Final Loss: {history['loss'][-1]:.4f}\n"
    info_text += f"Final LR: {history['lr'][-1]:.6f}"
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models (energy/diffusion) - M4 MPS Optimized")
    parser.add_argument("--model", type=str, choices=["energy", "diffusion", "both"], required=True,
                        help="Which model to train")
    parser.add_argument("--mode", type=str, choices=["quick", "fast", "full"], default="fast",
                        help="Training mode: quick (~25min), fast (~2h), full (6-8h)")
    parser.add_argument("--device", type=str, choices=["mps", "cpu"], default=("mps" if torch.backends.mps.is_available() else "cpu"))
    args = parser.parse_args()

    if args.model in ("energy", "both"):
        train_energy_model_cifar10(args)

    if args.model in ("diffusion", "both"):
        train_diffusion_model_cifar10(args)


if __name__ == "__main__":
    main()

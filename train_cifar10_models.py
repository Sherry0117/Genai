import os
import argparse
from typing import Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
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


def _cifar10_loader(batch_size: int, train: bool, resize_to_64: bool, num_workers: int = 8, pin_memory: bool = True) -> DataLoader:
    if resize_to_64:
        tfm = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
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


def _save_sample_grid(images: torch.Tensor, out_path: str, nrow: int = 4):
    images = images.clamp(0.0, 1.0)
    save_image(make_grid(images, nrow=nrow, padding=2, normalize=False), out_path)


def train_energy_model_cifar10(config):
    device = config.device
    model_dir = os.path.join("./models", "energy")
    samples_dir = os.path.join("./samples", "energy")
    _ensure_dirs(model_dir, samples_dir)

    epochs, max_images = _get_mode_config(config.mode)
    batch_size = 128

    train_loader = _cifar10_loader(batch_size=batch_size, train=True, resize_to_64=False)

    model = EnergyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.5, 0.999))
    scaler = GradScaler(enabled=(device.startswith("cuda")))

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Energy | Epoch {epoch+1}/{epochs}", ncols=120)
        seen = 0

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            seen += images.size(0)

            with autocast(enabled=(device.startswith("cuda"))):
                real_energy = model(images).mean()
                # sample() needs grad computation for Langevin dynamics, so don't wrap in no_grad
                # But we detach the result since we only need fake_images for forward pass
                fake_images = model.sample(num_samples=images.size(0), device=device, num_steps=20, step_size=10).detach()
                fake_energy = model(fake_images).mean()
                cd_loss = real_energy - fake_energy
                l2_reg = 1e-4 * sum(p.pow(2).sum() for p in model.parameters())
                loss = cd_loss + l2_reg

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            pbar.set_postfix({"Real_E": f"{real_energy.item():.3f}", "Fake_E": f"{fake_energy.item():.3f}", "Loss": f"{loss.item():.3f}"})

            if config.mode == "quick" and max_images > 0 and seen >= max_images:
                break

        model_path = os.path.join(model_dir, f"energy_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), model_path)

        model.eval()
        with torch.no_grad():
            samples = model.sample(num_samples=16, device=device)
        _save_sample_grid(samples, os.path.join(samples_dir, f"samples_epoch_{epoch+1}.png"), nrow=4)

    final_path = os.path.join(model_dir, "energy_final.pt")
    torch.save(model.state_dict(), final_path)
    return model


def train_diffusion_model_cifar10(config):
    device = config.device
    model_dir = os.path.join("./models", "diffusion")
    samples_dir = os.path.join("./samples", "diffusion")
    _ensure_dirs(model_dir, samples_dir)

    epochs, max_images = _get_mode_config(config.mode)
    batch_size = 128

    # Diffusion UNet was set up for variable image_size; we use 64 for better stability
    train_loader = _cifar10_loader(batch_size=batch_size, train=True, resize_to_64=True)

    diffusion_model: DiffusionModel = get_model("diffusion", device=device, image_size=64, num_channels=3)
    diffusion_model.train()

    optimizer = torch.optim.Adam(diffusion_model.network.parameters(), lr=2e-4)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=(device.startswith("cuda")))

    for epoch in range(epochs):
        pbar = tqdm(train_loader, desc=f"Diffusion | Epoch {epoch+1}/{epochs}", ncols=120)
        seen = 0

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            seen += images.size(0)

            # Re-implement train step with AMP and EMA update
            noises = torch.randn_like(images)
            diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
            noise_rates, signal_rates = diffusion_model.schedule_fn(diffusion_times)
            noisy_images = signal_rates * images + noise_rates * noises

            with autocast(enabled=(device.startswith("cuda"))):
                pred_noises = diffusion_model.network(noisy_images, noise_rates ** 2)
                loss = criterion(pred_noises, noises)

            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                for ema_param, param in zip(diffusion_model.ema_network.parameters(), diffusion_model.network.parameters()):
                    ema_param.copy_(diffusion_model.ema_decay * ema_param + (1. - diffusion_model.ema_decay) * param)

            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

            if config.mode == "quick" and max_images > 0 and seen >= max_images:
                break

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": diffusion_model.network.state_dict(),
            "ema_model_state_dict": diffusion_model.ema_network.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(ckpt, os.path.join(model_dir, f"diffusion_epoch_{epoch+1}.pth"))

        diffusion_model.eval()
        with torch.no_grad():
            # Prefer built-in generate with ema
            samples = diffusion_model.generate(num_images=16, diffusion_steps=100, image_size=64)
        _save_sample_grid(samples, os.path.join(samples_dir, f"samples_epoch_{epoch+1}.png"), nrow=4)
        diffusion_model.train()

    torch.save(
        {
            "model_state_dict": diffusion_model.network.state_dict(),
            "ema_model_state_dict": diffusion_model.ema_network.state_dict(),
        },
        os.path.join(model_dir, "diffusion_final.pth")
    )

    return diffusion_model


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10 models (energy/diffusion)")
    parser.add_argument("--model", type=str, choices=["energy", "diffusion", "both"], required=True,
                        help="Which model to train")
    parser.add_argument("--mode", type=str, choices=["quick", "fast", "full"], default="fast",
                        help="Training mode: quick (~25min), fast (~2h), full (6-8h)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    if args.model in ("energy", "both"):
        train_energy_model_cifar10(args)

    if args.model in ("diffusion", "both"):
        train_diffusion_model_cifar10(args)


if __name__ == "__main__":
    main()



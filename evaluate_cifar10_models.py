"""
Evaluate CIFAR-10 Energy Model and Diffusion Model
Generate evaluation samples and comparison plots
"""
import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from helper_lib.energy_model import EnergyModel
from helper_lib.model import DiffusionModel, get_model


def _denormalize_tensor(images: torch.Tensor) -> torch.Tensor:
    """Convert normalized images [-1, 1] back to [0, 1] for saving"""
    return (images + 1.0) / 2.0


def _save_sample_grid(images: torch.Tensor, out_path: str, nrow: int = 4):
    """Save sample grid, images should already be in [0, 1] range"""
    images = images.clamp(0.0, 1.0)
    save_image(make_grid(images, nrow=nrow, padding=2, normalize=False), out_path)


def evaluate_diffusion_model(device, model_path=None, output_dir="./evaluation_outputs"):
    """Evaluate Diffusion Model and generate samples"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating Diffusion Model on device: {device}")
    
    # Load model - use 32x32 for CIFAR-10
    image_size = 32
    diffusion_model = get_model('diffusion', schedule='ddpm_cosine', image_size=image_size, num_channels=3).to(device)
    
    # Load checkpoint if available
    if model_path and os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'ema_model_state_dict' in checkpoint:
            diffusion_model.ema_network.load_state_dict(checkpoint['ema_model_state_dict'])
        elif 'model_state_dict' in checkpoint:
            diffusion_model.network.load_state_dict(checkpoint['model_state_dict'])
        print("✓ Model loaded successfully")
    else:
        # Try to load from default location
        default_path = "./models/diffusion/diffusion_final.pth"
        if os.path.exists(default_path):
            print(f"Loading checkpoint from {default_path}")
            checkpoint = torch.load(default_path, map_location=device)
            if 'ema_model_state_dict' in checkpoint:
                diffusion_model.ema_network.load_state_dict(checkpoint['ema_model_state_dict'])
            elif 'model_state_dict' in checkpoint:
                diffusion_model.network.load_state_dict(checkpoint['model_state_dict'])
            print("✓ Model loaded successfully")
        else:
            print("⚠️  No checkpoint found, using untrained model")
    
    # Generate samples
    diffusion_model.eval()
    with torch.no_grad():
        print("Generating 16 samples with 1000 diffusion steps...")
        samples = diffusion_model.generate(num_images=16, diffusion_steps=1000, image_size=32)
        # Convert to [0, 1] for saving
        samples = _denormalize_tensor(samples)
    
    # Save samples
    output_path = os.path.join(output_dir, "diffusion_evaluation_samples.png")
    _save_sample_grid(samples, output_path, nrow=4)
    print(f"✓ Samples saved to {output_path}")
    
    return samples


def evaluate_energy_model(device, model_path=None, output_dir="./evaluation_outputs"):
    """Evaluate Energy Model and generate samples"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Evaluating Energy Model on device: {device}")
    
    # Load model
    model = EnergyModel().to(device)
    
    # Load checkpoint if available
    if model_path and os.path.exists(model_path):
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("✓ Model loaded successfully")
    else:
        # Try to load from default location
        default_path = "./models/energy/energy_final.pt"
        if os.path.exists(default_path):
            print(f"Loading checkpoint from {default_path}")
            checkpoint = torch.load(default_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✓ Model loaded successfully")
        else:
            print("⚠️  No checkpoint found, using untrained model")
    
    # Generate samples
    model.eval()
    with torch.no_grad():
        print("Generating 16 samples with 200 Langevin steps...")
        samples = model.sample(num_samples=16, device=device, num_steps=200, step_size=5)
    
    # Save samples
    output_path = os.path.join(output_dir, "energy_evaluation_samples.png")
    _save_sample_grid(samples, output_path, nrow=4)
    print(f"✓ Samples saved to {output_path}")
    
    return samples


def generate_epoch_comparison(device, epochs=[1, 5, 10, 15], model_type="diffusion", output_dir="./evaluation_outputs"):
    """Generate epoch comparison image showing samples from different training epochs"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {model_type} epoch comparison for epochs: {epochs}")
    
    if model_type == "diffusion":
        # Load model - use 32x32 for CIFAR-10
        image_size = 32
        diffusion_model = get_model('diffusion', schedule='ddpm_cosine', image_size=image_size, num_channels=3).to(device)
        
        all_samples = []
        epoch_labels = []
        
        for epoch in epochs:
            checkpoint_path = f"./models/diffusion/diffusion_epoch_{epoch}.pth"
            if not os.path.exists(checkpoint_path):
                print(f"⚠️  Warning: Checkpoint {checkpoint_path} not found, skipping epoch {epoch}")
                continue
            
            print(f"\nLoading epoch {epoch} checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'ema_model_state_dict' in checkpoint:
                diffusion_model.ema_network.load_state_dict(checkpoint['ema_model_state_dict'])
            elif 'model_state_dict' in checkpoint:
                diffusion_model.network.load_state_dict(checkpoint['model_state_dict'])
            
            # Generate samples
            diffusion_model.eval()
            with torch.no_grad():
                print(f"Generating 16 samples for epoch {epoch}...")
                samples = diffusion_model.generate(num_images=16, diffusion_steps=1000, image_size=image_size)
                samples = _denormalize_tensor(samples)
            
            all_samples.append(samples)
            epoch_labels.append(f"Epoch {epoch}")
        
        if not all_samples:
            print("❌ No valid checkpoints found!")
            return
        
        # Create comparison grid: 4 rows (epochs) x 4 columns (samples per epoch)
        # Each epoch row shows 4x4 grid of samples
        fig, axes = plt.subplots(len(all_samples), 1, figsize=(12, 3 * len(all_samples)))
        if len(all_samples) == 1:
            axes = [axes]
        
        for idx, (samples, label) in enumerate(zip(all_samples, epoch_labels)):
            # Create grid of samples for this epoch
            grid = make_grid(samples, nrow=4, padding=2, normalize=False)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            grid_np = grid_np.clip(0, 1)
            
            axes[idx].imshow(grid_np)
            axes[idx].axis('off')
            axes[idx].set_title(label, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "diffusion_epoch_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Epoch comparison saved to {output_path}")
        
    elif model_type == "energy":
        # Similar for energy model
        model = EnergyModel().to(device)
        
        all_samples = []
        epoch_labels = []
        
        for epoch in epochs:
            checkpoint_path = f"./models/energy/energy_epoch_{epoch}.pt"
            if not os.path.exists(checkpoint_path):
                print(f"⚠️  Warning: Checkpoint {checkpoint_path} not found, skipping epoch {epoch}")
                continue
            
            print(f"\nLoading epoch {epoch} checkpoint...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            # Generate samples
            model.eval()
            with torch.no_grad():
                print(f"Generating 16 samples for epoch {epoch}...")
                samples = model.sample(num_samples=16, device=device, num_steps=200, step_size=5)
            
            all_samples.append(samples)
            epoch_labels.append(f"Epoch {epoch}")
        
        if not all_samples:
            print("❌ No valid checkpoints found!")
            return
        
        # Create comparison grid
        fig, axes = plt.subplots(len(all_samples), 1, figsize=(12, 3 * len(all_samples)))
        if len(all_samples) == 1:
            axes = [axes]
        
        for idx, (samples, label) in enumerate(zip(all_samples, epoch_labels)):
            grid = make_grid(samples, nrow=4, padding=2, normalize=False)
            grid_np = grid.permute(1, 2, 0).cpu().numpy()
            grid_np = grid_np.clip(0, 1)
            
            axes[idx].imshow(grid_np)
            axes[idx].axis('off')
            axes[idx].set_title(label, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, "energy_epoch_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✓ Epoch comparison saved to {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 models")
    parser.add_argument("--model", type=str, choices=["energy", "diffusion", "both"], default="both",
                        help="Which model to evaluate")
    parser.add_argument("--device", type=str, choices=["mps", "cpu"], 
                        default=("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--output-dir", type=str, default="./evaluation_outputs",
                        help="Output directory for evaluation results")
    parser.add_argument("--epoch-comparison", action="store_true",
                        help="Generate epoch comparison images")
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 5, 10, 15],
                        help="Epochs to compare (default: 1 5 10 15)")
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    if args.epoch_comparison:
        if args.model in ("energy", "both"):
            generate_epoch_comparison(device, epochs=args.epochs, model_type="energy", output_dir=args.output_dir)
        if args.model in ("diffusion", "both"):
            generate_epoch_comparison(device, epochs=args.epochs, model_type="diffusion", output_dir=args.output_dir)
    else:
        if args.model in ("energy", "both"):
            evaluate_energy_model(device, output_dir=args.output_dir)
        
        if args.model in ("diffusion", "both"):
            evaluate_diffusion_model(device, output_dir=args.output_dir)
    
    print("\n✓ Evaluation completed!")


if __name__ == "__main__":
    main()


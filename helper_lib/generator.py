import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def generate_samples(model, device, num_samples=20, latent_dim=2):
    import math
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decoder(z)
        samples = samples.cpu().numpy()

    # Dynamically determine =grid size to fit all samples
    cols = 6
    rows = math.ceil(num_samples / cols)
    _fig, _axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))

    # Normalize axes to a flat iterable
    axes_iter = (_axes if isinstance(_axes, (list, tuple)) else _axes)
    axes_iter = getattr(axes_iter, "flat", [axes_iter])

    max_to_show = min(num_samples, rows * cols)
    for _i, _ax in enumerate(axes_iter):
        if _i >= max_to_show:
            _ax.axis('off')
            continue

        img = samples[_i]
        # Handle shapes: (H,W), (1,H,W), (C,H,W), (H,W,C)
        if img.ndim == 3 and img.shape[0] in (1, 3):
            # (C,H,W) -> (H,W) or (H,W,C)
            if img.shape[0] == 1:
                img_to_show = img[0]
                _ax.imshow(img_to_show, cmap='gray')
            else:
                img_to_show = img.transpose(1, 2, 0)
                _ax.imshow(img_to_show)
        elif img.ndim == 2:
            _ax.imshow(img, cmap='gray')
        elif img.ndim == 3 and img.shape[-1] in (1, 3):
            if img.shape[-1] == 1:
                _ax.imshow(img[..., 0], cmap='gray')
            else:
                _ax.imshow(img)
        else:
            # Fallback to squeeze then gray
            _ax.imshow(img.squeeze(), cmap='gray')

        _ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_wgan_grid(gen, fixed_noise, epoch):
    with torch.no_grad():
        fake = gen(fixed_noise).detach().cpu()
    grid = make_grid(fake, normalize=True)
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(f"Epoch {epoch}")
    plt.axis("off")
    plt.show()


def generate_diffusion_samples(model, device, num_samples=10, diffusion_steps=100, image_size=64):
    """
    Generate samples using trained Diffusion model
    
    Args:
        model: Trained DiffusionModel
        device: Device
        num_samples: Number of samples to generate
        diffusion_steps: Number of reverse diffusion steps (more steps = better quality but slower)
        image_size: Image size
    """
    
    # Move model to specified device and set to evaluation mode
    model.to(device)
    model.eval()
    
    print(f"Generating {num_samples} samples with {diffusion_steps} diffusion steps...")
    
    # Generate initial noise from standard normal distribution
    initial_noise = torch.randn(
        (num_samples, model.network.num_channels, image_size, image_size),
        device=device
    )
    
    # Perform reverse diffusion process to generate images
    with torch.no_grad():
        generated_images = model.generate(
            num_images=num_samples,
            diffusion_steps=diffusion_steps,
            image_size=image_size,
            initial_noise=initial_noise
        )
    
    # Move to CPU for plotting
    generated_images = generated_images.cpu()
    
    # Calculate grid layout
    nrow = int(num_samples ** 0.5)
    if nrow * nrow < num_samples:
        nrow += 1
    
    # Create image grid
    grid = make_grid(generated_images, nrow=nrow, normalize=True, padding=2)
    
    # Plot
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray' if model.network.num_channels == 1 else None)
    plt.axis('off')
    plt.title(f'Generated Samples (diffusion_steps={diffusion_steps})')
    plt.tight_layout()
    plt.show()
    
    return generated_images


def generate_and_save(model, device, num_samples=10, diffusion_steps=100, 
                     image_size=64, save_path='generated_samples.png'):
    """
    Generate samples and save to file
    
    Args:
        model: Trained DiffusionModel
        device: Device
        num_samples: Number of samples to generate
        diffusion_steps: Number of reverse diffusion steps
        image_size: Image size
        save_path: Save path
    """
    
    # Generate images without displaying them
    model.to(device)
    model.eval()
    
    print(f"Generating {num_samples} samples with {diffusion_steps} diffusion steps...")
    
    initial_noise = torch.randn(
        (num_samples, model.network.num_channels, image_size, image_size),
        device=device
    )
    
    with torch.no_grad():
        generated_images = model.generate(
            num_images=num_samples,
            diffusion_steps=diffusion_steps,
            image_size=image_size,
            initial_noise=initial_noise
        )
    
    generated_images = generated_images.cpu()
    
    # Calculate grid layout
    nrow = int(num_samples ** 0.5)
    if nrow * nrow < num_samples:
        nrow += 1
    
    # Create image grid and save
    grid = make_grid(generated_images, nrow=nrow, normalize=True, padding=2)
    
    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap='gray' if generated_images.shape[1] == 1 else None)
    plt.axis('off')
    plt.title(f'Generated Samples (diffusion_steps={diffusion_steps})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Samples saved to {save_path}")
    
    return generated_images


def generate_energy_samples(model, device, num_samples=16):
    """
    Generate samples from an Energy-based model using its Langevin sampler.
    The returned images are clamped/normalized into [0, 1] for visualization.

    Args:
        model: EnergyModel instance with .sample()
        device: torch device
        num_samples: number of images to generate

    Returns:
        Tensor of shape (num_samples, 3, 32, 32) in [0, 1]
    """
    model.to(device)
    model.eval()
    with torch.no_grad():
        images = model.sample(num_samples=num_samples, device=device)
    return images.clamp(0.0, 1.0)


def generate_diffusion_samples(diffusion_model, device, num_samples=16):
    """
    Generate samples from a Diffusion model.

    Prefers diffusion_model.sample() if available; otherwise falls back to
    diffusion_model.generate() using default steps and inferred image size.

    Args:
        diffusion_model: DiffusionModel instance
        device: torch device
        num_samples: number of images to generate

    Returns:
        Tensor of generated images normalized/clamped into [0, 1]
    """
    diffusion_model.to(device)
    diffusion_model.eval()

    with torch.no_grad():
        if hasattr(diffusion_model, 'sample') and callable(getattr(diffusion_model, 'sample')):
            images = diffusion_model.sample(num_samples=num_samples, device=device)
        else:
            # Fallback to .generate() from our DiffusionModel
            default_steps = getattr(diffusion_model, 'default_diffusion_steps', 100)
            image_size = getattr(diffusion_model.network, 'image_size', 64)
            images = diffusion_model.generate(
                num_images=num_samples,
                diffusion_steps=default_steps,
                image_size=image_size
            )

    return images.clamp(0.0, 1.0)

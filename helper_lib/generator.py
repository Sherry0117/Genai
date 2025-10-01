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

    # Dynamically determine grid size to fit all samples
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

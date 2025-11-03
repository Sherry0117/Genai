import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyModel(nn.Module):
    """
    Energy-based model for CIFAR-10 sized images (3x32x32).

    - Forward returns a scalar energy per image (lower means more realistic).
    - sample() generates images via Langevin dynamics.
    """

    def __init__(self):
        super().__init__()

        # CNN feature extractor: 3x32x32 -> feature maps
        self.features = nn.Sequential(
            # 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2),  # 32x32 -> 16x16

            # 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2),  # 16x16 -> 8x8

            # 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AvgPool2d(kernel_size=2),  # 8x8 -> 4x4

            # 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Fully connected head to scalar energy
        self.energy_head = nn.Sequential(
            nn.Flatten(),                # 256 x 4 x 4 -> 4096
            nn.Linear(256 * 4 * 4, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),          # scalar energy
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.2)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N, 3, 32, 32)
        Returns:
            energies: Tensor of shape (N,) where lower energy implies more realistic
        """
        feats = self.features(x)
        energy = self.energy_head(feats)  # (N, 1)
        return energy.squeeze(-1)

    @torch.no_grad()
    def _clamp_images_(self, x):
        # Clamp to a reasonable image range; CIFAR-10 commonly in [0,1] after ToTensor
        return x.clamp_(0.0, 1.0)

    def sample(self, num_samples, device, num_steps=60, step_size=10):
        """
        Generate samples using Langevin dynamics.

        x_{t+1} = x_t - 0.5 * s^2 * dE/dx + s * N(0, I)

        Args:
            num_samples: number of images to generate
            device: torch device
            num_steps: Langevin steps
            step_size: noise scale (float); effective step is step_size / 255 to match image scale

        Returns:
            Tensor of generated images (num_samples, 3, 32, 32) in [0,1]
        """
        self.eval()

        # Start from random noise in [0,1]
        x = torch.rand(num_samples, 3, 32, 32, device=device, dtype=torch.float32)

        # Use smaller step in image scale for stability
        s = float(step_size) / 255.0

        for _ in range(int(num_steps)):
            x.requires_grad_(True)
            
            # Compute energy and its gradient
            # Use torch.enable_grad() context to ensure gradients can be computed
            with torch.enable_grad():
                energy = self.forward(x).sum()
                grad_x, = torch.autograd.grad(energy, x, create_graph=False, retain_graph=False)

            # Langevin update
            noise = torch.randn_like(x)
            x = x - 0.5 * (s ** 2) * grad_x + s * noise

            # Detach and clamp to valid range
            x = x.detach()
            self._clamp_images_(x)

        return x


__all__ = ["EnergyModel"]



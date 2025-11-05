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
        # Architecture: 32 -> 64 -> 128 -> 256 channels
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
        """Initialize weights using Kaiming initialization"""
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
        Compute energy score for input images.
        
        Args:
            x: Tensor of shape (N, 3, 32, 32) in [0, 1] range
        Returns:
            energies: Tensor of shape (N,) where lower energy implies more realistic
        """
        feats = self.features(x)
        energy = self.energy_head(feats)  # (N, 1)
        energy = torch.tanh(energy) * 10.0  # Limit energy to [-10, 10] to prevent exp(-E) overflow
        return energy.squeeze(-1)
    
    def sample(self, num_samples, device, num_steps=60, step_size=10):
        """
        Generate samples using Langevin dynamics.
        
        Standard Langevin update formula:
        x_{t+1} = x_t - step_size * grad_E(x_t) + sqrt(2 * step_size) * N(0, I)
        
        Args:
            num_samples: number of images to generate
            device: torch device
            num_steps: number of Langevin steps (more steps = better quality, but slower)
            step_size: step size for Langevin dynamics (typically 0.01-0.1 for images in [0,1])
        
        Returns:
            Tensor of generated images (num_samples, 3, 32, 32) in [0,1]
        """
        self.eval()
        
        # Start from random noise in [0,1]
        x = torch.rand(num_samples, 3, 32, 32, device=device, dtype=torch.float32)
        
        # Convert step_size parameter to actual Langevin step size
        # For images in [0, 1] range, typical step size is 0.01-0.1
        lambda_t = float(step_size) / 100.0  # step_size=10 -> 0.1, step_size=1 -> 0.01
        
        for step in range(int(num_steps)):
            x.requires_grad_(True)
            
            # Compute energy and its gradient
            with torch.enable_grad():
                energy = self.forward(x).sum()
                grad_x, = torch.autograd.grad(energy, x, create_graph=False, retain_graph=False)
            
            # Standard Langevin update: x_{t+1} = x_t - lambda * grad_E + sqrt(2*lambda) * noise
            with torch.no_grad():
                noise = torch.randn_like(x)
                lambda_tensor = torch.tensor(lambda_t, device=x.device, dtype=x.dtype)
                x = x - lambda_t * grad_x + torch.sqrt(2.0 * lambda_tensor) * noise
                
                # Clamp to valid image range [0, 1]
                x = torch.clamp(x, 0.0, 1.0)
        
        return x.detach()


__all__ = ["EnergyModel"]

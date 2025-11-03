import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F
import math
import copy
from .energy_model import EnergyModel

class SimpleFCNN(nn.Module):
    def __init__(self, input_dim=224 * 224 * 3, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.classifier(x)


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)   
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) 
        self.fc1 = nn.Linear(32 * 8 * 8, 128)        
        self.fc2 = nn.Linear(128, 10)                    

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)   # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        # Conv Layer 1 with BatchNorm
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv Layer 2 with BatchNorm
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # Conv Layer 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Conv Layer 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Fully connected + Dropout
        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = x.view(-1, 128 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def get_model(model_name, num_classes=10, device='cpu', **kwargs):
    """Return a model instance by name.

    Supported models:
    - FCNN: Fully-connected on 224x224 RGB images
    - CNN: Small convnet
    - EnhancedCNN: Deeper convnet with BatchNorm and Dropout
    - SpecifiedCNN: Exact architecture as specified in requirements (64x64)
    - Diffusion / DiffusionModel: Diffusion model with UNet architecture
    - EnergyModel: Energy-based model for CIFAR-10 (3x32x32), outputs scalar energy
    """
    name = (model_name or "").strip().lower()

    if name == "fcnn":
        model = SimpleFCNN(num_classes=num_classes)
    elif name == "cnn":
        model = SimpleCNN(num_classes=num_classes)
    elif name == "enhancedcnn":
        model = EnhancedCNN(num_classes=num_classes)
    elif name == "specifiedcnn":
        model = SpecifiedCNN(num_classes=num_classes)
    elif name in ("diffusion", "diffusionmodel"):
        image_size = kwargs.get('image_size', 64)
        num_channels = kwargs.get('num_channels', 3)
        schedule = kwargs.get('schedule', 'cosine')  # 'linear', 'cosine', 'offset_cosine'
        
        # Select diffusion schedule function
        if schedule == 'linear':
            schedule_fn = linear_diffusion_schedule
        elif schedule == 'cosine':
            schedule_fn = cosine_diffusion_schedule
        elif schedule == 'offset_cosine':
            schedule_fn = offset_cosine_diffusion_schedule
        else:
            schedule_fn = cosine_diffusion_schedule
        
        unet = UNet(image_size=image_size, num_channels=num_channels)
        model = DiffusionModel(unet, schedule_fn)
    elif name == "energymodel":
        model = EnergyModel()
    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. Choose from FCNN, CNN, EnhancedCNN, "
            f"SpecifiedCNN, Diffusion, DiffusionModel, EnergyModel"
        )

    return model.to(device)

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dim=2):
        super(VariationalEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar   

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(2, 128 * 4 * 4)
        self.convtrans1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.convtrans3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.convtrans1(x))
        x = F.relu(self.convtrans2(x))
        x = torch.sigmoid(self.convtrans3(x))
        return x

class VAE(nn.Module):
    def __init__(self, encoder, decoder, beta=1.0):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(z)
        return reconstruction, mu, logvar


class Critic(nn.Module):
    def __init__(self):        
        super(Critic, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.act3 = nn.LeakyReLU(0.2, inplace=True)

        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(512)
        self.act4 = nn.LeakyReLU(0.2, inplace=True)

        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.conv1(x); x = self.batchnorm1(x); x = self.act1(x)
        x = self.conv2(x); x = self.batchnorm2(x); x = self.act2(x)
        x = self.conv3(x); x = self.batchnorm3(x); x = self.act3(x)
        x = self.conv4(x); x = self.batchnorm4(x); x = self.act4(x)
        x = self.conv5(x); x = self.flatten(x)
        return x

class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.reshape = lambda x: x.view(x.size(0), z_dim, 1, 1)

        self.deconv1 = nn.ConvTranspose2d(z_dim, 512, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512, momentum=0.9)
        self.act1 = nn.ReLU(True)

        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.act2 = nn.ReLU(True)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)
        self.act3 = nn.ReLU(True)

        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64, momentum=0.9)
        self.act4 = nn.ReLU(True)

        self.deconv5 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.reshape(x)
        x = self.deconv1(x); x = self.bn1(x); x = self.act1(x)
        x = self.deconv2(x); x = self.bn2(x); x = self.act2(x)
        x = self.deconv3(x); x = self.bn3(x); x = self.act3(x)
        x = self.deconv4(x); x = self.bn4(x); x = self.act4(x)
        x = self.deconv5(x); x = self.tanh(x)
        return x


# ===== GAN for MNIST =====
class GANGenerator(nn.Module):
    """Generator for MNIST GAN
    Input: Random noise vector (batch_size, 100)
    Output: Generated image (batch_size, 1, 28, 28)
    """
    def __init__(self, z_dim=100):
        super(GANGenerator, self).__init__()
        self.z_dim = z_dim
        
        # Fully connected layer: z_dim -> 128 * 7 * 7
        self.fc = nn.Linear(z_dim, 128 * 7 * 7)
        
        # Deconvolution layer 1: 128 -> 64, (7,7) -> (14,14)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Deconvolution layer 2: 64 -> 1, (14,14) -> (28,28)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)
        
    def forward(self, x):
        # FC layer and reshape
        x = self.fc(x)  # (batch_size, 128*7*7)
        x = x.view(-1, 128, 7, 7)  # (batch_size, 128, 7, 7)
        
        # Deconv 1 + BatchNorm + ReLU
        x = self.deconv1(x)  # (batch_size, 64, 14, 14)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Deconv 2 + Tanh
        x = self.deconv2(x)  # (batch_size, 1, 28, 28)
        x = torch.tanh(x)
        
        return x


class GANDiscriminator(nn.Module):
    """Discriminator for MNIST GAN
    Input: Image (batch_size, 1, 28, 28)
    Output: Probability of being real (batch_size, 1)
    """
    def __init__(self):
        super(GANDiscriminator, self).__init__()
        
        # Conv layer 1: 1 -> 64, (28,28) -> (14,14)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        
        # Conv layer 2: 64 -> 128, (14,14) -> (7,7)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Fully connected layer: 128 * 7 * 7 -> 1
        self.fc = nn.Linear(128 * 7 * 7, 1)
        
    def forward(self, x):
        # Conv 1 + LeakyReLU
        x = self.conv1(x)  # (batch_size, 64, 14, 14)
        x = F.leaky_relu(x, 0.2)
        
        # Conv 2 + BatchNorm + LeakyReLU
        x = self.conv2(x)  # (batch_size, 128, 7, 7)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        # Flatten and FC
        x = x.view(-1, 128 * 7 * 7)  # (batch_size, 128*7*7)
        x = self.fc(x)  # (batch_size, 1)
        x = torch.sigmoid(x)
        
        return x


# ===== Diffusion Model Components =====
class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding for encoding noise variance
    - num_frequencies: number of frequencies, default 16
    - input shape: (B, 1, 1, 1)
    - output shape: (B, 1, 1, 2 * num_frequencies)
    """
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))
    
    def forward(self, x):
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


class ResidualBlock(nn.Module):
    """
    Residual block: basic building block for UNet
    - Contains two convolutional layers and residual connection
    - Uses Swish activation function
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
        
        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def swish(self, x):
        return x * torch.sigmoid(x)
    
    def forward(self, x):
        residual = self.proj(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """
    Downsampling block: encoder part of UNet
    - Contains multiple ResidualBlocks
    - Uses AvgPool2d for downsampling
    - Saves skip connections
    """
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)
    
    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    """
    Upsampling block: decoder part of UNet
    - Uses bilinear interpolation for upsampling
    - Concatenates with skip connections
    """
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width
    
    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """
    UNet architecture: for predicting added noise
    - Input: noisy image + noise variance
    - Output: predicted noise
    """
    def __init__(self, image_size, num_channels, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)
        
        self.down1 = DownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)
        
        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)
        
        self.up1 = UpBlock(96, in_channels=128, block_depth=2)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)
        
        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)
    
    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        
        noise_emb = self.embedding(noise_variances)
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        x = torch.cat([x, noise_emb], dim=1)
        
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        
        x = self.mid1(x)
        x = self.mid2(x)
        
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        
        return self.final(x)


# Diffusion Schedule functions
def linear_diffusion_schedule(diffusion_times, min_rate=1e-4, max_rate=0.02):
    """Linear diffusion schedule"""
    diffusion_times = diffusion_times.to(dtype=torch.float32)
    betas = min_rate + diffusion_times * (max_rate - min_rate)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    signal_rates = torch.sqrt(alpha_bars)
    noise_rates = torch.sqrt(1.0 - alpha_bars)
    return noise_rates, signal_rates


def cosine_diffusion_schedule(diffusion_times):
    """Cosine diffusion schedule"""
    signal_rates = torch.cos(diffusion_times * math.pi / 2)
    noise_rates = torch.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """Offset cosine diffusion schedule"""
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()
    
    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32))
    
    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)
    
    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)
    
    return noise_rates, signal_rates


class DiffusionModel(nn.Module):
    """
    Complete diffusion model wrapper
    - Contains main network and EMA network
    - Implements forward diffusion and reverse diffusion
    - Handles training and generation
    """
    def __init__(self, model, schedule_fn):
        super().__init__()
        self.network = model
        self.ema_network = copy.deepcopy(model)
        self.ema_network.eval()
        self.ema_decay = 0.8
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0
    
    def to(self, device):
        super().to(device)
        self.ema_network.to(device)
        return self
    
    def set_normalizer(self, mean, std):
        self.normalizer_mean = mean
        self.normalizer_std = std
    
    def denormalize(self, x):
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)
    
    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()
        
        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images
    
    def reverse_diffusion(self, initial_noise, diffusion_steps):
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            
            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * torch.randn_like(pred_images)
        
        return pred_images
    
    def generate(self, num_images, diffusion_steps, image_size=64, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn((num_images, self.network.num_channels, image_size, image_size), device=next(self.network.parameters()).device)
        
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))
    
    def train_step(self, images, optimizer, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)
        
        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        
        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        loss = loss_fn(pred_noises, noises)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA network
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1. - self.ema_decay) * param)
        
        return loss.item()
    
    def test_step(self, images, loss_fn):
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)
        
        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        
        with torch.no_grad():
            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            loss = loss_fn(pred_noises, noises)
        
        return loss.item()


class SpecifiedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SpecifiedCNN, self).__init__()
        
        # Conv Layer 1: 3 -> 16, kernel 3x3, stride=1, padding=1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Conv Layer 2: 16 -> 32, kernel 3x3, stride=1, padding=1
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flatten
        self.flatten = nn.Flatten()
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 16 * 16, 100)  # After 2 pooling layers: 64->32->16
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(100, num_classes)
        
    def forward(self, x):
        # Conv1 + ReLU + MaxPool
        x = self.conv1(x)  # (batch_size, 16, 64, 64)
        x = self.relu1(x)
        x = self.pool1(x)  # (batch_size, 16, 32, 32)
        
        # Conv2 + ReLU + MaxPool
        x = self.conv2(x)  # (batch_size, 32, 32, 32)
        x = self.relu2(x)
        x = self.pool2(x)  # (batch_size, 32, 16, 16)
        
        # Flatten
        x = self.flatten(x)  # (batch_size, 32*16*16)
        
        # Fully connected layers
        x = self.fc1(x)  # (batch_size, 100)
        x = self.relu3(x)
        x = self.fc2(x)  # (batch_size, num_classes)
        
        return x

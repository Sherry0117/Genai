import torch
import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

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
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Input: 3, Output: 16
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)        # Pooling layer, halves dimensions
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # Input: 16, Output: 32
        self.fc1 = nn.Linear(32 * 8 * 8, 128)                    # Fully connected layer
        self.fc2 = nn.Linear(128, 10)                            # Output for 10 classes

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


def get_model(model_name, num_classes=10, device='cpu'):
    """Return a model instance by name: FCNN, CNN, EnhancedCNN, or SpecifiedCNN.

    - FCNN: Fully-connected on 224x224 RGB images
    - CNN: Small convnet
    - EnhancedCNN: ResNet18 backbone with linear head
    - SpecifiedCNN: Exact architecture as specified in requirements
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
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Choose from FCNN, CNN, EnhancedCNN, SpecifiedCNN")

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


class SpecifiedCNN(nn.Module):
    """Exact CNN architecture as specified in the requirements.
    
    Architecture:
    - Input: RGB image, size 64×64×3
    - Conv2D: 16 filters, kernel 3×3, stride=1, padding=1
    - ReLU
    - MaxPooling2D: kernel 2×2, stride=2
    - Conv2D: 32 filters, kernel 3×3, stride=1, padding=1
    - ReLU
    - MaxPooling2D: kernel 2×2, stride=2
    - Flatten
    - Fully connected (Linear): 100 units
    - ReLU
    - Fully connected (Linear): 10 units (10 classes)
    """
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

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim


def train_model(model, data_loader, criterion, optimizer, device='cpu',
                epochs=10, val_loader=None, verbose=True):
    """Train a PyTorch model.
    
    Args:
        model: PyTorch model to train
        data_loader: Training DataLoader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on ('cpu' or 'cuda')
        epochs: Number of training epochs
        val_loader: Optional validation DataLoader
        verbose: If True, print training progress
    
    Returns:
        Trained model
    """
    model.train()
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if verbose and batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(data_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        # Calculate average loss and accuracy for this epoch
        avg_loss = running_loss / len(data_loader)
        accuracy = 100. * correct / total
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(accuracy)
        
        if verbose:
            print(f'Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}%')
        
        # Validation if val_loader is provided
        if val_loader is not None:
            from .evaluator import evaluate_model
            val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if verbose:
                print(f'Epoch {epoch+1}/{epochs} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
    
    if verbose:
        print('Training completed!')
    
    return model, history

def train_vae_model(model, optimizer, loss_function, train_loader, device, epochs=10):
    # Train the model
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        train_loader_with_progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for data in train_loader_with_progress:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(inputs)
            loss = loss_function(recon, inputs, mu, logvar)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_loader_with_progress.set_postfix(loss=f'{loss.item():.4f}')
        # Optionally print/return average loss per epoch
        # avg_loss = running_loss / max(1, len(train_loader))
        # print(f"Epoch {epoch+1}: avg loss {avg_loss:.4f}")
    print("Finished Training")

def vae_loss_function(recon_x, x, mu, logvar):
    beta = 500
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * BCE + KLD

def train_wgan(gen, critic, dataloader, device, z_dim=100, lr=5e-5, n_critic=5, clip_value=0.01, epochs=20):
    datalogs = []
    opt_gen = optim.RMSprop(gen.parameters(), lr=lr)
    opt_critic = optim.RMSprop(critic.parameters(), lr=lr)

    for epoch in range(epochs):
        train_loader_with_progress = tqdm(
            iterable=dataloader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        for batch_number, (real, _) in enumerate(train_loader_with_progress):
            real = real.to(device)
            batch_size = real.size(0)

            # === Train Critic ===
            for _ in range(n_critic):
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake = gen(noise).detach()
                critic_real = critic(real).mean()
                critic_fake = critic(fake).mean()
                loss_critic = -(critic_real - critic_fake)

                critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                # Weight clipping
                for p in critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            # === Train Generator ===
            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = gen(noise)
            loss_gen = -critic(fake).mean()

            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_number % 100 == 0:
                train_loader_with_progress.set_postfix(
                    {"Batch": f"{batch_number}/{len(dataloader)}",
                     "D loss": f"{loss_critic.item():.4f}",
                     "G loss": f"{loss_gen.item():.4f}"}
                )
                datalogs.append(
                    {"epoch": epoch + batch_number / len(dataloader),
                     "Batch": batch_number/len(dataloader),
                     "D loss": loss_critic.item(),
                     "G loss": loss_gen.item()}
                )

    return datalogs


def train_gan(generator, discriminator, dataloader, device, z_dim=100, lr=0.0002, beta1=0.5, epochs=20):
    """Train a GAN model on MNIST dataset.
    
    Args:
        generator: GANGenerator model
        discriminator: GANDiscriminator model
        dataloader: DataLoader for training data (MNIST)
        device: Device to train on ('cpu' or 'cuda')
        z_dim: Dimension of noise vector (default: 100)
        lr: Learning rate (default: 0.0002)
        beta1: Beta1 parameter for Adam optimizer (default: 0.5)
        epochs: Number of training epochs
    
    Returns:
        Dictionary containing training logs with losses
    """
    # Loss function
    criterion = torch.nn.BCELoss()
    
    # Optimizers
    opt_gen = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
    
    # Training logs
    logs = []
    
    generator.train()
    discriminator.train()
    
    for epoch in range(epochs):
        train_loader_with_progress = tqdm(
            iterable=dataloader, ncols=120, desc=f"Epoch {epoch+1}/{epochs}"
        )
        
        for batch_idx, (real_images, _) in enumerate(train_loader_with_progress):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # ===== Train Discriminator =====
            opt_disc.zero_grad()
            
            # Real images
            real_output = discriminator(real_images)
            loss_real = criterion(real_output, real_labels)
            
            # Fake images
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            loss_fake = criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            loss_disc = loss_real + loss_fake
            loss_disc.backward()
            opt_disc.step()
            
            # ===== Train Generator =====
            opt_gen.zero_grad()
            
            # Generate fake images and try to fool discriminator
            noise = torch.randn(batch_size, z_dim).to(device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images)
            loss_gen = criterion(fake_output, real_labels)  # Want discriminator to think they're real
            
            loss_gen.backward()
            opt_gen.step()
            
            # Update progress bar
            if batch_idx % 50 == 0:
                train_loader_with_progress.set_postfix({
                    "D_loss": f"{loss_disc.item():.4f}",
                    "G_loss": f"{loss_gen.item():.4f}"
                })
                
                logs.append({
                    "epoch": epoch + 1,
                    "batch": batch_idx,
                    "D_loss": loss_disc.item(),
                    "G_loss": loss_gen.item()
                })
    
    print("Finished Training GAN")
    return logs


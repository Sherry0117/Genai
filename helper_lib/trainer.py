from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import os


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


def train_diffusion(model, data_loader, criterion, optimizer, device='cpu', epochs=10, 
                   val_loader=None, checkpoint_dir='checkpoints'):
    """
    Train Diffusion model
    
    Args:
        model: DiffusionModel instance
        data_loader: Training data loader
        criterion: Loss function (typically MSELoss)
        optimizer: Optimizer
        device: Training device
        epochs: Number of training epochs
        val_loader: Validation data loader (optional)
        checkpoint_dir: Checkpoint save directory
    
    Returns:
        Trained model
    """
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move model to specified device
    model.to(device)
    
    # Calculate dataset mean and std for normalization
    if hasattr(model, 'normalizer_mean'):
        print("Using existing normalizer values")
    else:
        print("Computing dataset statistics for normalization...")
        all_images = []
        for images, _ in data_loader:
            all_images.append(images)
            if len(all_images) >= 10:  # Estimate using first 10 batches
                break
        all_images = torch.cat(all_images, dim=0)
        mean = all_images.mean()
        std = all_images.std()
        model.set_normalizer(mean, std)
        print(f"Normalizer set: mean={mean:.4f}, std={std:.4f}")
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        
        loader_with_progress = tqdm(data_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs}')
        
        for images, _ in loader_with_progress:
            images = images.to(device)
            
            # Use model's train_step method
            loss = model.train_step(images, optimizer, criterion)
            train_losses.append(loss)
            
            loader_with_progress.set_postfix(loss=f'{loss:.4f}')
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_losses = []
            
            for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", ncols=120):
                images = images.to(device)
                loss = model.test_step(images, criterion)
                val_losses.append(loss)
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_checkpoint_path = os.path.join(checkpoint_dir, 'diffusion_best.pth')
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.network.state_dict(),
                    'ema_model_state_dict': model.ema_network.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'normalizer_mean': model.normalizer_mean,
                    'normalizer_std': model.normalizer_std
                }
                
                torch.save(checkpoint, best_checkpoint_path)
                print(f"New best model saved with val_loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")
        
        # Save checkpoint every epoch
        checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_epoch_{epoch+1}.pth')
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.network.state_dict(),
            'ema_model_state_dict': model.ema_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'normalizer_mean': model.normalizer_mean,
            'normalizer_std': model.normalizer_std
        }
        
        if val_loader is not None:
            checkpoint['val_loss'] = avg_val_loss
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")
    
    return model


def load_diffusion_checkpoint(model, optimizer, checkpoint_path, device='cpu'):
    """
    Load saved checkpoint
    
    Args:
        model: DiffusionModel instance
        optimizer: Optimizer
        checkpoint_path: Checkpoint file path
        device: Device
    
    Returns:
        Starting epoch number
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.network.load_state_dict(checkpoint['model_state_dict'])
    model.ema_network.load_state_dict(checkpoint['ema_model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Restore normalizer settings
    model.normalizer_mean = checkpoint['normalizer_mean']
    model.normalizer_std = checkpoint['normalizer_std']
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Train Loss: {checkpoint['train_loss']:.4f}")
    
    if 'val_loss' in checkpoint:
        print(f"Val Loss: {checkpoint['val_loss']:.4f}")
    
    return checkpoint['epoch']


def train_energy_model(model, data_loader, optimizer, device='cuda', epochs=10):
    """
    Train Energy Model using Contrastive Divergence.
    
    The training objective is to minimize the energy of real images while maximizing
    the energy of generated (fake) images. This is achieved through:
    - Computing energy for real images from the dataset
    - Generating fake images using MCMC sampling (Langevin dynamics)
    - Computing energy for fake images
    - Loss = E(real) - E(fake) + regularization term
    
    Args:
        model: EnergyModel instance
        data_loader: DataLoader for real images (CIFAR-10 format: 3x32x32)
        optimizer: PyTorch optimizer (e.g., Adam)
        device: Training device ('cpu' or 'cuda')
        epochs: Number of training epochs
        
    Returns:
        Trained EnergyModel
    """
    model.train()
    model = model.to(device)
    
    # Training history
    history = {
        'real_energy': [],
        'fake_energy': [],
        'loss': []
    }
    
    print(f"Starting Energy Model training on {device}")
    print(f"Total epochs: {epochs}")
    
    for epoch in range(epochs):
        epoch_real_energy = 0.0
        epoch_fake_energy = 0.0
        epoch_loss = 0.0
        num_batches = 0
        
        # Progress bar for current epoch
        loader_with_progress = tqdm(data_loader, desc=f'Epoch {epoch+1}/{epochs}', ncols=120)
        
        for batch_idx, (real_images, _) in enumerate(loader_with_progress):
            real_images = real_images.to(device)
            batch_size = real_images.size(0)
            
            # Forward pass for real images
            real_energy = model(real_images)  # Shape: (batch_size,)
            avg_real_energy = real_energy.mean()
            
            # Generate fake images using MCMC sampling
            # Use fewer steps during training for efficiency
            fake_images = model.sample(
                num_samples=batch_size, 
                device=device, 
                num_steps=20,  # Fewer steps during training
                step_size=10
            )
            
            # Forward pass for fake images
            fake_energy = model(fake_images)  # Shape: (batch_size,)
            avg_fake_energy = fake_energy.mean()
            
            # Contrastive Divergence loss: E(real) - E(fake)
            # We want to minimize real energy and maximize fake energy
            cd_loss = avg_real_energy - avg_fake_energy
            
            # Add L2 regularization to prevent energy explosion
            l2_reg = 1e-4 * sum(p.pow(2).sum() for p in model.parameters())
            total_loss = cd_loss + l2_reg
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track metrics
            epoch_real_energy += avg_real_energy.item()
            epoch_fake_energy += avg_fake_energy.item()
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Update progress bar every 50 steps
            if batch_idx % 50 == 0:
                loader_with_progress.set_postfix({
                    'Real_E': f'{avg_real_energy.item():.4f}',
                    'Fake_E': f'{avg_fake_energy.item():.4f}',
                    'Loss': f'{total_loss.item():.4f}'
                })
        
        # Calculate epoch averages
        avg_real_energy_epoch = epoch_real_energy / max(num_batches, 1)
        avg_fake_energy_epoch = epoch_fake_energy / max(num_batches, 1)
        avg_loss_epoch = epoch_loss / max(num_batches, 1)
        
        # Store in history
        history['real_energy'].append(avg_real_energy_epoch)
        history['fake_energy'].append(avg_fake_energy_epoch)
        history['loss'].append(avg_loss_epoch)
        
        print(f"Epoch {epoch+1}/{epochs} - Real Energy: {avg_real_energy_epoch:.4f}, "
              f"Fake Energy: {avg_fake_energy_epoch:.4f}, Loss: {avg_loss_epoch:.4f}")
    
    print("Energy Model training completed!")
    return model, history


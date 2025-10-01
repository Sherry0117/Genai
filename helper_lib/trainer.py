from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim


def train_model(model, data_loader, criterion, optimizer, device='cpu',
                epochs=10):
    # TODO: run several iterations of the training loop (based on epochs
    # parameter) and return the model
    return model

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


<div align="center">

# 🎨 Generative Vision

### *GANs, Diffusion Models, VAEs*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1generative)

</div>

---

**Navigation:** [← 3D Vision](../14_3D_Vision/) | [🏠 Home](../README.md) | [Vision + Language →](../16_Vision_Language/)

---

## 📖 Table of Contents
- [Visual Overview](#-visual-overview)
- [Complete Colab Code](#-complete-colab-code)
- [VAE](#-variational-autoencoders)
- [GAN](#-generative-adversarial-networks)
- [Diffusion](#-diffusion-models)
- [Interview Q&A](#-interview-questions--answers)

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/generative_models.svg" alt="Generative Models" width="100%"/>
</div>

---

## 📓 Complete Colab Code

```python
#@title 🎨 Generative Models - Complete Tutorial
#@markdown VAE, GAN, Diffusion - All from scratch!

!pip install torch torchvision matplotlib numpy tqdm diffusers transformers accelerate -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Load MNIST for demos
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
mnist = datasets.MNIST('./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=128, shuffle=True)
print("📦 MNIST loaded!")

#@title 1️⃣ Variational Autoencoder (VAE)

class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(200, latent_dim)
        self.fc_var = nn.Linear(200, latent_dim)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Sigmoid(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_var(h)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z).view(-1, 1, 28, 28)
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    # Reconstruction loss
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), 
                                        (x.view(-1, 784) + 1) / 2, 
                                        reduction='sum')
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + kl_loss

# Train VAE
vae = VAE(latent_dim=20).to(device)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)

print("Training VAE...")
vae.train()
for epoch in range(3):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, log_var = vae(data)
        loss = vae_loss(recon, data, mu, log_var)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}: Loss = {total_loss/len(dataloader.dataset):.4f}")

# Generate samples
vae.eval()
with torch.no_grad():
    z = torch.randn(64, 20).to(device)
    samples = vae.decode(z)

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Generated samples
grid = make_grid(samples, nrow=8, normalize=True)
axes[0, 0].imshow(grid.cpu().permute(1, 2, 0))
axes[0, 0].set_title('VAE Generated Samples')
axes[0, 0].axis('off')

# Reconstructions
with torch.no_grad():
    sample_data = next(iter(dataloader))[0][:16].to(device)
    recon, _, _ = vae(sample_data)

grid_orig = make_grid(sample_data, nrow=4, normalize=True)
grid_recon = make_grid(recon, nrow=4, normalize=True)

axes[0, 1].imshow(grid_orig.cpu().permute(1, 2, 0))
axes[0, 1].set_title('Original')
axes[0, 1].axis('off')

axes[1, 0].imshow(grid_recon.cpu().permute(1, 2, 0))
axes[1, 0].set_title('Reconstructed')
axes[1, 0].axis('off')

# Latent space interpolation
with torch.no_grad():
    z1 = torch.randn(1, 20).to(device)
    z2 = torch.randn(1, 20).to(device)
    interp = []
    for alpha in np.linspace(0, 1, 8):
        z = (1 - alpha) * z1 + alpha * z2
        interp.append(vae.decode(z))
    interp = torch.cat(interp)
    grid_interp = make_grid(interp, nrow=8, normalize=True)
    axes[1, 1].imshow(grid_interp.cpu().permute(1, 2, 0))
    axes[1, 1].set_title('Latent Space Interpolation')
    axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
print("✅ VAE complete!")

#@title 2️⃣ GAN (Generative Adversarial Network)

class Generator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.model(z).view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Train GAN
latent_dim = 100
G = Generator(latent_dim).to(device)
D = Discriminator().to(device)
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
criterion = nn.BCELoss()

print("Training GAN...")
G_losses, D_losses = [], []

for epoch in range(5):
    for batch_idx, (real_imgs, _) in enumerate(dataloader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(device)
        
        # Labels
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        
        # Train Discriminator
        opt_D.zero_grad()
        real_loss = criterion(D(real_imgs), real_labels)
        
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = G(z)
        fake_loss = criterion(D(fake_imgs.detach()), fake_labels)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        opt_D.step()
        
        # Train Generator
        opt_G.zero_grad()
        g_loss = criterion(D(fake_imgs), real_labels)
        g_loss.backward()
        opt_G.step()
        
    G_losses.append(g_loss.item())
    D_losses.append(d_loss.item())
    print(f"  Epoch {epoch+1}: G_loss={g_loss.item():.4f}, D_loss={d_loss.item():.4f}")

# Generate samples
G.eval()
with torch.no_grad():
    z = torch.randn(64, latent_dim).to(device)
    fake_samples = G(z)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

grid = make_grid(fake_samples, nrow=8, normalize=True)
axes[0].imshow(grid.cpu().permute(1, 2, 0))
axes[0].set_title('GAN Generated Samples')
axes[0].axis('off')

# Loss curves
axes[1].plot(G_losses, label='Generator')
axes[1].plot(D_losses, label='Discriminator')
axes[1].legend()
axes[1].set_title('Training Loss')
axes[1].set_xlabel('Epoch')

# Noise interpolation
with torch.no_grad():
    z1 = torch.randn(1, latent_dim).to(device)
    z2 = torch.randn(1, latent_dim).to(device)
    interp = []
    for alpha in np.linspace(0, 1, 8):
        z = (1 - alpha) * z1 + alpha * z2
        interp.append(G(z))
    interp = torch.cat(interp)
    grid_interp = make_grid(interp, nrow=8, normalize=True)
    axes[2].imshow(grid_interp.cpu().permute(1, 2, 0))
    axes[2].set_title('GAN Interpolation')
    axes[2].axis('off')

plt.tight_layout()
plt.show()
print("✅ GAN complete!")

#@title 3️⃣ Simple Diffusion Model (DDPM)

class SimpleDiffusion(nn.Module):
    """Simplified denoising network"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784 + 1, 512),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
        )
    
    def forward(self, x, t):
        # t: timestep embedding (simple scalar)
        t = t.view(-1, 1)
        x_flat = x.view(-1, 784)
        return self.net(torch.cat([x_flat, t], dim=1)).view(-1, 1, 28, 28)

# Diffusion parameters
T = 100  # timesteps
betas = torch.linspace(1e-4, 0.02, T).to(device)
alphas = 1 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

def add_noise(x, t):
    """Add noise at timestep t"""
    sqrt_alpha_bar = torch.sqrt(alpha_bars[t]).view(-1, 1, 1, 1)
    sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bars[t]).view(-1, 1, 1, 1)
    noise = torch.randn_like(x)
    return sqrt_alpha_bar * x + sqrt_one_minus_alpha_bar * noise, noise

# Train diffusion
diffusion = SimpleDiffusion().to(device)
optimizer = optim.Adam(diffusion.parameters(), lr=1e-3)

print("Training Diffusion Model...")
for epoch in range(3):
    total_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        # Random timesteps
        t = torch.randint(0, T, (data.size(0),)).to(device)
        
        # Add noise
        noisy, noise = add_noise(data, t)
        
        # Predict noise
        pred_noise = diffusion(noisy, t.float() / T)
        
        # Loss
        loss = F.mse_loss(pred_noise, noise)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"  Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}")

# Sample from diffusion (simplified)
@torch.no_grad()
def sample_diffusion(model, n_samples=16):
    model.eval()
    x = torch.randn(n_samples, 1, 28, 28).to(device)
    
    for t in reversed(range(T)):
        t_tensor = torch.full((n_samples,), t, device=device).float() / T
        pred_noise = model(x, t_tensor)
        
        alpha = alphas[t]
        alpha_bar = alpha_bars[t]
        
        if t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
        
        x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_bar) * pred_noise)
        x = x + torch.sqrt(betas[t]) * noise
    
    return x

samples = sample_diffusion(diffusion, 64)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

grid = make_grid(samples, nrow=8, normalize=True)
axes[0].imshow(grid.cpu().permute(1, 2, 0), cmap='gray')
axes[0].set_title('Diffusion Generated Samples')
axes[0].axis('off')

# Show denoising process
with torch.no_grad():
    x = torch.randn(1, 1, 28, 28).to(device)
    steps = []
    for t in [99, 75, 50, 25, 10, 0]:
        x_copy = x.clone()
        for s in reversed(range(t+1)):
            t_tensor = torch.full((1,), s, device=device).float() / T
            pred_noise = diffusion(x_copy, t_tensor)
            if s > 0:
                x_copy = (1/torch.sqrt(alphas[s])) * (x_copy - (1-alphas[s])/torch.sqrt(1-alpha_bars[s]) * pred_noise)
                x_copy += torch.sqrt(betas[s]) * torch.randn_like(x_copy)
        steps.append(x_copy)
    
    grid_steps = make_grid(torch.cat(steps), nrow=6, normalize=True)
    axes[1].imshow(grid_steps.cpu().permute(1, 2, 0), cmap='gray')
    axes[1].set_title('Denoising Process: t=99 → t=0')
    axes[1].axis('off')

plt.tight_layout()
plt.show()
print("✅ Diffusion model complete!")

#@title 4️⃣ Stable Diffusion (Using diffusers library)

from diffusers import StableDiffusionPipeline
import torch

print("Loading Stable Diffusion (this may take a moment)...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipe.to(device)
    
    # Generate image
    prompt = "a beautiful sunset over mountains, digital art"
    image = pipe(prompt, num_inference_steps=20).images[0]
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f'Prompt: "{prompt}"')
    plt.axis('off')
    plt.show()
    print("✅ Stable Diffusion generation complete!")
    
except Exception as e:
    print(f"⚠️ Stable Diffusion skipped (requires GPU/memory): {e}")

#@title 5️⃣ Model Comparison

print("\n📊 Generative Model Comparison:")
print("="*70)
print(f"{'Model':<15} {'Quality':<12} {'Speed':<12} {'Training':<15} {'Mode Collapse'}")
print("="*70)
print(f"{'VAE':<15} {'Blurry':<12} {'Fast':<12} {'Stable':<15} {'No'}")
print(f"{'GAN':<15} {'Sharp':<12} {'Fast':<12} {'Unstable':<15} {'Yes'}")
print(f"{'Diffusion':<15} {'Best':<12} {'Slow':<12} {'Stable':<15} {'No'}")
print("="*70)

print("\n" + "="*50)
print("✅ ALL GENERATIVE MODELS COMPLETE!")
print("="*50)
```

---

## ❓ Interview Questions & Answers

### Q1: GAN vs Diffusion - pros/cons?
| GAN | Diffusion |
|-----|-----------|
| Fast sampling | Slow (1000 steps) |
| Mode collapse | Better diversity |
| Hard to train | Stable training |
| Sharp images | Best quality |

### Q2: What is mode collapse?
**Answer:** Generator produces limited variety of outputs that fool discriminator.

### Q3: How does classifier-free guidance work?
**Answer:** Train both conditional/unconditional, interpolate at inference.

### Q4: VAE vs GAN for generation?
| VAE | GAN |
|-----|-----|
| Blurry | Sharp |
| Stable | Unstable |
| Explicit latent | Implicit |

### Q5: What is latent diffusion?
**Answer:** Run diffusion in compressed VAE latent space (faster, smaller).

---

<div align="center">

**[← 3D Vision](../14_3D_Vision/) | [🏠 Home](../README.md) | [Vision + Language →](../16_Vision_Language/)**

</div>

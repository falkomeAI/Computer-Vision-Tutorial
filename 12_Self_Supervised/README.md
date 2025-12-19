<div align="center">

# 🔄 Self-Supervised Learning

### *SimCLR, DINO, MAE, Contrastive Learning*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ssl)

</div>

---

**Navigation:** [← Vision Transformers](../11_Vision_Transformers/) | [🏠 Home](../README.md) | [Video & Temporal →](../13_Video_Temporal/)

---

## 📖 Table of Contents
- [Visual Overview](#-visual-overview)
- [Complete Colab Code](#-complete-colab-code)
- [Contrastive Learning](#-contrastive-learning)
- [DINO](#-dino)
- [MAE](#-masked-autoencoders)
- [Interview Q&A](#-interview-questions--answers)

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/ssl_methods.svg" alt="SSL Methods" width="100%"/>
</div>

---

## 📓 Complete Colab Code

```python
#@title 🔄 Self-Supervised Learning - Complete Tutorial
#@markdown SimCLR, BYOL, DINO concepts + MAE!

!pip install torch torchvision matplotlib numpy lightly timm transformers -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# CIFAR-10 for demos
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])

cifar = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
dataloader = DataLoader(cifar, batch_size=128, shuffle=True, num_workers=2)
print("📦 CIFAR-10 loaded!")

#@title 1️⃣ SimCLR - Contrastive Learning

class SimCLRAugmentation:
    """Two random augmented views of same image"""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)

class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()
        # Backbone
        self.encoder = base_encoder
        self.encoder.fc = nn.Identity()
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)
        z = self.projection(h)
        return F.normalize(z, dim=1)

def info_nce_loss(z1, z2, temperature=0.5):
    """NT-Xent (InfoNCE) loss"""
    batch_size = z1.size(0)
    
    # Compute similarities
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.mm(z, z.t()) / temperature  # (2B, 2B)
    
    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -1e9)
    
    # Positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(batch_size)
    ]).to(z.device)
    
    loss = F.cross_entropy(sim, labels)
    return loss

# Demo SimCLR
print("Training SimCLR (simplified)...")
base_encoder = models.resnet18(weights=None)
base_encoder.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)  # For CIFAR
base_encoder.maxpool = nn.Identity()

simclr = SimCLR(base_encoder).to(device)
optimizer = optim.Adam(simclr.parameters(), lr=3e-4)

# Create augmented dataloader
aug_transform = SimCLRAugmentation()
cifar_aug = datasets.CIFAR10('./data', train=True, download=True, transform=aug_transform)
aug_loader = DataLoader(cifar_aug, batch_size=128, shuffle=True, drop_last=True)

losses = []
for epoch in range(2):
    total_loss = 0
    for (x1, x2), _ in tqdm(aug_loader, desc=f"Epoch {epoch+1}"):
        x1, x2 = x1.to(device), x2.to(device)
        
        optimizer.zero_grad()
        z1 = simclr(x1)
        z2 = simclr(x2)
        loss = info_nce_loss(z1, z2)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    losses.append(total_loss / len(aug_loader))
    print(f"  Epoch {epoch+1}: Loss = {losses[-1]:.4f}")

print("✅ SimCLR training complete!")

#@title 2️⃣ BYOL - Bootstrap Your Own Latent

class BYOL(nn.Module):
    """BYOL: No negative samples needed!"""
    def __init__(self, base_encoder, hidden_dim=256, proj_dim=128):
        super().__init__()
        
        # Online network
        self.online_encoder = base_encoder
        self.online_encoder.fc = nn.Identity()
        
        self.online_projector = nn.Sequential(
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )
        
        # Target network (EMA of online)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projector = copy.deepcopy(self.online_projector)
        
        # Freeze target
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False
    
    def forward(self, x1, x2):
        # Online
        z1_online = self.predictor(self.online_projector(self.online_encoder(x1)))
        z2_online = self.predictor(self.online_projector(self.online_encoder(x2)))
        
        # Target (no grad)
        with torch.no_grad():
            z1_target = self.target_projector(self.target_encoder(x1))
            z2_target = self.target_projector(self.target_encoder(x2))
        
        return z1_online, z2_online, z1_target.detach(), z2_target.detach()
    
    @torch.no_grad()
    def update_target(self, tau=0.99):
        """EMA update of target network"""
        for online, target in zip(self.online_encoder.parameters(), 
                                   self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data
        for online, target in zip(self.online_projector.parameters(), 
                                   self.target_projector.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data

def byol_loss(pred, target):
    """Cosine similarity loss"""
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 2 - 2 * (pred * target).sum(dim=-1).mean()

print("BYOL concept demo:")
print("  • No negative samples needed")
print("  • Target network = EMA of online network")
print("  • Prevents collapse via asymmetric architecture")
print("✅ BYOL explained!")

#@title 3️⃣ DINO - Self-Distillation with No Labels

class DINOHead(nn.Module):
    """DINO projection head with centering"""
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.last_layer = nn.Linear(out_dim, out_dim, bias=False)
        self.last_layer.weight.data.copy_(torch.eye(out_dim))
    
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)

print("DINO Key Concepts:")
print("="*50)
print("""
1. Self-distillation: Student learns from Teacher
2. Teacher = EMA of Student (no backprop)
3. Centering: Subtract moving average of teacher output
   → Prevents collapse to uniform distribution
4. Sharpening: Low temperature for teacher

Key insight: Emergent attention maps!
  • [CLS] token attention reveals semantic segmentation
  • No supervision needed
""")

# Visualize DINO attention (using pretrained)
print("\nLoading pretrained DINO for attention visualization...")
try:
    import timm
    dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
    dino_model.eval().to(device)
    
    # Get sample image
    test_img = datasets.CIFAR10('./data', train=False, download=True)
    img, label = test_img[0]
    
    # Prepare
    transform_dino = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform_dino(img).unsqueeze(0).to(device)
    
    # Get attention
    with torch.no_grad():
        attentions = dino_model.get_last_selfattention(img_tensor)
    
    # CLS attention to patches
    nh = attentions.shape[1]  # Number of heads
    cls_attn = attentions[0, :, 0, 1:].reshape(nh, 28, 28)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Input Image')
    axes[0, 0].axis('off')
    
    for i, ax in enumerate(axes.flatten()[1:7]):
        ax.imshow(cls_attn[i].cpu(), cmap='viridis')
        ax.set_title(f'Head {i+1}')
        ax.axis('off')
    
    # Mean attention
    mean_attn = cls_attn.mean(0)
    axes[1, 3].imshow(mean_attn.cpu(), cmap='viridis')
    axes[1, 3].set_title('Mean Attention')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    print("✅ DINO attention visualization complete!")

except Exception as e:
    print(f"⚠️ DINO visualization skipped: {e}")

#@title 4️⃣ MAE - Masked Autoencoders

class SimpleMAE(nn.Module):
    """Simplified MAE for understanding"""
    def __init__(self, img_size=32, patch_size=4, embed_dim=192, mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        # Encoder (only visible patches)
        self.patch_embed = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 4, embed_dim*4, batch_first=True),
            num_layers=4
        )
        
        # Decoder
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, 4, embed_dim*4, batch_first=True),
            num_layers=2
        )
        self.decoder_pred = nn.Linear(embed_dim, patch_size**2 * 3)
    
    def patchify(self, imgs):
        """imgs: (B, C, H, W) -> patches: (B, N, patch_size^2 * C)"""
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(imgs.shape[0], h*w, -1)
        return x
    
    def random_masking(self, x):
        """Random mask patches, return visible and mask"""
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        mask = torch.ones(N, L, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, imgs):
        # Patchify and embed
        x = self.patch_embed(imgs).flatten(2).transpose(1, 2)
        
        # Mask
        x_visible, mask, ids_restore = self.random_masking(x)
        
        # Encode visible
        x_encoded = self.encoder(x_visible)
        
        # Add mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x_encoded.shape[1], 1)
        x_full = torch.cat([x_encoded, mask_tokens], dim=1)
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        
        # Decode
        x_decoded = self.decoder(x_full)
        pred = self.decoder_pred(x_decoded)
        
        # Loss on masked patches
        target = self.patchify(imgs)
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss, pred, mask

# Train MAE
print("Training MAE...")
mae = SimpleMAE().to(device)
optimizer = optim.AdamW(mae.parameters(), lr=1e-3)

cifar_mae = datasets.CIFAR10('./data', train=True, download=True,
                             transform=transforms.ToTensor())
mae_loader = DataLoader(cifar_mae, batch_size=128, shuffle=True)

for epoch in range(3):
    total_loss = 0
    for imgs, _ in tqdm(mae_loader, desc=f"Epoch {epoch+1}"):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        loss, _, _ = mae(imgs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}: Loss = {total_loss/len(mae_loader):.4f}")

# Visualize reconstruction
mae.eval()
sample_imgs = next(iter(mae_loader))[0][:8].to(device)

with torch.no_grad():
    _, pred, mask = mae(sample_imgs)
    
# Unpatchify
def unpatchify(x, p=4):
    h = w = int(x.shape[1]**0.5)
    x = x.reshape(x.shape[0], h, w, p, p, 3)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(x.shape[0], 3, h*p, w*p)
    return x

pred_imgs = unpatchify(pred)

fig, axes = plt.subplots(3, 8, figsize=(16, 6))
for i in range(8):
    axes[0, i].imshow(sample_imgs[i].cpu().permute(1, 2, 0).clip(0, 1))
    axes[0, i].axis('off')
    if i == 0: axes[0, i].set_title('Original')
    
    # Masked view
    masked = sample_imgs[i].cpu().clone()
    m = mask[i].view(8, 8)
    for pi in range(8):
        for pj in range(8):
            if m[pi, pj] == 1:
                masked[:, pi*4:(pi+1)*4, pj*4:(pj+1)*4] = 0.5
    axes[1, i].imshow(masked.permute(1, 2, 0).clip(0, 1))
    axes[1, i].axis('off')
    if i == 0: axes[1, i].set_title('Masked (75%)')
    
    axes[2, i].imshow(pred_imgs[i].cpu().permute(1, 2, 0).clip(0, 1))
    axes[2, i].axis('off')
    if i == 0: axes[2, i].set_title('Reconstructed')

plt.tight_layout()
plt.show()
print("✅ MAE complete!")

#@title 5️⃣ Linear Probing (Evaluate Representations)

def linear_probe(encoder, train_loader, test_loader, num_classes=10):
    """Evaluate encoder with frozen backbone + linear classifier"""
    encoder.eval()
    
    # Extract features
    def get_features(loader):
        features, labels = [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                feat = encoder(imgs.to(device))
                features.append(feat.cpu())
                labels.append(lbls)
        return torch.cat(features), torch.cat(labels)
    
    train_feat, train_labels = get_features(train_loader)
    test_feat, test_labels = get_features(test_loader)
    
    # Linear classifier
    classifier = nn.Linear(train_feat.shape[1], num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    
    # Train
    for epoch in range(10):
        classifier.train()
        perm = torch.randperm(len(train_feat))
        for i in range(0, len(train_feat), 256):
            idx = perm[i:i+256]
            optimizer.zero_grad()
            logits = classifier(train_feat[idx].to(device))
            loss = F.cross_entropy(logits, train_labels[idx].to(device))
            loss.backward()
            optimizer.step()
    
    # Evaluate
    classifier.eval()
    with torch.no_grad():
        preds = classifier(test_feat.to(device)).argmax(1).cpu()
        acc = (preds == test_labels).float().mean()
    
    return acc.item()

print("\n📊 Linear Probing Results:")
print("-" * 40)

# Random baseline
random_encoder = models.resnet18(weights=None)
random_encoder.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
random_encoder.maxpool = nn.Identity()
random_encoder.fc = nn.Identity()
random_encoder.to(device)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])
train_probe = DataLoader(datasets.CIFAR10('./data', train=True, transform=test_transform), 
                         batch_size=256, shuffle=False)
test_probe = DataLoader(datasets.CIFAR10('./data', train=False, transform=test_transform), 
                        batch_size=256, shuffle=False)

random_acc = linear_probe(random_encoder, train_probe, test_probe)
print(f"  Random Init: {random_acc:.2%}")

simclr_acc = linear_probe(simclr.encoder, train_probe, test_probe)
print(f"  SimCLR (2 epochs): {simclr_acc:.2%}")

print("\n" + "="*50)
print("✅ ALL SSL METHODS COMPLETE!")
print("="*50)
```

---

## ❓ Interview Questions & Answers

### Q1: SimCLR vs BYOL - key difference?
| SimCLR | BYOL |
|--------|------|
| Needs negatives | No negatives |
| Large batch critical | Smaller batch OK |
| InfoNCE loss | MSE loss |
| Simpler | EMA teacher |

### Q2: How does DINO avoid collapse?
**Answer:**
1. **Centering**: Subtract EMA of teacher outputs
2. **Sharpening**: Low temperature for teacher
3. **Stop gradient**: Don't backprop through teacher

### Q3: Why does MAE work with 75% masking?
**Answer:** Forces learning of high-level semantics, not just local patterns. Low-level cues are insufficient with so much missing.

### Q4: What is the InfoNCE loss?
```python
loss = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```
Maximize positive pair similarity, minimize negatives.

### Q5: Contrastive vs Generative SSL?
| Contrastive | Generative |
|-------------|------------|
| Instance discrimination | Reconstruction |
| SimCLR, MoCo, BYOL | MAE, BEiT |
| Global features | Local+Global |

---

<div align="center">

**[← Vision Transformers](../11_Vision_Transformers/) | [🏠 Home](../README.md) | [Video & Temporal →](../13_Video_Temporal/)**

</div>

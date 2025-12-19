# Vision Transformer (ViT)

> **Level:** 🟠 Advanced | **Time:** 4 hours | **Prerequisites:** CNNs, Attention

---

**Navigation:** [🏠 Module Home](./README.md) | [DeiT →](./DeiT.md)

---

## 📋 Summary

Vision Transformer (ViT) applies the Transformer architecture to images by treating image patches as tokens. Instead of convolutions, ViT uses self-attention to capture global relationships. With sufficient data, ViT outperforms CNNs on image classification.

---

## 🔢 Key Formulas

### Patch Embedding
$$
\mathbf{z}_0 = [\mathbf{x}_\text{class}; \mathbf{x}_p^1\mathbf{E}; \mathbf{x}_p^2\mathbf{E}; \cdots; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_\text{pos}
$$

where $N = HW/P^2$ patches, $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$

### Multi-Head Self-Attention
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

### Transformer Block
$$
\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}
$$
$$
\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell
$$

---

## 🎨 Visual Diagram

<div align="center">
<img src="./svg_figs/vit_architecture.svg" alt="ViT Architecture" width="100%"/>
</div>

---

## 💻 Google Colab - Ready to Run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_vit)

```python
#@title 🔮 Vision Transformer (ViT) - Complete from Scratch
#@markdown Click **Runtime → Run all** to execute everything!

# ============================================
# SETUP
# ============================================
!pip install torch torchvision einops timm -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# ============================================
# 1. PATCH EMBEDDING
# ============================================
print("\n" + "="*50)
print("1️⃣ PATCH EMBEDDING")
print("="*50)

class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Conv2d is equivalent to: unfold → linear projection
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)           # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)           # (B, embed_dim, N)
        x = x.transpose(1, 2)      # (B, N, embed_dim)
        return x

# Test
patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
dummy_img = torch.randn(1, 3, 224, 224)
patches = patch_embed(dummy_img)
print(f"Input: {dummy_img.shape}")
print(f"Patches: {patches.shape}  # {(224//16)**2} = 196 patches")

# ============================================
# 2. MULTI-HEAD SELF-ATTENTION
# ============================================
print("\n" + "="*50)
print("2️⃣ MULTI-HEAD SELF-ATTENTION")
print("="*50)

class Attention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5  # 1/sqrt(d_k)
        
        # Single linear for Q, K, V (efficient!)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn  # Return attention for visualization

# Test
attn = Attention(dim=768, n_heads=12)
x = torch.randn(1, 197, 768)  # 196 patches + 1 CLS
out, attn_weights = attn(x)
print(f"Input: {x.shape}")
print(f"Output: {out.shape}")
print(f"Attention weights: {attn_weights.shape}  # (B, heads, N, N)")

# ============================================
# 3. MLP BLOCK
# ============================================
print("\n" + "="*50)
print("3️⃣ MLP BLOCK")
print("="*50)

class MLP(nn.Module):
    """MLP with GELU activation"""
    def __init__(self, dim, hidden_dim=None, drop=0.):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

mlp = MLP(dim=768)
print(f"MLP params: {sum(p.numel() for p in mlp.parameters()):,}")
print(f"Hidden dim: 768 × 4 = 3072")

# ============================================
# 4. TRANSFORMER BLOCK
# ============================================
print("\n" + "="*50)
print("4️⃣ TRANSFORMER BLOCK")
print("="*50)

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block"""
    def __init__(self, dim, n_heads, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), drop)
    
    def forward(self, x):
        # Pre-norm: LN → Attn → Add
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out
        
        # Pre-norm: LN → MLP → Add
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights

block = TransformerBlock(dim=768, n_heads=12)
x = torch.randn(1, 197, 768)
out, _ = block(x)
print(f"TransformerBlock: {x.shape} → {out.shape}")

# ============================================
# 5. COMPLETE VIT
# ============================================
print("\n" + "="*50)
print("5️⃣ COMPLETE VISION TRANSFORMER")
print("="*50)

class ViT(nn.Module):
    """Vision Transformer"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 n_classes=1000, embed_dim=768, depth=12, n_heads=12,
                 mlp_ratio=4., drop=0.):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop)
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, drop)
            for _ in range(depth)
        ])
        
        # Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x, return_attention=False):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        
        # Add position embedding
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks
        attn_weights = []
        for block in self.blocks:
            x, attn = block(x)
            attn_weights.append(attn)
        
        # Classification head
        x = self.norm(x)
        cls_out = x[:, 0]  # CLS token
        logits = self.head(cls_out)
        
        if return_attention:
            return logits, attn_weights
        return logits

# Create different ViT variants
def vit_tiny(num_classes=10):
    return ViT(img_size=32, patch_size=4, n_classes=num_classes,
               embed_dim=192, depth=12, n_heads=3)

def vit_small(num_classes=10):
    return ViT(img_size=32, patch_size=4, n_classes=num_classes,
               embed_dim=384, depth=12, n_heads=6)

def vit_base(num_classes=1000):
    return ViT(img_size=224, patch_size=16, n_classes=num_classes,
               embed_dim=768, depth=12, n_heads=12)

# Test
model = vit_tiny(num_classes=10).to(device)
x = torch.randn(1, 3, 32, 32).to(device)
out = model(x)
print(f"\nViT-Tiny for CIFAR:")
print(f"  Input: {x.shape}")
print(f"  Output: {out.shape}")
print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# 6. TRAIN ON CIFAR-10
# ============================================
print("\n" + "="*50)
print("6️⃣ TRAINING ON CIFAR-10")
print("="*50)

# Data with augmentation
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])

trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
testset = datasets.CIFAR10('./data', train=False, transform=transform_test)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, num_workers=2)

# Model, optimizer
model = vit_tiny(num_classes=10).to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
criterion = nn.CrossEntropyLoss()

# Training
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(trainloader):.4f}, Acc: {100*correct/total:.2f}%")

# Test
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

print(f"\n🎯 Test Accuracy: {100*correct/total:.2f}%")

# ============================================
# 7. ATTENTION VISUALIZATION
# ============================================
print("\n" + "="*50)
print("7️⃣ ATTENTION VISUALIZATION")
print("="*50)

# Get sample image
img, label = testset[0]
img_tensor = img.unsqueeze(0).to(device)

# Forward with attention
model.eval()
with torch.no_grad():
    _, attn_weights = model(img_tensor, return_attention=True)

# Get attention from last layer (CLS token attending to patches)
attn = attn_weights[-1][0]  # (heads, N+1, N+1)
cls_attn = attn[:, 0, 1:].mean(0)  # Average over heads, CLS to patches

# Reshape to image grid
n_patches = int(cls_attn.shape[0] ** 0.5)
attn_map = cls_attn.reshape(n_patches, n_patches).cpu()

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

# Original image
axes[0].imshow(img.permute(1, 2, 0).numpy() * 0.5 + 0.5)
axes[0].set_title('Input Image')
axes[0].axis('off')

# Attention map
axes[1].imshow(attn_map, cmap='viridis')
axes[1].set_title('CLS Attention')
axes[1].axis('off')

# Overlay (upsample attention to image size)
attn_resized = F.interpolate(attn_map.unsqueeze(0).unsqueeze(0), 
                             size=(32, 32), mode='bilinear')[0, 0]
axes[2].imshow(img.permute(1, 2, 0).numpy() * 0.5 + 0.5)
axes[2].imshow(attn_resized.numpy(), cmap='jet', alpha=0.5)
axes[2].set_title('Attention Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("✅ ViT TUTORIAL COMPLETE!")
print("="*50)
```

---

## 📊 ViT Variants

| Model | Layers | Hidden | Heads | Params | ImageNet |
|-------|--------|--------|-------|--------|----------|
| ViT-Tiny | 12 | 192 | 3 | 5.7M | 72.2% |
| ViT-Small | 12 | 384 | 6 | 22M | 79.9% |
| ViT-Base | 12 | 768 | 12 | 86M | 77.9% |
| ViT-Large | 24 | 1024 | 16 | 307M | 76.5% |

---

## ⚠️ Common Pitfalls

| Mistake | Solution |
|---------|----------|
| ViT needs huge data | Use pretrained or DeiT distillation |
| Position embedding size mismatch | Interpolate for different resolutions |
| Training instability | Use AdamW, warmup, lower LR |
| No inductive bias | Add conv stem (hybrid ViT) |

---

## 🛠️ Mini-Project: ViT Attention Analysis

**Goal:** Visualize what ViT "sees" at different layers

```python
# Exercise: Extract attention maps from all 12 layers
# Compare early vs late layer attention patterns
# Find which heads focus on different semantic regions
```

---

## ❓ Interview Questions

### Q1: Why does ViT need more data than CNNs?
**Answer:** ViT lacks inductive biases (locality, translation equivariance). It must learn spatial relationships from scratch, requiring more data. With JFT-300M pretraining, ViT surpasses CNNs.

### Q2: What is the role of the [CLS] token?
**Answer:** A learnable token prepended to patches that aggregates global information through self-attention. The final [CLS] representation is used for classification.

### Q3: How does position embedding work in ViT?
**Answer:** Learnable 1D position embeddings added to patch embeddings. Can be interpolated for different input sizes. Unlike sinusoidal, learned embeddings can adapt to data.

---

## 📚 Further Reading

- [Original Paper: An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
- [Annotated ViT](https://github.com/lucidrains/vit-pytorch)
- [HuggingFace ViT](https://huggingface.co/docs/transformers/model_doc/vit)

---

<div align="center">

**[🏠 Module Home](./README.md) | [DeiT →](./DeiT.md)**

</div>


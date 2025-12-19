# 🔮 Vision Transformers

> **Level:** 🟠 Advanced | **Prerequisites:** CNNs, Attention Mechanisms

---

**Navigation:** [← Vision Tasks](../10_Vision_Tasks/) | [🏠 Home](../README.md) | [Self-Supervised Learning →](../12_Self_Supervised/)

---


## 📋 Summary

Vision Transformers (ViT) apply the Transformer architecture—originally designed for NLP—to images by treating image patches as tokens. This module covers **ViT fundamentals** (patch embedding, position encoding, [CLS] token), **hierarchical variants** (Swin Transformer with window attention), **efficient designs** (DeiT, PVT), and **self-supervised approaches** (DINO, MAE). ViTs have achieved state-of-the-art results across vision tasks when trained with sufficient data.

---

## 📊 Key Concepts Table

| Model | Year | Key Innovation | ImageNet Acc | Speed |
|-------|------|----------------|--------------|-------|
| **ViT-B/16** | 2020 | Patch tokens + Transformer | 77.9% | Medium |
| **DeiT-S** | 2021 | Knowledge distillation | 79.8% | Fast |
| **Swin-T** | 2021 | Window attention + shift | 81.3% | Fast |
| **PVT** | 2021 | Pyramid + spatial reduction | 79.8% | Fast |
| **BEiT** | 2021 | BERT-style pretraining | 83.2% | Medium |
| **MAE** | 2022 | Masked autoencoder | 83.6% | Slow train |
| **DINOv2** | 2023 | Self-supervised foundation | 86.5% | Medium |

---

## 🔢 Math / Formulas

### Patch Embedding
$$
\mathbf{z}_0 = [\mathbf{x}_\text{class}; \mathbf{x}_p^1\mathbf{E}; \mathbf{x}_p^2\mathbf{E}; \cdots; \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_\text{pos}
$$
where $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is patch projection, $N = HW/P^2$ patches

### Multi-Head Self-Attention
$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$$
\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

### Transformer Block
$$
\mathbf{z}'_\ell = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}
$$
$$
\mathbf{z}_\ell = \text{MLP}(\text{LN}(\mathbf{z}'_\ell)) + \mathbf{z}'_\ell
$$

### Window Attention (Swin)
$$
\text{Attention}(Q, K, V) = \text{SoftMax}(QK^T/\sqrt{d} + B)V
$$
where $B$ is relative position bias

---

## 🎨 Visual / Diagram

<div align="center">
<img src="./svg_figs/vit_architecture.svg" alt="ViT Architecture" width="100%"/>
</div>

---

## 💻 Code Practice

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vit)

```python
#@title 🔮 Vision Transformers - Complete Implementation
#@markdown Build ViT from scratch + Attention visualization!

!pip install torch torchvision matplotlib einops timm transformers -q

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import urllib.request

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Download sample image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
urllib.request.urlretrieve(url, "test.jpg")
image = Image.open("test.jpg")
print("📷 Image loaded!")

#@title 1️⃣ Patch Embedding
class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Equivalent to: split into patches + linear projection
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                              kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)           # (B, embed_dim, H/P, W/P)
        x = x.flatten(2)           # (B, embed_dim, N)
        x = x.transpose(1, 2)      # (B, N, embed_dim)
        return x

# Demo
patch_embed = PatchEmbedding(img_size=224, patch_size=16, embed_dim=768)
dummy_img = torch.randn(1, 3, 224, 224)
patches = patch_embed(dummy_img)
print(f"Input: {dummy_img.shape}")
print(f"Patches: {patches.shape}  # {224//16}×{224//16} = 196 patches")

#@title 2️⃣ Multi-Head Self-Attention
class Attention(nn.Module):
    """Multi-Head Self-Attention"""
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
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
        
        # Apply to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x, attn

# Demo
attn = Attention(dim=768, n_heads=12)
x = torch.randn(1, 197, 768)  # 196 patches + 1 CLS token
out, attn_weights = attn(x)
print(f"Attention input: {x.shape}")
print(f"Attention output: {out.shape}")
print(f"Attention weights: {attn_weights.shape}  # (B, heads, N, N)")

#@title 3️⃣ Transformer Block
class TransformerBlock(nn.Module):
    """Transformer Encoder Block"""
    def __init__(self, dim, n_heads, mlp_ratio=4., drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, n_heads=n_heads, attn_drop=drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        # Pre-norm architecture
        attn_out, attn_weights = self.attn(self.norm1(x))
        x = x + attn_out  # Residual
        x = x + self.mlp(self.norm2(x))  # Residual
        return x, attn_weights

#@title 4️⃣ Complete ViT
class ViT(nn.Module):
    """Vision Transformer"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 n_classes=1000, embed_dim=768, depth=12, n_heads=12):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        # Learnable tokens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, N, D)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)
        
        # Add position embedding
        x = x + self.pos_embed
        
        # Transformer blocks
        attn_weights_list = []
        for block in self.blocks:
            x, attn = block(x)
            attn_weights_list.append(attn)
        
        # Classification head
        x = self.norm(x)
        cls_out = x[:, 0]  # CLS token output
        return self.head(cls_out), attn_weights_list

# Create model
vit = ViT(img_size=224, patch_size=16, n_classes=1000, 
          embed_dim=384, depth=6, n_heads=6).to(device)

print(f"\n📊 ViT Model Summary:")
print(f"   Parameters: {sum(p.numel() for p in vit.parameters()):,}")
print(f"   Patch size: 16×16")
print(f"   Number of patches: {(224//16)**2} = 196")
print(f"   Embedding dim: 384")
print(f"   Transformer blocks: 6")

#@title 5️⃣ Attention Visualization
def visualize_attention(image, model):
    """Visualize attention from CLS token to patches"""
    # Prepare image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        _, attn_list = model(img_tensor)
    
    # Get attention from last layer, average over heads
    attn = attn_list[-1][0]  # (heads, N+1, N+1)
    cls_attn = attn[:, 0, 1:].mean(0)  # CLS to patches, avg heads
    
    # Reshape to 2D
    n_patches = int(cls_attn.shape[0] ** 0.5)
    cls_attn = cls_attn.reshape(n_patches, n_patches).cpu()
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(cls_attn, cmap='viridis')
    axes[1].set_title('CLS Attention Map')
    axes[1].axis('off')
    
    # Overlay
    attn_resized = torch.nn.functional.interpolate(
        cls_attn.unsqueeze(0).unsqueeze(0), 
        size=image.size[::-1], 
        mode='bilinear'
    ).squeeze().numpy()
    
    axes[2].imshow(image)
    axes[2].imshow(attn_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('Attention Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_attention(image, vit)
print("✅ Attention visualization complete!")

#@title 6️⃣ Pretrained ViT (HuggingFace)
from transformers import ViTImageProcessor, ViTForImageClassification

print("\n🤗 Loading pretrained ViT from HuggingFace...")
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
pretrained_vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
pretrained_vit.eval()

# Inference
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = pretrained_vit(**inputs)
    logits = outputs.logits

predicted_class = logits.argmax(-1).item()
confidence = F.softmax(logits, dim=-1).max().item()

print(f"   Predicted: {pretrained_vit.config.id2label[predicted_class]}")
print(f"   Confidence: {confidence:.2%}")

#@title 7️⃣ Model Comparison
print("\n📊 ViT Model Variants:")
print("-" * 60)
print(f"{'Model':<15} {'Layers':<10} {'Dim':<10} {'Heads':<10} {'Params':<12}")
print("-" * 60)
variants = [
    ("ViT-Tiny", 12, 192, 3, "5.7M"),
    ("ViT-Small", 12, 384, 6, "22M"),
    ("ViT-Base", 12, 768, 12, "86M"),
    ("ViT-Large", 24, 1024, 16, "307M"),
    ("Swin-Tiny", "[2,2,6,2]", 96, "[3,6,12,24]", "28M"),
    ("DeiT-Small", 12, 384, 6, "22M"),
]
for name, layers, dim, heads, params in variants:
    print(f"{name:<15} {str(layers):<10} {str(dim):<10} {str(heads):<10} {params:<12}")

print("\n" + "="*50)
print("✅ Vision Transformers Complete!")
print("="*50)
```

---

## ⚠️ Common Pitfalls / Tips

| Pitfall | Solution |
|---------|----------|
| ViT needs huge data | Use pretrained models or DeiT distillation |
| Position embedding size mismatch | Interpolate when using different resolutions |
| Memory issues with global attention | Use Swin's window attention or PVT |
| CLS vs average pooling | CLS works well for classification, GAP for dense tasks |
| Slow training | Use mixed precision, gradient checkpointing |

---

## 🛠️ Mini-Project Ideas

### Project 1: ViT from Scratch (Advanced)
- Implement full ViT architecture
- Train on CIFAR-10/100
- Compare with CNN baseline

### Project 2: Attention Visualization Tool (Advanced)
- Extract attention maps from different layers/heads
- Build interactive visualization
- Analyze what ViT "sees" vs CNN

### Project 3: Fine-tune for Custom Task (Advanced)
- Fine-tune pretrained ViT on your dataset
- Compare different fine-tuning strategies
- Measure transfer learning effectiveness

---

## ❓ Interview Questions & Answers

### Q1: ViT vs CNN - key differences?

| ViT | CNN |
|-----|-----|
| Global attention (all tokens interact) | Local receptive field |
| Learned position encoding | Inherent spatial inductive bias |
| Needs more data/compute | Data efficient |
| O(N²) attention complexity | O(N) conv complexity |
| Better scaling with data | Saturates earlier |

### Q2: What is the [CLS] token and why use it?

**Answer:**
- Learnable token prepended to patch sequence
- Aggregates global information through self-attention
- Final [CLS] representation used for classification
- Alternative: Global average pooling of all patch tokens

### Q3: How does Swin Transformer reduce complexity?

**Answer:**
1. **Window attention**: Compute attention only within M×M windows → O(M²N) vs O(N²)
2. **Shifted windows**: Alternate between regular and shifted partitions for cross-window connections
3. **Hierarchical**: Patch merging creates multi-scale features like CNN

### Q4: Why does ViT need more data than CNN?

**Answer:**
- No inductive bias for locality or translation equivariance
- Must learn spatial relationships from scratch
- CNNs have built-in local processing via convolution
- With enough data (JFT-300M), ViT outperforms CNNs

### Q5: Explain MAE (Masked Autoencoder) pretraining.

**Answer:**
1. **Mask 75%** of image patches randomly
2. **Encode only visible patches** (efficient!)
3. Add mask tokens and **decode to reconstruct pixels**
4. **MSE loss** on masked patches only

Key insight: High masking ratio forces learning semantic representations, not just local patterns.

---

## 📚 References / Further Reading

### Original Papers
- ViT: "An Image is Worth 16×16 Words" (Dosovitskiy 2020)
- DeiT: "Training Data-Efficient Image Transformers" (Touvron 2021)
- Swin: "Swin Transformer: Hierarchical Vision Transformer" (Liu 2021)
- MAE: "Masked Autoencoders Are Scalable Vision Learners" (He 2022)
- DINO: "Emerging Properties in Self-Supervised Vision Transformers" (Caron 2021)

### Online Resources
- [HuggingFace ViT](https://huggingface.co/docs/transformers/model_doc/vit)
- [timm Vision Models](https://github.com/huggingface/pytorch-image-models)
- [Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)

---

<div align="center">

**[← Vision Tasks](../10_Vision_Tasks/) | [🏠 Home](../README.md) | [Self-Supervised Learning →](../12_Self_Supervised/)**

</div>

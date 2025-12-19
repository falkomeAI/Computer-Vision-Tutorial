# ResNet & Skip Connections

> **Level:** 🟡 Intermediate | **Time:** 3 hours | **Prerequisites:** CNN basics

---

**Navigation:** [← Classic CNNs](./Classic_CNNs.md) | [🏠 Module Home](./README.md) | [EfficientNet →](./EfficientNet.md)

---

## 📋 Summary

ResNet (Residual Networks) introduced **skip connections** that revolutionized deep learning by enabling training of very deep networks (100+ layers). The key insight: learning residual functions F(x) = H(x) - x is easier than learning H(x) directly.

---

## 🔢 Key Formulas

### Residual Learning
$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

### Gradient Flow (why it works)
$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{y}} \cdot \left(1 + \frac{\partial \mathcal{F}}{\partial \mathbf{x}}\right)
$$

The **+1** ensures gradients flow directly through the skip connection!

---

## 🎨 Visual Diagram

<div align="center">
<img src="./svg_figs/resnet_block.svg" alt="ResNet Block" width="100%"/>
</div>

---

## 💻 Google Colab - Ready to Run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_resnet)

```python
#@title 🧠 ResNet from Scratch - Complete Tutorial
#@markdown Click **Runtime → Run all** to execute everything!

# ============================================
# SETUP
# ============================================
!pip install torch torchvision matplotlib -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# ============================================
# 1. BASIC RESIDUAL BLOCK
# ============================================
print("\n" + "="*50)
print("1️⃣ BASIC RESIDUAL BLOCK")
print("="*50)

class BasicBlock(nn.Module):
    """ResNet Basic Block for ResNet-18/34"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # First conv: may downsample (stride > 1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 
                                stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second conv: always stride=1
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (identity or projection)
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # Main path: Conv → BN → ReLU → Conv → BN
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Shortcut path (may need dimension matching)
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Skip connection: add identity!
        out += identity  # ← THE KEY INNOVATION!
        out = F.relu(out)
        
        return out

# Test basic block
block = BasicBlock(64, 64)
x = torch.randn(1, 64, 32, 32)
print(f"Input: {x.shape}")
print(f"Output: {block(x).shape}")
print(f"Parameters: {sum(p.numel() for p in block.parameters()):,}")

# ============================================
# 2. BOTTLENECK BLOCK (ResNet-50+)
# ============================================
print("\n" + "="*50)
print("2️⃣ BOTTLENECK BLOCK (ResNet-50+)")
print("="*50)

class Bottleneck(nn.Module):
    """ResNet Bottleneck Block for ResNet-50/101/152"""
    expansion = 4  # Output channels = input × 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        # 1×1 reduce
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 3×3 process
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                                stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 1×1 expand
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                                1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.downsample = downsample
        
    def forward(self, x):
        identity = x
        
        # 1×1 → 3×3 → 1×1 (bottleneck structure)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out

# Compare parameters: BasicBlock vs Bottleneck
basic = BasicBlock(256, 256)
bottle = Bottleneck(256, 64)  # 64 → 64 → 256 (expansion=4)

print(f"BasicBlock (256→256→256): {sum(p.numel() for p in basic.parameters()):,} params")
print(f"Bottleneck (256→64→64→256): {sum(p.numel() for p in bottle.parameters()):,} params")

# ============================================
# 3. FULL ResNet-18 FROM SCRATCH
# ============================================
print("\n" + "="*50)
print("3️⃣ FULL ResNet-18 IMPLEMENTATION")
print("="*50)

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        # Stem: Conv → BN → ReLU → MaxPool
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 4 stages of residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Head: Global Average Pool → FC
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                         1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Stem
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Head
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# Create ResNet variants
def resnet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)

def resnet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

# Test
model = resnet18(num_classes=10).to(device)
x = torch.randn(1, 3, 224, 224).to(device)
out = model(x)
print(f"ResNet-18 output: {out.shape}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================
# 4. TRAIN ON CIFAR-10
# ============================================
print("\n" + "="*50)
print("4️⃣ TRAINING ON CIFAR-10")
print("="*50)

# Data
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])

trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
]))

trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=128, num_workers=2)

# Modified ResNet-18 for CIFAR-10 (32×32 images)
class ResNetCIFAR(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super().__init__()
        self.in_channels = 64
        
        # Smaller stem for 32×32 images
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = ResNetCIFAR(BasicBlock, [2, 2, 2, 2]).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Training loop (shortened for demo)
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
    
    scheduler.step()
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
# 5. VISUALIZE FEATURE MAPS
# ============================================
print("\n" + "="*50)
print("5️⃣ VISUALIZE FEATURE MAPS")
print("="*50)

# Get a sample image
img, label = testset[0]
img = img.unsqueeze(0).to(device)

# Hook to capture intermediate outputs
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

model.layer1.register_forward_hook(get_activation('layer1'))
model.layer2.register_forward_hook(get_activation('layer2'))
model.layer3.register_forward_hook(get_activation('layer3'))

# Forward pass
_ = model(img)

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Original image
axes[0].imshow(img.cpu().squeeze().permute(1, 2, 0).numpy() * 0.5 + 0.5)
axes[0].set_title('Input Image')

# Feature maps from different layers
for ax, (name, act) in zip(axes[1:], activations.items()):
    # Show mean across channels
    feature_map = act[0].mean(0).cpu().numpy()
    ax.imshow(feature_map, cmap='viridis')
    ax.set_title(f'{name}: {act.shape[1]} channels')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("✅ ResNet TUTORIAL COMPLETE!")
print("="*50)
```

---

## 📊 ResNet Architecture Variants

| Model | Layers | Params | Top-1 Acc |
|-------|--------|--------|-----------|
| ResNet-18 | 18 | 11.7M | 69.8% |
| ResNet-34 | 34 | 21.8M | 73.3% |
| ResNet-50 | 50 | 25.6M | 76.1% |
| ResNet-101 | 101 | 44.5M | 77.4% |
| ResNet-152 | 152 | 60.2M | 78.3% |

---

## ⚠️ Common Pitfalls

| Mistake | Solution |
|---------|----------|
| Vanishing gradients in deep networks | Use skip connections (ResNet) |
| Dimension mismatch in skip | Use 1×1 conv for projection |
| Not using BatchNorm | Always BN after conv, before ReLU |
| Learning rate too high | Start with 0.1, decay at plateaus |

---

## 🛠️ Mini-Project: ResNet Ablation Study

**Goal:** Compare ResNet with/without skip connections

```python
# Exercise: Implement PlainNet (ResNet without skip connections)
# Compare training on CIFAR-10 at depths: 18, 34, 56 layers
# Plot training curves and observe degradation problem
```

---

## ❓ Interview Questions

### Q1: Why do skip connections help training?
**Answer:** 
1. Gradient flows directly (the +1 term)
2. Easy to learn identity (F(x)=0)
3. Acts like ensemble of shallow networks
4. Prevents degradation problem

### Q2: BasicBlock vs Bottleneck - when to use which?
**Answer:** BasicBlock for shallower networks (ResNet-18/34), Bottleneck for deeper (ResNet-50+). Bottleneck uses 1×1→3×3→1×1 to reduce computation while maintaining accuracy.

### Q3: Why pre-activation ResNet (BN→ReLU→Conv) is better?
**Answer:** Cleaner forward signal, better gradient flow, improved regularization. Identity mappings are more direct.

---

## 📚 Further Reading

- [Original Paper: Deep Residual Learning](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)
- [PyTorch ResNet Implementation](https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py)

---

<div align="center">

**[← Classic CNNs](./Classic_CNNs.md) | [🏠 Module Home](./README.md) | [EfficientNet →](./EfficientNet.md)**

</div>


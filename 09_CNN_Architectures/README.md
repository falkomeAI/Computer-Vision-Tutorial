# 🧠 CNN Architectures

> **Level:** 🟡 Intermediate | **Prerequisites:** Neural Networks, Linear Algebra

---

**Navigation:** [← Neural Networks](../08_Neural_Networks/) | [🏠 Home](../README.md) | [Vision Tasks →](../10_Vision_Tasks/)

---


## 📋 Summary

Convolutional Neural Networks revolutionized computer vision starting with LeNet (1998) and AlexNet (2012). This module covers the evolution of CNN architectures from **LeNet → AlexNet → VGG → ResNet → DenseNet → EfficientNet → ConvNeXt**. You'll learn the key innovations at each stage: ReLU, dropout, skip connections, batch normalization, depthwise separable convolutions, and compound scaling.

---

## 📊 Key Concepts Table

| Architecture | Year | Key Innovation | Parameters | Top-1 Acc |
|--------------|------|----------------|------------|-----------|
| **LeNet-5** | 1998 | First CNN | 60K | - |
| **AlexNet** | 2012 | ReLU, Dropout, GPU | 60M | 63.3% |
| **VGG-16** | 2014 | 3×3 kernels only | 138M | 74.4% |
| **GoogLeNet** | 2014 | Inception modules | 7M | 74.8% |
| **ResNet-50** | 2015 | Skip connections | 25M | 76.0% |
| **DenseNet-121** | 2017 | Dense connections | 8M | 74.9% |
| **EfficientNet-B0** | 2019 | Compound scaling | 5M | 77.1% |
| **ConvNeXt-T** | 2022 | Modernized ResNet | 29M | 82.1% |

---

## 🔢 Math / Formulas

### Convolution Operation
$$
\text{Output}[i,j] = \sum_{m}\sum_{n} \text{Input}[i+m, j+n] \cdot \text{Kernel}[m,n] + \text{bias}
$$

### Output Size Formula
$$
\text{Output Size} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$
where W=input size, K=kernel size, P=padding, S=stride

### Residual Connection (ResNet)
$$
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
$$

### Dense Connection (DenseNet)
$$
\mathbf{x}_\ell = H_\ell([\mathbf{x}_0, \mathbf{x}_1, ..., \mathbf{x}_{\ell-1}])
$$

### Compound Scaling (EfficientNet)
$$
\text{depth: } d = \alpha^\phi, \quad \text{width: } w = \beta^\phi, \quad \text{resolution: } r = \gamma^\phi
$$

---

## 🎨 Visual / Diagram

<div align="center">
<img src="./svg_figs/cnn_evolution.svg" alt="CNN Evolution" width="100%"/>
</div>

---

## 💻 Code Practice

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cnn_architectures)

```python
#@title 🧠 CNN Architectures - Complete Implementation
#@markdown Build LeNet → ResNet → ConvNeXt from scratch!

!pip install torch torchvision matplotlib timm -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
])
trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128)
print("📦 CIFAR-10 loaded!")

#@title 1️⃣ LeNet-5 (1998) - The Pioneer
class LeNet5(nn.Module):
    """LeNet-5: The original CNN architecture"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)      # 32→28
        self.pool = nn.AvgPool2d(2)           # 28→14
        self.conv2 = nn.Conv2d(6, 16, 5)     # 14→10, then pool→5
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    
    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)

lenet = LeNet5().to(device)
print(f"LeNet-5 parameters: {sum(p.numel() for p in lenet.parameters()):,}")

#@title 2️⃣ VGG Block (2014) - Deeper is Better
def make_vgg_block(in_channels, out_channels, num_convs):
    """VGG uses repeated 3×3 conv blocks"""
    layers = []
    for _ in range(num_convs):
        layers.extend([
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ])
        in_channels = out_channels
    layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class VGG11(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            make_vgg_block(3, 64, 1),
            make_vgg_block(64, 128, 1),
            make_vgg_block(128, 256, 2),
            make_vgg_block(256, 512, 2),
            make_vgg_block(512, 512, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

vgg = VGG11().to(device)
print(f"VGG-11 parameters: {sum(p.numel() for p in vgg.parameters()):,}")

#@title 3️⃣ ResNet (2015) - Skip Connections
class BasicBlock(nn.Module):
    """ResNet Basic Block with skip connection"""
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Shortcut for dimension matching
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # ← Skip connection!
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

resnet = ResNet18().to(device)
print(f"ResNet-18 parameters: {sum(p.numel() for p in resnet.parameters()):,}")

#@title 4️⃣ DenseNet Block (2017) - Dense Connections
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=False)
    
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)  # ← Concatenate!

# Demo
print("\nDenseNet connection demo:")
print("  Input: 64 channels")
print("  After 4 layers (growth_rate=32): 64 + 4×32 = 192 channels")

#@title 5️⃣ ConvNeXt Block (2022) - Modern CNN
class ConvNeXtBlock(nn.Module):
    """ConvNeXt: CNN modernized with Transformer ideas"""
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)  # Depthwise
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4*dim)  # Inverted bottleneck
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4*dim, dim)
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))  # Layer scale
    
    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW → NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
        return shortcut + x

print("\nConvNeXt modernizations:")
print("  • 7×7 depthwise convolution")
print("  • LayerNorm instead of BatchNorm")
print("  • GELU activation")
print("  • Inverted bottleneck (expand 4×)")
print("  • Layer scale initialization")

#@title 6️⃣ Train and Compare
def train_model(model, name, epochs=3):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
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
        
        print(f"  {name} Epoch {epoch+1}: Loss={total_loss/len(trainloader):.4f}, Acc={100*correct/total:.1f}%")
    
    # Test
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100 * correct / total

print("\n🏋️ Training models (3 epochs each)...")
results = {}
for name, model in [("LeNet", LeNet5()), ("ResNet-18", ResNet18())]:
    model = model.to(device)
    acc = train_model(model, name)
    results[name] = acc
    print(f"  {name} Test Accuracy: {acc:.1f}%\n")

# Visualize
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=['blue', 'orange'])
plt.ylabel('Test Accuracy (%)')
plt.title('Architecture Comparison (3 epochs)')
for i, (name, acc) in enumerate(results.items()):
    plt.text(i, acc + 1, f'{acc:.1f}%', ha='center')
plt.show()

#@title 7️⃣ Load Pretrained Models
print("\n📦 Loading pretrained models from timm:")
for model_name in ['resnet18', 'efficientnet_b0', 'convnext_tiny']:
    model = timm.create_model(model_name, pretrained=False)
    params = sum(p.numel() for p in model.parameters())
    print(f"  {model_name}: {params/1e6:.1f}M parameters")

print("\n" + "="*50)
print("✅ CNN Architectures Complete!")
print("="*50)
```

---

## ⚠️ Common Pitfalls / Tips

| Pitfall | Solution |
|---------|----------|
| Vanishing gradients in deep networks | Use skip connections (ResNet), BatchNorm |
| Overfitting on small datasets | Dropout, data augmentation, transfer learning |
| Wrong output size after conv | Use formula: (W-K+2P)/S + 1 |
| BatchNorm issues at inference | Set model.eval() before testing |
| GPU memory overflow | Reduce batch size, use gradient checkpointing |

---

## 🛠️ Mini-Project Ideas

### Project 1: Build Your Own Architecture (Intermediate)
- Design a custom CNN for CIFAR-10
- Experiment with kernel sizes, depths, skip connections
- Target: 85%+ accuracy with <1M parameters

### Project 2: Architecture Ablation Study (Intermediate)
- Compare ResNet vs VGG vs DenseNet on same task
- Measure accuracy, speed, memory usage
- Visualize feature maps and gradients

### Project 3: Transfer Learning Pipeline (Intermediate)
- Fine-tune pretrained models on custom dataset
- Compare different unfreezing strategies
- Implement learning rate scheduling

---

## ❓ Interview Questions & Answers

### Q1: Why do skip connections help training?

**Answer:**
1. **Gradient flow**: Gradients can bypass layers via shortcut
2. **Identity mapping**: Easy to learn if no transformation needed
3. **Ensemble effect**: Acts like ensemble of shallow networks
4. **Prevents degradation**: Accuracy doesn't drop with more layers

### Q2: What's the difference between 1×1, 3×3, and 7×7 convolutions?

| Kernel | Use Case |
|--------|----------|
| **1×1** | Channel mixing, dimension reduction |
| **3×3** | Standard spatial feature extraction |
| **7×7** | Large receptive field, stem layers |

### Q3: Explain depthwise separable convolution.

**Answer:** Splits standard conv into:
1. **Depthwise**: One filter per channel (spatial)
2. **Pointwise**: 1×1 conv (channel mixing)

**Cost reduction**: From O(K²×C_in×C_out) to O(K²×C_in + C_in×C_out)

### Q4: Why does EfficientNet use compound scaling?

**Answer:** Balances depth/width/resolution together:
- Deeper: More complex features
- Wider: More channels per layer
- Higher resolution: More fine-grained patterns

Scaling one alone gives diminishing returns.

### Q5: VGG vs ResNet - which is better and why?

| VGG | ResNet |
|-----|--------|
| Simpler design | Skip connections |
| 138M params | 25M params |
| Max 19 layers | 100+ layers |
| No skip connections | Better gradient flow |

**ResNet is better** because skip connections enable much deeper networks with fewer parameters.

---

## 📚 References / Further Reading

### Original Papers
- LeNet: "Gradient-Based Learning Applied to Document Recognition" (LeCun 1998)
- AlexNet: "ImageNet Classification with Deep CNNs" (Krizhevsky 2012)
- VGG: "Very Deep Convolutional Networks" (Simonyan 2014)
- ResNet: "Deep Residual Learning" (He 2015)
- DenseNet: "Densely Connected CNNs" (Huang 2017)
- EfficientNet: "Rethinking Model Scaling" (Tan 2019)
- ConvNeXt: "A ConvNet for the 2020s" (Liu 2022)

### Online Resources
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
- [timm (PyTorch Image Models)](https://github.com/huggingface/pytorch-image-models)
- [Papers With Code - Image Classification](https://paperswithcode.com/task/image-classification)

---

<div align="center">

**[← Neural Networks](../08_Neural_Networks/) | [🏠 Home](../README.md) | [Vision Tasks →](../10_Vision_Tasks/)**

</div>

<div align="center">

# 👁️ Computer Vision Complete

**Learn Computer Vision from Zero to Hero**

<br/>

```
📚 20 Modules  •  🎨 46 Diagrams  •  💻 Ready-to-Run Code
```

<br/>

[Get Started](#-quick-start) • [View Modules](#-all-modules) • [Learning Path](#-learning-path)

</div>

---

## 🎯 What You'll Learn

| Level | Topics | Time |
|:-----:|--------|:----:|
| 🟢 **Beginner** | Math, Image Processing, Filters, Features | 4 weeks |
| 🟡 **Intermediate** | CNNs, Detection, Segmentation | 6 weeks |
| 🟠 **Advanced** | Transformers, GANs, 3D Vision | 4 weeks |
| 🔴 **Research** | CLIP, Diffusion, Foundation Models | 2 weeks |

---

## 🚀 Quick Start

**1. Pick a module** → Click any link below

**2. Read the README** → Each module has explanations + diagrams

**3. Run the code** → Copy to Google Colab and run!

---

## 📚 All Modules

### 🟢 Foundations (Weeks 1-4)

| # | Module | What You'll Learn |
|:-:|--------|-------------------|
| 01 | [**Math Foundations**](./01_Mathematical_Foundations/) | Vectors, matrices, calculus, probability |
| 02 | [**Transforms**](./02_Transform_Methods/) | Fourier, wavelets, DCT, compression |
| 03 | [**Image Formation**](./03_Image_Formation/) | Cameras, sensors, color spaces |
| 04 | [**Image Processing**](./04_Low_Level_Processing/) | Filters, edges, histograms, noise |

### 🟡 Classical CV (Weeks 5-8)

| # | Module | What You'll Learn |
|:-:|--------|-------------------|
| 05 | [**Features**](./05_Features_Detection/) | SIFT, Harris, HOG, ORB |
| 06 | [**Geometry**](./06_Geometry_MultiView/) | Homography, stereo, 3D reconstruction |
| 07 | [**Classical ML**](./07_Classical_ML/) | PCA, SVM, clustering |
| 08 | [**Neural Networks**](./08_Neural_Networks/) | MLP, backprop, optimization |

### 🟡 Deep Learning (Weeks 9-12)

| # | Module | What You'll Learn |
|:-:|--------|-------------------|
| 09 | [**CNNs**](./09_CNN_Architectures/) | LeNet → ResNet → EfficientNet |
| 10 | [**Vision Tasks**](./10_Vision_Tasks/) | Classification, detection, segmentation |
| 11 | [**Transformers**](./11_Vision_Transformers/) | ViT, Swin, attention |
| 12 | [**Self-Supervised**](./12_Self_Supervised/) | SimCLR, DINO, MAE |

### 🟠 Advanced (Weeks 13-14)

| # | Module | What You'll Learn |
|:-:|--------|-------------------|
| 13 | [**Video**](./13_Video_Temporal/) | Optical flow, action recognition |
| 14 | [**3D Vision**](./14_3D_Vision/) | Depth, NeRF, point clouds |
| 15 | [**Generative**](./15_Generative_Vision/) | VAE, GAN, diffusion |
| 16 | [**Vision+Language**](./16_Vision_Language/) | CLIP, captioning, VQA |

### 🔴 Production & Research (Weeks 15-16)

| # | Module | What You'll Learn |
|:-:|--------|-------------------|
| 17 | [**Computational Photo**](./17_Computational_Photography/) | HDR, super-resolution |
| 18 | [**Deployment**](./18_Deployment_Systems/) | Quantization, ONNX, TensorRT |
| 19 | [**Ethics & Safety**](./19_Ethics_Safety/) | Adversarial attacks, fairness |
| 20 | [**Research**](./20_Research_Frontiers/) | SAM, foundation models |

---

## 🗺️ Learning Path

```
Start Here!
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  🟢 BEGINNER                                                │
│  Modules 1-4: Math → Transforms → Images → Processing       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  🟡 INTERMEDIATE                                            │
│  Modules 5-10: Features → Geometry → ML → CNNs → Tasks      │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  🟠 ADVANCED                                                │
│  Modules 11-16: Transformers → SSL → Video → 3D → Gen       │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  🔴 RESEARCH                                                │
│  Modules 17-20: Deploy → Ethics → Frontiers                 │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 What's Inside Each Module

```
Module_Name/
├── README.md      ← Explanations, formulas, code
├── svg_figs/      ← Visual diagrams
└── Topic.md       ← Detailed sub-topics (some modules)
```

---

## 🎨 Visual Diagrams

All modules include **clean, minimal SVG diagrams**:

<table>
<tr>
<td align="center"><b>Image Processing</b><br/>Filters, Edges, Color</td>
<td align="center"><b>Neural Networks</b><br/>CNN, ResNet, ViT</td>
<td align="center"><b>Advanced</b><br/>NeRF, Diffusion, CLIP</td>
</tr>
</table>

---

## 💻 Code Examples

Every module has **ready-to-run Python code**:

```python
# Just copy to Google Colab and run!
import torch
import torchvision

model = torchvision.models.resnet50(pretrained=True)
# ... full examples in each module
```

---

## 📊 Key Formulas

<table>
<tr>
<td>

**Convolution**
```
(f * g)[n] = Σ f[m] · g[n-m]
```

</td>
<td>

**Attention**
```
Attn(Q,K,V) = softmax(QKᵀ/√d)V
```

</td>
<td>

**Cross-Entropy**
```
L = -Σ yᵢ log(ŷᵢ)
```

</td>
</tr>
</table>

---

## ⭐ Tips for Success

1. **Follow the order** — Modules build on each other
2. **Run the code** — Learning by doing works best
3. **Look at diagrams** — They explain complex concepts visually
4. **Practice Q&A** — Each module has interview questions

---

<div align="center">

### Ready to Start?

**[👉 Begin with Module 1: Math Foundations](./01_Mathematical_Foundations/)**

---

Made with ❤️ for the CV community

**Star ⭐ this repo if you find it helpful!**

</div>

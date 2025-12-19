<div align="center">

# 🎬 Video & Temporal Vision

### *Optical Flow, Action Recognition, Tracking*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/intermediate_source/video_api_tutorial.ipynb)

</div>

---

**Navigation:** [← Self-Supervised Learning](../12_Self_Supervised/) | [🏠 Home](../README.md) | [3D Vision →](../14_3D_Vision/)

---

## 📖 Topics Covered
- Optical Flow
- Action Recognition
- Video Classification
- Object Tracking
- Video Transformers

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/optical_flow.svg" alt="Optical Flow" width="100%"/>
</div>

---

## 🌊 Optical Flow

```python
import cv2

# Lucas-Kanade (sparse)
p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)

# Farneback (dense)
flow = cv2.calcOpticalFlowFarneback(
    prev_gray, gray, None, 
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)

# RAFT (deep learning)
from torchvision.models.optical_flow import raft_small
model = raft_small(pretrained=True)
flow = model(frame1, frame2)
```

---

## 🏃 Action Recognition

### 3D CNNs

```python
from torchvision.models.video import r3d_18, r2plus1d_18

# R3D: 3D ResNet
model = r3d_18(pretrained=True)
# Input: (B, C, T, H, W) = (B, 3, 16, 112, 112)

# R(2+1)D: Factorized 3D conv
model = r2plus1d_18(pretrained=True)
# Separate spatial (2D) and temporal (1D) convolutions
```

### Two-Stream Networks

```python
# Two-Stream architecture:
# 1. Spatial stream: single RGB frame → appearance
# 2. Temporal stream: optical flow stack → motion
# 3. Fusion: average or late fusion

class TwoStream(nn.Module):
    def __init__(self):
        self.spatial = resnet50()  # RGB input
        self.temporal = resnet50()  # Flow input (stacked)
    
    def forward(self, rgb, flow):
        spatial_out = self.spatial(rgb)
        temporal_out = self.temporal(flow)
        return (spatial_out + temporal_out) / 2
```

### Video Transformers

```python
# TimeSformer: Divided space-time attention
# ViViT: Factorized encoder

from transformers import VideoMAEForVideoClassification
model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base")
```

---

## 🎯 Object Tracking

```python
# Single object tracking
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)
success, bbox = tracker.update(new_frame)

# Multi-object tracking (MOT)
# SORT: Simple Online Realtime Tracking
# DeepSORT: + appearance features
# ByteTrack: + low-confidence detections
```

---

## ❓ Interview Questions & Answers

### Q1: Dense vs Sparse optical flow?
| Dense | Sparse |
|-------|--------|
| Every pixel | Selected points |
| Slower | Faster |
| Complete motion field | Feature tracking |
| Farneback, RAFT | Lucas-Kanade |

### Q2: 2D CNN + LSTM vs 3D CNN for video?
| 2D + LSTM | 3D CNN |
|-----------|--------|
| Separate spatial/temporal | Joint modeling |
| Sequential processing | Parallel |
| Longer sequences | Short clips |
| Less params | More params |

### Q3: What is temporal stride in video models?
**Answer:**
- Sample every N-th frame
- Trades temporal resolution for coverage
- stride=2: 32 frames covers 64 frame span
- Reduces computation

### Q4: How does DeepSORT work?
**Answer:**
1. Detect objects in each frame
2. Extract appearance features (ReID)
3. Kalman filter for motion prediction
4. Hungarian matching (IoU + appearance)
5. Handle occlusions with track memory

### Q5: Video MAE vs Image MAE?
| Video MAE | Image MAE |
|-----------|-----------|
| Tube masking | Patch masking |
| Space-time reconstruction | Spatial only |
| Learns motion | Static features |
| 90% masking | 75% masking |

---

## 📓 Colab Notebooks

| Topic | Link |
|-------|------|
| Optical Flow | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/vision/blob/main/references/video_classification/train.py) |
| Action Recognition | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/intermediate_source/video_api_tutorial.ipynb) |
| RAFT | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/princeton-vl/RAFT/blob/master/demo.ipynb) |

---

<div align="center">

**[← Self-Supervised Learning](../12_Self_Supervised/) | [🏠 Home](../README.md) | [3D Vision →](../14_3D_Vision/)**

</div>

<div align="center">

# 🎬 Video & Temporal Vision

### *Optical Flow, Action Recognition, Tracking*

| Level | Time | Prerequisites |
|:-----:|:----:|:-------------:|
| 🟠 Intermediate-Advanced | 3 hours | CNNs, Image Processing |

</div>

---

**Navigation:** [← Self-Supervised](../12_Self_Supervised/) | [🏠 Home](../README.md) | [3D Vision →](../14_3D_Vision/)

---

## 📖 Table of Contents
- [Key Concepts](#-key-concepts)
- [Mathematical Foundations](#-mathematical-foundations)
- [Algorithms](#-algorithms)
- [Visual Overview](#-visual-overview)
- [Interview Q&A](#-interview-questions--answers)

---

## 🎯 Key Concepts

| Task | Input | Output | Key Methods |
|:-----|:------|:-------|:------------|
| **Optical Flow** | Frame t, Frame t+1 | Motion vectors (u,v) | Lucas-Kanade, RAFT |
| **Action Recognition** | Video clip | Action class | 3D CNN, Video Transformer |
| **Object Tracking** | Video + detection | Trajectories | SORT, DeepSORT |
| **Video Segmentation** | Video | Per-frame masks | SAM 2, XMem |

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/optical_flow.svg" alt="Optical Flow" width="100%"/>
</div>

---

## 🔢 Mathematical Foundations

### 1. Optical Flow Constraint Equation

```
┌─────────────────────────────────────────────────────┐
│  BRIGHTNESS CONSTANCY ASSUMPTION                    │
│                                                     │
│  I(x, y, t) = I(x+u, y+v, t+1)                     │
│                                                     │
│  Taylor expansion:                                  │
│  I(x+u, y+v, t+1) ≈ I + Iₓu + Iᵧv + Iₜ            │
│                                                     │
│  OPTICAL FLOW EQUATION:                             │
│                                                     │
│  Iₓu + Iᵧv + Iₜ = 0                                │
│                                                     │
│  Or: ∇I · [u,v]ᵀ + Iₜ = 0                          │
│                                                     │
│  Problem: 1 equation, 2 unknowns (aperture problem) │
└─────────────────────────────────────────────────────┘
```

### 2. Lucas-Kanade Method

```
┌─────────────────────────────────────────────────────┐
│  ASSUMPTION: Flow is constant in local window       │
│                                                     │
│  For n pixels in window:                            │
│  [Iₓ₁ Iᵧ₁]   [u]   [-Iₜ₁]                          │
│  [Iₓ₂ Iᵧ₂]   [v] = [-Iₜ₂]                          │
│  [...  ...]         [...]                           │
│  [Iₓₙ Iᵧₙ]         [-Iₜₙ]                          │
│                                                     │
│       A      ·  d  =   b                            │
│                                                     │
│  Least squares solution:                            │
│  d = (AᵀA)⁻¹Aᵀb                                    │
│                                                     │
│  AᵀA = [ΣIₓ²   ΣIₓIᵧ]  = Structure tensor M        │
│        [ΣIₓIᵧ  ΣIᵧ² ]                              │
└─────────────────────────────────────────────────────┘
```

**Solvability:** Need AᵀA to be invertible → corner points work best

### 3. Horn-Schunck Method (Dense Flow)

```
┌─────────────────────────────────────────────────────┐
│  GLOBAL ENERGY MINIMIZATION                         │
│                                                     │
│  E = ∫∫ [(Iₓu + Iᵧv + Iₜ)² + α²(|∇u|² + |∇v|²)] dxdy│
│         └─────────────────┘   └──────────────────┘  │
│          Data term            Smoothness term       │
│                                                     │
│  α controls smoothness vs data fidelity             │
│  Large α → smoother flow                            │
│                                                     │
│  Solved via Euler-Lagrange equations               │
└─────────────────────────────────────────────────────┘
```

### 4. Multi-Scale Pyramid

```
┌─────────────────────────────────────────────────────┐
│  COARSE-TO-FINE ESTIMATION                          │
│                                                     │
│  Problem: Large motions violate linearization       │
│                                                     │
│  Solution:                                          │
│  1. Build image pyramid (downsample)                │
│  2. Compute flow at coarsest level                  │
│  3. Warp image, compute residual flow               │
│  4. Upsample and refine at next level               │
│  5. Repeat until finest level                       │
│                                                     │
│  Level L:  I_L ────→ Flow_L                        │
│              ↓         ↓                            │
│  Level L-1: I_{L-1} → Warp → Residual → Flow_{L-1} │
└─────────────────────────────────────────────────────┘
```

### 5. Action Recognition Formulations

| Approach | Representation | Formula |
|:---------|:---------------|:--------|
| **Two-Stream** | RGB + Flow | P = f_rgb + f_flow |
| **3D CNN** | Spatio-temporal | y = C3D(V[t-k:t+k]) |
| **LSTM** | Sequential features | hₜ = LSTM(CNN(Iₜ), hₜ₋₁) |
| **Transformer** | Patch tokens | y = ViViT([CLS] + patches) |

### 6. Object Tracking - State Estimation

```
┌─────────────────────────────────────────────────────┐
│  KALMAN FILTER (Linear Motion Model)                │
│                                                     │
│  State: x = [x, y, w, h, ẋ, ẏ, ẇ, ḣ]ᵀ              │
│                                                     │
│  Predict:                                           │
│    x̂ₖ|ₖ₋₁ = Fxₖ₋₁                                  │
│    Pₖ|ₖ₋₁ = FPₖ₋₁Fᵀ + Q                            │
│                                                     │
│  Update:                                            │
│    K = Pₖ|ₖ₋₁Hᵀ(HPₖ|ₖ₋₁Hᵀ + R)⁻¹                  │
│    x̂ₖ = x̂ₖ|ₖ₋₁ + K(zₖ - Hx̂ₖ|ₖ₋₁)                  │
│    Pₖ = (I - KH)Pₖ|ₖ₋₁                             │
│                                                     │
│  F: motion model, H: observation model              │
│  Q: process noise, R: measurement noise             │
└─────────────────────────────────────────────────────┘
```

### 7. Data Association (Hungarian Algorithm)

```
┌─────────────────────────────────────────────────────┐
│  COST MATRIX                                        │
│                                                     │
│  C[i,j] = distance(track_i, detection_j)           │
│                                                     │
│  Common distances:                                  │
│  - IoU: 1 - IoU(bbox_track, bbox_det)              │
│  - Euclidean: ||center_track - center_det||        │
│  - Mahalanobis: (x-μ)ᵀΣ⁻¹(x-μ) (uses Kalman cov)  │
│  - Cosine: 1 - cosine(appearance_emb)              │
│                                                     │
│  Hungarian algorithm finds optimal assignment       │
│  Complexity: O(n³)                                  │
└─────────────────────────────────────────────────────┘
```

---

## ⚙️ Algorithms

### Algorithm 1: Lucas-Kanade Optical Flow

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Image I₁, I₂, window size w                │
│  OUTPUT: Flow field (u, v)                         │
│                                                     │
│  1. Compute gradients: Iₓ, Iᵧ, Iₜ                  │
│  2. FOR each pixel (x, y):                          │
│     3. Extract window W centered at (x,y)           │
│     4. Build A = [Iₓ, Iᵧ] for pixels in W          │
│     5. Build b = -Iₜ for pixels in W               │
│     6. Solve: [u,v]ᵀ = (AᵀA)⁻¹Aᵀb                  │
│     7. Store flow(x,y) = (u, v)                     │
│  8. RETURN flow field                               │
│                                                     │
│  Note: Only compute at corner points for efficiency │
└─────────────────────────────────────────────────────┘
```

### Algorithm 2: SORT (Simple Online Realtime Tracking)

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Detections per frame                        │
│  OUTPUT: Tracks with IDs                            │
│                                                     │
│  Initialize: tracks = []                            │
│  FOR each frame:                                    │
│    1. PREDICT: Kalman predict for all tracks       │
│    2. ASSOCIATE:                                    │
│       - Compute IoU(tracks, detections)             │
│       - Hungarian algorithm for assignment          │
│       - Threshold to reject bad matches             │
│    3. UPDATE:                                       │
│       - Matched: Kalman update with detection       │
│       - Unmatched track: increment miss count       │
│       - Unmatched detection: create new track       │
│    4. MANAGE:                                       │
│       - Delete tracks with miss > max_age           │
│       - Confirm tracks with hits > min_hits         │
│                                                     │
│  RETURN tracks                                      │
└─────────────────────────────────────────────────────┘
```

### Algorithm 3: Two-Stream Action Recognition

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Video frames {I₁, ..., Iₜ}                 │
│  OUTPUT: Action class prediction                    │
│                                                     │
│  SPATIAL STREAM:                                    │
│  1. Sample single frame Iₜ                          │
│  2. f_spatial = CNN_rgb(Iₜ)                        │
│                                                     │
│  TEMPORAL STREAM:                                   │
│  3. Compute optical flow: {F₁, ..., Fₜ₋₁}          │
│  4. Stack L consecutive flows                       │
│  5. f_temporal = CNN_flow(stack)                   │
│                                                     │
│  FUSION:                                            │
│  6. Late fusion: P = softmax(f_spatial + f_temporal)│
│  7. OR Early fusion: concatenate features          │
│                                                     │
│  RETURN argmax(P)                                   │
└─────────────────────────────────────────────────────┘
```

---

## ❓ Interview Questions & Answers

<details>
<summary><b>Q1: Explain the aperture problem in optical flow.</b></summary>

**Answer:**

**Problem:** Looking through a small window, we can only measure motion perpendicular to edges, not along them.

**Mathematically:** Iₓu + Iᵧv + Iₜ = 0 is one equation with two unknowns (u, v)

**Why corners work:** At corners, we have gradients in both x and y directions, making AᵀA invertible.

**Solutions:**
- Lucas-Kanade: Use larger window (local constraint)
- Horn-Schunck: Add global smoothness constraint
- Deep learning: Learn to resolve ambiguity

</details>

<details>
<summary><b>Q2: Lucas-Kanade vs Horn-Schunck?</b></summary>

**Answer:**

| Aspect | Lucas-Kanade | Horn-Schunck |
|:-------|:-------------|:-------------|
| Type | Local (sparse) | Global (dense) |
| Constraint | Constant flow in window | Smoothness |
| Result | Flow at corners | Flow everywhere |
| Speed | Fast | Slower |
| Large motion | Needs pyramid | Needs pyramid |
| Discontinuities | Handles well | Over-smooths |

</details>

<details>
<summary><b>Q3: How does RAFT improve optical flow?</b></summary>

**Answer:**

**RAFT (Recurrent All-Pairs Field Transforms):**

1. **All-pairs correlation:** Compute 4D correlation volume between all pixel pairs
2. **Iterative refinement:** Update flow estimate recurrently using GRU
3. **Multi-scale:** Correlation pyramid, not image pyramid

**Key innovations:**
- No coarse-to-fine warping
- Learns to update flow iteratively
- State-of-the-art accuracy

</details>

<details>
<summary><b>Q4: How does DeepSORT improve over SORT?</b></summary>

**Answer:**

**SORT:** Uses only IoU and Kalman filter

**DeepSORT adds:**
1. **Appearance features:** CNN embedding for each detection
2. **Cosine distance:** Match by appearance similarity
3. **Cascade matching:** Prioritize recent tracks
4. **Mahalanobis distance:** Use Kalman uncertainty

**Result:** Better handling of:
- Occlusions (re-identification)
- Camera motion
- Similar-looking objects

</details>

<details>
<summary><b>Q5: 3D CNN vs Two-Stream for action recognition?</b></summary>

**Answer:**

| Aspect | 3D CNN (C3D, I3D) | Two-Stream |
|:-------|:------------------|:-----------|
| Motion | Learned implicitly | Explicit (optical flow) |
| Computation | Higher (3D conv) | 2x models |
| Pretraining | Kinetics, etc. | ImageNet (2D) |
| Accuracy | Good | Competitive |
| Real-time | Harder | Possible |

**Modern approach:** Video Transformers (ViViT, TimeSformer) - patch-based, flexible

</details>

<details>
<summary><b>Q6: What is the difference between tracking and detection?</b></summary>

**Answer:**

| Aspect | Detection | Tracking |
|:-------|:----------|:---------|
| Input | Single frame | Video |
| Output | Bounding boxes | Trajectories with IDs |
| Temporal | No | Yes |
| Identity | No | Yes (same ID over time) |

**Tracking methods:**
- **Tracking-by-detection:** Detect + associate
- **Single-object tracking:** Given init box, follow
- **Multi-object tracking:** Multiple objects + IDs

</details>

<details>
<summary><b>Q7: Explain the Kalman filter for tracking.</b></summary>

**Answer:**

**State:** Position + velocity [x, y, w, h, ẋ, ẏ, ẇ, ḣ]

**Predict step:**
- Use motion model (constant velocity)
- Uncertainty increases

**Update step:**
- Get measurement (detection)
- Compute Kalman gain (trust measurement vs prediction)
- Update state and reduce uncertainty

**Key formulas:**
- Predict: x̂ = Fx, P = FPFᵀ + Q
- Update: K = PHᵀ(HPHᵀ + R)⁻¹

</details>

<details>
<summary><b>Q8: How to handle occlusion in tracking?</b></summary>

**Answer:**

**Strategies:**
1. **Keep predicting:** Use Kalman filter to predict trajectory
2. **Track management:** Don't delete immediately (max_age parameter)
3. **Re-identification:** Use appearance features to re-match
4. **Motion model:** Longer-term prediction with uncertainty

**DeepSORT approach:**
- Keep track alive for T frames without detection
- Use appearance embedding for re-identification
- Cascade matching: prefer recent matches

</details>

---

## 📚 Key Formulas Reference

| Formula | Description |
|:--------|:------------|
| Iₓu + Iᵧv + Iₜ = 0 | Optical flow constraint |
| d = (AᵀA)⁻¹Aᵀb | Lucas-Kanade solution |
| E = ∫(Data + αSmooth)dA | Horn-Schunck energy |
| x̂ = Fx + Kz | Kalman filter update |
| C[i,j] = 1 - IoU(i,j) | SORT cost matrix |

---

## 📓 Practice

See the Colab notebook: [`colab_tutorial.ipynb`](./colab_tutorial.ipynb)

---

<div align="center">

**[← Self-Supervised](../12_Self_Supervised/) | [🏠 Home](../README.md) | [3D Vision →](../14_3D_Vision/)**

</div>

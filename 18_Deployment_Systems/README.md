<div align="center">

<br/>

<a href="../17_Computational_Photography/README.md"><img src="https://img.shields.io/badge/◀__Photo-0f172a?style=for-the-badge&labelColor=1e293b" height="35"/></a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="../README.md"><img src="https://img.shields.io/badge/🏠__HOME-FBBF24?style=for-the-badge&labelColor=0f172a" height="35"/></a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="../19_Ethics_Safety/README.md"><img src="https://img.shields.io/badge/Ethics__▶-0f172a?style=for-the-badge&labelColor=1e293b" height="35"/></a>

<br/><br/>

---

<br/>

# ⚡ DEPLOYMENT

### 🌙 *Lab to Production*

<br/>

<img src="https://img.shields.io/badge/📚__MODULE__18/20-FBBF24?style=for-the-badge&labelColor=0f172a" height="40"/>
&nbsp;&nbsp;
<img src="https://img.shields.io/badge/⏱️__2_HOURS-FBBF24?style=for-the-badge&labelColor=0f172a" height="40"/>
&nbsp;&nbsp;
<img src="https://img.shields.io/badge/📓__NOTEBOOK_READY-34D399?style=for-the-badge&labelColor=0f172a" height="40"/>

<br/><br/>

---

</div>

<br/>

## 🎯 Key Concepts

| Technique | Size Reduction | Speed | Accuracy |
| :--- | :--- | :--- | :--- |
| **Quantization** | 4× (FP32→INT8) | 2-4× | ~1% drop |
| **Pruning** | 2-10× | 1.5-3× | 1-3% drop |
| **Distillation** | Student smaller | Varies | 1-2% drop |
| **Architecture** | Design efficient | Native | Varies |

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/model_optimization.svg" alt="Model Optimization" width="100%"/>
</div>

---

## 🔢 Mathematical Foundations

### 1. Quantization

```
┌─────────────────────────────────────────────────────┐
│  LINEAR QUANTIZATION                                │
│                                                     │
│  Quantize: q = round(x / scale) + zero_point        │
│  Dequantize: x' = (q - zero_point) × scale          │
│                                                     │
│  SYMMETRIC (signed)                                 │
│  scale = max(|x|) / 127                             │
│  zero_point = 0                                     │
│                                                     │
│  ASYMMETRIC (unsigned)                              │
│  scale = (max - min) / 255                          │
│  zero_point = round(-min / scale)                   │
│                                                     │
│  INT8 GEMM:                                         │
│  Y = scale_a × scale_b × (Qₐ × Qᵦ) + bias           │
└─────────────────────────────────────────────────────┘
```

### 2. Quantization-Aware Training (QAT)

```
┌─────────────────────────────────────────────────────┐
│  FAKE QUANTIZATION (differentiable)                 │
│                                                     │
│  Forward: x̂ = dequant(quant(x))                    │
│  Backward: ∂L/∂x = ∂L/∂x̂ (straight-through)        │
│                                                     │
│  Simulates quantization during training             │
│  Allows network to adapt to quantization noise      │
│                                                     │
│  POST-TRAINING vs QAT                               │
│  PTQ: Faster, slight accuracy drop                  │
│  QAT: Requires retraining, better accuracy          │
└─────────────────────────────────────────────────────┘
```

### 3. Pruning

```
┌─────────────────────────────────────────────────────┐
│  UNSTRUCTURED PRUNING                               │
│                                                     │
│  Remove individual weights: W' = W ⊙ M              │
│  M[i,j] = 1 if |W[i,j]| > threshold, else 0         │
│                                                     │
│  STRUCTURED PRUNING                                 │
│                                                     │
│  Remove entire filters/channels/layers              │
│  More hardware-friendly                             │
│                                                     │
│  MAGNITUDE PRUNING                                  │
│  Score = |weight|                                   │
│  Prune lowest k% by magnitude                       │
│                                                     │
│  LOTTERY TICKET HYPOTHESIS                          │
│  Sparse subnetworks exist that train well alone     │
└─────────────────────────────────────────────────────┘
```

### 4. Knowledge Distillation

```
┌─────────────────────────────────────────────────────┐
│  HINTON'S DISTILLATION                              │
│                                                     │
│  L = α × L_hard + (1-α) × L_soft                    │
│                                                     │
│  L_hard = CE(student, labels)                       │
│  L_soft = KL(softmax(student/T), softmax(teacher/T))│
│                                                     │
│  Temperature T softens distributions                │
│  Higher T → more information from teacher           │
│                                                     │
│  FEATURE DISTILLATION                               │
│                                                     │
│  L_feat = ||f_student - f_teacher||²                │
│  Match intermediate feature maps                    │
└─────────────────────────────────────────────────────┘
```

### 5. Efficient Architectures

| Model | Key Innovation | MAdds | Top-1 |
| :--- | :--- | :--- | :--- |
| **MobileNetV1** | Depthwise separable conv | 569M | 70.6% |
| **MobileNetV2** | Inverted residual | 300M | 72.0% |
| **EfficientNet** | Compound scaling | 390M | 77.1% |
| **ShuffleNet** | Channel shuffle | 140M | 69.4% |

```
┌─────────────────────────────────────────────────────┐
│  DEPTHWISE SEPARABLE CONVOLUTION                    │
│                                                     │
│  Standard: K×K×Cᵢₙ×Cₒᵤₜ                             │
│                                                     │
│  Depthwise: K×K×1×Cᵢₙ (spatial)                     │
│  Pointwise: 1×1×Cᵢₙ×Cₒᵤₜ (channel mixing)           │
│                                                     │
│  Reduction: (K² + Cₒᵤₜ) / (K² × Cₒᵤₜ)               │
│  For 3×3, Cₒᵤₜ=256: ~9× fewer params                │
└─────────────────────────────────────────────────────┘
```

### 6. Mixed Precision Training

```
┌─────────────────────────────────────────────────────┐
│  FP16 + FP32 MIXED PRECISION                        │
│                                                     │
│  Forward: FP16 (faster, less memory)                │
│  Backward: FP16                                     │
│  Master weights: FP32 (for updates)                 │
│  Loss scaling: scale loss to avoid underflow        │
│                                                     │
│  LOSS SCALING                                       │
│  scaled_loss = loss × scale_factor                  │
│  Update in FP32, then convert back                  │
│                                                     │
│  Speedup: ~2× on modern GPUs (Tensor Cores)         │
└─────────────────────────────────────────────────────┘
```

---

## ⚙️ Algorithms

### Algorithm 1: Post-Training Quantization

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Trained FP32 model, calibration data        │
│  OUTPUT: INT8 model                                 │
│                                                     │
│  1. Run calibration data through model              │
│  2. FOR each layer:                                 │
│     3. Collect activation statistics (min, max)     │
│     4. Compute scale = (max - min) / 255            │
│     5. Compute zero_point = round(-min / scale)     │
│  6. Quantize weights:                               │
│     q_w = round(w / scale_w)                        │
│  7. Replace FP32 ops with INT8 ops                  │
│                                                     │
│  Calibration methods:                               │
│  - MinMax: use observed min/max                     │
│  - Histogram: percentile clipping                   │
│  - Entropy: minimize KL divergence                  │
└─────────────────────────────────────────────────────┘
```

### Algorithm 2: Iterative Pruning

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Trained model, target sparsity s            │
│  OUTPUT: Pruned model                               │
│                                                     │
│  1. Train to convergence                            │
│  2. FOR each pruning step:                          │
│     3. Compute importance scores (magnitude)        │
│     4. Prune lowest p% of weights                   │
│     5. Fine-tune for k epochs                       │
│     6. IF sparsity >= s: break                      │
│                                                     │
│  GRADUAL PRUNING SCHEDULE                           │
│  sₜ = sₓ + (s - sₓ)(1 - (t-t₀)/(T-t₀))³             │
│                                                     │
│  Start sparse at t₀, reach target at T              │
└─────────────────────────────────────────────────────┘
```

### Algorithm 3: Knowledge Distillation

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Teacher model T, student architecture S     │
│  OUTPUT: Trained student                            │
│                                                     │
│  1. Train or load teacher T                         │
│  2. Initialize student S randomly                   │
│  3. FOR each batch (x, y):                          │
│     4. z_t = T(x), z_s = S(x)                       │
│     5. p_t = softmax(z_t / T)                       │
│     6. p_s = softmax(z_s / T)                       │
│     7. L_hard = CE(z_s, y)                          │
│     8. L_soft = KL(p_s, p_t) × T²                   │
│     9. L = α × L_hard + (1-α) × L_soft              │
│    10. Update S using L                             │
│                                                     │
│  Temperature T typically 2-20                       │
│  α typically 0.5-0.9                                │
└─────────────────────────────────────────────────────┘
```

---

## ❓ Interview Questions & Answers

<details>
<summary><b>Q1: What is the difference between PTQ and QAT?</b></summary>

**Answer:**

| Aspect | Post-Training (PTQ) | Quantization-Aware (QAT) |
| :--- | :--- | :--- |
| Training | No retraining | Retraining required |
| Time | Fast (minutes) | Slow (hours/days) |
| Accuracy | Lower | Higher |
| Use case | Quick deployment | Production quality |

**QAT** simulates quantization during training, allowing the model to adapt.

</details>

<details>
<summary><b>Q2: Why is structured pruning more practical than unstructured?</b></summary>

**Answer:**

**Unstructured:** Random zeros → need sparse matrix libraries

**Structured:** Remove entire channels/filters → standard dense ops

| Aspect | Unstructured | Structured |
| :--- | :--- | :--- |
| Granularity | Individual weights | Channels, filters |
| Sparsity | Very high (90%+) | Moderate (50-80%) |
| Speedup | Limited (sparse libs) | Direct (smaller matrix) |
| Hardware | Specialized | Standard |

</details>

<details>
<summary><b>Q3: How does knowledge distillation work?</b></summary>

**Answer:**

**Teacher:** Large, accurate model
**Student:** Small, efficient model

**Key insight:** Soft labels (teacher probabilities) contain more information than hard labels

**Temperature:** Higher T → softer distribution → more "dark knowledge"

**Loss:** α × Hard_loss + (1-α) × KL(student, teacher)

**Why it works:** Student learns class relationships, not just correct answer

</details>

<details>
<summary><b>Q4: Explain depthwise separable convolution.</b></summary>

**Answer:**

**Standard conv:** K×K×Cᵢₙ×Cₒᵤₜ operations

**Depthwise separable:**
1. **Depthwise:** K×K conv per channel (K×K×Cᵢₙ)
2. **Pointwise:** 1×1 conv to mix channels (Cᵢₙ×Cₒᵤₜ)

**Savings:** (K² + Cₒᵤₜ)/(K²×Cₒᵤₜ) ≈ 1/Cₒᵤₜ + 1/K²

For 3×3, 256 channels: ~9× reduction

</details>

<details>
<summary><b>Q5: What is loss scaling in mixed precision?</b></summary>

**Answer:**

**Problem:** FP16 has limited range → small gradients underflow to 0

**Solution:** Scale loss before backward, unscale gradients after

1. loss_scaled = loss × scale (e.g., 1024)
2. Compute gradients in FP16
3. Unscale: grad = grad_fp16 / scale
4. Update in FP32

**Dynamic scaling:** Increase scale until overflow, then reduce

</details>

<details>
<summary><b>Q6: How does TensorRT optimize inference?</b></summary>

**Answer:**

**Optimizations:**
1. **Layer fusion:** Conv+BN+ReLU → single kernel
2. **Precision:** FP16/INT8 with calibration
3. **Kernel auto-tuning:** Select best CUDA kernels
4. **Memory:** Optimize tensor memory layout
5. **Batching:** Dynamic batching for throughput

**Speedup:** Typically 2-10× over PyTorch

</details>

<details>
<summary><b>Q7: What is the lottery ticket hypothesis?</b></summary>

**Answer:**

**Claim:** Dense networks contain sparse subnetworks (winning tickets) that can train to same accuracy alone.

**Finding:** 
- Prune + reinitialize to original weights
- These sparse networks train as well as dense

**Implication:** Dense networks may be overparameterized for training, not just inference

**Limitation:** Finding tickets requires training dense network first

</details>

<details>
<summary><b>Q8: Compare ONNX, TensorRT, and CoreML.</b></summary>

**Answer:**

| Aspect | ONNX | TensorRT | CoreML |
| :--- | :--- | :--- | :--- |
| Purpose | Interchange format | NVIDIA inference | Apple inference |
| Hardware | Generic | NVIDIA GPU | Apple Neural Engine |
| Optimization | Minimal | Heavy | Heavy |
| Platform | Cross-platform | Linux, Windows | macOS, iOS |

**Typical pipeline:** PyTorch → ONNX → TensorRT/CoreML

</details>

---

## 📚 Key Formulas Reference

| Formula | Description |
| :--- | :--- |
| q = round(x/scale) + zp | Quantization |
| x' = (q - zp) × scale | Dequantization |
| W' = W ⊙ M | Pruning (mask) |
| L = α·CE + (1-α)·KL | Distillation loss |
| p_soft = softmax(z/T) | Temperature softmax |


---

<br/>

<div align="center">

## 📓 PRACTICE

### 🚀 *Ready to code? Let's get started!*

<br/>

### 🚀 Open in Google Colab

<br/>

<p align="center">
  <a href="https://colab.research.google.com/github/falkomeAI/Computer-Vision-Tutorial/blob/main/18_Deployment_Systems/colab_tutorial.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="60"/>
  </a>
</p>

<br/>

<p align="center">
  <strong>✨ Click the badge above to open this notebook directly in Google Colab!</strong>
</p>

<br/>


</div>

<br/>


---

<br/>

<div align="center">

| | | |
| :--- |:---:|---:|
| **[◀ Photo](../17_Computational_Photography/README.md)** | **[🏠 HOME](../README.md)** | **[Ethics ▶](../19_Ethics_Safety/README.md)** |

<br/>

---

🌙 Part of **[Computer Vision Complete](../README.md)**

<p align="center">
  Made with ❤️ by <a href="https://github.com/falkomeAI">falkomeAI</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/⭐_Star_this_repo_if_helpful-60A5FA?style=for-the-badge&logo=github&logoColor=white" alt="Star"/>
</p>

<br/>

</div>

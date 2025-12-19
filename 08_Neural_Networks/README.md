<div align="center">

<br/>

<a href="../07_Classical_ML/README.md"><img src="https://img.shields.io/badge/◀__Classical ML-0f172a?style=for-the-badge&labelColor=1e293b" height="35"/></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="../README.md"><img src="https://img.shields.io/badge/🏠__HOME-A78BFA?style=for-the-badge&labelColor=0f172a" height="35"/></a>
&nbsp;&nbsp;&nbsp;&nbsp;
<a href="../09_CNN_Architectures/README.md"><img src="https://img.shields.io/badge/CNNs__▶-0f172a?style=for-the-badge&labelColor=1e293b" height="35"/></a>

<br/><br/>

---

<br/>

# 🧠 NEURAL NETWORKS

### 🌙 *Deep Learning Foundations*

<br/>

<img src="https://img.shields.io/badge/📚__MODULE__08/20-A78BFA?style=for-the-badge&labelColor=0f172a" height="40"/>
&nbsp;&nbsp;
<img src="https://img.shields.io/badge/⏱️__2_HOURS-FBBF24?style=for-the-badge&labelColor=0f172a" height="40"/>
&nbsp;&nbsp;
<img src="https://img.shields.io/badge/📓__NOTEBOOK_READY-34D399?style=for-the-badge&labelColor=0f172a" height="40"/>

<br/><br/>

---

</div>

<br/>

## 🎯 Key Concepts

| Concept | Formula | Description |
| :--- | :--- | :--- |
| **Perceptron** | y = σ(wᵀx + b) | Single neuron, linear classifier |
| **Forward Pass** | aˡ = σ(Wˡaˡ⁻¹ + bˡ) | Layer-by-layer computation |
| **Loss Function** | L = -Σylog(ŷ) | Cross-entropy for classification |
| **Gradient** | ∂L/∂W = ∂L/∂a · ∂a/∂W | Chain rule application |
| **Update Rule** | W ← W - η∇L | Gradient descent step |

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/backpropagation.svg" alt="Backpropagation" width="100%"/>
</div>

---

## 🔢 Mathematical Foundations

### 1. Single Neuron (Perceptron)

```
┌─────────────────────────────────────────────────────┐
│  PERCEPTRON                                         │
│                                                     │
│  z = Σᵢ wᵢxᵢ + b = wᵀx + b                          │
│                                                     │
│  y = σ(z)  where σ is activation function           │
│                                                     │
│  Decision boundary: wᵀx + b = 0 (hyperplane)        │
└─────────────────────────────────────────────────────┘
```

### 2. Activation Functions

| Function | Formula | Derivative | Properties |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | σ(x) = 1/(1+e⁻ˣ) | σ(x)(1-σ(x)) | Range [0,1], vanishing gradient |
| **Tanh** | tanh(x) = (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | 1-tanh²(x) | Range [-1,1], zero-centered |
| **ReLU** | max(0,x) | 0 if x<0, 1 if x>0 | No vanishing gradient, sparse |
| **Leaky ReLU** | max(αx, x) | α if x<0, 1 if x>0 | No dead neurons |
| **GELU** | x·Φ(x) | Complex | Smooth, used in Transformers |
| **Softmax** | eˣⁱ/Σeˣʲ | pᵢ(δᵢⱼ - pⱼ) | Multi-class probabilities |

### 3. Multi-Layer Perceptron (MLP)

```
┌─────────────────────────────────────────────────────┐
│  FORWARD PROPAGATION                                │
│                                                     │
│  Layer l:                                           │
│    zˡ = Wˡaˡ⁻¹ + bˡ                                 │
│    aˡ = σ(zˡ)                                       │
│                                                     │
│  Where:                                             │
│    Wˡ ∈ ℝⁿˡ×ⁿˡ⁻¹  (weight matrix)                   │
│    bˡ ∈ ℝⁿˡ       (bias vector)                     │
│    aˡ ∈ ℝⁿˡ       (activations)                     │
└─────────────────────────────────────────────────────┘
```

### 4. Loss Functions

| Loss | Formula | Use Case |
| :--- | :--- | :--- |
| **MSE** | L = (1/n)Σ(y-ŷ)² | Regression |
| **Cross-Entropy** | L = -Σylog(ŷ) | Classification |
| **Binary CE** | L = -[ylog(ŷ) + (1-y)log(1-ŷ)] | Binary classification |
| **Hinge** | L = max(0, 1-y·ŷ) | SVM-like margin |

### 5. Backpropagation (Chain Rule)

```
┌─────────────────────────────────────────────────────┐
│  BACKWARD PROPAGATION                               │
│                                                     │
│  Output layer L:                                    │
│    δᴸ = ∂L/∂aᴸ ⊙ σ'(zᴸ)                             │
│                                                     │
│  Hidden layer l:                                    │
│    δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ σ'(zˡ)                        │
│                                                     │
│  Gradients:                                         │
│    ∂L/∂Wˡ = δˡ(aˡ⁻¹)ᵀ                               │
│    ∂L/∂bˡ = δˡ                                      │
│                                                     │
│  ⊙ = element-wise multiplication                    │
└─────────────────────────────────────────────────────┘
```

### 6. Optimization Algorithms

| Optimizer | Update Rule | Properties |
| :--- | :--- | :--- |
| **SGD** | W ← W - η∇L | Simple, may oscillate |
| **Momentum** | v ← βv + ∇L, W ← W - ηv | Accelerates in consistent direction |
| **RMSprop** | s ← ρs + (1-ρ)(∇L)², W ← W - η∇L/√(s+ε) | Adaptive learning rate |
| **Adam** | m ← β₁m + (1-β₁)∇L, v ← β₂v + (1-β₂)(∇L)², W ← W - ηm̂/√(v̂+ε) | Combines momentum + RMSprop |

**Adam Details:**
```
m̂ = m/(1-β₁ᵗ)  (bias correction for 1st moment)
v̂ = v/(1-β₂ᵗ)  (bias correction for 2nd moment)
Default: β₁=0.9, β₂=0.999, ε=10⁻⁸
```

### 7. Weight Initialization

| Method | Formula | Best For |
| :--- | :--- | :--- |
| **Xavier/Glorot** | W ~ U[-√(6/(nᵢₙ+nₒᵤₜ)), √(6/(nᵢₙ+nₒᵤₜ))] | Sigmoid, Tanh |
| **He/Kaiming** | W ~ N(0, 2/nᵢₙ) | ReLU |
| **LeCun** | W ~ N(0, 1/nᵢₙ) | SELU |

**Why?** Maintain variance across layers: Var(aˡ) ≈ Var(aˡ⁻¹)

### 8. Regularization Techniques

| Technique | Effect | Formula/Method |
| :--- | :--- | :--- |
| **L2 (Weight Decay)** | Penalize large weights | L' = L + λΣw² |
| **L1 (Lasso)** | Encourage sparsity | L' = L + λΣ\|w\| |
| **Dropout** | Random neuron dropping | p(keep) = 1-p, scale by 1/(1-p) |
| **Batch Norm** | Normalize activations | x̂ = (x-μ)/σ, y = γx̂+β |
| **Early Stopping** | Stop before overfit | Monitor validation loss |

---

## ⚙️ Algorithms

### Algorithm 1: Stochastic Gradient Descent (SGD)

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Training data {(xᵢ, yᵢ)}, learning rate η   │
│  OUTPUT: Trained weights W, b                       │
│                                                     │
│  1. Initialize W, b randomly                        │
│  2. FOR epoch = 1 to num_epochs:                    │
│     3. Shuffle training data                        │
│     4. FOR each mini-batch B:                       │
│        5. Forward: ŷ = f(x; W, b)                   │
│        6. Compute loss: L = Loss(ŷ, y)              │
│        7. Backward: compute ∂L/∂W, ∂L/∂b            │
│        8. Update: W ← W - η·∂L/∂W                   │
│                   b ← b - η·∂L/∂b                   │
│  9. RETURN W, b                                     │
└─────────────────────────────────────────────────────┘
```

### Algorithm 2: Backpropagation

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Network with L layers, input x, target y    │
│  OUTPUT: Gradients ∂L/∂Wˡ, ∂L/∂bˡ for all l         │
│                                                     │
│  FORWARD PASS:                                      │
│  1. a⁰ = x                                          │
│  2. FOR l = 1 to L:                                 │
│     3. zˡ = Wˡaˡ⁻¹ + bˡ                             │
│     4. aˡ = σ(zˡ)                                   │
│                                                     │
│  BACKWARD PASS:                                     │
│  5. δᴸ = ∇ₐL(aᴸ, y) ⊙ σ'(zᴸ)                        │
│  6. FOR l = L-1 to 1:                               │
│     7. δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ σ'(zˡ)                    │
│                                                     │
│  COMPUTE GRADIENTS:                                 │
│  8. FOR l = 1 to L:                                 │
│     9. ∂L/∂Wˡ = δˡ(aˡ⁻¹)ᵀ                           │
│    10. ∂L/∂bˡ = δˡ                                  │
│                                                     │
│  RETURN all gradients                               │
└─────────────────────────────────────────────────────┘
```

### Algorithm 3: Dropout (Training)

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Activation a, dropout probability p         │
│  OUTPUT: Masked activation                          │
│                                                     │
│  TRAINING:                                          │
│  1. m ~ Bernoulli(1-p)  (mask of 0s and 1s)         │
│  2. ã = a ⊙ m           (apply mask)                │
│  3. ã = ã / (1-p)       (scale to maintain E[a])    │
│                                                     │
│  INFERENCE:                                         │
│  1. Use all neurons (no dropout)                    │
│  2. No scaling needed (inverted dropout)            │
└─────────────────────────────────────────────────────┘
```

---

## ❓ Interview Questions & Answers

<details>
<summary><b>Q1: Explain the vanishing gradient problem.</b></summary>

**Answer:**
In deep networks, gradients become exponentially small as they backpropagate:

- **Cause**: Chain rule multiplication: ∂L/∂W¹ = ∂L/∂aᴸ × ∂aᴸ/∂aᴸ⁻¹ × ... × ∂a²/∂a¹ × ∂a¹/∂W¹
- **Sigmoid**: max derivative = 0.25, so after n layers: 0.25ⁿ → 0
- **Effect**: Early layers learn very slowly or not at all

**Solutions:**
1. ReLU activation (gradient = 1 for x > 0)
2. Residual connections (skip connections)
3. Proper initialization (He/Xavier)
4. Batch normalization
5. LSTM/GRU for RNNs

</details>

<details>
<summary><b>Q2: Why ReLU over Sigmoid?</b></summary>

**Answer:**

| Aspect | ReLU | Sigmoid |
| :--- | :--- | :--- |
| Gradient | 0 or 1 (no saturation for x>0) | 0-0.25 (saturates) |
| Computation | max(0,x) - fast | exp() - slow |
| Sparsity | ~50% neurons inactive | All active |
| Zero-centered | No | No |
| Dead neurons | Possible (if x<0 always) | No |

**When to use Sigmoid:** Output layer for binary classification (probability)

</details>

<details>
<summary><b>Q3: What is batch normalization and why does it help?</b></summary>

**Answer:**

**What:**
1. Normalize: x̂ = (x - μ_batch) / √(σ²_batch + ε)
2. Scale & shift: y = γx̂ + β (learnable parameters)

**Why it helps:**
- **Faster training**: Allows higher learning rates
- **Regularization**: Adds noise via mini-batch statistics
- **Reduces internal covariate shift**: Stabilizes layer inputs
- **Allows deeper networks**: Prevents gradient issues

**Training vs Inference:**
- Training: Use batch statistics (μ_batch, σ_batch)
- Inference: Use running average statistics

</details>

<details>
<summary><b>Q4: Compare SGD, Momentum, and Adam.</b></summary>

**Answer:**

| Optimizer | Pros | Cons | When to Use |
| :--- | :--- | :--- | :--- |
| **SGD** | Simple, good generalization | Slow, oscillates | Fine-tuning |
| **SGD+Momentum** | Faster, reduces oscillation | Still needs LR tuning | Most cases |
| **Adam** | Adaptive LR, works out-of-box | May generalize worse | Prototyping |
| **AdamW** | Proper weight decay | Slightly more complex | Transformers |

**Adam formula:**
- m_t = β₁m_{t-1} + (1-β₁)g_t  (1st moment)
- v_t = β₂v_{t-1} + (1-β₂)g_t² (2nd moment)
- θ = θ - η·m̂_t/(√v̂_t + ε)

</details>

<details>
<summary><b>Q5: How does dropout prevent overfitting?</b></summary>

**Answer:**

**Mechanism:**
1. Randomly drop neurons with probability p during training
2. Forces network to be redundant - no single neuron is essential
3. Ensemble effect: like training 2^n different networks

**Mathematics:**
- Training: a' = a × mask / (1-p)
- Inference: use full network (no dropout)

**Key insight:** Prevents co-adaptation of neurons

**Typical values:** p = 0.2-0.5 (higher for larger layers)

</details>

<details>
<summary><b>Q6: Why is weight initialization important?</b></summary>

**Answer:**

**Problem:** Bad initialization → vanishing/exploding gradients

**Xavier/Glorot:** For sigmoid/tanh
```
Var(W) = 2 / (n_in + n_out)
```

**He/Kaiming:** For ReLU
```
Var(W) = 2 / n_in
```

**Goal:** Keep variance constant across layers
- Var(aˡ) ≈ Var(aˡ⁻¹)
- Var(∂L/∂aˡ) ≈ Var(∂L/∂aˡ⁺¹)

</details>

<details>
<summary><b>Q7: What is the difference between L1 and L2 regularization?</b></summary>

**Answer:**

| Aspect | L1 (Lasso) | L2 (Ridge) |
| :--- | :--- | :--- |
| Penalty | λΣ\|w\| | λΣw² |
| Gradient | ±λ (constant) | 2λw (proportional) |
| Effect | Sparse weights (some = 0) | Small weights (none = 0) |
| Feature selection | Yes | No |
| Solution | Not differentiable at 0 | Smooth |

**L2 in optimizers:** Called "weight decay" - W ← W(1-λη) - η∇L

</details>

<details>
<summary><b>Q8: Explain the universal approximation theorem.</b></summary>

**Answer:**

**Theorem:** A neural network with one hidden layer of sufficient width can approximate any continuous function on compact subsets of ℝⁿ.

**Implications:**
- MLPs are theoretically powerful
- Width matters more than depth (in theory)
- BUT: may need exponentially many neurons

**Practice:**
- Deeper networks are more efficient
- Need proper training (optimization matters)
- Doesn't guarantee we can FIND the approximation

</details>

---

## 📚 Key Formulas Reference

| Formula | Description |
| :--- | :--- |
| y = σ(Wx + b) | Neuron output |
| L = -Σylog(ŷ) | Cross-entropy loss |
| ∂L/∂W = δ·aᵀ | Weight gradient |
| δˡ = (Wˡ⁺¹)ᵀδˡ⁺¹ ⊙ σ'(zˡ) | Error backpropagation |
| W ← W - η∇L | SGD update |
| m = β₁m + (1-β₁)∇L | Adam 1st moment |
| Var(W) = 2/n_in | He initialization |


---

<br/>

<div align="center">

## 📓 PRACTICE

<br/>

### 🚀 Click to Open Directly in Google Colab

<br/>

<a href="https://colab.research.google.com/github/USERNAME/computer_vision_complete/blob/main/08_Neural_Networks/colab_tutorial.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" height="50"/>
</a>

<br/><br/>

> ⚠️ **First time?** Push this repo to GitHub, then replace `USERNAME` in the link above with your GitHub username.

<br/>

**Or manually:** [📥 Download](./colab_tutorial.ipynb) → [🌐 Colab](https://colab.research.google.com) → Upload

</div>

<br/>




---

<br/>

<div align="center">

| | | |
|:---|:---:|---:|
| **[◀ Classical ML](../07_Classical_ML/README.md)** | **[🏠 HOME](../README.md)** | **[CNNs ▶](../09_CNN_Architectures/README.md)** |

<br/>

---

🌙 Part of **[Computer Vision Complete](../README.md)** · Made with ❤️

<br/>

</div>

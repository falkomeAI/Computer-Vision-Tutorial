<div align="center">

<br/>

<a href="../06_Geometry_MultiView/README.md"><img src="https://img.shields.io/badge/◀__Geometry-0f172a?style=for-the-badge&labelColor=1e293b" height="35"/></a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="../README.md"><img src="https://img.shields.io/badge/🏠__HOME-A78BFA?style=for-the-badge&labelColor=0f172a" height="35"/></a>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="../08_Neural_Networks/README.md"><img src="https://img.shields.io/badge/Neural Nets__▶-0f172a?style=for-the-badge&labelColor=1e293b" height="35"/></a>

<br/><br/>

---

<br/>

# 📊 CLASSICAL ML

### 🌙 *Before Deep Learning*

<br/>

<img src="https://img.shields.io/badge/📚__MODULE__07/20-A78BFA?style=for-the-badge&labelColor=0f172a" height="40"/>
&nbsp;&nbsp;
<img src="https://img.shields.io/badge/⏱️__2_HOURS-FBBF24?style=for-the-badge&labelColor=0f172a" height="40"/>
&nbsp;&nbsp;
<img src="https://img.shields.io/badge/📓__NOTEBOOK_READY-34D399?style=for-the-badge&labelColor=0f172a" height="40"/>

<br/><br/>

---

</div>

<br/>

## 🎯 Key Concepts

| Method | Type | Objective | Use Case |
| :--- | :--- | :--- | :--- |
| **PCA** | Unsupervised | max Var(Xw), \|\|w\|\|=1 | Dimensionality reduction |
| **SVM** | Supervised | min \|\|w\|\|² + CΣξ | Classification |
| **K-Means** | Unsupervised | min Σ\|\|x-μₖ\|\|² | Clustering |
| **KNN** | Supervised | Majority vote of k neighbors | Classification |
| **Random Forest** | Supervised | Ensemble of trees | Classification/Regression |

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/svm_kernel.svg" alt="SVM Kernel" width="100%"/>
</div>

---

## 🔢 Mathematical Foundations

### 1. Principal Component Analysis (PCA)

```
┌─────────────────────────────────────────────────────┐
│  GOAL: Find directions of maximum variance          │
│                                                     │
│  1. Center data: X̄ = X - mean(X)                   │
│                                                     │
│  2. Covariance matrix: C = (1/n)X̄ᵀX̄               │
│                                                     │
│  3. Eigendecomposition: C = VΛVᵀ                    │
│     - V: eigenvectors (principal components)        │
│     - Λ: eigenvalues (variance explained)           │
│                                                     │
│  4. Project: X_pca = X̄V[:,:k]                      │
│                                                     │
│  Variance explained: λᵢ / Σλⱼ                       │
└─────────────────────────────────────────────────────┘
```

**Properties:**
- Principal components are orthogonal
- First PC captures maximum variance
- Used for visualization, denoising, compression

### 2. Support Vector Machine (SVM)

```
┌─────────────────────────────────────────────────────┐
│  HARD MARGIN (linearly separable)                   │
│                                                     │
│  min  (1/2)||w||²                                   │
│  s.t. yᵢ(wᵀxᵢ + b) ≥ 1  ∀i                          │
│                                                     │
│  SOFT MARGIN (with slack variables)                 │
│                                                     │
│  min  (1/2)||w||² + C Σξᵢ                           │
│  s.t. yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ                         │
│       ξᵢ ≥ 0                                        │
│                                                     │
│  Margin = 2 / ||w||                                 │
└─────────────────────────────────────────────────────┘
```

**Kernel Trick:**
| Kernel | Formula | Use Case |
| :--- | :--- | :--- |
| Linear | K(x,y) = xᵀy | Linearly separable |
| RBF | K(x,y) = exp(-γ\|\|x-y\|\|²) | Non-linear, default |
| Polynomial | K(x,y) = (γxᵀy + r)^d | Polynomial boundary |

### 3. K-Means Clustering

```
┌─────────────────────────────────────────────────────┐
│  OBJECTIVE: min Σₖ Σₓ∈Cₖ ||x - μₖ||²                │
│                                                     │
│  Where:                                             │
│    Cₖ = cluster k                                   │
│    μₖ = centroid of cluster k                       │
│                                                     │
│  Update rules:                                      │
│    Assignment: cᵢ = argmin_k ||xᵢ - μₖ||²           │
│    Centroid:   μₖ = (1/|Cₖ|) Σₓ∈Cₖ x                │
└─────────────────────────────────────────────────────┘
```

### 4. K-Nearest Neighbors (KNN)

```
┌─────────────────────────────────────────────────────┐
│  CLASSIFICATION:                                    │
│    ŷ = mode({yⱼ : xⱼ ∈ Nₖ(x)})                      │
│                                                     │
│  REGRESSION:                                        │
│    ŷ = (1/k) Σⱼ∈Nₖ(x) yⱼ                            │
│                                                     │
│  Distance metrics:                                  │
│    Euclidean: d(x,y) = √(Σ(xᵢ-yᵢ)²)                 │
│    Manhattan: d(x,y) = Σ|xᵢ-yᵢ|                     │
│    Cosine:    d(x,y) = 1 - (xᵀy)/(||x||||y||)       │
└─────────────────────────────────────────────────────┘
```

### 5. Decision Trees

```
┌─────────────────────────────────────────────────────┐
│  SPLITTING CRITERIA                                 │
│                                                     │
│  Entropy: H(S) = -Σpᵢlog₂(pᵢ)                       │
│                                                     │
│  Information Gain: IG = H(S) - Σ(|Sᵥ|/|S|)H(Sᵥ)     │
│                                                     │
│  Gini Impurity: G = 1 - Σpᵢ²                        │
│                                                     │
│  Choose split that maximizes IG or minimizes Gini   │
└─────────────────────────────────────────────────────┘
```

### 6. Ensemble Methods

| Method | Technique | Formula |
| :--- | :--- | :--- |
| **Bagging** | Bootstrap + Aggregate | ŷ = (1/B)Σfᵦ(x) |
| **Random Forest** | Bagging + random features | ŷ = mode(tree predictions) |
| **Boosting** | Sequential weighted | ŷ = Σαₘhₘ(x) |
| **AdaBoost** | Exponential loss | αₘ = (1/2)ln((1-εₘ)/εₘ) |

---

## ⚙️ Algorithms

### Algorithm 1: K-Means

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Data X, number of clusters K                │
│  OUTPUT: Cluster assignments, centroids             │
│                                                     │
│  1. Initialize centroids μ₁,...,μₖ randomly         │
│  2. REPEAT until convergence:                       │
│     3. Assignment step:                             │
│        FOR each xᵢ:                                 │
│          cᵢ = argmin_k ||xᵢ - μₖ||²                 │
│     4. Update step:                                 │
│        FOR each k:                                  │
│          μₖ = mean({xᵢ : cᵢ = k})                   │
│  5. RETURN clusters, centroids                      │
│                                                     │
│  Convergence: centroids don't change                │
│  Complexity: O(nKd) per iteration                   │
└─────────────────────────────────────────────────────┘
```

### Algorithm 2: PCA

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Data X ∈ ℝⁿˣᵈ, target dimensions k          │
│  OUTPUT: Projected data X_pca ∈ ℝⁿˣᵏ                │
│                                                     │
│  1. Center: X̄ = X - mean(X, axis=0)                │
│  2. Covariance: C = (1/n)X̄ᵀX̄                      │
│  3. Eigendecomposition: C = VΛVᵀ                    │
│  4. Sort eigenvectors by eigenvalue (descending)    │
│  5. Select top k eigenvectors: Vₖ                   │
│  6. Project: X_pca = X̄Vₖ                           │
│  7. RETURN X_pca                                    │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Algorithm 3: SVM (SMO sketch)

```
┌─────────────────────────────────────────────────────┐
│  INPUT: Data (xᵢ, yᵢ), kernel K, C                  │
│  OUTPUT: Support vectors, weights                   │
│                                                     │
│  Dual problem:                                      │
│  max Σαᵢ - (1/2)ΣΣαᵢαⱼyᵢyⱼK(xᵢ,xⱼ)                  │
│  s.t. 0 ≤ αᵢ ≤ C, Σαᵢyᵢ = 0                         │
│                                                     │
│  Decision function:                                 │
│  f(x) = sign(Σαᵢyᵢ K(xᵢ,x) + b)                     │
│                                                     │
│  Support vectors: points where 0 < αᵢ ≤ C           │
└─────────────────────────────────────────────────────┘
```

---

## ❓ Interview Questions & Answers

<details>
<summary><b>Q1: How does PCA work? What are its limitations?</b></summary>

**Answer:**

**How it works:**
1. Find directions of maximum variance
2. Project data onto these directions
3. Keeps most information with fewer dimensions

**Limitations:**
- Only linear transformations
- Sensitive to scaling (standardize first!)
- May not capture class-discriminative features
- Outliers affect results significantly

</details>

<details>
<summary><b>Q2: Explain the kernel trick in SVM.</b></summary>

**Answer:**

**Problem:** Data not linearly separable in original space

**Solution:** Map to higher dimension where it becomes separable

**Kernel trick:** Never explicitly compute φ(x), only K(x,y) = φ(x)ᵀφ(y)

**Example - RBF kernel:**
- Implicitly maps to infinite dimensions
- K(x,y) = exp(-γ||x-y||²)
- γ controls decision boundary complexity

**Key insight:** Dual formulation only uses dot products → can kernelize

</details>

<details>
<summary><b>Q3: How to choose K in K-Means?</b></summary>

**Answer:**

**Methods:**
1. **Elbow method:** Plot inertia vs K, find "elbow"
2. **Silhouette score:** Measures cluster separation, maximize
3. **Gap statistic:** Compare to null reference distribution
4. **Domain knowledge:** Sometimes K is known

**Inertia formula:** Σₖ Σₓ∈Cₖ ||x - μₖ||²

**Silhouette:** s = (b-a) / max(a,b)
- a = mean intra-cluster distance
- b = mean nearest-cluster distance

</details>

<details>
<summary><b>Q4: Random Forest vs single Decision Tree?</b></summary>

**Answer:**

| Aspect | Single Tree | Random Forest |
| :--- | :--- | :--- |
| Variance | High (overfit) | Low (averaged) |
| Bias | Low | Low |
| Interpretability | High | Low |
| Training time | Fast | Slower |
| Feature importance | Yes | Yes (averaged) |

**Why RF works:**
- Bagging reduces variance
- Random feature selection decorrelates trees
- Ensemble averages out individual errors

</details>

<details>
<summary><b>Q5: What is the bias-variance tradeoff?</b></summary>

**Answer:**

```
Total Error = Bias² + Variance + Noise
```

| Model | Bias | Variance | Example |
| :--- | :--- | :--- | :--- |
| Simple | High | Low | Linear regression |
| Complex | Low | High | Deep tree |

**Goal:** Find sweet spot

**Solutions:**
- Cross-validation to tune complexity
- Regularization (increase bias, decrease variance)
- Ensemble methods (decrease variance)

</details>

<details>
<summary><b>Q6: KNN - how to choose K?</b></summary>

**Answer:**

**Guidelines:**
- Small K: Low bias, high variance (noisy)
- Large K: High bias, low variance (smooth)
- Odd K for binary classification (avoid ties)
- Rule of thumb: K = √n

**Cross-validation:** Try different K, pick best

**Distance weighting:** Give closer neighbors more weight

</details>

<details>
<summary><b>Q7: How does AdaBoost work?</b></summary>

**Answer:**

1. **Initialize** weights wᵢ = 1/n
2. **For each round m:**
   - Train weak learner hₘ on weighted data
   - Compute error: εₘ = Σwᵢ𝟙[hₘ(xᵢ)≠yᵢ]
   - Compute weight: αₘ = (1/2)ln((1-εₘ)/εₘ)
   - Update weights: wᵢ ← wᵢ exp(-αₘyᵢhₘ(xᵢ))
   - Normalize weights
3. **Final prediction:** H(x) = sign(Σαₘhₘ(x))

**Key:** Focuses on misclassified samples each round

</details>

<details>
<summary><b>Q8: LDA vs PCA?</b></summary>

**Answer:**

| Aspect | PCA | LDA |
| :--- | :--- | :--- |
| Type | Unsupervised | Supervised |
| Goal | Max variance | Max class separation |
| Uses labels | No | Yes |
| Max components | min(n,d) | C-1 (C=classes) |

**LDA objective:** Maximize between-class / within-class variance

**When to use:**
- PCA: General dimensionality reduction
- LDA: When you have labels and want classification

</details>

---

## 📚 Key Formulas Reference

| Formula | Description |
| :--- | :--- |
| C = (1/n)XᵀX | Covariance matrix |
| K(x,y) = exp(-γ\|\|x-y\|\|²) | RBF kernel |
| J = Σₖ Σₓ∈Cₖ \|\|x - μₖ\|\|² | K-means objective |
| H(S) = -Σpᵢlog₂(pᵢ) | Entropy |
| G = 1 - Σpᵢ² | Gini impurity |
| s = (b-a) / max(a,b) | Silhouette score |


---

<br/>

<div align="center">

## 📓 PRACTICE

### 🚀 *Ready to code? Let's get started!*

<br/>

### 🚀 Open in Google Colab

<br/>

<p align="center">
  <a href="https://colab.research.google.com/github/falkomeAI/computer_vision_complete/blob/main/07_Classical_ML/colab_tutorial.ipynb">
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
| **[◀ Geometry](../06_Geometry_MultiView/README.md)** | **[🏠 HOME](../README.md)** | **[Neural Nets ▶](../08_Neural_Networks/README.md)** |

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

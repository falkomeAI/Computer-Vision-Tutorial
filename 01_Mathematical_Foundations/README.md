# 📐 Mathematical Foundations for Computer Vision

> **Level:** 🟢 Beginner | **Prerequisites:** Basic calculus, programming

---

**Navigation:** [🏠 Home](../README.md) | [Transform Methods →](../02_Transform_Methods/)

---


## 📋 Summary

Mathematical foundations form the backbone of all computer vision algorithms. This module covers **linear algebra** (vectors, matrices, eigendecomposition, SVD), **probability and statistics** (distributions, Bayes' theorem, MLE), **optimization** (gradient descent, convexity), and **signal processing basics** (convolution, sampling). Understanding these concepts is essential for both classical and deep learning approaches.

---

## 📊 Key Concepts Table

| Concept | Definition | CV Application |
|---------|------------|----------------|
| **Matrix Multiplication** | Linear transformation | Neural network layers |
| **Eigendecomposition** | A = VΛV⁻¹ | PCA, covariance analysis |
| **SVD** | A = UΣVᵀ | Image compression, pseudo-inverse |
| **Gradient** | Vector of partial derivatives | Backpropagation |
| **Convolution** | Sliding window operation | Filtering, CNNs |
| **Probability Distribution** | P(x) over outcomes | Uncertainty modeling |

---

## 🔢 Math / Formulas

### Linear Algebra

**Matrix-Vector Multiplication:**
$$
\mathbf{y} = \mathbf{W}\mathbf{x} \quad \text{where} \quad y_i = \sum_j W_{ij} x_j
$$

**Eigendecomposition (for symmetric matrices):**
$$
\mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^T \quad \text{where} \quad \mathbf{A}\mathbf{v}_i = \lambda_i \mathbf{v}_i
$$

**Singular Value Decomposition:**
$$
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T
$$

### Probability

**Bayes' Theorem:**
$$
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}
$$

**Gaussian Distribution:**
$$
\mathcal{N}(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
$$

### Optimization

**Gradient Descent Update:**
$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)
$$

### Convolution

**2D Discrete Convolution:**
$$
(f * g)[m, n] = \sum_{i}\sum_{j} f[i, j] \cdot g[m-i, n-j]
$$

---

## 🎨 Visual / Diagram

<div align="center">
<img src="./svg_figs/linear_algebra_overview.svg" alt="Linear Algebra Overview" width="100%"/>
</div>

---

<div align="center">
<img src="./svg_figs/convolution_operation.svg" alt="Convolution Operation" width="100%"/>
</div>

---

## 💻 Code Practice

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1mathematical_foundations)

```python
#@title 📐 Mathematical Foundations - Complete Tutorial
#@markdown Linear Algebra, Probability, Optimization for Computer Vision!

!pip install numpy matplotlib scipy scikit-learn torch -q

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, stats
from sklearn.decomposition import PCA
import torch
import torch.nn as nn

print("✅ Setup complete!")

#@title 1️⃣ Vectors and Matrices
def vectors_matrices_demo():
    """Basic vector and matrix operations for CV"""
    # Vectors represent images, features, etc.
    image_flat = np.random.randn(784)  # Flattened 28x28 image
    feature_vector = np.random.randn(128)  # Feature from CNN
    
    # Matrices - Linear transformation
    transformation = np.random.randn(64, 128)  # Linear layer weights
    
    # Key operations
    dot_product = np.dot(feature_vector, feature_vector)  # Similarity
    matrix_mult = transformation @ feature_vector  # Linear transformation
    
    # Norms
    l1_norm = np.linalg.norm(feature_vector, ord=1)
    l2_norm = np.linalg.norm(feature_vector, ord=2)
    
    print(f"Feature vector shape: {feature_vector.shape}")
    print(f"After transformation: {matrix_mult.shape}")
    print(f"L2 norm: {l2_norm:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(feature_vector[:50])
    axes[0].set_title('Feature Vector (first 50 dims)')
    
    axes[1].imshow(transformation, cmap='RdBu', aspect='auto')
    axes[1].set_title(f'Weight Matrix {transformation.shape}')
    
    axes[2].bar(['L1', 'L2'], [l1_norm, l2_norm])
    axes[2].set_title('Vector Norms')
    
    plt.tight_layout()
    plt.show()

vectors_matrices_demo()

#@title 2️⃣ Eigendecomposition & SVD
def eigen_svd_demo():
    """SVD for image compression"""
    from sklearn.datasets import load_sample_image
    
    # Create sample data
    data = np.random.randn(100, 10)
    cov_matrix = data.T @ data / 100
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    
    # SVD
    U, S, Vt = np.linalg.svd(data, full_matrices=False)
    
    # Low-rank approximation
    k = 3
    data_approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    error = np.linalg.norm(data - data_approx, 'fro') / np.linalg.norm(data, 'fro')
    
    print(f"Rank-{k} approximation error: {error:.2%}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].imshow(cov_matrix, cmap='viridis')
    axes[0].set_title('Covariance Matrix')
    
    axes[1].bar(range(len(eigenvalues)), eigenvalues)
    axes[1].set_title('Eigenvalues (sorted)')
    
    cumvar = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    axes[2].plot(cumvar, 'o-')
    axes[2].axhline(0.95, color='r', linestyle='--', label='95%')
    axes[2].set_title('Cumulative Variance')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

eigen_svd_demo()

#@title 3️⃣ Probability Distributions
def probability_demo():
    """Common distributions in CV"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = np.linspace(-5, 5, 1000)
    
    # Gaussian
    for sigma in [0.5, 1, 2]:
        y = stats.norm.pdf(x, 0, sigma)
        axes[0].plot(x, y, label=f'σ={sigma}')
    axes[0].set_title('Gaussian Distribution')
    axes[0].legend()
    
    # Softmax
    logits = np.array([2.0, 1.0, 0.1])
    def softmax(x, T=1):
        e_x = np.exp(x / T)
        return e_x / e_x.sum()
    
    for T in [0.5, 1.0, 2.0]:
        probs = softmax(logits, T)
        offset = [0.5, 1.0, 2.0].index(T) * 0.25
        axes[1].bar(np.arange(3) + offset, probs, width=0.2, label=f'T={T}')
    axes[1].set_title('Softmax (Temperature)')
    axes[1].legend()
    
    # Cross-entropy loss
    p_range = np.linspace(0.01, 0.99, 100)
    ce_loss = -np.log(p_range)
    axes[2].plot(p_range, ce_loss)
    axes[2].set_xlabel('Predicted prob for true class')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('Cross-Entropy Loss')
    
    plt.tight_layout()
    plt.show()

probability_demo()

#@title 4️⃣ Gradient Descent
def optimization_demo():
    """Gradient descent visualization"""
    # 2D loss landscape
    def f(x, y):
        return (1 - x)**2 + 100*(y - x**2)**2  # Rosenbrock
    
    def grad_f(x, y):
        dx = -2*(1 - x) - 400*x*(y - x**2)
        dy = 200*(y - x**2)
        return np.array([dx, dy])
    
    # Gradient descent
    x = np.array([-1.0, 1.0])
    path = [x.copy()]
    lr = 0.001
    
    for _ in range(1000):
        grad = grad_f(x[0], x[1])
        x = x - lr * grad
        path.append(x.copy())
    
    path = np.array(path)
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    
    xx = np.linspace(-2, 2, 100)
    yy = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(xx, yy)
    Z = f(X, Y)
    
    ax.contour(X, Y, np.log(Z + 1), levels=30, cmap='viridis')
    ax.plot(path[:, 0], path[:, 1], 'r.-', markersize=1, label='GD path')
    ax.scatter(1, 1, c='green', s=100, marker='*', label='Minimum')
    ax.scatter(-1, 1, c='red', s=100, marker='o', label='Start')
    ax.set_title('Gradient Descent on Rosenbrock Function')
    ax.legend()
    
    plt.show()

optimization_demo()

#@title 5️⃣ Convolution
def convolution_demo():
    """1D and 2D convolution"""
    # 1D convolution
    signal = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
    kernel = np.array([1, 0, -1])  # Edge detector
    conv_result = np.convolve(signal, kernel, mode='same')
    
    # 2D convolution
    from scipy.signal import convolve2d
    image = np.random.rand(8, 8)
    kernel_2d = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    conv_2d = convolve2d(image, kernel_2d, mode='same')
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    axes[0, 0].plot(signal, 'o-', label='Signal')
    axes[0, 0].set_title('1D Signal')
    
    axes[0, 1].bar(range(len(kernel)), kernel)
    axes[0, 1].set_title('Kernel [1, 0, -1]')
    
    axes[0, 2].plot(conv_result, 'o-')
    axes[0, 2].set_title('Convolution Result')
    
    axes[1, 0].imshow(image, cmap='gray')
    axes[1, 0].set_title('2D Image')
    
    axes[1, 1].imshow(kernel_2d, cmap='RdBu')
    axes[1, 1].set_title('Prewitt Kernel')
    
    axes[1, 2].imshow(conv_2d, cmap='RdBu')
    axes[1, 2].set_title('2D Convolution (edges)')
    
    plt.tight_layout()
    plt.show()

convolution_demo()

print("\n" + "="*50)
print("✅ Mathematical Foundations Complete!")
print("="*50)
```

---

## ⚠️ Common Pitfalls / Tips

| Pitfall | Solution |
|---------|----------|
| Numerical instability in matrix operations | Use SVD instead of inverse; add small ε to denominators |
| Vanishing gradients | Use proper initialization; batch normalization |
| Confusing row vs column vectors | Be consistent; PyTorch uses (batch, features) |
| Forgetting to normalize | Always normalize features for distance-based methods |
| Wrong axis in NumPy operations | Use explicit `axis=` parameter |

---

## 🛠️ Mini-Project Ideas

### Project 1: PCA Image Compression (Beginner)
- Implement PCA from scratch using SVD
- Compress images at different ranks
- Visualize reconstruction quality vs compression ratio

### Project 2: Gradient Descent Visualizer (Beginner)
- Implement GD, SGD, Adam from scratch
- Compare convergence on different loss landscapes
- Visualize optimization paths

### Project 3: Convolution Calculator (Beginner)
- Build an interactive convolution demo
- Support different kernels (blur, sharpen, edge)
- Compare with NumPy/SciPy implementations

---

## ❓ Interview Questions & Answers

### Q1: What is the difference between eigenvalues and singular values?

| Eigenvalues | Singular Values |
|-------------|-----------------|
| Square matrices only | Any matrix shape |
| Can be negative/complex | Always non-negative |
| Av = λv | Av = σu |
| Used in PCA (on covariance) | Used in SVD (on data) |

**Answer:** Eigenvalues exist only for square matrices and satisfy Av = λv. Singular values are always non-negative and come from the SVD decomposition A = UΣVᵀ. For symmetric positive semi-definite matrices, eigenvalues equal singular values.

### Q2: Why is convolution used in CNNs?

**Answer:**
1. **Translation equivariance**: If input shifts, output shifts same amount
2. **Parameter sharing**: Same kernel applied everywhere → fewer parameters
3. **Local connectivity**: Each output depends on local input region
4. **Hierarchy**: Stacking layers builds up receptive field

### Q3: Explain gradient descent vs. stochastic gradient descent.

| Full-batch GD | Stochastic GD (SGD) |
|---------------|---------------------|
| Uses all samples | Uses 1 or mini-batch |
| Exact gradient | Noisy gradient estimate |
| Slow per update | Fast per update |
| May get stuck in local minima | Noise helps escape |

### Q4: What is the softmax function and when is it used?

**Answer:**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

- Converts logits to probabilities (sum to 1)
- Used in multi-class classification output layer
- Temperature parameter controls "sharpness"

### Q5: How does SVD relate to PCA?

**Answer:** PCA finds principal components by:
1. Center data: X̄ = X - mean
2. Compute covariance: C = X̄ᵀX̄/(n-1)
3. Eigendecompose C

OR equivalently: SVD of centered data X̄ = UΣVᵀ
- Principal components = columns of V
- Eigenvalues = Σ²/(n-1)

---

## 📚 References / Further Reading

### Textbooks
- *Linear Algebra Done Right* - Sheldon Axler
- *Pattern Recognition and Machine Learning* - Bishop (Ch. 1-2)
- *Mathematics for Machine Learning* - Deisenroth et al. (free online)

### Online Resources
- [3Blue1Brown Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability)
- [Stanford CS229 Math Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)

### Papers
- "Matrix Cookbook" - Technical reference for matrix calculus
- "A Tutorial on Principal Component Analysis" - Shlens

---

<div align="center">

**[🏠 Home](../README.md) | [Transform Methods →](../02_Transform_Methods/)**

</div>

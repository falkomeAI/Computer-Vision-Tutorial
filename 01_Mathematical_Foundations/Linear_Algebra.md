# Linear Algebra for Computer Vision

> **Level:** 🟢 Beginner | **Time:** 2 hours | **Prerequisites:** Basic math

---

**Navigation:** [🏠 Module Home](./README.md) | [Probability & Statistics →](./Probability_Statistics.md)

---

## 📋 Summary

Linear algebra is the foundation of computer vision. Every image is a matrix, every transformation is matrix multiplication, and every neural network layer is a linear operation followed by non-linearity.

---

## 🔢 Key Concepts

### Vectors
$$
\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix} \in \mathbb{R}^n
$$

### Matrix Multiplication
$$
\mathbf{C} = \mathbf{A}\mathbf{B} \quad \text{where} \quad C_{ij} = \sum_k A_{ik} B_{kj}
$$

### Eigendecomposition
$$
\mathbf{A}\mathbf{v} = \lambda\mathbf{v} \quad \Rightarrow \quad \mathbf{A} = \mathbf{V}\mathbf{\Lambda}\mathbf{V}^{-1}
$$

### Singular Value Decomposition (SVD)
$$
\mathbf{A} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T
$$

---

## 🎨 Visual Diagram

<div align="center">
<img src="./svg_figs/linear_algebra_overview.svg" alt="Linear Algebra Overview" width="100%"/>
</div>

---

## 💻 Google Colab - Ready to Run

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_linear_algebra)

```python
#@title 📐 Linear Algebra for Computer Vision - Complete Tutorial
#@markdown Click **Runtime → Run all** to execute everything!

# ============================================
# SETUP - Run this cell first!
# ============================================
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

print("✅ Setup complete!")
print(f"NumPy version: {np.__version__}")

# ============================================
# 1. VECTORS & MATRICES
# ============================================
print("\n" + "="*50)
print("1️⃣ VECTORS & MATRICES")
print("="*50)

# Create vectors (image pixels, features)
pixel_vector = np.array([128, 64, 255])  # RGB pixel
feature_vector = np.random.randn(128)    # CNN feature

print(f"RGB pixel: {pixel_vector}")
print(f"Feature vector shape: {feature_vector.shape}")

# Create matrix (grayscale image)
image = np.random.randint(0, 256, size=(4, 4))
print(f"\n4x4 'image':\n{image}")

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"\nA:\n{A}")
print(f"B:\n{B}")
print(f"A @ B (matrix multiplication):\n{A @ B}")
print(f"A * B (element-wise):\n{A * B}")

# ============================================
# 2. NORMS & DISTANCES
# ============================================
print("\n" + "="*50)
print("2️⃣ NORMS & DISTANCES")
print("="*50)

v = np.array([3, 4])
print(f"Vector v = {v}")
print(f"L1 norm (Manhattan): {np.linalg.norm(v, ord=1)}")
print(f"L2 norm (Euclidean): {np.linalg.norm(v, ord=2)}")
print(f"L∞ norm (Max): {np.linalg.norm(v, ord=np.inf)}")

# Cosine similarity (used in CLIP, embedding matching)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
cosine_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"\nCosine similarity: {cosine_sim:.4f}")

# ============================================
# 3. EIGENVALUES & PCA
# ============================================
print("\n" + "="*50)
print("3️⃣ EIGENVALUES & PCA")
print("="*50)

# Covariance matrix (symmetric)
data = np.random.randn(100, 3)  # 100 samples, 3 features
cov_matrix = np.cov(data.T)
print(f"Covariance matrix:\n{cov_matrix}")

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
idx = eigenvalues.argsort()[::-1]  # Sort descending
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"\nEigenvalues (sorted): {eigenvalues}")
print(f"Explained variance ratio: {eigenvalues / eigenvalues.sum()}")

# PCA projection
k = 2  # Keep top 2 components
principal_components = eigenvectors[:, :k]
projected = data @ principal_components
print(f"\nOriginal shape: {data.shape} → Projected: {projected.shape}")

# ============================================
# 4. SVD FOR IMAGE COMPRESSION
# ============================================
print("\n" + "="*50)
print("4️⃣ SVD FOR IMAGE COMPRESSION")
print("="*50)

# Create sample image
np.random.seed(42)
image = np.random.randn(64, 64)

# SVD decomposition
U, S, Vt = np.linalg.svd(image, full_matrices=False)
print(f"U: {U.shape}, S: {S.shape}, Vt: {Vt.shape}")

# Low-rank approximation
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
ranks = [64, 20, 10, 5]

for ax, k in zip(axes, ranks):
    # Reconstruct with k singular values
    approx = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    error = np.linalg.norm(image - approx, 'fro') / np.linalg.norm(image, 'fro')
    
    ax.imshow(approx, cmap='gray')
    ax.set_title(f'Rank {k}\nError: {error:.2%}')
    ax.axis('off')

plt.suptitle('SVD Image Compression', fontsize=14)
plt.tight_layout()
plt.show()

# ============================================
# 5. TRANSFORMATIONS (ROTATION, SCALING)
# ============================================
print("\n" + "="*50)
print("5️⃣ GEOMETRIC TRANSFORMATIONS")
print("="*50)

# 2D Rotation matrix
theta = np.pi / 4  # 45 degrees
rotation = np.array([
    [np.cos(theta), -np.sin(theta)],
    [np.sin(theta), np.cos(theta)]
])
print(f"Rotation matrix (45°):\n{rotation}")

# Scale matrix
scale = np.array([[2, 0], [0, 0.5]])
print(f"\nScale matrix:\n{scale}")

# Apply to points
points = np.array([[1, 0], [0, 1], [1, 1]]).T  # 2x3
rotated = rotation @ points
scaled = scale @ points

print(f"\nOriginal points:\n{points.T}")
print(f"Rotated:\n{rotated.T}")
print(f"Scaled:\n{scaled.T}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
colors = ['red', 'green', 'blue']

for ax, pts, title in zip(axes, [points, rotated, scaled], 
                           ['Original', 'Rotated 45°', 'Scaled (2x, 0.5y)']):
    ax.scatter(pts[0], pts[1], c=colors, s=100)
    ax.set_xlim(-2, 3)
    ax.set_ylim(-2, 3)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_title(title)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("✅ LINEAR ALGEBRA TUTORIAL COMPLETE!")
print("="*50)
```

---

## 📊 Key Concepts Table

| Concept | Formula | CV Application |
|---------|---------|----------------|
| Matrix Multiplication | $C = AB$ | Neural network layers |
| Transpose | $A^T_{ij} = A_{ji}$ | Attention: $QK^T$ |
| Inverse | $AA^{-1} = I$ | Camera calibration |
| Eigenvalues | $Av = \lambda v$ | PCA, face recognition |
| SVD | $A = U\Sigma V^T$ | Image compression |
| Norm | $\|v\|_2 = \sqrt{\sum v_i^2}$ | Loss functions |

---

## ⚠️ Common Pitfalls

| Mistake | Solution |
|---------|----------|
| Matrix dimension mismatch | Check shapes: (m,n) @ (n,p) → (m,p) |
| Confusing @ and * | `@` = matrix mult, `*` = element-wise |
| Singular matrix inversion | Use pseudo-inverse or add regularization |
| Not normalizing before PCA | Center data (subtract mean) first |

---

## 🛠️ Mini-Project: Image Compression with SVD

**Goal:** Compress an image using different SVD ranks

```python
# Try this! Load your own image and compress it
from PIL import Image
import requests
from io import BytesIO

# Load image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
img = np.array(Image.open(BytesIO(requests.get(url).content)).convert('L'))

# Compress with different ranks
for rank in [100, 50, 20, 10]:
    U, S, Vt = np.linalg.svd(img, full_matrices=False)
    compressed = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    compression_ratio = (rank * (img.shape[0] + img.shape[1] + 1)) / img.size
    print(f"Rank {rank}: {compression_ratio:.1%} of original size")
```

---

## ❓ Interview Questions

### Q1: Why is SVD useful for image processing?
**Answer:** SVD decomposes an image into orthogonal components ordered by importance (singular values). Keeping top-k components gives optimal rank-k approximation, enabling compression and denoising.

### Q2: What's the difference between eigenvalues and singular values?
**Answer:** Eigenvalues exist only for square matrices. Singular values exist for any matrix. For symmetric positive semi-definite matrices, eigenvalues = singular values.

### Q3: Why do we normalize features before computing cosine similarity?
**Answer:** Cosine similarity measures angle, not magnitude. Normalizing ensures we compare directions only, making it robust to feature scaling.

---

## 📚 Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra)
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf)
- *Linear Algebra Done Right* - Sheldon Axler

---

<div align="center">

**[🏠 Module Home](./README.md) | [Probability & Statistics →](./Probability_Statistics.md)**

</div>


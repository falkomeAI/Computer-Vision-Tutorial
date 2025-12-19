<div align="center">

# 🤖 Classical Machine Learning for Vision

### *PCA, SVM, k-means, Random Forests*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ageron/handson-ml2/blob/master/08_dimensionality_reduction.ipynb)

</div>

---

**Navigation:** [← Geometry & Multi-View](../06_Geometry_MultiView/) | [🏠 Home](../README.md) | [Neural Networks →](../08_Neural_Networks/)

---

## 📖 Topics Covered
- Dimensionality Reduction (PCA, LDA)
- Clustering (k-means, GMM)
- Classification (SVM, Random Forest)
- Graphical Models (HMM, CRF)

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/svm_kernel.svg" alt="SVM and Kernel Methods" width="100%"/>
</div>

---

## 📉 Dimensionality Reduction

### PCA

```python
from sklearn.decomposition import PCA

# Fit PCA
pca = PCA(n_components=50)
X_reduced = pca.fit_transform(X)

# Explained variance
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# Eigenfaces
pca = PCA(n_components=100)
eigenfaces = pca.fit_transform(face_data)
```

### LDA (Linear Discriminant Analysis)

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LDA(n_components=min(n_classes-1, n_features))
X_lda = lda.fit_transform(X, y)
```

---

## 🎯 Clustering

### k-Means

```python
from sklearn.cluster import KMeans

# Cluster image colors
pixels = img.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=42)
labels = kmeans.fit_predict(pixels)
quantized = kmeans.cluster_centers_[labels].reshape(img.shape)
```

### Gaussian Mixture Models

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X)
labels = gmm.predict(X)
probabilities = gmm.predict_proba(X)
```

---

## 📊 Classification

### SVM

```python
from sklearn.svm import SVC

# RBF kernel SVM
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)
predictions = svm.predict(X_test)

# HOG + SVM for pedestrian detection
hog_features = compute_hog(images)
svm.fit(hog_features, labels)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, max_depth=10)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
```

---

## 🔗 Graphical Models

### Hidden Markov Models

```python
from hmmlearn import hmm

model = hmm.GaussianHMM(n_components=3, covariance_type="full")
model.fit(observations)
hidden_states = model.predict(observations)
```

---

## ❓ Interview Questions & Answers

### Q1: PCA vs LDA?
| PCA | LDA |
|-----|-----|
| Unsupervised | Supervised |
| Max variance | Max class separation |
| No label info | Uses labels |
| Up to d components | Up to c-1 components |

### Q2: How does SVM work for image classification?
**Answer:**
1. Extract features (HOG, SIFT, CNN)
2. Flatten to vectors
3. Find hyperplane maximizing margin
4. Kernel trick for non-linear (RBF, polynomial)

### Q3: When to use k-means vs GMM?
| k-means | GMM |
|---------|-----|
| Hard assignment | Soft (probabilistic) |
| Spherical clusters | Elliptical clusters |
| Faster | More flexible |
| Sensitive to outliers | More robust |

### Q4: What is the kernel trick?
**Answer:** Map data to higher dimensions without explicit computation:
- K(x, y) = φ(x)·φ(y)
- RBF: K(x,y) = exp(-γ||x-y||²)
- Polynomial: K(x,y) = (x·y + c)^d

### Q5: How does AdaBoost work?
**Answer:**
1. Train weak classifier on weighted data
2. Increase weights for misclassified samples
3. Combine weak classifiers with weights
4. Final: H(x) = sign(Σ α_t h_t(x))

---

## 📓 Colab Notebooks

| Topic | Link |
|-------|------|
| PCA | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ageron/handson-ml2/blob/master/08_dimensionality_reduction.ipynb) |
| SVM | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ageron/handson-ml2/blob/master/05_support_vector_machines.ipynb) |
| Clustering | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ageron/handson-ml2/blob/master/09_unsupervised_learning.ipynb) |

---

<div align="center">

**[← Geometry & Multi-View](../06_Geometry_MultiView/) | [🏠 Home](../README.md) | [Neural Networks →](../08_Neural_Networks/)**

</div>

<div align="center">

# 🧠 Neural Network Foundations

### *MLP, Backpropagation, Optimization*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/beginner_source/basics/buildmodel_tutorial.ipynb)

</div>

---

**Navigation:** [← Classical ML](../07_Classical_ML/) | [🏠 Home](../README.md) | [CNN Architectures →](../09_CNN_Architectures/)

---

## 📖 Topics Covered
- Perceptron & MLP
- Activation Functions
- Backpropagation
- Optimization (SGD, Adam)
- Regularization
- Initialization

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/backpropagation.svg" alt="Backpropagation" width="100%"/>
</div>

---

## 🔮 Perceptron & MLP

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.layers(x)
```

---

## ⚡ Activation Functions

```python
# ReLU: max(0, x)
relu = nn.ReLU()

# Leaky ReLU: max(αx, x)
leaky = nn.LeakyReLU(0.01)

# GELU: x * Φ(x)
gelu = nn.GELU()

# Sigmoid: 1 / (1 + e^-x)
sigmoid = nn.Sigmoid()

# Softmax: e^xi / Σe^xj
softmax = nn.Softmax(dim=1)
```

---

## 🔄 Backpropagation

```python
# Forward pass
y_pred = model(x)
loss = criterion(y_pred, y)

# Backward pass
optimizer.zero_grad()
loss.backward()  # Compute gradients
optimizer.step()  # Update weights

# Chain rule: ∂L/∂w = ∂L/∂y × ∂y/∂w
```

---

## 📈 Optimization

```python
# SGD with momentum
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Adam (adaptive moment estimation)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# AdamW (decoupled weight decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
```

---

## 🛡️ Regularization

```python
# Dropout
dropout = nn.Dropout(p=0.5)

# L2 regularization (weight decay)
optimizer = torch.optim.Adam(params, weight_decay=1e-4)

# Batch normalization
bn = nn.BatchNorm1d(num_features)

# Label smoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
```

---

## 🎲 Initialization

```python
# Xavier/Glorot
nn.init.xavier_uniform_(layer.weight)

# Kaiming/He (for ReLU)
nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')

# Why? Maintain variance across layers
# Xavier: Var(w) = 2/(fan_in + fan_out)
# Kaiming: Var(w) = 2/fan_in
```

---

## ❓ Interview Questions & Answers

### Q1: Why ReLU over Sigmoid?
| ReLU | Sigmoid |
|------|---------|
| No vanishing gradient (for x>0) | Saturates at 0, 1 |
| Sparse activation | Always non-zero |
| Computationally simple | exp() is expensive |
| Dead neurons possible | No dead neurons |

### Q2: What is the vanishing gradient problem?
**Answer:** In deep networks, gradients become tiny through chain rule multiplication:
- Sigmoid derivative max = 0.25
- After n layers: 0.25^n → 0
- Solution: ReLU, skip connections, proper init

### Q3: SGD vs Adam?
| SGD | Adam |
|-----|------|
| One learning rate | Per-parameter adaptive |
| Needs tuning | Works out of box |
| Better generalization | Faster convergence |
| Simpler | More memory |

### Q4: What is batch normalization?
**Answer:**
1. Normalize activations: x̂ = (x - μ) / σ
2. Scale and shift: y = γx̂ + β
3. Benefits: faster training, higher LR, regularization

### Q5: Why weight decay/L2 regularization?
**Answer:**
- Penalty: L + λ||w||²
- Prevents large weights
- Encourages simpler models
- Reduces overfitting

---

## 📓 Colab Notebooks

| Topic | Link |
|-------|------|
| Build Model | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/beginner_source/basics/buildmodel_tutorial.ipynb) |
| Optimization | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/beginner_source/basics/optimization_tutorial.ipynb) |
| Autograd | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/beginner_source/basics/autogradqs_tutorial.ipynb) |

---

<div align="center">

**[← Classical ML](../07_Classical_ML/) | [🏠 Home](../README.md) | [CNN Architectures →](../09_CNN_Architectures/)**

</div>

<div align="center">

# ⚖️ Ethics, Safety & Robustness

### *Bias, Adversarial, Privacy, Fairness*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/beginner_source/fgsm_tutorial.ipynb)

</div>

---

**Navigation:** [← Deployment & Systems](../18_Deployment_Systems/) | [🏠 Home](../README.md) | [Research Frontiers →](../20_Research_Frontiers/)

---

## 📖 Topics Covered
- Adversarial Attacks
- Robustness & Defense
- Fairness & Bias
- Privacy-Preserving Vision
- Explainability

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/adversarial_attacks.svg" alt="Adversarial Attacks" width="100%"/>
</div>

---

## ⚔️ Adversarial Attacks

### FGSM (Fast Gradient Sign Method)

```python
def fgsm_attack(image, epsilon, gradient):
    """
    x_adv = x + ε * sign(∇_x L(θ, x, y))
    """
    perturbation = epsilon * gradient.sign()
    adv_image = image + perturbation
    return torch.clamp(adv_image, 0, 1)

# Generate adversarial example
image.requires_grad = True
output = model(image)
loss = criterion(output, target)
loss.backward()
adv_image = fgsm_attack(image, epsilon=0.03, gradient=image.grad)
```

### PGD (Projected Gradient Descent)

```python
def pgd_attack(model, image, target, epsilon=0.03, alpha=0.01, iters=40):
    adv = image.clone().detach()
    
    for _ in range(iters):
        adv.requires_grad = True
        output = model(adv)
        loss = criterion(output, target)
        loss.backward()
        
        # Update
        adv = adv + alpha * adv.grad.sign()
        
        # Project back to epsilon-ball
        perturbation = torch.clamp(adv - image, -epsilon, epsilon)
        adv = torch.clamp(image + perturbation, 0, 1).detach()
    
    return adv
```

---

## 🛡️ Defenses

### Adversarial Training

```python
def adversarial_training_step(model, images, labels, epsilon=0.03):
    # Generate adversarial examples
    adv_images = pgd_attack(model, images, labels, epsilon)
    
    # Train on mix of clean and adversarial
    clean_output = model(images)
    adv_output = model(adv_images)
    
    loss = 0.5 * criterion(clean_output, labels) + 0.5 * criterion(adv_output, labels)
    return loss
```

### Certified Defense

```python
# Randomized smoothing
def certified_predict(model, x, sigma=0.25, n=1000):
    # Sample noise
    noise = torch.randn(n, *x.shape) * sigma
    noisy_inputs = x + noise
    
    # Get predictions
    predictions = model(noisy_inputs).argmax(dim=1)
    
    # Majority vote
    counts = torch.bincount(predictions)
    return counts.argmax()
```

---

## ⚖️ Fairness & Bias

```python
# Check demographic parity
def demographic_parity(predictions, protected_attribute):
    """
    P(Ŷ=1 | A=0) = P(Ŷ=1 | A=1)
    """
    group_0 = predictions[protected_attribute == 0].mean()
    group_1 = predictions[protected_attribute == 1].mean()
    return abs(group_0 - group_1)

# Equalized odds
def equalized_odds(predictions, labels, protected_attribute):
    """
    P(Ŷ=1 | Y=y, A=0) = P(Ŷ=1 | Y=y, A=1) for y ∈ {0,1}
    """
    tpr_diff = abs(
        (predictions[(labels==1) & (protected_attribute==0)]).mean() -
        (predictions[(labels==1) & (protected_attribute==1)]).mean()
    )
    return tpr_diff
```

---

## 🔒 Privacy-Preserving

```python
# Differential Privacy with PyTorch
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, dataloader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=dataloader,
    noise_multiplier=1.0,  # Privacy-utility tradeoff
    max_grad_norm=1.0,
)

# Training gives (ε, δ)-differential privacy
epsilon = privacy_engine.get_epsilon(delta=1e-5)
```

---

## 🔍 Explainability

```python
# Grad-CAM
def grad_cam(model, image, target_class):
    # Get activations and gradients
    activations = model.get_activations(image)
    model.zero_grad()
    output = model(image)
    output[0, target_class].backward()
    gradients = model.get_gradients()
    
    # Global average pooling of gradients
    weights = gradients.mean(dim=[2, 3], keepdim=True)
    
    # Weighted combination of activations
    cam = F.relu((weights * activations).sum(dim=1))
    return cam

# SHAP for feature importance
import shap
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_images)
```

---

## ❓ Interview Questions & Answers

### Q1: Why do adversarial examples exist?
**Answer:**
- Linear hypothesis: DNNs are too linear in high dimensions
- Small perturbation × many dimensions = large effect
- Models exploit spurious correlations
- Not robust to distribution shift

### Q2: FGSM vs PGD?
| FGSM | PGD |
|------|-----|
| One step | Multiple steps |
| Faster | Stronger attack |
| Less effective | Gold standard |
| ε-bounded | ε-bounded + projection |

### Q3: What is model bias in CV?
**Answer:**
- Training data imbalance
- Historical bias in labels
- Representation bias
- Examples: face recognition accuracy varies by demographics
- Mitigation: balanced data, fair loss functions, auditing

### Q4: How does differential privacy work?
**Answer:**
- Add calibrated noise to gradients
- Limit individual sample influence
- (ε, δ)-guarantee: bounded privacy loss
- Trade-off: more noise = more privacy = less accuracy

### Q5: What is the robustness-accuracy trade-off?
**Answer:**
- Adversarially trained models have lower clean accuracy
- Standard models: ~97% clean, ~0% adversarial
- Robust models: ~87% clean, ~50% adversarial
- Fundamental trade-off, not just optimization issue

---

## 📓 Colab Notebooks

| Topic | Link |
|-------|------|
| FGSM Attack | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/beginner_source/fgsm_tutorial.ipynb) |
| Grad-CAM | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jacobgil/pytorch-grad-cam/blob/master/tutorials/Class%20Activation%20Maps%20for%20Object%20Detection.ipynb) |
| Fairness | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Trusted-AI/AIF360/blob/master/examples/tutorial_medical_expenditure.ipynb) |

---

<div align="center">

**[← Deployment & Systems](../18_Deployment_Systems/) | [🏠 Home](../README.md) | [Research Frontiers →](../20_Research_Frontiers/)**

</div>

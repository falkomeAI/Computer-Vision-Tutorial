<div align="center">

# 🔬 Research Frontiers

### *Foundation Models, World Models, Open-Vocabulary*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb)

</div>

---

**Navigation:** [← Ethics & Safety](../19_Ethics_Safety/) | [🏠 Home](../README.md)

---

## 📖 Topics Covered
- Foundation Models (SAM, DINOv2)
- Open-Vocabulary Detection
- World Models
- Continual Learning
- Neuro-Symbolic Vision

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/foundation_models.svg" alt="Foundation Models" width="100%"/>
</div>

---

## 🏛️ Foundation Models

### Segment Anything (SAM)

```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)

predictor.set_image(image)

# Point prompt
masks, scores, _ = predictor.predict(
    point_coords=np.array([[500, 375]]),
    point_labels=np.array([1]),  # 1 = foreground
    multimask_output=True
)

# Box prompt
masks, _, _ = predictor.predict(box=np.array([100, 100, 400, 400]))
```

### DINOv2

```python
import torch
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# Extract features
with torch.no_grad():
    features = dinov2(images)  # (B, 1536)
    
# Dense features for segmentation
features = dinov2.forward_features(images)
patch_tokens = features['x_norm_patchtokens']  # (B, N, D)
```

---

## 🔓 Open-Vocabulary Detection

```python
# OWL-ViT: Open-World Localization
from transformers import OwlViTProcessor, OwlViTForObjectDetection

processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")

# Detect any object by text
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)

# Grounding DINO
from groundingdino.util.inference import predict
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption="person . dog . car",  # Any text queries
    box_threshold=0.35
)
```

---

## 🌍 World Models

```python
# World models learn environment dynamics
# x_{t+1} = f(x_t, a_t) - predict next state given action

class WorldModel(nn.Module):
    def __init__(self):
        self.encoder = ViTEncoder()  # Observation → latent
        self.dynamics = RSSM()       # Latent dynamics
        self.decoder = CNNDecoder()  # Latent → observation
        self.reward = RewardHead()   # Predict rewards
    
    def imagine(self, state, actions):
        """Imagine future states without environment"""
        imagined_states = [state]
        for action in actions:
            state = self.dynamics(state, action)
            imagined_states.append(state)
        return imagined_states

# Dreamer: Learn policy in imagination
# GAIA-1: World model for autonomous driving
```

---

## 🔄 Continual Learning

```python
# Elastic Weight Consolidation (EWC)
class EWC:
    def __init__(self, model, dataset, importance=1000):
        self.params = {n: p.clone() for n, p in model.named_parameters()}
        self.fisher = self._compute_fisher(model, dataset)
        self.importance = importance
    
    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            loss += (self.fisher[n] * (p - self.params[n])**2).sum()
        return self.importance * loss

# PackNet: Prune and pack networks for each task
# LoRA: Efficient adaptation with low-rank updates
```

---

## 🧩 Neuro-Symbolic Vision

```python
# Combine neural perception with symbolic reasoning
class NeuroSymbolic(nn.Module):
    def __init__(self):
        self.perception = DETR()  # Object detection
        self.reasoning = LogicModule()  # Symbolic reasoning
    
    def forward(self, image, query):
        # Perceive objects
        objects = self.perception(image)
        
        # Symbolic scene graph
        scene_graph = self.build_graph(objects)
        
        # Reason about query
        answer = self.reasoning(scene_graph, query)
        return answer

# Visual Programming: LLM generates code to compose modules
# VISPROG: "Find the red object to the left of the dog"
```

---

## 🎯 Zero/Few-Shot Learning

```python
# In-context learning for vision
def visual_prompting(model, support_images, support_labels, query_image):
    """
    Similar to language prompting but for images
    """
    # Create visual prompt: interleave support examples
    prompt = []
    for img, label in zip(support_images, support_labels):
        prompt.append(img)
        prompt.append(label_token(label))
    prompt.append(query_image)
    
    # Model completes the pattern
    prediction = model(prompt)
    return prediction

# Segment Anything: zero-shot segmentation
# CLIP: zero-shot classification
```

---

## ❓ Interview Questions & Answers

### Q1: What makes SAM a foundation model?
**Answer:**
- Trained on 11M images, 1.1B masks
- Promptable: points, boxes, text
- Zero-shot generalization
- Task-agnostic backbone
- Enables many downstream tasks

### Q2: Open-vocabulary vs closed-set detection?
| Closed-set | Open-vocabulary |
|------------|-----------------|
| Fixed classes | Any text query |
| Training classes only | Novel classes |
| Softmax output | CLIP-like matching |
| Limited flexibility | Flexible |

### Q3: Why are world models important?
**Answer:**
- Learn environment dynamics
- Plan in imagination (sample efficient)
- Transfer across tasks
- Key for embodied AI / robotics
- Examples: Dreamer, GAIA-1

### Q4: What is catastrophic forgetting?
**Answer:**
- Neural networks forget old tasks when learning new
- Weights overwritten for new data
- Solutions: EWC, PackNet, replay, LoRA
- Active research area

### Q5: How does visual prompting work?
**Answer:**
- Show examples in input (like language ICL)
- Model learns pattern from examples
- No weight updates needed
- Enables few-shot adaptation
- Examples: Painter, SegGPT

---

## 📓 Colab Notebooks

| Topic | Link |
|-------|------|
| SAM | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb) |
| DINOv2 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb) |
| Grounding DINO | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IDEA-Research/GroundingDINO/blob/main/demo/inference_on_a_image.ipynb) |

---

<div align="center">

**[← Ethics & Safety](../19_Ethics_Safety/) | [🏠 Home](../README.md)**

</div>

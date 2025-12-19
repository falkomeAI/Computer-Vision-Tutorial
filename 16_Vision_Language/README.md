<div align="center">

# 🗣️ Vision + Language

### *CLIP, VQA, Captioning, Multimodal*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb)

</div>

---

**Navigation:** [← Generative Vision](../15_Generative_Vision/) | [🏠 Home](../README.md) | [Computational Photography →](../17_Computational_Photography/)

---

## 📖 Topics Covered
- CLIP (Contrastive Language-Image)
- Image Captioning
- Visual Question Answering
- Grounding & Detection
- Multimodal LLMs

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/clip_architecture.svg" alt="CLIP Architecture" width="100%"/>
</div>

---

## 🔗 CLIP

```python
import clip
import torch

model, preprocess = clip.load("ViT-B/32")

# Zero-shot classification
image = preprocess(Image.open("dog.jpg")).unsqueeze(0)
text = clip.tokenize(["a dog", "a cat", "a bird"])

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    # Cosine similarity
    similarity = (image_features @ text_features.T).softmax(dim=-1)
    print(similarity)  # [0.95, 0.03, 0.02]
```

### CLIP Training

```python
# Contrastive loss on image-text pairs
def clip_loss(image_features, text_features, temperature=0.07):
    # Normalize
    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)
    
    # Similarity matrix
    logits = image_features @ text_features.T / temperature
    
    # Labels: diagonal is positive
    labels = torch.arange(len(logits), device=logits.device)
    
    # Symmetric loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2
```

---

## 📝 Image Captioning

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

inputs = processor(image, return_tensors="pt")
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)
```

---

## ❓ Visual Question Answering

```python
from transformers import ViltProcessor, ViltForQuestionAnswering

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

encoding = processor(image, question, return_tensors="pt")
outputs = model(**encoding)
answer = model.config.id2label[outputs.logits.argmax(-1).item()]
```

---

## 🎯 Grounding & Detection

```python
# Grounding DINO: Open-vocabulary detection
from groundingdino.util.inference import load_model, predict

model = load_model("GroundingDINO_SwinT_OGC")
boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption="dog . cat . person",  # Text queries
    box_threshold=0.35
)
```

---

## 🤖 Multimodal LLMs

```python
# LLaVA
from transformers import LlavaProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

prompt = "USER: <image>\nWhat is in this image?\nASSISTANT:"
inputs = processor(prompt, image, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
```

---

## ❓ Interview Questions & Answers

### Q1: How does CLIP enable zero-shot classification?
**Answer:**
1. Encode image and text prompts separately
2. Compute cosine similarity
3. Pick class with highest similarity
4. No training on target classes needed
5. Works because of large-scale pretraining

### Q2: CLIP vs BLIP?
| CLIP | BLIP |
|------|------|
| Contrastive only | Contrastive + Generative |
| Dual encoder | Encoder-decoder |
| Classification | Generation tasks |
| Zero-shot | Captioning, VQA |

### Q3: What is visual grounding?
**Answer:** Localize objects/regions described in text:
- Input: image + text ("the red car on the left")
- Output: bounding box or segmentation mask
- Models: GroundingDINO, GLIPv2

### Q4: How do multimodal LLMs handle images?
**Answer:**
- Visual encoder → image tokens
- Project to LLM embedding space
- Interleave with text tokens
- LLM processes jointly
- Examples: LLaVA, GPT-4V, Gemini

### Q5: What is instruction tuning for vision?
**Answer:**
- Fine-tune on diverse vision-language tasks
- Follow natural language instructions
- "Describe this image", "Find the dog"
- Enables flexible, general-purpose models

---

## 📓 Colab Notebooks

| Topic | Link |
|-------|------|
| CLIP | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb) |
| BLIP | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/salesforce/BLIP/blob/main/demo.ipynb) |
| LLaVA | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/haotian-liu/LLaVA/blob/main/llava_demo.ipynb) |

---

<div align="center">

**[← Generative Vision](../15_Generative_Vision/) | [🏠 Home](../README.md) | [Computational Photography →](../17_Computational_Photography/)**

</div>

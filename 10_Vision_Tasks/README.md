<div align="center">

# 🎯 Vision Tasks

### *Detection, Segmentation, Pose Estimation*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1_GdGFTfLJECh_hcRqKqnIRYp9P8zrMmx)

</div>

---

**Navigation:** [← CNN Architectures](../09_CNN_Architectures/) | [🏠 Home](../README.md) | [Vision Transformers →](../11_Vision_Transformers/)

---

## 📖 Table of Contents
- [Visual Overview](#-visual-overview)
- [Complete Colab Code](#-complete-colab-code)
- [Object Detection](#-object-detection)
- [Segmentation](#-segmentation)
- [Interview Q&A](#-interview-questions--answers)

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/detection_architectures.svg" alt="Detection Architectures" width="100%"/>
</div>

---

## 📓 Complete Colab Code

### Copy this entire block to run in Google Colab:

```python
#@title 🎯 Vision Tasks - Complete Tutorial
#@markdown Run all cells to see Detection, Segmentation, and more!

# Install dependencies
!pip install torch torchvision opencv-python-headless matplotlib pillow ultralytics -q

import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import urllib.request
import cv2

# Download sample image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
urllib.request.urlretrieve(url, "test_image.jpg")

image = Image.open("test_image.jpg").convert("RGB")
print("✅ Setup complete! Image size:", image.size)

#@title 1️⃣ Object Detection with Faster R-CNN

def run_detection(image):
    """Run Faster R-CNN object detection"""
    
    # Load pretrained model
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    
    # Transform
    transform = weights.transforms()
    img_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    # Get class names
    categories = weights.meta["categories"]
    
    # Visualize
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw boxes
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if score > 0.5:  # Confidence threshold
            x1, y1, x2, y2 = box.numpy()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{categories[label]}: {score:.2f}", 
                   fontsize=10, color='red', weight='bold',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    ax.set_title("Faster R-CNN Object Detection")
    ax.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print detections
    print("\n📦 Detected Objects:")
    for label, score in zip(predictions['labels'], predictions['scores']):
        if score > 0.5:
            print(f"  - {categories[label]}: {score:.2%}")
    
    return predictions

predictions = run_detection(image)
print("\n✅ Detection complete!")

#@title 2️⃣ YOLOv8 Detection (State-of-the-Art)

def run_yolo():
    """Run YOLOv8 detection"""
    from ultralytics import YOLO
    
    # Load YOLOv8
    model = YOLO('yolov8n.pt')  # nano model
    
    # Run inference
    results = model("test_image.jpg")
    
    # Visualize
    fig, ax = plt.subplots(1, figsize=(12, 8))
    
    # Get annotated image
    annotated = results[0].plot()
    ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax.set_title("YOLOv8 Detection")
    ax.axis('off')
    plt.show()
    
    # Print results
    print("\n🎯 YOLOv8 Results:")
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f"  - {model.names[cls]}: {conf:.2%}")

run_yolo()
print("\n✅ YOLOv8 complete!")

#@title 3️⃣ Semantic Segmentation with DeepLabV3

def run_segmentation(image):
    """Run DeepLabV3 semantic segmentation"""
    
    # Load model
    weights = DeepLabV3_ResNet101_Weights.DEFAULT
    model = deeplabv3_resnet101(weights=weights)
    model.eval()
    
    # Transform
    transform = weights.transforms()
    img_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        output = model(img_tensor)['out'][0]
    
    # Get prediction mask
    pred_mask = output.argmax(0).numpy()
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap='tab20')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    # Overlay
    mask_colored = plt.cm.tab20(pred_mask / 20)[:, :, :3]
    overlay = np.array(image.resize(pred_mask.shape[::-1])) / 255 * 0.5 + mask_colored * 0.5
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print classes found
    unique_classes = np.unique(pred_mask)
    categories = weights.meta["categories"]
    print("\n🎨 Segmentation Classes Found:")
    for cls_id in unique_classes:
        print(f"  - {categories[cls_id]}")
    
    return pred_mask

mask = run_segmentation(image)
print("\n✅ Segmentation complete!")

#@title 4️⃣ Instance Segmentation with Mask R-CNN

def run_instance_segmentation(image):
    """Run Mask R-CNN for instance segmentation"""
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    
    # Load model
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights)
    model.eval()
    
    # Transform
    transform = weights.transforms()
    img_tensor = transform(image).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        predictions = model(img_tensor)[0]
    
    categories = weights.meta["categories"]
    
    # Visualize
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)
    
    # Draw masks and boxes
    colors = plt.cm.rainbow(np.linspace(0, 1, len(predictions['masks'])))
    
    for idx, (mask, box, label, score) in enumerate(zip(
        predictions['masks'], predictions['boxes'], predictions['labels'], predictions['scores']
    )):
        if score > 0.5:
            # Mask
            mask_np = mask[0].numpy() > 0.5
            colored_mask = np.zeros((*mask_np.shape, 4))
            colored_mask[mask_np] = [*colors[idx][:3], 0.5]
            ax.imshow(colored_mask)
            
            # Box
            x1, y1, x2, y2 = box.numpy()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=2, edgecolor=colors[idx], facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, f"{categories[label]}: {score:.2f}", 
                   fontsize=10, color='white', weight='bold',
                   bbox=dict(boxstyle='round', facecolor=colors[idx], alpha=0.8))
    
    ax.set_title("Mask R-CNN Instance Segmentation")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

run_instance_segmentation(image)
print("\n✅ Instance segmentation complete!")

#@title 5️⃣ Metrics: IoU and mAP

def calculate_iou(box1, box2):
    """Calculate Intersection over Union"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# Example
box_pred = [100, 100, 200, 200]
box_gt = [110, 110, 210, 210]
iou = calculate_iou(box_pred, box_gt)
print(f"📐 IoU between boxes: {iou:.2%}")

# Visualize IoU
fig, ax = plt.subplots(figsize=(8, 6))
rect1 = patches.Rectangle((100, 100), 100, 100, linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3, label='Predicted')
rect2 = patches.Rectangle((110, 110), 100, 100, linewidth=2, edgecolor='red', facecolor='red', alpha=0.3, label='Ground Truth')
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.set_xlim(0, 300)
ax.set_ylim(0, 300)
ax.legend()
ax.set_title(f"IoU Visualization: {iou:.2%}")
plt.gca().invert_yaxis()
plt.show()

print("\n" + "="*50)
print("✅ ALL VISION TASKS COMPLETE!")
print("="*50)
```

---

## 🖼️ Object Detection

### Two-Stage vs One-Stage

| Feature | Two-Stage (Faster R-CNN) | One-Stage (YOLO) |
|---------|-------------------------|------------------|
| **Pipeline** | Region proposal → Classification | Direct prediction |
| **Speed** | ~5 FPS | ~100+ FPS |
| **Accuracy** | Higher on small objects | Comparable |
| **Use Case** | Accuracy-critical | Real-time |

### Key Metrics

```python
# IoU (Intersection over Union)
IoU = Area_Intersection / Area_Union

# mAP (mean Average Precision)
mAP = mean(AP per class)
# AP@0.5 = AP at IoU threshold 0.5
# AP@0.5:0.95 = average over [0.5, 0.55, ..., 0.95]
```

---

## 🎨 Segmentation

### Types Comparison

| Type | Output | Example |
|------|--------|---------|
| **Semantic** | Class per pixel | All "person" same label |
| **Instance** | Object per pixel | Each person separate |
| **Panoptic** | Both | Everything labeled + instances |

---

## ❓ Interview Questions & Answers

### Q1: How does Non-Maximum Suppression (NMS) work?
**Answer:**
1. Sort boxes by confidence score
2. Select box with highest score, add to output
3. Remove all boxes with IoU > threshold with selected
4. Repeat until no boxes remain
```python
def nms(boxes, scores, threshold=0.5):
    indices = scores.argsort()[::-1]
    keep = []
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        ious = calculate_ious(boxes[i], boxes[indices[1:]])
        indices = indices[1:][ious <= threshold]
    return keep
```

### Q2: What is Feature Pyramid Network (FPN)?
**Answer:** Multi-scale feature extraction:
- Bottom-up: backbone extracts features
- Top-down: upsample and merge
- Lateral connections: combine features
- Detect at multiple scales for different object sizes

### Q3: YOLO vs Faster R-CNN trade-offs?
| YOLO | Faster R-CNN |
|------|--------------|
| Single pass | Two passes |
| Grid-based | Region-based |
| Real-time | More accurate |
| Struggles with small | Better for small objects |

### Q4: What is anchor-free detection?
**Answer:** Predict boxes without predefined anchors:
- **FCOS**: Predict from each feature point
- **CenterNet**: Detect object centers + size
- **CornerNet**: Detect corner pairs
- Benefits: No anchor tuning, simpler

### Q5: Semantic vs Instance vs Panoptic?
| Semantic | Instance | Panoptic |
|----------|----------|----------|
| Class labels | Object instances | Both |
| "Sky is sky" | "Person 1, Person 2" | All combined |
| FCN, DeepLab | Mask R-CNN | Panoptic FPN |

---

## 📓 More Colab Resources

| Topic | Link |
|-------|------|
| TorchVision Detection | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pytorch/tutorials/blob/main/intermediate_source/torchvision_tutorial.ipynb) |
| YOLOv8 Tutorial | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb) |
| Segment Anything | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb) |

---

<div align="center">

**[← CNN Architectures](../09_CNN_Architectures/) | [🏠 Home](../README.md) | [Vision Transformers →](../11_Vision_Transformers/)**

</div>

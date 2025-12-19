<div align="center">

# 🔧 Low-Level Image Processing

### *Filtering, Enhancement, Restoration*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dZ0RnPnLxPHzZCq0V8lJHdVl0lJEzxHa)

</div>

---

**Navigation:** [← Image Formation](../03_Image_Formation/) | [🏠 Home](../README.md) | [Feature Detection →](../05_Features_Detection/)

---

## 📖 Table of Contents
- [Visual Overview](#-visual-overview)
- [Complete Colab Code](#-complete-colab-code)
- [Histogram Processing](#-histogram-processing)
- [Spatial Filtering](#-spatial-filtering)
- [Edge Detection](#-edge-detection)
- [Interview Q&A](#-interview-questions--answers)

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/filtering_types.svg" alt="Filtering Types" width="100%"/>
</div>

---

## 📓 Complete Colab Code

### Copy this entire block to run in Google Colab:

```python
#@title 🔧 Low-Level Image Processing - Complete Tutorial
#@markdown Run this cell to install dependencies and set up

!pip install opencv-python-headless matplotlib numpy scipy scikit-image -q

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import data, filters, exposure
from google.colab import files
from io import BytesIO
from PIL import Image
import urllib.request

# Download sample image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/300px-PNG_transparency_demonstration_1.png"
urllib.request.urlretrieve(url, "sample.png")
image = cv2.imread("sample.png")
if image is None:
    # Fallback: create synthetic image
    image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

print("✅ Setup complete! Image shape:", image.shape)

#@title 1️⃣ Histogram Equalization
def histogram_demo(img):
    """Demonstrate histogram equalization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Original histogram
    axes[1, 0].hist(img.ravel(), bins=256, range=(0, 256), color='blue', alpha=0.7)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    
    # Standard equalization
    equalized = cv2.equalizeHist(img)
    axes[0, 1].imshow(equalized, cmap='gray')
    axes[0, 1].set_title('Histogram Equalized')
    axes[0, 1].axis('off')
    
    axes[1, 1].hist(equalized.ravel(), bins=256, range=(0, 256), color='green', alpha=0.7)
    axes[1, 1].set_title('Equalized Histogram')
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(img)
    axes[0, 2].imshow(clahe_img, cmap='gray')
    axes[0, 2].set_title('CLAHE (Adaptive)')
    axes[0, 2].axis('off')
    
    axes[1, 2].hist(clahe_img.ravel(), bins=256, range=(0, 256), color='red', alpha=0.7)
    axes[1, 2].set_title('CLAHE Histogram')
    
    plt.tight_layout()
    plt.show()
    
    return equalized, clahe_img

eq_img, clahe_img = histogram_demo(gray)
print("📊 Histogram equalization complete!")

#@title 2️⃣ Spatial Filtering (Blur, Sharpen, Denoise)
def filtering_demo(img):
    """Demonstrate different filtering operations"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Box blur
    box_blur = cv2.blur(img, (5, 5))
    axes[0, 1].imshow(box_blur, cmap='gray')
    axes[0, 1].set_title('Box Blur (5×5)')
    axes[0, 1].axis('off')
    
    # Gaussian blur
    gaussian_blur = cv2.GaussianBlur(img, (5, 5), sigmaX=1.5)
    axes[0, 2].imshow(gaussian_blur, cmap='gray')
    axes[0, 2].set_title('Gaussian Blur (σ=1.5)')
    axes[0, 2].axis('off')
    
    # Median filter
    median_blur = cv2.medianBlur(img, 5)
    axes[0, 3].imshow(median_blur, cmap='gray')
    axes[0, 3].set_title('Median Filter (5×5)')
    axes[0, 3].axis('off')
    
    # Bilateral filter
    bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    axes[1, 0].imshow(bilateral, cmap='gray')
    axes[1, 0].set_title('Bilateral Filter')
    axes[1, 0].axis('off')
    
    # Sharpening
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel_sharpen)
    axes[1, 1].imshow(sharpened, cmap='gray')
    axes[1, 1].set_title('Sharpening')
    axes[1, 1].axis('off')
    
    # Unsharp masking
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    unsharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    axes[1, 2].imshow(unsharp, cmap='gray')
    axes[1, 2].set_title('Unsharp Mask')
    axes[1, 2].axis('off')
    
    # Non-local means denoising
    denoised = cv2.fastNlMeansDenoising(img, h=10, templateWindowSize=7, searchWindowSize=21)
    axes[1, 3].imshow(denoised, cmap='gray')
    axes[1, 3].set_title('Non-Local Means')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
filtering_demo(gray)
print("🎛️ Spatial filtering complete!")

#@title 3️⃣ Edge Detection
def edge_detection_demo(img):
    """Demonstrate edge detection methods"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Sobel X
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    axes[0, 1].imshow(np.abs(sobel_x), cmap='gray')
    axes[0, 1].set_title('Sobel X (Vertical edges)')
    axes[0, 1].axis('off')
    
    # Sobel Y
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    axes[0, 2].imshow(np.abs(sobel_y), cmap='gray')
    axes[0, 2].set_title('Sobel Y (Horizontal edges)')
    axes[0, 2].axis('off')
    
    # Sobel magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    axes[0, 3].imshow(magnitude, cmap='gray')
    axes[0, 3].set_title('Sobel Magnitude')
    axes[0, 3].axis('off')
    
    # Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    axes[1, 0].imshow(np.abs(laplacian), cmap='gray')
    axes[1, 0].set_title('Laplacian')
    axes[1, 0].axis('off')
    
    # Canny
    canny = cv2.Canny(img, 50, 150)
    axes[1, 1].imshow(canny, cmap='gray')
    axes[1, 1].set_title('Canny (50-150)')
    axes[1, 1].axis('off')
    
    # Canny with different thresholds
    canny2 = cv2.Canny(img, 100, 200)
    axes[1, 2].imshow(canny2, cmap='gray')
    axes[1, 2].set_title('Canny (100-200)')
    axes[1, 2].axis('off')
    
    # Gradient direction
    direction = np.arctan2(sobel_y, sobel_x)
    axes[1, 3].imshow(direction, cmap='hsv')
    axes[1, 3].set_title('Gradient Direction')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
edge_detection_demo(gray)
print("🔍 Edge detection complete!")

#@title 4️⃣ Morphological Operations
def morphology_demo(img):
    """Demonstrate morphological operations"""
    # Create binary image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(binary, cmap='gray')
    axes[0, 0].set_title('Binary Image')
    axes[0, 0].axis('off')
    
    # Erosion
    erosion = cv2.erode(binary, kernel, iterations=1)
    axes[0, 1].imshow(erosion, cmap='gray')
    axes[0, 1].set_title('Erosion')
    axes[0, 1].axis('off')
    
    # Dilation
    dilation = cv2.dilate(binary, kernel, iterations=1)
    axes[0, 2].imshow(dilation, cmap='gray')
    axes[0, 2].set_title('Dilation')
    axes[0, 2].axis('off')
    
    # Opening
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    axes[0, 3].imshow(opening, cmap='gray')
    axes[0, 3].set_title('Opening (Erode→Dilate)')
    axes[0, 3].axis('off')
    
    # Closing
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    axes[1, 0].imshow(closing, cmap='gray')
    axes[1, 0].set_title('Closing (Dilate→Erode)')
    axes[1, 0].axis('off')
    
    # Gradient
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    axes[1, 1].imshow(gradient, cmap='gray')
    axes[1, 1].set_title('Morphological Gradient')
    axes[1, 1].axis('off')
    
    # Top Hat
    tophat = cv2.morphologyEx(binary, cv2.MORPH_TOPHAT, kernel)
    axes[1, 2].imshow(tophat, cmap='gray')
    axes[1, 2].set_title('Top Hat')
    axes[1, 2].axis('off')
    
    # Black Hat
    blackhat = cv2.morphologyEx(binary, cv2.MORPH_BLACKHAT, kernel)
    axes[1, 3].imshow(blackhat, cmap='gray')
    axes[1, 3].set_title('Black Hat')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

morphology_demo(gray)
print("🔲 Morphological operations complete!")

#@title 5️⃣ Custom Convolution Kernels
def custom_kernels_demo(img):
    """Demonstrate custom convolution kernels"""
    
    # Define various kernels
    kernels = {
        'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        'Edge Detect': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'Emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        'Box Blur': np.ones((3, 3)) / 9,
        'Gaussian 3x3': np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16,
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
        'Prewitt X': np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
        'Prewitt Y': np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (name, kernel) in enumerate(kernels.items()):
        result = cv2.filter2D(img, -1, kernel)
        axes[idx].imshow(result, cmap='gray')
        axes[idx].set_title(f'{name}\n{kernel.shape[0]}×{kernel.shape[1]}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print kernels
    print("\n📋 Kernel Values:")
    for name, kernel in kernels.items():
        print(f"\n{name}:")
        print(kernel)

custom_kernels_demo(gray)
print("🎯 Custom kernels complete!")

print("\n" + "="*50)
print("✅ ALL DEMOS COMPLETE!")
print("="*50)
```

---

## 📊 Histogram Processing

### Theory
- **Histogram**: Distribution of pixel intensities (0-255)
- **Equalization**: Spread values to use full range
- **CLAHE**: Adaptive equalization for local contrast

```python
# Manual histogram equalization
def manual_equalize(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf.max()
    equalized = cdf_normalized[img]
    return equalized.astype(np.uint8)
```

---

## 🎛️ Spatial Filtering

### Common Kernels Reference

| Filter | Kernel | Effect |
|--------|--------|--------|
| **Box** | all 1/9 | Simple blur |
| **Gaussian** | weighted | Smooth blur |
| **Median** | sort + middle | Salt-pepper noise |
| **Bilateral** | space + range | Edge-preserving |

---

## 🔍 Edge Detection

### Canny Algorithm Steps
1. **Gaussian blur** - reduce noise
2. **Gradient calculation** - Sobel operators
3. **Non-maximum suppression** - thin edges
4. **Double thresholding** - strong/weak edges
5. **Hysteresis** - connect weak to strong

---

## ❓ Interview Questions & Answers

### Q1: Gaussian vs Median filter?
| Gaussian | Median |
|----------|--------|
| Linear | Non-linear |
| Blurs edges | Preserves edges |
| Gaussian noise | Salt-pepper noise |
| Fast (FFT) | Slower (sorting) |

### Q2: How does bilateral filtering work?
**Answer:** Uses TWO Gaussians:
- **Spatial**: weights by distance
- **Range**: weights by intensity difference
- Only averages similar nearby pixels → preserves edges

### Q3: Why does Canny use double thresholding?
**Answer:**
- **High threshold**: Find strong edges (confident)
- **Low threshold**: Find weak edges (candidates)
- **Hysteresis**: Keep weak edges connected to strong ones
- Removes noise while keeping complete edges

### Q4: Opening vs Closing?
| Opening | Closing |
|---------|---------|
| Erode → Dilate | Dilate → Erode |
| Removes small bright spots | Fills small dark holes |
| Smooths contours outward | Smooths contours inward |

### Q5: What is morphological gradient?
**Answer:** `Dilation - Erosion` = edge outline of objects. Shows boundaries without direction information.

---

## 📓 More Colab Notebooks

| Topic | Direct Code |
|-------|-------------|
| scikit-image Filters | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scikit-image/skimage-tutorials/blob/main/lectures/1_image_filters.ipynb) |
| OpenCV Morphology | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opencv/opencv/blob/master/samples/python/morphology.py) |
| Image Restoration | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scikit-image/skimage-tutorials/blob/main/lectures/4_restoration.ipynb) |

---

<div align="center">

**[← Image Formation](../03_Image_Formation/) | [🏠 Home](../README.md) | [Feature Detection →](../05_Features_Detection/)**

</div>

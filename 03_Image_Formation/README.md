<div align="center">

# 📷 Image Formation & Physics

### *Sensors, Optics, Color & Illumination*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scikit-image/skimage-tutorials/blob/main/lectures/0_color_and_exposure.ipynb)

</div>

---

**Navigation:** [← Transform Methods](../02_Transform_Methods/) | [🏠 Home](../README.md) | [Low-Level Processing →](../04_Low_Level_Processing/)

---

## 📖 Topics Covered
- Image Sensors (CCD, CMOS)
- Color Filter Arrays (Bayer)
- Radiometry & Photometry
- BRDF & Reflectance
- Illumination Models
- HDR Imaging

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/camera_model.svg" alt="Camera Model" width="100%"/>
</div>

---

## 📸 Image Sensors

### CCD vs CMOS

| Feature | CCD | CMOS |
|---------|-----|------|
| **Readout** | Sequential | Parallel |
| **Noise** | Lower | Higher (improving) |
| **Power** | Higher | Lower |
| **Speed** | Slower | Faster |
| **Cost** | Higher | Lower |
| **Use** | Scientific | Consumer |

### Bayer Pattern

```python
# Demosaicing - convert Bayer to RGB
import cv2
raw = cv2.imread('raw.bayer', cv2.IMREAD_UNCHANGED)
rgb = cv2.cvtColor(raw, cv2.COLOR_BAYER_BG2RGB)

# Bayer pattern: RGGB, GRBG, GBRG, BGGR
# Each pixel has only one color - interpolate others
```

---

## 🎨 Color Spaces

```python
# RGB to other color spaces
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

# LAB: L=lightness, A=green-red, B=blue-yellow
# HSV: Hue, Saturation, Value
# YCrCb: Luminance, Chrominance (JPEG uses this)
```

### Color Space Comparison

| Space | Use Case |
|-------|----------|
| **RGB** | Display, storage |
| **HSV** | Color-based segmentation |
| **LAB** | Perceptually uniform |
| **YCrCb** | Compression |
| **XYZ** | Device-independent |

---

## 💡 Illumination & Reflectance

### BRDF (Bidirectional Reflectance Distribution Function)

```
f_r(ωi, ωo) = dL_o(ωo) / dE_i(ωi)
```

### Lambertian Model
```python
# Diffuse reflection - intensity proportional to cos(angle)
I = I_ambient + I_diffuse * max(0, dot(N, L))
```

### Phong Model
```python
# Adds specular highlight
I = I_ambient + I_diffuse * (N·L) + I_specular * (R·V)^n
```

---

## 🌅 HDR Imaging

```python
import cv2
# Load multiple exposures
exposures = [cv2.imread(f'exp_{i}.jpg') for i in range(3)]
times = np.array([1/30, 1/8, 1/2], dtype=np.float32)

# Create HDR
calibrate = cv2.createCalibrateDebevec()
response = calibrate.process(exposures, times)
merge = cv2.createMergeDebevec()
hdr = merge.process(exposures, times, response)

# Tone mapping
tonemap = cv2.createTonemap(gamma=2.2)
ldr = tonemap.process(hdr)
```

---

## ❓ Interview Questions & Answers

### Q1: What is the Bayer pattern and why is it used?
**Answer:** Bayer pattern is a color filter array (CFA) with 50% green, 25% red, 25% blue. Green has more filters because human eyes are most sensitive to green. Demosaicing interpolates missing colors.

### Q2: Explain the difference between radiometry and photometry.
**Answer:**
- **Radiometry**: Measures electromagnetic radiation (Watts)
- **Photometry**: Weighted by human eye sensitivity (Lumens)
- Photometry = Radiometry × V(λ) luminosity function

### Q3: What is gamma correction?
**Answer:** Compensates for non-linear display response.
- Encoding: I_out = I_in^(1/γ)
- Display: I_display = I_encoded^γ
- Standard gamma ≈ 2.2

### Q4: How does white balance work?
**Answer:** Adjusts color channels to make neutral colors appear gray under different illuminants.
```python
# Gray world assumption
avg_r, avg_g, avg_b = img.mean(axis=(0,1))
scale = [avg_g/avg_r, 1, avg_g/avg_b]
balanced = img * scale
```

### Q5: What causes image noise?
**Answer:**
- **Shot noise**: Photon counting (Poisson)
- **Read noise**: Sensor electronics (Gaussian)
- **Dark current**: Thermal electrons
- **Quantization noise**: ADC discretization

---

## 📓 Colab Notebooks

| Topic | Link |
|-------|------|
| Color & Exposure | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scikit-image/skimage-tutorials/blob/main/lectures/0_color_and_exposure.ipynb) |
| HDR Imaging | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opencv/opencv/blob/master/samples/python/tutorial_code/photo/hdr_imaging/hdr_imaging.py) |

---

<div align="center">

**[← Transform Methods](../02_Transform_Methods/) | [🏠 Home](../README.md) | [Low-Level Processing →](../04_Low_Level_Processing/)**

</div>

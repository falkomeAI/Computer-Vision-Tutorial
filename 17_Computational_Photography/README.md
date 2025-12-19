<div align="center">

# 📸 Computational Photography

### *HDR, Deblurring, Low-light, Super-Resolution*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scikit-image/skimage-tutorials/blob/main/lectures/4_restoration.ipynb)

</div>

---

**Navigation:** [← Vision + Language](../16_Vision_Language/) | [🏠 Home](../README.md) | [Deployment & Systems →](../18_Deployment_Systems/)

---

## 📖 Topics Covered
- HDR Imaging
- Image Deblurring
- Super-Resolution
- Low-light Enhancement
- Panorama Stitching
- Inpainting

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/hdr_pipeline.svg" alt="HDR Pipeline" width="100%"/>
</div>

---

## 🌅 HDR Imaging

```python
import cv2

# Load multiple exposures
images = [cv2.imread(f'exp_{i}.jpg') for i in range(3)]
times = np.array([1/30, 1/8, 1/2], dtype=np.float32)

# Merge exposures
merge_debevec = cv2.createMergeDebevec()
hdr = merge_debevec.process(images, times)

# Tone mapping
tonemap = cv2.createTonemap(gamma=2.2)
ldr = tonemap.process(hdr) * 255

# Or use Reinhard
tonemap_reinhard = cv2.createTonemapReinhard(gamma=2.2, intensity=0, light_adapt=0)
ldr_reinhard = tonemap_reinhard.process(hdr) * 255
```

---

## 🔍 Image Deblurring

```python
# Wiener deconvolution
from scipy.signal import wiener
deblurred = wiener(blurred, psf, noise_var)

# Deep learning approach
from torchvision.transforms.functional import gaussian_blur

# Train deblurring network
class DeblurNet(nn.Module):
    def __init__(self):
        self.encoder = UNetEncoder()
        self.decoder = UNetDecoder()
    
    def forward(self, blurred):
        features = self.encoder(blurred)
        sharp = self.decoder(features)
        return sharp
```

---

## 🔎 Super-Resolution

```python
# ESRGAN for 4x upscaling
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet

model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23)
model.load_state_dict(torch.load('ESRGAN_x4.pth'))

# Upscale
lr_image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0) / 255.0
sr_image = model(lr_image)

# Real-ESRGAN for real-world images
from realesrgan import RealESRGANer
upsampler = RealESRGANer(scale=4, model_path='RealESRGAN_x4plus.pth')
output = upsampler.enhance(image)
```

---

## 🌙 Low-light Enhancement

```python
# Zero-DCE (Zero-Reference Deep Curve Estimation)
# No paired training data needed!

# CLAHE for simple enhancement
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Convert to LAB, apply to L channel
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
lab[:,:,0] = clahe.apply(lab[:,:,0])
enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Gamma correction
gamma = 0.5  # < 1 brightens
enhanced = np.power(image / 255.0, gamma) * 255
```

---

## 🖼️ Panorama Stitching

```python
# Feature-based stitching
stitcher = cv2.Stitcher_create()
status, pano = stitcher.stitch(images)

# Manual approach
# 1. Detect features (SIFT/ORB)
# 2. Match features
# 3. Compute homography (RANSAC)
# 4. Warp and blend

sift = cv2.SIFT_create()
kp1, desc1 = sift.detectAndCompute(img1, None)
kp2, desc2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)
good = [m for m,n in matches if m.distance < 0.75*n.distance]

H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
result = cv2.warpPerspective(img1, H, (w, h))
```

---

## ❓ Interview Questions & Answers

### Q1: How does HDR imaging work?
**Answer:**
1. Capture multiple exposures (bracket)
2. Estimate camera response curve
3. Merge to radiance map
4. Tone map for display
5. Result: more dynamic range

### Q2: Blind vs non-blind deblurring?
| Non-blind | Blind |
|-----------|-------|
| Known PSF | Unknown PSF |
| Wiener, Richardson-Lucy | Estimate PSF first |
| Simpler | More complex |
| Better if PSF known | Real-world motion blur |

### Q3: Why PSNR is not enough for SR evaluation?
**Answer:**
- PSNR measures pixel-wise error
- Doesn't capture perceptual quality
- Blurry can have high PSNR
- Better: LPIPS, SSIM, FID
- Human evaluation matters

### Q4: What is the checkerboard artifact in SR?
**Answer:**
- Caused by transposed convolution
- Uneven overlap in upsampling
- Solutions: resize + conv, PixelShuffle
- Sub-pixel convolution avoids this

### Q5: How does seam carving work?
**Answer:**
1. Compute energy map (gradient magnitude)
2. Find minimum energy seam (DP)
3. Remove seam (content-aware resize)
4. Repeat for desired size
5. Preserves important content

---

## 📓 Colab Notebooks

| Topic | Link |
|-------|------|
| HDR | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opencv/opencv/blob/master/samples/python/tutorial_code/photo/hdr_imaging/hdr_imaging.py) |
| Super-Resolution | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/xinntao/Real-ESRGAN/blob/master/inference_realesrgan.ipynb) |
| Restoration | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/scikit-image/skimage-tutorials/blob/main/lectures/4_restoration.ipynb) |

---

<div align="center">

**[← Vision + Language](../16_Vision_Language/) | [🏠 Home](../README.md) | [Deployment & Systems →](../18_Deployment_Systems/)**

</div>

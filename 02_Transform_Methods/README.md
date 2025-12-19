<div align="center">

# 🌊 Transform Domain Methods

### *Fourier, Wavelets, DCT & Beyond*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1transform_methods)

</div>

---

**Navigation:** [← Mathematical Foundations](../01_Mathematical_Foundations/) | [🏠 Home](../README.md) | [Image Formation →](../03_Image_Formation/)

---

## 📖 Table of Contents
- [Visual Overview](#-visual-overview)
- [Complete Colab Code](#-complete-colab-code)
- [Fourier Transform](#-fourier-transform)
- [Wavelets](#-wavelets)
- [Interview Q&A](#-interview-questions--answers)

---

## 🌀 Visual Overview

<div align="center">
<img src="./svg_figs/fourier_transform.svg" alt="Fourier Transform" width="100%"/>
</div>

<div align="center">
<img src="./svg_figs/wavelet_decomposition.svg" alt="Wavelet Decomposition" width="100%"/>
</div>

---

## 📓 Complete Colab Code

### Copy and run in Google Colab:

```python
#@title 🌊 Transform Methods - Complete Tutorial
#@markdown Fourier Transform, Wavelets, DCT - All Use Cases!

!pip install opencv-python-headless numpy matplotlib scipy PyWavelets scikit-image -q

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.fftpack import dct, idct
import pywt
from skimage import data, color
import urllib.request

# Download sample image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
urllib.request.urlretrieve(url, "sample.png")
image = cv2.imread("sample.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    image = data.camera()  # Fallback
image = cv2.resize(image, (256, 256))
print("✅ Setup complete! Image shape:", image.shape)

#@title 1️⃣ 2D Fourier Transform (FFT)

def fourier_analysis(img):
    """Complete Fourier Transform analysis"""
    # Compute FFT
    f_transform = fft2(img)
    f_shift = fftshift(f_transform)
    
    # Magnitude and Phase spectrum
    magnitude = 20 * np.log(np.abs(f_shift) + 1)
    phase = np.angle(f_shift)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(magnitude, cmap='gray')
    axes[0, 1].set_title('Magnitude Spectrum (log scale)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(phase, cmap='hsv')
    axes[0, 2].set_title('Phase Spectrum')
    axes[0, 2].axis('off')
    
    # Reconstruct from magnitude only (random phase)
    random_phase = np.exp(1j * np.random.uniform(-np.pi, np.pi, f_shift.shape))
    mag_only = np.abs(f_shift) * random_phase
    recon_mag = np.abs(ifft2(ifftshift(mag_only)))
    axes[1, 0].imshow(recon_mag, cmap='gray')
    axes[1, 0].set_title('Magnitude Only (random phase)')
    axes[1, 0].axis('off')
    
    # Reconstruct from phase only (unit magnitude)
    phase_only = np.exp(1j * np.angle(f_shift))
    recon_phase = np.abs(ifft2(ifftshift(phase_only)))
    axes[1, 1].imshow(recon_phase, cmap='gray')
    axes[1, 1].set_title('Phase Only (unit magnitude)')
    axes[1, 1].axis('off')
    
    # Perfect reconstruction
    recon = np.abs(ifft2(ifftshift(f_shift)))
    axes[1, 2].imshow(recon, cmap='gray')
    axes[1, 2].set_title('Perfect Reconstruction')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return f_shift, magnitude, phase

f_shift, mag, phase = fourier_analysis(image)
print("📊 Phase carries structural info, magnitude carries energy!")

#@title 2️⃣ Frequency Domain Filtering

def frequency_filtering(img):
    """Low-pass, High-pass, Band-pass filters"""
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    
    # FFT
    f_shift = fftshift(fft2(img))
    
    # Create filters
    def create_gaussian_filter(shape, cutoff, filter_type='low'):
        rows, cols = shape
        x = np.arange(cols) - cols // 2
        y = np.arange(rows) - rows // 2
        X, Y = np.meshgrid(x, y)
        D = np.sqrt(X**2 + Y**2)
        
        if filter_type == 'low':
            return np.exp(-(D**2) / (2 * cutoff**2))
        elif filter_type == 'high':
            return 1 - np.exp(-(D**2) / (2 * cutoff**2))
        elif filter_type == 'band':
            return np.exp(-(D**2) / (2 * cutoff[1]**2)) - np.exp(-(D**2) / (2 * cutoff[0]**2))
    
    # Apply filters
    low_pass = create_gaussian_filter(img.shape, 30, 'low')
    high_pass = create_gaussian_filter(img.shape, 30, 'high')
    band_pass = create_gaussian_filter(img.shape, (20, 50), 'band')
    
    # Filter results
    low_result = np.abs(ifft2(ifftshift(f_shift * low_pass)))
    high_result = np.abs(ifft2(ifftshift(f_shift * high_pass)))
    band_result = np.abs(ifft2(ifftshift(f_shift * band_pass)))
    
    # Visualize
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(low_pass, cmap='gray')
    axes[0, 1].set_title('Low-Pass Filter (σ=30)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(high_pass, cmap='gray')
    axes[0, 2].set_title('High-Pass Filter')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(band_pass, cmap='gray')
    axes[0, 3].set_title('Band-Pass Filter (20-50)')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(20*np.log(np.abs(f_shift)+1), cmap='gray')
    axes[1, 0].set_title('Magnitude Spectrum')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(low_result, cmap='gray')
    axes[1, 1].set_title('Low-Pass Result (Blur)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(high_result, cmap='gray')
    axes[1, 2].set_title('High-Pass Result (Edges)')
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(band_result, cmap='gray')
    axes[1, 3].set_title('Band-Pass Result')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()

frequency_filtering(image)
print("✅ Frequency filtering complete!")

#@title 3️⃣ Discrete Wavelet Transform (DWT)

def wavelet_analysis(img):
    """Complete wavelet decomposition and reconstruction"""
    
    # Single level decomposition
    coeffs = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs
    
    # Multi-level decomposition
    coeffs_multi = pywt.wavedec2(img, 'db4', level=3)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(LL, cmap='gray')
    axes[0, 1].set_title('LL (Approximation)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(np.abs(LH), cmap='gray')
    axes[0, 2].set_title('LH (Horizontal Details)')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(np.abs(HL), cmap='gray')
    axes[0, 3].set_title('HL (Vertical Details)')
    axes[0, 3].axis('off')
    
    axes[1, 0].imshow(np.abs(HH), cmap='gray')
    axes[1, 0].set_title('HH (Diagonal Details)')
    axes[1, 0].axis('off')
    
    # Show multi-level
    arr, slices = pywt.coeffs_to_array(coeffs_multi)
    axes[1, 1].imshow(np.abs(arr), cmap='gray')
    axes[1, 1].set_title('3-Level DWT (db4)')
    axes[1, 1].axis('off')
    
    # Reconstruction
    reconstructed = pywt.idwt2(coeffs, 'haar')
    axes[1, 2].imshow(reconstructed, cmap='gray')
    axes[1, 2].set_title('Reconstructed (IDWT)')
    axes[1, 2].axis('off')
    
    # Error
    error = np.abs(img[:reconstructed.shape[0], :reconstructed.shape[1]] - reconstructed)
    axes[1, 3].imshow(error, cmap='hot')
    axes[1, 3].set_title(f'Reconstruction Error (max={error.max():.2e})')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return coeffs, coeffs_multi

coeffs, coeffs_multi = wavelet_analysis(image)
print("🌊 Wavelet analysis complete!")

#@title 4️⃣ Wavelet Denoising

def wavelet_denoising(img):
    """Denoise using wavelet thresholding"""
    # Add noise
    noise_sigma = 25
    noisy = img + noise_sigma * np.random.randn(*img.shape)
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Different thresholding methods
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy, cmap='gray')
    axes[0, 1].set_title(f'Noisy (σ={noise_sigma})')
    axes[0, 1].axis('off')
    
    # Soft thresholding
    coeffs = pywt.wavedec2(noisy, 'db4', level=3)
    threshold = noise_sigma * np.sqrt(2 * np.log(img.size))
    
    denoised_coeffs = [coeffs[0]]
    for detail_coeffs in coeffs[1:]:
        denoised_coeffs.append(
            tuple(pywt.threshold(c, threshold, mode='soft') for c in detail_coeffs)
        )
    denoised_soft = pywt.waverec2(denoised_coeffs, 'db4')
    
    axes[0, 2].imshow(denoised_soft[:img.shape[0], :img.shape[1]], cmap='gray')
    axes[0, 2].set_title('Soft Thresholding')
    axes[0, 2].axis('off')
    
    # Hard thresholding
    denoised_coeffs = [coeffs[0]]
    for detail_coeffs in coeffs[1:]:
        denoised_coeffs.append(
            tuple(pywt.threshold(c, threshold, mode='hard') for c in detail_coeffs)
        )
    denoised_hard = pywt.waverec2(denoised_coeffs, 'db4')
    
    axes[1, 0].imshow(denoised_hard[:img.shape[0], :img.shape[1]], cmap='gray')
    axes[1, 0].set_title('Hard Thresholding')
    axes[1, 0].axis('off')
    
    # BayesShrink
    denoised_bayes = pywt.threshold(noisy, threshold * 0.8, mode='soft')
    axes[1, 1].imshow(denoised_bayes, cmap='gray')
    axes[1, 1].set_title('BayesShrink-like')
    axes[1, 1].axis('off')
    
    # PSNR comparison
    def psnr(original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    psnr_noisy = psnr(img, noisy)
    psnr_soft = psnr(img, denoised_soft[:img.shape[0], :img.shape[1]])
    
    axes[1, 2].bar(['Noisy', 'Denoised'], [psnr_noisy, psnr_soft], color=['red', 'green'])
    axes[1, 2].set_ylabel('PSNR (dB)')
    axes[1, 2].set_title(f'Quality: {psnr_noisy:.1f}dB → {psnr_soft:.1f}dB')
    
    plt.tight_layout()
    plt.show()

wavelet_denoising(image.astype(float))
print("🔇 Denoising complete!")

#@title 5️⃣ Discrete Cosine Transform (DCT) - JPEG

def dct_analysis(img):
    """DCT for image compression (JPEG-style)"""
    
    def dct2(block):
        return dct(dct(block.T, norm='ortho').T, norm='ortho')
    
    def idct2(block):
        return idct(idct(block.T, norm='ortho').T, norm='ortho')
    
    # JPEG quantization matrix
    Q = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])
    
    # Process 8x8 blocks
    h, w = img.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    img_crop = img[:h8, :w8].astype(float)
    
    # Different quality levels
    qualities = [10, 50, 90]
    results = []
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    axes[0, 0].imshow(img_crop, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    for idx, quality in enumerate(qualities):
        # Scale Q matrix
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality
        Q_scaled = np.clip(np.floor((Q * scale + 50) / 100), 1, 255)
        
        # Process blocks
        compressed = np.zeros_like(img_crop)
        for i in range(0, h8, 8):
            for j in range(0, w8, 8):
                block = img_crop[i:i+8, j:j+8] - 128
                dct_block = dct2(block)
                quantized = np.round(dct_block / Q_scaled) * Q_scaled
                compressed[i:i+8, j:j+8] = idct2(quantized) + 128
        
        compressed = np.clip(compressed, 0, 255)
        results.append(compressed)
        
        # Calculate compression ratio (simplified)
        original_bits = h8 * w8 * 8
        compressed_bits = np.count_nonzero(compressed) * 8  # Simplified
        
        axes[0, idx+1].imshow(compressed, cmap='gray')
        axes[0, idx+1].set_title(f'Quality={quality}')
        axes[0, idx+1].axis('off')
    
    # Show DCT basis functions
    basis = np.zeros((64, 64))
    for i in range(8):
        for j in range(8):
            block = np.zeros((8, 8))
            block[i, j] = 1
            basis[i*8:(i+1)*8, j*8:(j+1)*8] = idct2(block)
    
    axes[1, 0].imshow(basis, cmap='gray')
    axes[1, 0].set_title('DCT Basis Functions')
    axes[1, 0].axis('off')
    
    # Show single block DCT
    block = img_crop[64:72, 64:72]
    dct_block = dct2(block - 128)
    
    axes[1, 1].imshow(block, cmap='gray')
    axes[1, 1].set_title('8x8 Block')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.abs(dct_block), cmap='hot')
    axes[1, 2].set_title('DCT Coefficients')
    axes[1, 2].axis('off')
    
    # PSNR comparison
    psnrs = [cv2.PSNR(img_crop.astype(np.uint8), r.astype(np.uint8)) for r in results]
    axes[1, 3].bar([f'Q={q}' for q in qualities], psnrs, color=['red', 'yellow', 'green'])
    axes[1, 3].set_ylabel('PSNR (dB)')
    axes[1, 3].set_title('Quality vs PSNR')
    
    plt.tight_layout()
    plt.show()

dct_analysis(image)
print("🗜️ DCT/JPEG analysis complete!")

#@title 6️⃣ Gabor Filters for Texture

def gabor_analysis(img):
    """Gabor filters for texture analysis"""
    
    # Different orientations and frequencies
    orientations = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    frequencies = [0.1, 0.2, 0.3]
    
    fig, axes = plt.subplots(len(frequencies)+1, len(orientations)+1, figsize=(15, 12))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Show filter banks and responses
    for i, freq in enumerate(frequencies):
        axes[i+1, 0].text(0.5, 0.5, f'f={freq}', ha='center', va='center', fontsize=12)
        axes[i+1, 0].axis('off')
        
        for j, theta in enumerate(orientations):
            # Create Gabor kernel
            kernel = cv2.getGaborKernel(
                (31, 31), sigma=4.0, theta=theta, 
                lambd=1/freq, gamma=0.5, psi=0
            )
            
            # Filter response
            response = cv2.filter2D(img, cv2.CV_64F, kernel)
            
            if i == 0:
                axes[0, j+1].set_title(f'θ={int(np.degrees(theta))}°')
                axes[0, j+1].axis('off')
            
            axes[i+1, j+1].imshow(np.abs(response), cmap='hot')
            axes[i+1, j+1].axis('off')
    
    plt.suptitle('Gabor Filter Bank Responses', fontsize=14)
    plt.tight_layout()
    plt.show()

gabor_analysis(image)
print("🎨 Gabor texture analysis complete!")

#@title 7️⃣ Compare All Wavelets

def compare_wavelets(img):
    """Compare different wavelet families"""
    wavelets = ['haar', 'db4', 'sym4', 'coif2', 'bior2.2']
    
    fig, axes = plt.subplots(2, len(wavelets)+1, figsize=(18, 8))
    
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    for idx, wavelet in enumerate(wavelets):
        coeffs = pywt.wavedec2(img, wavelet, level=2)
        arr, slices = pywt.coeffs_to_array(coeffs)
        
        axes[0, idx+1].imshow(np.abs(arr), cmap='gray')
        axes[0, idx+1].set_title(f'{wavelet}')
        axes[0, idx+1].axis('off')
        
        # Compression: keep only top 10% coefficients
        threshold = np.percentile(np.abs(arr), 90)
        arr_compressed = np.where(np.abs(arr) > threshold, arr, 0)
        coeffs_compressed = pywt.array_to_coeffs(arr_compressed, slices, output_format='wavedec2')
        reconstructed = pywt.waverec2(coeffs_compressed, wavelet)
        
        axes[1, idx+1].imshow(reconstructed[:img.shape[0], :img.shape[1]], cmap='gray')
        axes[1, idx+1].set_title(f'90% compressed')
        axes[1, idx+1].axis('off')
    
    plt.tight_layout()
    plt.show()

compare_wavelets(image)
print("📊 Wavelet comparison complete!")

print("\n" + "="*50)
print("✅ ALL TRANSFORM METHODS COMPLETE!")
print("="*50)
```

---

## ❓ Interview Questions & Answers

### Q1: Why use Fourier Transform in image processing?
**Answer:**
- Filtering = multiplication (faster for large kernels)
- Understand frequency content (edges = high freq)
- Remove periodic noise (notch filtering)
- Convolution theorem: F{f*g} = F{f}·F{g}

### Q2: Fourier vs Wavelet - when to use which?
| Fourier | Wavelet |
|---------|---------|
| Global frequency | Local time-frequency |
| Stationary signals | Non-stationary |
| Periodic patterns | Transients, edges |
| JPEG compression | JPEG2000 |

### Q3: What is the Nyquist theorem?
**Answer:** Sampling rate must be > 2× max frequency to avoid aliasing.
- fs > 2·fmax
- In images: pixel spacing determines max resolvable frequency

### Q4: How does JPEG compression work?
**Answer:**
1. Convert RGB → YCbCr
2. Downsample chroma (4:2:0)
3. 8×8 block DCT
4. Quantize (lossy step)
5. Zigzag scan + RLE + Huffman

### Q5: What is multi-resolution analysis in wavelets?
**Answer:** Decompose signal into approximation (low-freq) and details (high-freq) at multiple scales. Each level halves resolution and captures different frequency bands.

---

<div align="center">

**[← Mathematical Foundations](../01_Mathematical_Foundations/) | [🏠 Home](../README.md) | [Image Formation →](../03_Image_Formation/)**

</div>

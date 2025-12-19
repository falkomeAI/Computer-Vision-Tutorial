<div align="center">

# 🎯 Feature Detection & Description

### *SIFT, ORB, HOG & Feature Matching*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1features)

</div>

---

**Navigation:** [← Low-Level Processing](../04_Low_Level_Processing/) | [🏠 Home](../README.md) | [Geometry & Multi-View →](../06_Geometry_MultiView/)

---

## 📖 Table of Contents
- [Visual Overview](#-visual-overview)
- [Complete Colab Code](#-complete-colab-code)
- [Corner Detection](#-corner-detection)
- [Feature Matching](#-feature-matching)
- [Interview Q&A](#-interview-questions--answers)

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/feature_pipeline.svg" alt="Feature Pipeline" width="100%"/>
</div>

---

## 📓 Complete Colab Code

```python
#@title 🎯 Feature Detection - Complete Tutorial
#@markdown SIFT, ORB, Harris, HOG, Feature Matching!

!pip install opencv-python-headless opencv-contrib-python numpy matplotlib scikit-image -q

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import urllib.request

# Download sample images
url1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/280px-Cute_dog.jpg"
urllib.request.urlretrieve(url1, "img1.png")
urllib.request.urlretrieve(url2, "img2.jpg")

img1 = cv2.imread("img1.png")
img2 = cv2.imread("img2.jpg")
if img1 is None: img1 = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
if img2 is None: img2 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
print("✅ Setup complete!")

#@title 1️⃣ Harris Corner Detection

def harris_corners(img, gray):
    """Detect corners using Harris"""
    # Harris corner detection
    harris = cv2.cornerHarris(gray.astype(np.float32), blockSize=2, ksize=3, k=0.04)
    
    # Dilate for marking
    harris_dilated = cv2.dilate(harris, None)
    
    # Threshold
    threshold = 0.01 * harris.max()
    corners = harris > threshold
    
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(harris, cmap='hot')
    axes[1].set_title('Harris Response')
    axes[1].axis('off')
    
    # Mark corners
    img_corners = img.copy()
    img_corners[corners] = [0, 255, 0]
    axes[2].imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'Corners ({np.sum(corners)} detected)')
    axes[2].axis('off')
    
    # Shi-Tomasi corners
    corners_st = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    img_st = img.copy()
    if corners_st is not None:
        for corner in corners_st:
            x, y = corner.ravel()
            cv2.circle(img_st, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    axes[3].imshow(cv2.cvtColor(img_st, cv2.COLOR_BGR2RGB))
    axes[3].set_title('Shi-Tomasi Corners')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()

harris_corners(img1, gray1)
print("📍 Harris corners detected!")

#@title 2️⃣ SIFT Features

def sift_features(img, gray):
    """SIFT keypoint detection and description"""
    # Create SIFT
    sift = cv2.SIFT_create()
    
    # Detect and compute
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    
    # Draw keypoints
    img_kp = cv2.drawKeypoints(img, keypoints, None, 
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'SIFT Keypoints ({len(keypoints)})')
    axes[1].axis('off')
    
    # Keypoint properties
    if len(keypoints) > 0:
        sizes = [kp.size for kp in keypoints]
        angles = [kp.angle for kp in keypoints]
        
        axes[2].hist(sizes, bins=30, alpha=0.7, label='Size')
        axes[2].hist(angles, bins=30, alpha=0.7, label='Angle')
        axes[2].legend()
        axes[2].set_title('Keypoint Distribution')
    
    plt.tight_layout()
    plt.show()
    
    print(f"  Descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")
    return keypoints, descriptors

kp1, desc1 = sift_features(img1, gray1)
print("🔍 SIFT features extracted!")

#@title 3️⃣ ORB Features (Fast Alternative)

def orb_features(img, gray):
    """ORB: Oriented FAST and Rotated BRIEF"""
    # Create ORB
    orb = cv2.ORB_create(nfeatures=500)
    
    # Detect and compute
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    
    # Draw
    img_kp = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0),
                               flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Compare with FAST
    fast = cv2.FastFeatureDetector_create(threshold=25)
    fast_kp = fast.detect(gray, None)
    img_fast = cv2.drawKeypoints(img, fast_kp, None, color=(255, 0, 0))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'FAST Corners ({len(fast_kp)})')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'ORB Features ({len(keypoints)})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"  ORB descriptor: {descriptors.shape if descriptors is not None else 'None'} (binary)")
    return keypoints, descriptors

kp_orb, desc_orb = orb_features(img1, gray1)
print("⚡ ORB features extracted!")

#@title 4️⃣ HOG Features (Histogram of Oriented Gradients)

def hog_features(img, gray):
    """HOG descriptor for object detection"""
    # Resize for consistent size
    resized = cv2.resize(gray, (128, 128))
    
    # Compute HOG
    fd, hog_image = hog(resized, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=True, 
                       block_norm='L2-Hys')
    
    # Rescale for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(resized, cmap='gray')
    axes[1].set_title('Resized (128x128)')
    axes[1].axis('off')
    
    axes[2].imshow(hog_image_rescaled, cmap='gray')
    axes[2].set_title('HOG Visualization')
    axes[2].axis('off')
    
    # Show gradient magnitudes
    gx = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)
    
    axes[3].imshow(magnitude, cmap='hot')
    axes[3].set_title('Gradient Magnitude')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"  HOG descriptor length: {len(fd)}")
    return fd

hog_desc = hog_features(img1, gray1)
print("📊 HOG features computed!")

#@title 5️⃣ Feature Matching

def feature_matching(img1, img2, gray1, gray2):
    """Match features between two images"""
    # Detect features
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)
    
    if desc1 is None or desc2 is None:
        print("No features found!")
        return
    
    # BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]
    
    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches_flann = flann.knnMatch(desc1, desc2, k=2)
    
    # Ratio test
    good_matches = []
    for m, n in matches_flann:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # BF matches
    img_bf = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axes[0, 0].imshow(cv2.cvtColor(img_bf, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'BFMatcher ({len(matches)} matches)')
    axes[0, 0].axis('off')
    
    # FLANN + ratio test
    img_flann = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:30], None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    axes[0, 1].imshow(cv2.cvtColor(img_flann, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title(f'FLANN + Ratio Test ({len(good_matches)} good)')
    axes[0, 1].axis('off')
    
    # Distance histogram
    distances = [m.distance for m in matches]
    axes[1, 0].hist(distances, bins=30, color='blue', alpha=0.7)
    axes[1, 0].set_xlabel('Match Distance')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Match Distance Distribution')
    
    # Keypoint comparison
    axes[1, 1].bar(['Image 1', 'Image 2'], [len(kp1), len(kp2)], color=['blue', 'orange'])
    axes[1, 1].set_ylabel('Keypoints')
    axes[1, 1].set_title('Keypoint Count')
    
    plt.tight_layout()
    plt.show()
    
    return matches, good_matches

matches, good = feature_matching(img1, img2, gray1, gray2)
print("🔗 Feature matching complete!")

#@title 6️⃣ Homography with RANSAC

def ransac_homography(img1, img2, gray1, gray2):
    """Find homography using RANSAC"""
    # Detect features
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)
    
    if desc1 is None or desc2 is None:
        print("No features!")
        return
    
    # Match
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Ratio test
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    if len(good) < 4:
        print("Not enough matches for homography")
        return
    
    # Get points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    
    # RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print("Homography failed")
        return
    
    # Warp image
    h, w = img1.shape[:2]
    warped = cv2.warpPerspective(img1, H, (w, h))
    
    # Draw inliers
    inliers = mask.ravel().tolist()
    inlier_matches = [good[i] for i in range(len(good)) if inliers[i]]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Source Image')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Warped (H found with {sum(inliers)} inliers)')
    axes[1].axis('off')
    
    axes[2].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Target Image')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"  Homography matrix:\n{H}")
    print(f"  RANSAC inliers: {sum(inliers)}/{len(good)}")

ransac_homography(img1, img2, gray1, gray2)
print("🎯 Homography computed!")

#@title 7️⃣ Compare All Detectors

def compare_detectors(img, gray):
    """Compare detection speed and count"""
    import time
    
    detectors = {
        'Harris': lambda: cv2.cornerHarris(gray.astype(np.float32), 2, 3, 0.04),
        'FAST': lambda: cv2.FastFeatureDetector_create().detect(gray),
        'ORB': lambda: cv2.ORB_create().detectAndCompute(gray, None),
        'SIFT': lambda: cv2.SIFT_create().detectAndCompute(gray, None),
        'AKAZE': lambda: cv2.AKAZE_create().detectAndCompute(gray, None),
    }
    
    results = {}
    for name, detector in detectors.items():
        start = time.time()
        result = detector()
        elapsed = time.time() - start
        
        if name == 'Harris':
            count = np.sum(result > 0.01 * result.max())
        elif name == 'FAST':
            count = len(result)
        else:
            count = len(result[0]) if result[0] is not None else 0
        
        results[name] = {'time': elapsed * 1000, 'count': count}
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    names = list(results.keys())
    times = [results[n]['time'] for n in names]
    counts = [results[n]['count'] for n in names]
    
    axes[0].bar(names, times, color='steelblue')
    axes[0].set_ylabel('Time (ms)')
    axes[0].set_title('Detection Speed')
    axes[0].tick_params(axis='x', rotation=45)
    
    axes[1].bar(names, counts, color='coral')
    axes[1].set_ylabel('Keypoints')
    axes[1].set_title('Keypoint Count')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print("\n📊 Detector Comparison:")
    for name, data in results.items():
        print(f"  {name}: {data['count']} keypoints in {data['time']:.1f}ms")

compare_detectors(img1, gray1)

print("\n" + "="*50)
print("✅ ALL FEATURE DETECTION COMPLETE!")
print("="*50)
```

---

## ❓ Interview Questions & Answers

### Q1: SIFT vs ORB - when to use which?
| SIFT | ORB |
|------|-----|
| 128D float descriptor | 256-bit binary |
| Scale + rotation invariant | Rotation invariant |
| Slower | ~100x faster |
| More accurate | Good enough for real-time |

### Q2: How does Harris corner detection work?
**Answer:**
1. Compute gradients Ix, Iy
2. Build structure tensor M
3. R = det(M) - k·trace(M)²
4. Large R = corner

### Q3: What is the ratio test in matching?
**Answer:** Lowe's ratio test: if best/2nd_best < 0.75 → good match

### Q4: How does RANSAC work?
**Answer:**
1. Sample minimal set (4 for homography)
2. Fit model
3. Count inliers
4. Repeat, keep best

### Q5: Why is HOG good for pedestrian detection?
**Answer:** Captures edge orientations in local cells, robust to illumination, proven effective for human shape detection.

---

<div align="center">

**[← Low-Level Processing](../04_Low_Level_Processing/) | [🏠 Home](../README.md) | [Geometry & Multi-View →](../06_Geometry_MultiView/)**

</div>

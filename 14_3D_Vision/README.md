<div align="center">

# 🌐 3D & Spatial Vision

### *Depth, Point Clouds, NeRF, 3D Gaussian Splatting*

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13dvision)

</div>

---

**Navigation:** [← Video & Temporal](../13_Video_Temporal/) | [🏠 Home](../README.md) | [Generative Vision →](../15_Generative_Vision/)

---

## 📖 Table of Contents
- [Visual Overview](#-visual-overview)
- [Complete Colab Code](#-complete-colab-code)
- [Depth Estimation](#-depth-estimation)
- [Point Clouds](#-point-clouds)
- [NeRF](#-nerf)
- [Interview Q&A](#-interview-questions--answers)

---

## 🎨 Visual Overview

<div align="center">
<img src="./svg_figs/nerf_architecture.svg" alt="NeRF Architecture" width="100%"/>
</div>

---

## 📓 Complete Colab Code

```python
#@title 🌐 3D Vision - Complete Tutorial
#@markdown Depth, Point Clouds, NeRF concepts!

!pip install torch torchvision numpy matplotlib opencv-python-headless open3d pillow transformers -q

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
from PIL import Image
import urllib.request

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# Download sample image
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Cute_dog.jpg/320px-Cute_dog.jpg"
urllib.request.urlretrieve(url, "sample.jpg")
image = Image.open("sample.jpg")
print("📷 Sample image loaded!")

#@title 1️⃣ Monocular Depth Estimation (MiDaS)

from transformers import DPTImageProcessor, DPTForDepthEstimation

def run_depth_estimation(image):
    """Run MiDaS depth estimation"""
    print("Loading MiDaS model...")
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    model.eval()
    
    # Process
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth
    
    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    
    depth = prediction.squeeze().cpu().numpy()
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    im = axes[1].imshow(depth, cmap='plasma')
    axes[1].set_title("Depth Map")
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    
    # 3D visualization
    h, w = depth.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    ax3d = fig.add_subplot(1, 3, 3, projection='3d')
    ax3d.scatter(x[::10, ::10].flatten(), 
                 y[::10, ::10].flatten(), 
                 -depth[::10, ::10].flatten(),
                 c=depth[::10, ::10].flatten(), 
                 cmap='plasma', s=1)
    ax3d.set_title("3D Point Cloud")
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Depth')
    
    plt.tight_layout()
    plt.show()
    
    return depth

depth_map = run_depth_estimation(image)
print("✅ Depth estimation complete!")

#@title 2️⃣ Stereo Depth Estimation

def stereo_depth_demo():
    """Demonstrate stereo depth calculation"""
    # Create synthetic stereo pair
    h, w = 240, 320
    
    # Create simple 3D scene
    depth_gt = np.ones((h, w)) * 100
    depth_gt[80:160, 100:220] = 50  # Closer object
    
    # Disparity (inversely proportional to depth)
    baseline = 0.1  # meters
    focal = 500  # pixels
    disparity = baseline * focal / depth_gt
    
    # Create left and right images
    left = np.zeros((h, w), dtype=np.uint8)
    left[80:160, 100:220] = 200  # Object
    left[20:60, 20:80] = 128     # Background object
    
    # Shift for right image
    right = np.zeros_like(left)
    for y in range(h):
        for x in range(w):
            x_shift = int(disparity[y, x])
            if 0 <= x - x_shift < w:
                right[y, x - x_shift] = left[y, x]
    
    # Compute disparity using OpenCV
    stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
    computed_disparity = stereo.compute(left, right).astype(np.float32) / 16
    
    # Convert to depth
    computed_depth = np.where(computed_disparity > 0, 
                              baseline * focal / computed_disparity, 0)
    
    # Visualize
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(left, cmap='gray')
    axes[0, 0].set_title("Left Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(right, cmap='gray')
    axes[0, 1].set_title("Right Image")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(disparity, cmap='plasma')
    axes[0, 2].set_title("Ground Truth Disparity")
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(computed_disparity, cmap='plasma')
    axes[1, 0].set_title("Computed Disparity (StereoBM)")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(depth_gt, cmap='viridis')
    axes[1, 1].set_title("Ground Truth Depth")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.clip(computed_depth, 0, 200), cmap='viridis')
    axes[1, 2].set_title("Computed Depth")
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("📐 Stereo geometry:")
    print(f"   depth = baseline × focal / disparity")
    print(f"   d = {baseline} × {focal} / disparity")

stereo_depth_demo()
print("✅ Stereo depth complete!")

#@title 3️⃣ Point Cloud Processing

def point_cloud_demo():
    """Basic point cloud operations"""
    import open3d as o3d
    
    # Create synthetic point cloud (sphere)
    n_points = 5000
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 1 + np.random.normal(0, 0.05, n_points)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    points = np.stack([x, y, z], axis=1)
    
    # Add noise
    noisy_points = points + np.random.normal(0, 0.02, points.shape)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(noisy_points)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # Visualize with matplotlib (since Open3D GUI doesn't work in Colab)
    fig = plt.figure(figsize=(16, 5))
    
    # Original points
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(x[::5], y[::5], z[::5], c=z[::5], cmap='viridis', s=1)
    ax1.set_title('Clean Sphere')
    
    # Noisy points
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(noisy_points[::5, 0], noisy_points[::5, 1], noisy_points[::5, 2], 
                c=noisy_points[::5, 2], cmap='viridis', s=1)
    ax2.set_title('Noisy Point Cloud')
    
    # With normals (subsample for visualization)
    normals = np.asarray(pcd.normals)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(noisy_points[::20, 0], noisy_points[::20, 1], noisy_points[::20, 2], 
                c='blue', s=5)
    ax3.quiver(noisy_points[::20, 0], noisy_points[::20, 1], noisy_points[::20, 2],
               normals[::20, 0], normals[::20, 1], normals[::20, 2], 
               length=0.1, color='red', alpha=0.5)
    ax3.set_title('Point Cloud with Normals')
    
    plt.tight_layout()
    plt.show()
    
    # Point cloud operations
    print("\n📊 Point Cloud Operations:")
    print(f"   Points: {len(pcd.points)}")
    print(f"   Has normals: {pcd.has_normals()}")
    
    # Downsampling
    pcd_down = pcd.voxel_down_sample(voxel_size=0.1)
    print(f"   After voxel downsampling: {len(pcd_down.points)} points")
    
    # Statistical outlier removal
    pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"   After outlier removal: {len(pcd_clean.points)} points")

point_cloud_demo()
print("✅ Point cloud processing complete!")

#@title 4️⃣ NeRF - Neural Radiance Fields (Simplified)

class SimpleNeRF(nn.Module):
    """Simplified NeRF MLP"""
    def __init__(self, pos_dim=60, dir_dim=24, hidden_dim=256):
        super().__init__()
        
        # Position encoding: sin/cos at different frequencies
        self.L_pos = 10  # 2*10*3 = 60
        self.L_dir = 4   # 2*4*3 = 24
        
        # Density MLP (depends on position only)
        self.density_mlp = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.density_out = nn.Linear(hidden_dim, 1)  # sigma
        self.feature_out = nn.Linear(hidden_dim, hidden_dim)
        
        # Color MLP (depends on position features + direction)
        self.color_mlp = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()  # RGB in [0, 1]
        )
    
    def positional_encoding(self, x, L):
        """Encode position with sin/cos at multiple frequencies"""
        freqs = 2 ** torch.linspace(0, L-1, L, device=x.device)
        x_proj = x.unsqueeze(-1) * freqs  # (B, 3, L)
        x_proj = x_proj.reshape(*x.shape[:-1], -1)  # (B, 3*L)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)  # (B, 6*L)
    
    def forward(self, positions, directions):
        """
        positions: (B, 3) - 3D coordinates
        directions: (B, 3) - viewing directions (normalized)
        Returns: rgb (B, 3), sigma (B, 1)
        """
        # Encode inputs
        pos_enc = self.positional_encoding(positions, self.L_pos)
        dir_enc = self.positional_encoding(directions, self.L_dir)
        
        # Get density and features
        h = self.density_mlp(pos_enc)
        sigma = F.relu(self.density_out(h))  # Density must be positive
        features = self.feature_out(h)
        
        # Get color (view-dependent)
        rgb = self.color_mlp(torch.cat([features, dir_enc], dim=-1))
        
        return rgb, sigma

def volume_rendering(rgb, sigma, z_vals):
    """
    Classic NeRF volume rendering
    rgb: (B, N_samples, 3)
    sigma: (B, N_samples, 1)
    z_vals: (B, N_samples) - depths along ray
    """
    # Compute distances between samples
    dists = z_vals[:, 1:] - z_vals[:, :-1]
    dists = torch.cat([dists, torch.full_like(dists[:, :1], 1e10)], dim=-1)
    
    # Compute alpha (opacity)
    alpha = 1 - torch.exp(-sigma.squeeze(-1) * dists)
    
    # Compute transmittance
    T = torch.cumprod(1 - alpha + 1e-10, dim=-1)
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=-1)
    
    # Compute weights
    weights = alpha * T
    
    # Composite color
    rgb_final = (weights.unsqueeze(-1) * rgb).sum(dim=1)
    
    return rgb_final, weights

# Demo
nerf = SimpleNeRF().to(device)

# Sample rays
batch_size = 64
n_samples = 64

# Random positions along rays
positions = torch.randn(batch_size, n_samples, 3).to(device)
directions = F.normalize(torch.randn(batch_size, 3), dim=-1).to(device)
directions = directions.unsqueeze(1).expand(-1, n_samples, -1)

# Forward pass
rgb, sigma = nerf(positions.reshape(-1, 3), directions.reshape(-1, 3))
rgb = rgb.reshape(batch_size, n_samples, 3)
sigma = sigma.reshape(batch_size, n_samples, 1)

# Volume render
z_vals = torch.linspace(0, 4, n_samples).unsqueeze(0).expand(batch_size, -1).to(device)
rendered_rgb, weights = volume_rendering(rgb, sigma, z_vals)

print("🎨 NeRF Demo:")
print(f"   Input positions: {positions.shape}")
print(f"   Input directions: {directions.shape}")
print(f"   Per-sample RGB: {rgb.shape}")
print(f"   Per-sample Sigma: {sigma.shape}")
print(f"   Rendered color: {rendered_rgb.shape}")

# Visualize weights distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(weights[0].cpu().detach().numpy())
axes[0].set_xlabel('Sample along ray')
axes[0].set_ylabel('Weight')
axes[0].set_title('Volume Rendering Weights (1 ray)')

# Rendered colors (random, just for demo)
ax1 = axes[1]
colors = rendered_rgb.cpu().detach().numpy()
ax1.bar(range(len(colors)), colors.mean(axis=1), color='gray')
ax1.set_xlabel('Ray index')
ax1.set_ylabel('Mean RGB')
ax1.set_title('Rendered Ray Colors')

plt.tight_layout()
plt.show()
print("✅ NeRF demo complete!")

#@title 5️⃣ 3D Gaussian Splatting Concepts

print("🔮 3D Gaussian Splatting:")
print("="*50)
print("""
Key Differences from NeRF:
------------------------
1. EXPLICIT representation (vs implicit MLP)
   - Millions of 3D Gaussians with learnable parameters
   - Each Gaussian: position, covariance, color, opacity

2. RASTERIZATION (vs ray marching)
   - Project Gaussians to 2D
   - Sort by depth
   - Alpha blend front-to-back
   - MUCH faster rendering!

3. Gaussian Parameters:
   - μ (mean): 3D position
   - Σ (covariance): 3D shape (anisotropic)
   - c (color): Spherical harmonics coefficients
   - α (opacity): Single scalar

4. Rendering Equation:
   C = Σ_i c_i α_i Π_{j<i}(1 - α_j)
   (Same as NeRF, but Gaussians sorted in order)

5. Optimization:
   - Differentiable rasterization
   - Adaptive density control (split/clone/prune)
   - Typically < 30 min training vs hours for NeRF
""")

# Visualize 2D Gaussians
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Single Gaussian
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)

# Isotropic
sigma = 1.0
Z1 = np.exp(-(X**2 + Y**2) / (2*sigma**2))
axes[0].contourf(X, Y, Z1, levels=20, cmap='viridis')
axes[0].set_title('Isotropic Gaussian')
axes[0].set_aspect('equal')

# Anisotropic
cov = np.array([[2, 0.8], [0.8, 0.5]])
cov_inv = np.linalg.inv(cov)
Z2 = np.zeros_like(X)
for i in range(len(x)):
    for j in range(len(y)):
        p = np.array([X[i,j], Y[i,j]])
        Z2[i,j] = np.exp(-0.5 * p @ cov_inv @ p)
axes[1].contourf(X, Y, Z2, levels=20, cmap='viridis')
axes[1].set_title('Anisotropic Gaussian')
axes[1].set_aspect('equal')

# Multiple Gaussians (like 3DGS)
Z3 = np.zeros_like(X)
centers = [(-1, -1), (1, 1), (0, 0.5)]
covs = [
    np.array([[0.3, 0], [0, 0.3]]),
    np.array([[0.5, 0.2], [0.2, 0.2]]),
    np.array([[0.2, -0.1], [-0.1, 0.4]])
]
colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
alphas = [0.8, 0.6, 0.9]

for (cx, cy), cov, color, alpha in zip(centers, covs, colors, alphas):
    cov_inv = np.linalg.inv(cov)
    for i in range(len(x)):
        for j in range(len(y)):
            p = np.array([X[i,j] - cx, Y[i,j] - cy])
            Z3[i,j] += alpha * np.exp(-0.5 * p @ cov_inv @ p)

axes[2].contourf(X, Y, Z3, levels=20, cmap='viridis')
axes[2].set_title('Multiple Gaussians (3DGS-like)')
axes[2].set_aspect('equal')

plt.tight_layout()
plt.show()
print("✅ 3DGS concepts explained!")

#@title 6️⃣ Camera Models

def camera_projection_demo():
    """Demonstrate camera projection"""
    # Camera intrinsics
    fx, fy = 500, 500  # Focal lengths
    cx, cy = 320, 240  # Principal point
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ])
    
    # 3D points (cube corners)
    cube = np.array([
        [-1, -1, 4],
        [1, -1, 4],
        [1, 1, 4],
        [-1, 1, 4],
        [-1, -1, 6],
        [1, -1, 6],
        [1, 1, 6],
        [-1, 1, 6]
    ]).T  # 3 x 8
    
    # Project to image
    cube_h = np.vstack([cube, np.ones(8)])  # Homogeneous
    proj = K @ cube[:3]  # Project
    proj_2d = proj[:2] / proj[2:]  # Normalize
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 3D view
    ax3d = fig.add_subplot(121, projection='3d')
    
    # Draw cube edges
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4),
             (0,4), (1,5), (2,6), (3,7)]
    for e in edges:
        ax3d.plot3D([cube[0,e[0]], cube[0,e[1]]],
                   [cube[1,e[0]], cube[1,e[1]]],
                   [cube[2,e[0]], cube[2,e[1]]], 'b-')
    
    ax3d.scatter(*cube, s=50, c='red')
    ax3d.scatter(0, 0, 0, s=100, c='green', marker='^', label='Camera')
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_title('3D Scene (Camera at Origin)')
    ax3d.legend()
    
    # 2D projection
    axes[1].scatter(proj_2d[0], proj_2d[1], s=50, c='red')
    for e in edges:
        axes[1].plot([proj_2d[0,e[0]], proj_2d[0,e[1]]],
                    [proj_2d[1,e[0]], proj_2d[1,e[1]]], 'b-')
    axes[1].set_xlim(0, 640)
    axes[1].set_ylim(480, 0)  # Image coordinates
    axes[1].set_xlabel('u (pixels)')
    axes[1].set_ylabel('v (pixels)')
    axes[1].set_title('2D Projection')
    axes[1].set_aspect('equal')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("📷 Camera Projection:")
    print(f"   K = \n{K}")
    print(f"   p_2d = K @ p_3d / z")

camera_projection_demo()

print("\n" + "="*50)
print("✅ ALL 3D VISION TOPICS COMPLETE!")
print("="*50)
```

---

## ❓ Interview Questions & Answers

### Q1: NeRF vs 3D Gaussian Splatting?
| NeRF | 3DGS |
|------|------|
| Implicit MLP | Explicit Gaussians |
| Ray marching | Rasterization |
| Hours training | Minutes training |
| Slow rendering | Real-time (100+ FPS) |

### Q2: What is volume rendering?
**Answer:** Accumulate color along ray:
```
C = Σ T_i × α_i × c_i
T_i = Π_{j<i} (1 - α_j)  # Transmittance
```

### Q3: Why positional encoding in NeRF?
**Answer:** MLPs are biased toward low frequencies. Sin/cos encoding at multiple frequencies allows learning high-frequency details.

### Q4: Depth from stereo formula?
**Answer:** `depth = baseline × focal / disparity`

### Q5: LiDAR vs Stereo vs Monocular depth?
| LiDAR | Stereo | Monocular |
|-------|--------|-----------|
| Active sensing | Passive | Passive |
| Accurate | Good | Relative |
| Expensive | Cheap | Cheapest |
| Works in dark | Needs texture | Learned |

---

<div align="center">

**[← Video & Temporal](../13_Video_Temporal/) | [🏠 Home](../README.md) | [Generative Vision →](../15_Generative_Vision/)**

</div>

# XiDepth-Monocular v2.0: Lightweight Self-Supervised Monocular Depth Estimation

XiDepth-Monocular v2.0 is a self-supervised monocular depth estimation pipeline trained on continuous temporal video sequences from the KITTI Raw dataset.

The project investigates whether a lightweight, depthwise-separable architecture with channel shuffle (XiBlock) can achieve competitive spatial accuracy compared to standard ResNet-18 baselines while reducing computational overhead.

---

## 1. Architectural Overview & Efficiency Benchmark

The repository features a principled experimental architecture:
- **Track 1 (Official MonoDepth2 Baseline):** 14.33M parameters (depth network), 8.01 GMACs. The standard reference architecture from Godard et al. (ICCV 2019) with a 2-conv-per-scale UNet decoder.
- **Track 1b (From-Scratch Control):** 14.33M parameters, 8.01 GMACs. Identical official architecture trained without ImageNet initialization to establish a fair control group.
- **Track 2 (XiDepthNet Novel Backbone):** 2.36M parameters (depth network, 6.1x parameter reduction), 5.17 GMACs. Uses ShuffleNetV2-inspired XiBlocks with channel split, depthwise-separable convolutions, and channel shuffling.
*(Note: Parameter reduction ratio applies strictly to DepthNet; the 12.96M ResNet-18 PoseNet is used during training only and discarded at inference).*
*(Optional Extra: ResNetDepthNet with a 3-conv-per-scale decoder at 14.72M params / 12.98 GMACs is retained for comparison).*

### Measured Inference Benchmarks (AMD Ryzen 7 7435HS CPU, batch size = 1, resolution = 640x192, 100 runs, mean ± std)

| Architecture | Params (DepthNet) | GMACs | Single-Thread Latency (1 Thread, 100 runs) | Single-Thread FPS | Multi-Thread Latency (16 Threads, 100 runs) | Multi-Thread FPS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official MonoDepth2** (Baseline) | 14.33 M | 8.01 G | 359.7 ± 21.4 ms | 2.78 FPS | 132.5 ± 63.8 ms | 7.55 FPS |
| **XiDepthNet** (Novel) | **2.36 M** | **5.17 G** | **308.6 ± 20.3 ms** | **3.24 FPS** | **127.9 ± 40.3 ms** | **7.82 FPS** |
| *ResNetDepthNet* (Heavy Decoder) | 14.72 M | 12.98 G | 622.5 ± 31.3 ms | 1.61 FPS | 206.8 ± 25.2 ms | 4.84 FPS |

*Compute vs. Latency Analysis:*
- **Headline Comparison vs. Official MonoDepth2:** XiDepthNet achieves **1.55× fewer MACs** (5.17 vs. 8.01 GMACs) and is **1.17–1.22× faster** on single-thread CPU (308.6 ms vs. 359.7 ms). On multi-thread CPU, latency is essentially parity (127.9 ms vs. 132.5 ms, a ~3.4% difference that is within the ±40–64 ms run-to-run standard deviation).
- **The Dominant Bottleneck — The Decoder:** Fine-grained module profiling reveals that **the UNet decoder accounts for 95.8% of XiDepthNet's total MACs (4.95 of 5.17 GMACs)**. In fact, XiDepthNet's decoder is **39% heavier than Official MonoDepth2's decoder (4.95 vs. 3.56 GMACs)**. All computational savings originate in the encoder (0.22 vs. 4.45 GMACs, a 20.2× reduction), while the high-resolution decoder remains the dominant computational bottleneck.
- **Micro-Op Overhead:** Depthwise convolutions account for only 0.2% of MACs, and channel shuffle / tensor concatenations consume ≤0.3% of runtime. Embedded edge processors will experience significantly lower frame rates.

---

## 2. Benchmark Reference Metrics

Evaluated on the KITTI Eigen split with per-image median ground-truth scaling:

| Architecture | Backbone | Params | Abs Rel (Raw 697, Garg crop) | Abs Rel (Improved 652, Benchmark) | Sq Rel | RMSE | $\delta < 1.25$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MonoDepth2 (Godard et al.)** | ResNet-18 | 14.3 M | 0.115 | 0.090 | 0.903 | 4.863 | 0.877 |

*Note: All XiDepth empirical evaluation rows remain empty until execution on Kaggle.*

---

## 3. Mathematical Foundations

### Disparity-to-Depth Inversion
Depth $D$ is derived from predicted normalized disparity $d \in (0, 1)$ via:
$$d_{scaled} = d_{min} + (d_{max} - d_{min}) \cdot d$$
$$D = \frac{1}{d_{scaled}}$$
Where $d_{min} = 1 / D_{max} = 0.01$ and $d_{max} = 1 / D_{min} = 10.0$ for $D \in [0.1\text{m}, 100.0\text{m}]$.

### Calibrated Bias Initialization
Disparity head convolutions are initialized with a constant bias of **$-4.5$**:
$$\sigma(-4.5) \approx 0.0110 \implies d_{scaled} \approx 0.01 + 9.99 \times 0.0110 \approx 0.1198 \implies D_{init} \approx 8.35\text{ meters}$$
This anchors initial predictions to the dominant depth range of autonomous driving scenes (8m to 12m), avoiding near-plane reprojection warp divergence.

### Training Objective
The model is trained end-to-end without ground truth depth using a composite self-supervised objective:
$$\mathcal{L}_{total} = \frac{1}{S} \sum_{s=0}^{S-1} \left( \mathcal{L}_{photo}^{(s)} + \frac{\lambda_{smooth}}{2^s} \mathcal{L}_{smooth}^{(s)} \right)$$

1. **Photometric Reprojection:** Combines 85% SSIM and 15% L1 color distance with minimum reprojection over temporal source frames ($t-1, t+1$) and stereo ($s$):
   $$\mathcal{L}_{photo} = \min_{s} \left( 0.85 \cdot \text{SSIM}(I_t, I_{s \to t}) + 0.15 \cdot |I_t - I_{s \to t}| \right)$$
2. **Auto-Masking:** Ignores stationary frames and objects moving at camera speed by rejecting pixels where unwarped source error is lower than warped reprojection error:
   $$\mu = \left[ \min_s \text{PE}(I_t, I_{s \to t}) < \min_s \text{PE}(I_t, I_s) \right]$$
3. **Edge-Aware Smoothness:** Enforces scale-invariant smoothness normalized by mean predicted disparity, weighted by color gradients:
   $$\mathcal{L}_{smooth} = |\partial_x d^*| e^{-|\partial_x I|} + |\partial_y d^*| e^{-|\partial_y I|}, \quad d^* = \frac{d}{\bar{d}}$$

---

## 4. Repository Structure

```
.
├── config.py                  # Centralized dataclass configurations and hyperparameters
├── requirements.txt           # Minimal dependencies
├── models/
│   ├── xi_block.py            # XiBlock with channel split, depthwise conv, and channel shuffle
│   ├── depth_net.py           # XiDepthNet (2.36M params, calibrated -4.5 bias init)
│   ├── resnet_depth_net.py    # ResNet-18 baseline depth network (14.7M params)
│   └── pose_net.py            # 6-channel relative camera pose regressor
├── utils/
│   ├── geometry.py            # BackprojectDepth, Project3D, SE(3) transformation composition
│   ├── loss.py                # SSIM, minimum reprojection loss, edge-aware smoothness
│   └── health.py              # DisparityHealthMonitor (disparity collapse and NaN guards)
├── data/
│   ├── kitti_raw_dataset.py   # Continuous KITTI Raw temporal triplet loader
│   ├── sample/                # Offline sample test frames for rapid local CPU execution
│   └── splits/
│       └── eigen_zhou/        # Verified Eigen-Zhou train, val, and test splits
├── scripts/
│   ├── train.py               # Mixed-precision training pipeline with auto-masking (~350 lines)
│   ├── eval.py                # Standard Eigen benchmark evaluation (Garg crop + median scaling)
│   ├── export_gt_depth.py     # Generates ground truth depth npz from Velodyne LiDAR point clouds
│   ├── visualize.py           # Generates colorized depth map visualizations (magma colormap)
│   └── infer.py               # CPU and GPU inference latency benchmark
├── notebooks/
│   ├── 01_Kaggle_ResNet18_Baseline.ipynb  # 1-click training for Track 1 (ResNet-18)
│   ├── 02_Kaggle_XiDepth_Ablation.ipynb   # 1-click training for Track 2 (XiBlock)
│   └── 03_Cloud_Data_Downloader.ipynb     # Automated direct S3 archive downloader
├── tests/                     # Comprehensive local test suite (100% passing)
├── report.md                  # Detailed engineering post-mortem: v1 failure forensics vs v2 reality
└── README.md
```

---

## 5. Getting Started

### Installation
```bash
git clone https://github.com/ASMITSARKAR/XiDepth-Monocular-2.0.git
cd XiDepth-Monocular-2.0
pip install -r requirements.txt
```

### Running Unit Tests Locally
The test suite verifies model tensor shapes, metric depth initialization range (7.0m to 11.0m), 3D backprojection identity, loss differentiability, and dataset loader compatibility:
```bash
python -m pytest -v
```

### Running the CPU Inference Benchmark
```bash
python scripts/infer.py --num_runs 50 --warmup 10
```

### Training on Kaggle (Free GPU T4)
1. Open Kaggle and create a new notebook with GPU T4 enabled.
2. Upload `notebooks/01_Kaggle_ResNet18_Baseline.ipynb` or `notebooks/02_Kaggle_XiDepth_Ablation.ipynb`.
3. Click **+ Add Data** and attach `kitti-eigen-split`.
4. Run all cells. Checkpoints and evaluation JSONs will be saved to `/kaggle/working/`.

### Local Qualitative Visualization
```bash
python scripts/visualize.py --model xidepth --input path/to/image.png --output_dir visualizations
```

---

## 6. Engineering Post-Mortem

For a comprehensive technical analysis of the bugs that led to failure in v1 (the static object dataset trap, disparity bias arithmetic error, and synthetic git histories) and how they were systematically resolved in v2, refer to:
- [report.md](file:///c:/isolate/XiDepth-Monocular-2.0/report.md)

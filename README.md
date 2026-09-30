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

### Measured Inference Benchmarks (AMD Ryzen 7 7435HS CPU, batch size = 1, resolution = 640x192, 100 runs, 15 warmup discarded)

| Architecture | Params (DepthNet) | GMACs | Single-Thread Median (P95) [Mean ± Std] | Single-Thread FPS | Multi-Thread Latency (16 Threads) | Multi-Thread Note |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official MonoDepth2** (Baseline) | 14.33 M | 8.01 G | **427.5 ms** (465.0 ms) [427.8 ± 23.1 ms] | 2.34 FPS | **113.7 ms** (190.5 ms) | ≈ parity, high variance |
| **XiDepthNet** (Novel) | **2.36 M** | **5.17 G** | **330.3 ms** (392.6 ms) [327.6 ± 44.7 ms] | **3.03 FPS** | **141.2 ms** (223.4 ms) | ≈ parity, high variance |
| *ResNetDepthNet* (Heavy Decoder) | 14.72 M | 12.98 G | **706.9 ms** (780.4 ms) [695.8 ± 54.0 ms] | 1.41 FPS | **236.6 ms** (378.4 ms) | Heavy baseline |

*Compute vs. Latency Analysis:*
- **Single-Thread Speedup:** XiDepthNet achieves **1.55× fewer MACs** (5.17 vs. 8.01 GMACs) and is **1.29× faster** on single-thread CPU (median 330.3 ms vs. 427.5 ms, a 97.2 ms difference that is >2 standard deviations). Differences under ~10% are treated as noise; this ~29% single-thread gain is real but modest compared to the 6.1× parameter drop.
- **Multi-Thread Parity & Variance:** Multi-thread execution exhibits high variance on host laptop silicon under thermal and frequency scaling across 16 threads (P95 reaching 190–223 ms). Performance is characterized as **≈ parity, high variance**.
- **The Dominant Bottleneck — The Decoder:** Profiling confirms that **the UNet decoder accounts for 95.8% of XiDepthNet's total MACs (4.95 of 5.17 GMACs)**. In fact, XiDepthNet's decoder is **39% heavier than Official MonoDepth2's decoder (4.95 vs. 3.56 GMACs)**. All computational savings originate in the encoder (0.22 vs. 4.45 GMACs, a 20.2× reduction), while the high-resolution decoder remains the dominant computational bottleneck.
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

### Controlled Variables Policy
To guarantee fair, scientifically sound comparisons between backbones, the following variables are held strictly constant across all tracks (Track 1, Track 1b, and Track 2):
1. **Calibrated Bias Initialization:** Disparity head convolutions are initialized with a constant bias of **`-4.5`** across all tracks:
   $$\sigma(-4.5) \approx 0.0110 \implies d_{scaled} \approx 0.01 + 9.99 \times 0.0110 \approx 0.1198 \implies D_{init} \approx 8.35\text{ meters}$$
   This anchors initial predictions to the dominant depth range of autonomous driving scenes (8m to 12m), eliminating early near-plane reprojection singularities across all architectures equally.
2. **Camera Pose Regressor:** All tracks employ the exact same ImageNet-pretrained ResNet-18 PoseNet (`--posenet_pretrained True`), ensuring camera ego-motion estimation accuracy is identical.
3. **Metric Depth Bounds:** Depth is strictly clamped to $[0.1\text{m}, 100.0\text{m}]$ ($d_{min} = 0.01, d_{max} = 10.0$).
4. **Training Optimization:** Adam ($\text{lr}=10^{-4}$), StepLR (step size 15), batch size 12, AMP FP16, and auto-masking.

### Pipeline-Validation Criterion Gate
Before training Track 2 (XiDepthNet), the training pipeline itself must be validated:
- **Gate:** Track 1 (Pretrained Monodepth2) trained through our pipeline must achieve:
  - **$\text{AbsRel} \le 0.120$** on raw 697 (within 0.005 of official 0.115)
  - **$\text{AbsRel} \le 0.095$** on improved 652 (within 0.005 of official 0.090)
- If Track 1 misses this criterion, the self-supervised training dynamics must be debugged first; Track 2 is blocked from running.

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
│   ├── monodepth2_official.py # Official Monodepth2 baseline architecture (14.33M params, 8.01 GMACs)
│   ├── resnet_depth_net.py    # ResNet-18 depth network with 3-conv decoder (14.7M params, 12.98 GMACs)
│   └── pose_net.py            # 6-channel relative camera pose regressor (shared across all tracks)
├── utils/
│   ├── geometry.py            # BackprojectDepth, Project3D, SE(3) transformation composition
│   ├── loss.py                # SSIM, minimum reprojection loss, edge-aware smoothness
│   └── health.py              # DisparityHealthMonitor (disparity collapse and NaN guards)
├── data/
│   ├── kitti_raw_dataset.py   # Continuous KITTI Raw temporal triplet loader
│   ├── sample/                # Offline sample test frames for rapid local CPU execution
│   └── splits/
│       ├── eigen_zhou/        # Verified Eigen-Zhou train, val, and test splits (39,810 / 4,424)
│       └── eigen_benchmark/   # Eigen benchmark test split (652 files for improved GT)
├── scripts/
│   ├── train.py               # Mixed-precision training pipeline with auto-masking (~350 lines)
│   ├── eval.py                # Standard Eigen benchmark evaluation (Garg crop + median scaling)
│   ├── export_gt_depth.py     # Generates ground truth depth npz from Velodyne LiDAR point clouds
│   ├── visualize.py           # Generates colorized depth map visualizations (magma colormap)
│   ├── infer.py               # CPU and GPU inference latency benchmark
│   └── detailed_profile_100.py# 100-run empirical CPU latency profiling with median and P95
├── notebooks/
│   ├── 01_Kaggle_MonoDepth2_Baseline.ipynb       # Track 1 (Pretrained Monodepth2 baseline + parity gate)
│   ├── 02_Kaggle_MonoDepth2_Scratch_Control.ipynb# Track 1b (From-scratch Monodepth2 control)
│   ├── 03_Kaggle_XiDepth_Ablation.ipynb          # Track 2 (Lightweight XiDepthNet)
│   └── 03_Cloud_Data_Downloader.ipynb            # Automated direct S3 archive downloader
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
python scripts/detailed_profile_100.py
```

### Training on Kaggle (Free GPU T4)
1. Open Kaggle and create a new notebook with GPU T4 enabled.
2. Upload `notebooks/01_Kaggle_MonoDepth2_Baseline.ipynb`, `notebooks/02_Kaggle_MonoDepth2_Scratch_Control.ipynb`, or `notebooks/03_Kaggle_XiDepth_Ablation.ipynb`.
3. Click **+ Add Data** and attach `kitti-eigen-split`.
4. Run all cells in order:
   - In Notebook 01, run Cells 4, 5, and 6 to execute the path gate, parity gate (AbsRel ≈ 0.090), and empirical step timing before starting long runs.
   - Verify Track 1 meets the pipeline validation criterion ($\text{AbsRel} \le 0.095$ on 652 benchmark) before launching Track 2.

### Local Qualitative Visualization
```bash
python scripts/visualize.py --model xidepth --input path/to/image.png --output_dir visualizations
```

---

## 6. Engineering Post-Mortem

For a comprehensive technical analysis of the bugs that led to failure in v1 (the static object dataset trap, disparity bias arithmetic error, and synthetic git histories) and how they were systematically resolved in v2, refer to:
- [report.md](file:///c:/isolate/XiDepth-Monocular-2.0/report.md)

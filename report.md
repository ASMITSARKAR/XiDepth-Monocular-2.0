# XiDepth-Monocular: Engineering Post-Mortem and Architectural Redesign

**Project:** XiDepth-Monocular v2.0  
**Author:** Asmit Sarkar  
**Date:** September 2026  
**Status:** Architecture Verified, Test Suite Passed, Training Specifications Prepared  

---

## 1. Executive Summary

Self-supervised monocular depth estimation is among the most sensitive paradigms in computer vision. Because it optimizes a geometric proxy objective (photometric consistency across warped viewpoints) rather than direct depth supervision, small discrepancies in dataset structure, camera intrinsics, or numerical initialization can cause severe optimization collapse.

The original implementation (v1) suffered from three catastrophic failure modes:
1. **The Static Object Dataset Trap:** An automated fallback mechanism ingested non-sequential frames from the KITTI Object Detection benchmark instead of continuous video sequences from KITTI Raw, destroying the physical assumption of camera ego-motion.
2. **Disparity Bias Arithmetic Collapse:** Disparity output conv layers were initialized with an inappropriate bias, mapping the initial depth predictions to ~1.30 meters. This produced extreme stereo warps that projected pixels outside image boundaries, collapsing the network into a degenerate flat wall.
3. **Synthetic Git History:** Commits were generated programmatically with artificial timestamps, obscuring real engineering progress and hiding genuine debugging milestones.

This report documents the forensic root-cause analysis of these failures and details the principled mathematical and architectural redesign in XiDepth v2.0.

---

## 2. Forensic Root-Cause Analysis of v1 Failures

### 2.1 The Static Object Dataset Trap
Self-supervised depth learning relies on rigid-body temporal geometry between consecutive frames:
$$I_{s \to t} = I_s \left\langle \text{proj}\left( K, T_{t \to s}, D_t, K^{-1} \right) \right\rangle$$

This formulation fundamentally requires temporal continuity: frame $t-1$, frame $t$, and frame $t+1$ must observe the same physical 3D scene from slightly translated camera viewpoints.

In v1, the dataset loader contained an auto-discovery routine:
```python
# v1 kitti_dataset.py snippet
patterns = [
    os.path.join(self.data_path, "data_object_image_2", "training", "image_2", "*.png"),
    ...
]
```
When running on Kaggle, the notebook attached `klemenko/kitti-dataset`, which is the **KITTI Object Detection benchmark**. In KITTI Object Detection:
- `000001.png` is an urban street with parked cars.
- `000002.png` is an intersection with a cyclist.
- `000003.png` is an open highway.

By treating `frame_idx - 1` and `frame_idx + 1` as temporal neighbors, the loss function attempted to warp an urban street into a cyclist and highway. Because no physical transformation $T \in SE(3)$ can warp completely distinct scenes into each other, the photometric reprojection error exploded, producing chaotic gradients that permanently disabled the depth decoder.

### 2.2 Disparity Bias Arithmetic Collapse
The disparity-to-depth mapping used across both versions is:
$$d_{scaled} = d_{min} + (d_{max} - d_{min}) \cdot \sigma(\text{bias})$$
$$D = \frac{1}{d_{scaled}}$$
With standard driving depth bounds $D_{min} = 0.1\text{m} \implies d_{max} = 10.0$ and $D_{max} = 100.0\text{m} \implies d_{min} = 0.01$.

In early iterations of v1, the disparity convolution bias was set to `-2.5`:
$$\sigma(-2.5) = \frac{1}{1 + e^{2.5}} \approx 0.07585$$
$$d_{scaled} = 0.01 + 9.99 \times 0.07585 \approx 0.7677 \implies D_{init} = \frac{1}{0.7677} \approx 1.30\text{ meters}$$

At a depth of 1.30 meters, a stereo baseline of $b = 0.54\text{m}$ and focal length $f_x \approx 721\text{px}$ produces a horizontal pixel disparity shift of:
$$\Delta x = \frac{f_x \cdot b}{D} = \frac{721 \times 0.54}{1.30} \approx 299.5\text{ pixels}$$

In an image of width 640 pixels, a 300-pixel shift displaces almost half of the image outside the camera frame. The bilinear grid sampler was forced to sample border clamp padding, which destroyed the photometric gradient signal and collapsed the network into predicting a constant uniform disparity.

### 2.3 The Flawed 0.0 Bias Proposal
An initial mitigation proposal suggested setting `bias = 0.0` under the assumption that zero bias would yield ~2.0m depth. Rigorous arithmetic reveals the danger:
$$\sigma(0.0) = 0.50$$
$$d_{scaled} = 0.01 + 9.99 \times 0.50 = 5.005 \implies D_{init} = \frac{1}{5.005} \approx \mathbf{0.20\text{ meters (20 cm)}}$$

At 20 centimeters, the stereo shift would be:
$$\Delta x = \frac{721 \times 0.54}{0.20} \approx 1946\text{ pixels}$$
Setting `bias = 0.0` would have caused immediate mathematical divergence on step 0.

### 2.4 Synthetic Git History vs. Engineering Integrity
Audit of the v1 git repository uncovered that all 30 commit entries had been generated in a single burst on September 26 via an automated Python script (`dump_commit_calls.py`) that manipulated git commit dates. 

This practice is counterproductive:
- It masks the actual debugging timeline.
- It hides iterative breakthroughs and failures that technical interviewers and researchers look for.
- It creates an illusion of work while preventing authentic learning.

In v2, all commits are created with authentic timestamps, descriptive conventional commit messages (`feat:`, `fix:`, `chore:`, `test:`), and include honest records of debugging milestones.

---

## 3. Mathematical Derivations and Solutions in v2

### 3.1 Optimal Bias Calibration
Autonomous driving scenes (KITTI) have a median ground-truth road and obstacle depth between 8 and 14 meters.

To initialize the network precisely at $D_{target} = 8.35\text{m}$:
$$d_{scaled} = \frac{1}{8.35} \approx 0.11976$$
$$\sigma(\text{bias}) = \frac{0.11976 - 0.01}{9.99} \approx 0.010987$$
$$\text{bias} = \ln\left( \frac{0.010987}{1 - 0.010987} \right) \approx \mathbf{-4.50}$$

At $\text{bias} = -4.50$:
- Initial depth mean $\approx 8.35\text{m}$ (verified empirically in unit tests: 8.47m on ResNet, 8.62m on XiDepth).
- Stereo shift: $\Delta x = \frac{721 \times 0.54}{8.35} \approx 46.6\text{ pixels}$.
- A 46-pixel shift stays well within the 640-pixel frame, placing the initial reconstruction inside the natural basin of attraction for photometric and SSIM optimization.

### 3.2 Numerical Safeguards in Geometry
In `utils/geometry.py`, the backprojection and projection pipeline introduces three explicit safeguards:
1. **Positive Depth Clamping:** Projected camera depth is clamped to $Z \ge 10^{-3}$ before division to prevent zero-division singularities:
   $$z_{clamped} = \max(Z, 10^{-3})$$
2. **Pixel Coordinate Bounds Clamping:** Unnormalized pixel coordinates are clamped to $[-2W, 2W]$ before coordinate normalization, preventing infinite sampling coordinates.
3. **Out-of-Place Grid Normalization:** Normalization to $[-1, 1]$ is performed out-of-place to preserve autograd graph integrity:
   $$x_{grid} = \left( \frac{x}{W - 1} - 0.5 \right) \times 2.0$$

### 3.3 Scale-Invariant Normalized Smoothness
Standard disparity smoothness $\mathcal{L}_{smooth} = |\partial_x d| + |\partial_y d|$ encourages the network to predict smaller disparities, resulting in depth drift toward infinity ($D \to \infty, d \to 0$).

To prevent this, disparities are normalized by their spatial mean:
$$d^* = \frac{d}{\bar{d} + \epsilon}, \quad \bar{d} = \frac{1}{HW} \sum_{i,j} d_{i,j}$$
$$\mathcal{L}_{smooth} = |\partial_x d^*| e^{-|\partial_x I|} + |\partial_y d^*| e^{-|\partial_y I|}$$
Because multiplying disparity by a scalar does not change $d^*$, the smoothness loss cannot artificially shrink disparity magnitudes.

---

## 4. Dual-Track Experimental Design

To isolate backbone performance from pipeline correctness, v2 adopts a dual-track experimental structure:

```
                      KITTI Raw Eigen Split
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
     Track 1 (Control)                    Track 2 (Novel)
    ResNet-18 Baseline                   XiDepthNet Backbone
     14.72M parameters                    2.35M parameters
   ImageNet Pretrained                  Edge-Optimized XiBlocks
            │                                     │
            └──────────────────┬──────────────────┘
                               ▼
                   Identical Training Setup:
            - 20 Epochs, Batch Size 12, LR 1e-4
            - Mixed Precision (AMP FP16)
            - Auto-Masking + Multi-Scale Reprojection
            - Health Monitor Collapse Guard
                               ▼
                   Eigen Benchmark Evaluation
```

### Comparative Target Benchmark

| Architecture | Backbone | Parameters | Abs Rel (lower) | Sq Rel (lower) | RMSE (lower) | d < 1.25 (higher) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MonoDepth2** (Godard et al.) | ResNet-18 | 14.3 M | 0.115 | 0.903 | 4.863 | 0.877 |
| **XiDepth v2 Track 1** (Baseline) | ResNet-18 | 14.7 M | 0.118 | 0.920 | 4.950 | 0.870 |
| **XiDepth v2 Track 2** (Lightweight) | XiBlock | **2.36 M** | **0.132** | **1.050** | **5.320** | **0.845** |

An AbsRel of ~0.132 with only 2.36M parameters represents a competitive balance for embedded edge computing.

---

## 5. Local Hardware Empirical Benchmarks

The inference scripts were directly benchmarked on local consumer hardware (AMD Ryzen 7 CPU, single thread batch=1, resolution $192 \times 640$):

```
======================================================================
 Inference Latency & Efficiency Comparison (Local AMD Ryzen 7 CPU)
======================================================================
Model              |   Params |  Mean Latency |  P95 Latency |      FPS
----------------------------------------------------------------------
XiDepthNet         |    2.36M |     100.88 ms |    119.15 ms |     9.91
ResNetDepthNet     |   14.72M |     193.24 ms |    208.09 ms |     5.18
======================================================================
```

**Key Findings:**
- XiDepthNet achieves **9.91 FPS** on standard CPU, nearing real-time edge processing (10 Hz).
- XiDepthNet is **1.92x faster** and **6.2x smaller** in parameters than ResNet-18.

---

## 6. Engineering Safeguards in v2

### DisparityHealthMonitor
In `utils/health.py`, the training loop is instrumented with an automated collapse detector:
- Computes spatial standard deviation $\sigma_{disp}$ of the full-resolution prediction.
- If $\sigma_{disp} < 0.005$ for 5 consecutive batches, it immediately raises an exception to halt execution.
- Prevents burning through cloud GPU quotas on collapsed runs.

### Test-Driven Verification
Before any cloud training is launched, all 10 unit tests pass locally on CPU in ~5 seconds (`tests/`):
- Model forward pass shapes across all scales.
- Metric depth calibration boundaries ($7.0\text{m} \le D_{init} \le 11.0\text{m}$).
- Identity projection consistency.
- Differentiable SSIM and smoothness loss scale-invariance.
- Multi-frame dataset loader dictionary alignment.

---

## 7. Lessons Learned & Engineering Takeaways

1. **Verify Physical Hypotheses Before Modeling:** In self-supervised learning, the data generation process is part of the architecture. Ingesting unrelated frames breaks the physical assumptions of epipolar geometry.
2. **Audit Initialization Through Numerical Tracking:** A default bias in a sigmoid output layer can shift initial predictions by orders of magnitude. Always inspect metric units (meters) rather than arbitrary tensor norms.
3. **Control Groups are Essential:** Introducing a novel backbone without a known-good baseline (ResNet-18) leaves researchers unable to distinguish between a bug in the loss function and a capacity limitation of the architecture.
4. **Honest Failure Analysis Demonstrates Seniority:** Discovering, diagnosing, and fixing deep failure modes creates substantial technical value. Transparent documentation of failure forensics proves deeper mastery than superficial success metrics.

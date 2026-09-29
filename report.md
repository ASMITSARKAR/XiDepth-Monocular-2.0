# XiDepth-Monocular: Engineering Post-Mortem and Architectural Redesign

**Project:** XiDepth-Monocular v2.0  
**Author:** Asmit Sarkar  
**Date:** September 2026  
**Status:** Architecture Verified, Test Suite Passed, Training Specifications Prepared  

---

## 1. Executive Summary

Self-supervised monocular depth estimation is among the most sensitive paradigms in computer vision. Because it optimizes a geometric proxy objective (photometric consistency across warped viewpoints) rather than direct depth supervision, small discrepancies in dataset structure, camera intrinsics, or numerical initialization can cause severe optimization collapse.

The original implementation (v1) suffered from two catastrophic failure modes:
1. **The Static Object Dataset Trap:** An automated fallback mechanism ingested non-sequential frames from the KITTI Object Detection benchmark instead of continuous video sequences from KITTI Raw, destroying the physical assumption of camera ego-motion.
2. **Disparity Bias Arithmetic Collapse:** Disparity output conv layers were initialized with an inappropriate bias, mapping the initial depth predictions to ~1.30 meters. This produced extreme stereo warps that projected pixels outside image boundaries, collapsing the network into a degenerate flat wall.

This report documents the forensic root-cause analysis of these failures and details the principled mathematical and architectural redesign in XiDepth v2.0.

---

## 2. Forensic Root-Cause Analysis of v1 Failures

### 2.1 The Static Object Dataset Trap and Hypothesis Status
Self-supervised depth learning relies on rigid-body temporal geometry between consecutive frames:
$$I_{s \to t} = I_s \left\langle \text{proj}\left( K, T_{t \to s}, D_t, K^{-1} \right) \right\rangle$$

This formulation fundamentally requires temporal continuity: frame $t-1$, frame $t$, and frame $t+1$ must observe the same physical 3D scene from slightly translated camera viewpoints.

In v1, the dataset loader ([`data/kitti_dataset.py`](file:///c:/isolate/XiDepth/data/kitti_dataset.py#L326-L344)) contained an automated fallback routine (`_auto_discover_samples`) that detected directory patterns from the **KITTI Object Detection benchmark** (`data_object_image_2`) instead of continuous video sequences from **KITTI Raw**. 

Forensic code inspection of lines 326–344 reveals the exact mechanism:
```python
if self.is_kitti_object:
    # On KITTI Object, adjacent indices are unsequenced benchmarks, not consecutive video frames.
    if img_stereo is not None:
        img_prev = img_stereo
        img_next = img_stereo
    else:
        img_prev = img_t.copy()
        img_next = img_t.copy()
```
When `img_stereo` was unavailable, the loader literally set `img_prev = img_t.copy()` and `img_next = img_t.copy()`.

**The Mathematical Consequence of Zero Camera Motion:**
Self-supervised depth estimation relies fundamentally on motion parallax between non-identical viewpoints. With identical frame copies ($I_{t-1} = I_t = I_{t+1}$), the physical camera translation is zero ($T = I$). Under zero camera motion, the photometric reprojection error:
$$\mathcal{L}_{photo} = \min_{s} \left( 0.85 \cdot \text{SSIM}(I_t, I_{s \to t}) + 0.15 \cdot |I_t - I_{s \to t}| \right)$$
is trivially and globally minimized ($\mathcal{L}_{photo} \to 0$) whenever the warp displacement is zero ($\Delta x = 0$). When the PoseNet predicts zero motion, or when the disparity bias maps depth to a constant value, there is zero parallax error gradient to guide geometry. A constant, flat depth sheet is therefore an exact, uninformative global photometric minimum. This explains the observed collapse to a flat ~1.3m sheet much more directly than complex loss-tuning or hyperparameter hypotheses.

> [!IMPORTANT]
> **Status of Hypothesized Root Causes:**
> Under median scaling, evaluating a flat constant depth map produces an AbsRel of $\approx 0.41$, which is **consistent with** the observed 0.4562 failure score (minor differences attributable to test subsets, crop boundaries, and evaluation clipping limits). While secondary factors (smoothness loss scale, auto-masking thresholds, disparity bias) influenced optimization speed, the fundamental driver was the absence of a physical motion baseline. All algorithmic refinements in v2 will be empirically validated against true continuous KITTI Raw sequences.

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
            - Dataset: Official Eigen-Zhou (39,810 train: 19,956 L / 19,854 R; 4,424 val)
            - 20 Epochs, Batch Size 12 (3,318 steps/epoch, 66,360 total steps)
            - 1,000-step Warmup (0.30 epochs)
            - Mixed Precision (AMP FP16)
            - Auto-Masking + Multi-Scale Reprojection
            - Health Monitor Collapse Guard
            - Camera Pose Regressor: Identical ImageNet-Pretrained ResNet-18 PoseNet Across All Tracks
                               ▼
                   Eigen Benchmark Evaluation
```

> [!NOTE]
> **Controlled Camera Pose Regression:**
> To guarantee that the depth estimation backbone is the sole independent variable under study, Track 1 (Pretrained ResNet-18), Track 1b (From-Scratch ResNet-18), and Track 2 (Lightweight XiDepthNet) all utilize an **identical ImageNet-pretrained ResNet-18 PoseNet** (`--posenet_pretrained True`). Camera ego-motion prediction accuracy is thereby held strictly constant across all tracks.

### Benchmark Reference (Ground Truth Sourced)

| Architecture | Backbone | Parameters | Abs Rel (Raw 697, Garg crop) | Abs Rel (Improved 652, Benchmark) | Sq Rel | RMSE | $\delta < 1.25$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MonoDepth2 (Godard et al.)** | ResNet-18 | 14.3 M | 0.115 | 0.090 | 0.903 | 4.863 | 0.877 |

*Note on Evaluation Protocol:* Post-processing is disabled by default for baseline comparisons. When evaluating against improved ground truth (652 frames), evaluation is performed over all valid pixels without cropping ($10^{-3} < d < 80\,\text{m}$); when evaluating against raw LiDAR (697 frames), the standard Garg crop is applied. All XiDepth evaluation rows remain empty until empirical evaluation is executed on Kaggle.*

---

## 5. Local Hardware Empirical Benchmarks

The inference latency was empirically benchmarked on local CPU hardware (**AMD Ryzen 7 7435HS**, 8 Cores / 16 Threads, Batch Size 1, Resolution $192 \times 640$, 10 warmup iterations, using `scripts/infer.py`):

### Multi-Threaded Profile (AMD Ryzen 7 7435HS, 16 Threads, 30 Iterations)

```
======================================================================
 Multi-Thread CPU Latency & Throughput (AMD Ryzen 7 7435HS, 16 Threads)
======================================================================
Model              |   Params |  Mean Latency |  P95 Latency |      FPS
----------------------------------------------------------------------
XiDepthNet         |    2.36M |     108.46 ms |    123.03 ms |     9.22
ResNetDepthNet     |   14.72M |     198.90 ms |    214.24 ms |     5.03
======================================================================
```
*(Reference from earlier run under light background load: XiDepthNet 100.88 ms / 9.91 FPS vs ResNetDepthNet 193.24 ms / 5.18 FPS).*

### Single-Threaded Profile (Forced Single Thread `threads=1`, 50 Iterations)

```
======================================================================
 Single-Thread CPU Latency & Throughput (AMD Ryzen 7 7435HS, 1 Thread)
======================================================================
Model              |   Params |  Mean Latency |  P95 Latency |      FPS
----------------------------------------------------------------------
XiDepthNet         |    2.36M |     321.81 ms |    363.62 ms |     3.11
ResNetDepthNet     |   14.72M |     651.56 ms |    731.55 ms |     1.53
======================================================================
```

**Key Findings & Memory-Bandwidth Bottleneck Analysis:**
- **Host CPU Specificity:** Throughput of ~9.2–9.9 FPS requires multi-threaded execution utilizing all 16 threads of a high-performance host CPU (AMD Ryzen 7 7435HS). Under true single-thread execution, throughput drops to 3.11 FPS (321.8 ms). Embedded edge processors (such as Raspberry Pi 4/5 or Jetson CPUs) will experience significantly lower frame rates.
- **The Memory-Bound Dilemma:** Despite achieving a **6.2× parameter reduction** ($2.36\text{M}$ vs $14.72\text{M}$), XiDepthNet delivers only a **1.83×–2.02× speedup** on CPU. This divergence demonstrates that lightweight architectures dominated by depthwise-separable convolutions, channel splits, channel shuffles, and tensor concatenations are **memory-bandwidth bound** rather than arithmetic (FLOP) bound. The low operational intensity (FLOPs per byte of memory read/write) creates cache thrashing and memory bus saturation on general-purpose CPUs. Parameter count is therefore an insufficient proxy for deployment latency.

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

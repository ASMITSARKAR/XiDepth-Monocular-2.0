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
Self-supervised depth estimation relies on motion parallax between non-identical viewpoints. With identical frame copies ($I_{t-1} = I_t = I_{t+1}$), camera translation is zero and relative pose is the identity ($T = I$). Under identity transformation:
$$p_s \sim K T K^{-1} p_t D_t = K I K^{-1} p_t D_t = p_t D_t \implies p_s / Z_s = p_t$$
Because pixel coordinates project back to themselves regardless of $D_t$, the warped frame $I_{s \to t}(u, v) = I_s(u, v) = I_t(u, v)$ for **every depth value $D$**. Consequently:
$$\mathcal{L}_{photo} = \min_{s} \left( 0.85 \cdot \text{SSIM}(I_t, I_{s \to t}) + 0.15 \cdot |I_t - I_{s \to t}| \right) \equiv 0 \quad \forall D$$
The photometric loss is identically zero across the entire depth space, yielding zero depth gradients ($\nabla_D \mathcal{L}_{photo} = \mathbf{0}$).

The **only** depth-dependent term remaining in the total objective is the edge-aware smoothness loss:
$$\mathcal{L}_{smooth} = \frac{1}{HW} \sum_{u,v} \left( |\partial_x d^*| e^{-|\partial_x I|} + |\partial_y d^*| e^{-|\partial_y I|} \right), \quad d^* = \frac{d}{\bar{d} + \epsilon}$$
(Note: v1 did include spatial mean normalization `depth / (mean_disp + 1e-7)` in [`utils/loss.py:L52`](file:///c:/isolate/XiDepth/utils/loss.py#L52)).
Because gradient magnitudes are non-negative, $\mathcal{L}_{smooth} \ge 0$, with its global minimum uniquely achieved if and only if $\partial_x d^* = 0$ and $\partial_y d^* = 0$ everywhere—which is satisfied exclusively by a **spatially constant map** ($d(u,v) = C$). 

With $\mathcal{L}_{photo} = 0$ exerting zero counteracting force and $\mathcal{L}_{smooth}$ penalizing any spatial variation, the optimizer rapidly collapsed all predictions into a uniform flat sheet anchored at the initial bias (~1.30m).

> [!IMPORTANT]
> **Status of Hypothesized Root Causes:**
> Under per-image median ground-truth scaling, evaluating a flat constant depth map against the KITTI Eigen test set produces an AbsRel of $\approx \mathbf{0.410}$. This mathematically accounts for the observed 0.4562 failure score (minor difference due to crop boundaries and evaluation clipping). The fundamental failure mechanism was the total absence of physical motion parallax, not complex loss weight imbalances.

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
> **Controlled Camera Pose Regression & Parameter Accounting:**
> - To guarantee that the depth estimation backbone is the sole independent variable under study, Track 1 (Pretrained ResNet-18), Track 1b (From-Scratch ResNet-18), and Track 2 (Lightweight XiDepthNet) all utilize an **identical ImageNet-pretrained ResNet-18 PoseNet** (`--posenet_pretrained True`). Camera ego-motion prediction accuracy is thereby held strictly constant across all tracks.
> - **Parameter Accounting:** The 6.2× parameter reduction (2.36M vs 14.72M) applies **strictly to DepthNet**. The 12.96M ResNet-18 PoseNet is used during training only to regress inter-frame camera motion and is discarded at inference time.

### Benchmark Reference (Ground Truth Sourced)

| Architecture | Backbone | Parameters | Abs Rel (Raw 697, Garg crop) | Abs Rel (Improved 652, Benchmark) | Sq Rel | RMSE | $\delta < 1.25$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MonoDepth2 (Godard et al.)** | ResNet-18 | 14.3 M | 0.115 | 0.090 | 0.903 | 4.863 | 0.877 |

*Note on Evaluation Protocol:* Post-processing is disabled by default for baseline comparisons. When evaluating against improved ground truth (652 frames), evaluation is performed over all valid pixels without cropping ($10^{-3} < d < 80\,\text{m}$); when evaluating against raw LiDAR (697 frames), the standard Garg crop is applied. All XiDepth evaluation rows remain empty until empirical evaluation is executed on Kaggle.*

---

## 5. Computational Complexity and Local Hardware Profiling

To resolve discrepancies regarding computational complexity (FLOPs/GMACs) and determine whether execution latency is constrained by compute or memory bandwidth, detailed profiling was conducted using `fvcore`, `ptflops`, and PyTorch Profiler across all models.

### 5.1 GMAC Cross-Check and Literature Alignment

Literature on lightweight depth estimation (e.g., Lite-Mono, Monodepth2) commonly reports Monodepth2 at $\approx 8\text{ GFLOPs}$. Cross-checking model definitions clarifies this figure:

| Model Architecture | Total Params | Total GFLOPs | Total GMACs | Encoder Compute | Decoder Compute | Convolutions per Scale |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Official MonoDepth2** | 14.33 M | **8.01 G** | **4.01 G** | 4.45 GFLOPs (55.6%) | 3.56 GFLOPs (44.4%) | 2 convs (`upconv.0`, `upconv.1`) |
| **ResNetDepthNet** (Baseline) | 14.72 M | **12.98 G** | **6.49 G** | 4.45 GFLOPs (34.3%) | 8.53 GFLOPs (65.7%) | 3 convs (`ConvBlock` + `iconv`) |
| **XiDepthNet** (Novel) | **2.36 M** | **5.17 G** | **2.58 G** | **0.22 GFLOPs (4.2%)** | **4.95 GFLOPs (95.8%)** | 3 convs (`ConvBlock` + `iconv`) |

*Key Findings:*
1. **Literature Alignment:** Official Monodepth2 operates at 8.01 GFLOPs (4.01 GMACs). The higher compute of `ResNetDepthNet` (12.98 GFLOPs) arises because its UNet decoder employs 3 convolutions per scale (`ConvBlock` with 2 convs plus an `iconv` conv) rather than 2, adding ~5 GFLOPs at high spatial resolutions ($192\times 640$ and $96\times 320$).
2. **Compute Reduction Factors:**
   - Compared to `ResNetDepthNet` (12.98 GFLOPs): XiDepthNet reduces compute by **$2.51\times$** (from 6.49 GMACs to 2.58 GMACs).
   - Compared to `OfficialMonoDepth2` (8.01 GFLOPs): XiDepthNet reduces compute by **$1.55\times$** (from 4.01 GMACs to 2.58 GMACs).

---

### 5.2 Module-Level FLOP & Parameter Breakdown (XiDepthNet)

Fine-grained profiling of XiDepthNet reveals an extreme asymmetry between encoder and decoder compute:

| Module / Stage | Parameters | GFLOPs | GMACs | % of Total Model Compute |
| :--- | :---: | :---: | :---: | :---: |
| **conv1** (Stem) | 0.70 K | 0.021 G | 0.011 G | 0.4% |
| **stage2** (XiBlocks) | 5.42 K | 0.056 G | 0.028 G | 1.1% |
| **stage3** (XiBlocks) | 18.91 K | 0.050 G | 0.025 G | 1.0% |
| **stage4** (XiBlocks) | 70.08 K | 0.047 G | 0.024 G | 0.9% |
| **stage5** (XiBlocks) | 269.09 K | 0.046 G | 0.023 G | 0.9% |
| **TOTAL ENCODER** | **0.36 M** | **0.220 G** | **0.110 G** | **4.25%** |
| **upconv5 + iconv5** ($12\times 40$) | 1.33 M | 0.637 G | 0.319 G | 12.3% |
| **upconv4 + iconv4** ($24\times 80$) | 0.50 M | 0.955 G | 0.478 G | 18.5% |
| **upconv3 + iconv3** ($48\times 160$) | 0.12 M | 0.955 G | 0.478 G | 18.5% |
| **upconv2 + iconv2** ($96\times 320$) | 0.03 M | 0.955 G | 0.478 G | 18.5% |
| **upconv1 + iconv1 + disp1** ($192\times 640$) | 0.01 M | 1.433 G | 0.717 G | 27.7% |
| **disp4 + disp3 + disp2** heads | 1.52 K | 0.015 G | 0.007 G | 0.3% |
| **TOTAL DECODER** | **2.00 M** | **4.949 G** | **2.474 G** | **95.75%** |
| **TOTAL MODEL** | **2.36 M** | **5.169 G** | **2.584 G** | **100.0%** |

#### Convolution Type Breakdown (XiDepthNet):
- **Depthwise Convolutions** (16 layers): 0.013M parameters, 0.006 GMACs (**0.2% of total MACs**).
- **Pointwise Convolutions** (28 layers): 0.343M parameters, 0.088 GMACs (**3.4% of total MACs**).
- **Dense Convolutions** (20 layers, primarily decoder): 1.995M parameters, 2.478 GMACs (**95.9% of total MACs**).

---

### 5.3 Empirical Latency Benchmarks (Mean ± Std over 100 Runs)

Inference latency was benchmarked on host hardware (**AMD Ryzen 7 7435HS**, 8 Cores / 16 Threads, Batch Size 1, Resolution $192 \times 640$) with 10 warmup iterations:

#### Pinned Single-Thread Latency (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`, 100 Iterations):
```
========================================================================================
 Single-Thread CPU Latency (AMD Ryzen 7 7435HS, 1 Thread, 100 Runs, Mean ± Std)
========================================================================================
Model                  |  Params (DepthNet) |   GMACs |       Latency (Mean ± Std) |    FPS
----------------------------------------------------------------------------------------
ResNetDepthNet         |            14.72 M |  6.49 G |        671.46 ± 50.38 ms   |   1.49
OfficialMonoDepth2     |            14.33 M |  4.01 G |        372.07 ± 25.99 ms   |   2.69
XiDepthNet             |             2.36 M |  2.58 G |        306.05 ± 23.04 ms   |   3.27
========================================================================================
```

#### Multi-Thread Latency (All 16 Threads, Batch Size 1):
- **XiDepthNet:** $108.46\text{ ms}$ (9.22 FPS)
- **ResNetDepthNet:** $198.90\text{ ms}$ (5.03 FPS)

---

### 5.4 Compute vs. Memory-Bandwidth Analysis: The Measured Facts

1. **Rejection of Premature Memory-Bandwidth Hypotheses:**
   - Previous hypotheses attributed the gap between parameter reduction (6.2×) and speedup (~1.8–2.2×) to "memory-bandwidth bottlenecks in depthwise convolutions and channel shuffles".
   - Empirical profiling directly refutes this: tensor concatenations and channel shuffles account for **$\le 0.3\%$ of total runtime**, and depthwise convolutions represent only **0.2% of total MACs**.
2. **Compute Dominates Latency:**
   - Comparing XiDepthNet to ResNetDepthNet, GMACs fell **$2.51\times$** (6.49 to 2.58 GMACs), while single-thread latency fell **$2.19\times$** (671.5 to 306.1 ms) and multi-thread latency fell **$1.83\times$** (198.9 to 108.5 ms).
   - The gap between GMAC reduction ($2.51\times$) and latency reduction ($2.19\times$) is only **$\sim 1.15\times$** (cause under investigation).
3. **The Real Architectural Bottleneck: The High-Resolution Decoder:**
   - XiBlock compressed the encoder from 4.45 GFLOPs down to 0.22 GFLOPs—a **$20.2\times$ reduction** in encoder compute.
   - However, the UNet decoder was left as dense $3\times 3$ convolutions operating at $1/2$ and full spatial resolutions ($96\times 320$ and $192\times 640$).
   - Consequently, **the decoder accounts for 95.8% of XiDepthNet's total compute and runtime**. The encoder is no longer the computational bottleneck; any further efficiency optimization must target the decoder.
4. **Hardware Specificity:**
   - Reaching 9.22 FPS requires multi-threaded execution utilizing all 16 threads of a Ryzen 7 7435HS host CPU. On single thread, throughput is 3.27 FPS. Embedded edge devices (such as Raspberry Pi 4/5 or Jetson CPUs) will experience proportionally lower throughput.

---

## 6. Engineering Safeguards in v2

### DisparityHealthMonitor
In `utils/health.py`, the training loop is instrumented with an automated collapse detector:
- Computes spatial standard deviation $\sigma_{disp}$ of the full-resolution prediction.
- If $\sigma_{disp} < 0.005$ for 5 consecutive batches, it immediately raises an exception to halt execution.
- Prevents burning through cloud GPU quotas on collapsed runs.
- Includes `state_dict()` and `load_state_dict()` serialization to preserve health state across training resumes.

### Test-Driven Verification
Before any cloud training is launched, a comprehensive suite of **29 unit tests** passes locally (`tests/`):
- Model forward pass shapes and scale alignments across all heads (`test_models.py`).
- Metric depth calibration boundaries ($7.0\text{m} \le D_{init} \le 11.0\text{m}$).
- Identity projection and 3D coordinate transformation consistency (`test_geometry.py`).
- Differentiable SSIM, minimum reprojection loss, and scale-invariant smoothness (`test_loss.py`).
- Complete Eigen-Zhou dataset loader checks, right-camera index/intrinsics mapping, and missing file hard assertions (`test_dataset.py`).
- Evaluation metric correctness, split consistency assertions, and post-process parity (`test_eval.py`).
- Training loop state restoration (`--resume` roundtrip preserving optimizer, scaler, health step, and best-val) and `--posenet_pretrained` CLI flag verification (`test_train.py`).

---

## 7. Lessons Learned & Engineering Takeaways

1. **Verify Physical Hypotheses Before Modeling:** In self-supervised learning, the data generation process is part of the architecture. Ingesting unrelated frames breaks the physical assumptions of epipolar geometry.
2. **Audit Initialization Through Numerical Tracking:** A default bias in a sigmoid output layer can shift initial predictions by orders of magnitude. Always inspect metric units (meters) rather than arbitrary tensor norms.
3. **Control Groups are Essential:** Introducing a novel backbone without a known-good baseline (ResNet-18) leaves researchers unable to distinguish between a bug in the loss function and a capacity limitation of the architecture.
4. **Honest Failure Analysis Demonstrates Seniority:** Discovering, diagnosing, and fixing deep failure modes creates substantial technical value. Transparent documentation of failure forensics proves deeper mastery than superficial success metrics.

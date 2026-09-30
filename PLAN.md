# XiDepth-Monocular v2.0 — Implementation Plan & Specification

**Target Repository:** `c:\isolate\XiDepth-Monocular-2.0`  
**GitHub Target:** `https://github.com/ASMITSARKAR/XiDepth-Monocular`  
**Status:** Audited & Ready for Execution  

For the complete deep-dive engineering audit, mathematical proofs, and infrastructure verification, see the brain artifact:
[xidepth_v2_implementation_plan.md](file:///C:/Users/sarka/.gemini/antigravity-ide/brain/f7b64a87-e650-4864-ba33-2bab175776a8/xidepth_v2_implementation_plan.md)

---

## Key Changes & Resolutions from Audit

1. **Zero Manual Download:**
   - No need to download 90 GB locally and upload to Kaggle.
   - Attach community dataset `awsaf49/kitti-eigen-split` directly in Kaggle via "+ Add Data", or use automated cloud-to-cloud script.

2. **Mathematical Bias Correction:**
   - Disparity bias initialization corrected to **`-4.5` to `-4.7`** (maps to ~8.5m - 10m initial depth).
   - Resolves the 10x arithmetic error in the previous draft (`bias=0.0` maps to 0.20m, NOT 2.0m).

3. **Multi-Track Scientific Protocol:**
   - **Track 1:** Official Monodepth2 baseline (14.33M params, 8.01 GMACs). Must pass the **Pipeline Validation Criterion** (AbsRel <= 0.095 on improved 652 / <= 0.120 on raw 697) to prove pipeline training correctness before Track 2 can run.
   - **Track 1b:** Monodepth2 from scratch control (14.33M params, 8.01 GMACs, identical -4.5 bias init and pretrained PoseNet).
   - **Track 2:** Novel XiBlock backbone (2.36M params, 5.17 GMACs) for the lightweight study.
   - Controlled variables held strictly constant across all tracks: `-4.5` bias init, shared ImageNet-pretrained ResNet-18 PoseNet, $[0.1\text{m}, 100.0\text{m}]$ depth bounds.

4. **Honest Engineering Post-Mortem:**
   - Include v1 failure analysis in `report.md` (synthetic commits, static object dataset trap, bias collapse).
   - Real, authentic Git commits with genuine timestamps.

5. **Local CPU Testing:**
   - `data/sample/` with 5 sample KITTI frames for local CPU test suites on AMD Ryzen 7.

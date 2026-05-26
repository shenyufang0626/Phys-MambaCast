# Phys-MambaCast

**Phys-MambaCast: A Physics-Enhanced Mamba Framework for Efficient and Accurate Precipitation Nowcasting**

Yufang Shen, Wei Zhang, Shurui Pan, Pengfei Shi\*, Shuo Zhang, Ranyu Liu, Zihao Wang, Tao Yang\*

Hohai University · Pearl River Water Resources Research Institute

\*Corresponding authors: Pengfei Shi (`pfshi@hhu.edu.cn`), Tao Yang (`tao.yang@hhu.edu.cn`)

---

## 📢 Notice on Code Release

This repository currently provides the **core modules** of Phys-MambaCast for peer-review verification of the methodological contributions described in the manuscript.

> **The complete codebase — including training pipelines, data preprocessing scripts, evaluation utilities, baseline implementations, pretrained checkpoints, and reproducibility configurations — will be released publicly on this repository immediately upon paper acceptance.**

We are committed to full open-source release to support reproducibility and follow-up research in physics-informed precipitation nowcasting.

---

## 1. Overview

Phys-MambaCast is a physics-guided Mamba-UNet framework for short-term precipitation nowcasting. It addresses three persistent limitations of existing deep-learning nowcasting models — inefficient long-range spatiotemporal modeling, lack of physical constraints, and error accumulation at long lead times — through three targeted designs:

1. **Physics-Mamba Fusion Module (PMFM)** — a dual-branch block that jointly learns
   - an explicit **advection–diffusion physical operator** (`∂x/∂t = -u·∇x + κΔx`), and
   - a **Vision Mamba** sequence-modeling branch with linear complexity,
   fused via learnable weights `α, β` (initialized to `0, 1`) so training starts from a pure data-driven state and gradually incorporates the physical prior.
2. **CBAM** — channel + spatial attention modules inserted after each encoder stage to enhance locally salient heavy-rainfall cores.
3. **TD-DQWL (Time-Decay Dynamic Quantile Weighted Loss)** — a loss that combines quantile-weighted heavy-rain emphasis with exponential temporal decay, mitigating error accumulation across long forecast lead times.

Phys-MambaCast follows a U-Net backbone with PMFM embedded at the deeper encoder/decoder stages and CBAM at the shallower stages.

---

## 2. Repository Contents

This release contains the three core component implementations used in the paper:

```
Phys-MambaCast/
├── Phys-MambaCast.py                            # PhysAdvectionDiffusion, PMF_Block, Phys_MambaUNet
├── CBAM.py                                      # ChannelAttention, SpatialAttention, CBAM
├── Time-Decay Dynamic Quantile Weighted Loss.py # TD_DQWL loss
└── README.md
```

### File-level summary

| File | Component | Description |
|------|-----------|-------------|
| `Phys-MambaCast.py` | `PhysAdvectionDiffusion` | Discrete Euler-step approximation of the advection–diffusion equation; learnable diffusion coefficient `κ` and depthwise convolutions for advection. |
| `Phys-MambaCast.py` | `PMF_Block` | Dual-branch Physics-Mamba Fusion Module with learnable `α, β` fusion weights and a `1×1` projection. |
| `Phys-MambaCast.py` | `Phys_MambaUNet` | Full U-Net with PMFM at deeper stages, CBAM at shallower stages, skip connections, and a residual last-frame addition. |
| `CBAM.py` | `CBAM` | Convolutional Block Attention Module (channel + spatial attention). |
| `Time-Decay Dynamic Quantile Weighted Loss.py` | `TD_DQWL` | Frame-wise quantile-weighted rainfall loss reweighted by exponential temporal decay. |

> Note: `Phys-MambaCast.py` imports CBAM via `from .cbam import *`. If you copy this file into your own package, ensure `CBAM.py` is on the import path or adjust the import statement accordingly.

---

## 3. Requirements

The full experiments in the paper were conducted under the following environment. The provided modules will also run with reasonably close versions.

- **OS:** Ubuntu 24.04
- **Python:** 3.10+
- **PyTorch:** 2.8.0
- **CUDA:** 12.8
- **GPU:** NVIDIA RTX 5090 (24 GB or higher recommended for the full model at `288 × 288`)

### Key dependencies

```bash
pip install torch==2.8.0 torchvision
pip install mamba-ssm causal-conv1d   # required by the Vision Mamba branch
pip install numpy
```

`mamba-ssm` and `causal-conv1d` require a CUDA toolkit and matching `nvcc`. Please follow the official [mamba-ssm](https://github.com/state-spaces/mamba) installation guide for your environment.

---

## 4. Quick Usage

The released modules can be instantiated and called directly. The example below mirrors the configuration used for the KNMI Radar dataset (`288 × 288`, 6-in / 6-out frames).

```python
import torch
from Phys_MambaCast import Phys_MambaUNet
from Time_Decay_Dynamic_Quantile_Weighted_Loss import TD_DQWL

# Model: 6 input frames -> 6 predicted frames
model = Phys_MambaUNet(
    input_frames=6,
    predicted_frames=6,
    c_list=[8, 16, 24, 32, 48, 64],
).cuda()

# Forward pass
x = torch.randn(2, 6, 288, 288).cuda()   # (B, T_in, H, W)
y = model(x)                              # (B, T_out, H, W)

# Loss
loss_fn = TD_DQWL(omega_t=0.5, alpha=1.0, beta=1.0, c=1.0, temporal_beta=0.2)
# loss_fn expects tensors shaped [T, C, H, W]; reshape your batch accordingly.
```

The PMFM is **modular** and can be embedded into other encoder–decoder backbones (U-Net, Swin-UNet, Mamba-UNet, etc.) without modification:

```python
from Phys_MambaCast import PMF_Block

block = PMF_Block(input_dim=32, output_dim=48)
y = block(torch.randn(2, 32, 36, 36).cuda())   # (B, 48, 36, 36)
```

---

## 5. Datasets

The paper evaluates Phys-MambaCast on two public precipitation datasets with distinct observation modalities and climatic regimes:

| Dataset | Source | Region | Resolution | Period | Input Size |
|---------|--------|--------|------------|--------|------------|
| **KNMI-20** | KNMI C-band radar composite | Netherlands | 1 km, 5 min | 2016–2019 (test: 2019) | `288 × 288` |
| **ERA5-Land-20** | ECMWF ERA5-Land reanalysis | Eastern China | 0.1°, hourly | 2013–2024 wet seasons (test: 2023–2024) | `256 × 256` |

Both datasets are filtered to retain samples whose precipitating pixels cover more than 20% of the spatial domain. Each sample contains 12 consecutive frames (6 input + 6 target), with autoregressive rolling used to extend the horizon up to 12 future steps during evaluation.

Data-loading, normalization, oversampling, and rolling-evaluation scripts will be released with the full code.

---

## 6. Training Hyperparameters (from the paper)

| Parameter | KNMI | ERA5-Land |
|-----------|------|-----------|
| `img_size` | 288 | 256 |
| `batch_size` | 6 | 36 |
| `max_epochs` | 100 | 100 |
| `learning_rate` | 1e-3 | 1e-3 |
| `lr_patience` | 4 | 4 |
| `num_input_frames` | 6 | 6 |
| `num_output_frames` | 6 | 6 |
| `es_patience` | 10 | 10 |
| `use_oversampled_dataset` | True | True |
| `valid_size` | 0.2 | 0.2 |
| `bilinear` | True | True |

Optimizer: **AdamW** with cosine annealing learning-rate schedule.

---

## 7. Baselines Compared in the Paper

Phys-MambaCast is compared against seven representative baselines covering CNN-, RNN-, Transformer-, Diffusion-, and Mamba-based families:

- SmaAt-UNet · ConvLSTM · LPT-QPN · TransUNet · AA-TransUNet · DiffCast (with PhyDNet backbone) · Mamba-UNet

Baseline implementations and our re-training scripts will be released with the full code.

---

## 8. Citation

A formal citation will be added once the paper is accepted. Please contact the corresponding authors for the latest pre-print reference.

```bibtex
@article{shen2025physmambacast,
  title   = {Phys-MambaCast: A Physics-Enhanced Mamba Framework for Efficient and Accurate Precipitation Nowcasting},
  author  = {Shen, Yufang and Zhang, Wei and Pan, Shurui and Shi, Pengfei and
             Zhang, Shuo and Liu, Ranyu and Wang, Zihao and Yang, Tao},
  journal = {Under Review},
  year    = {2025}
}
```

---

## 9. Contact

For questions about the method, code, or future code release, please contact:
- Yufang Shen - `250201010012@hhu.edu.cn`
- Pengfei Shi — `pfshi@hhu.edu.cn`
- Tao Yang — `tao.yang@hhu.edu.cn`

---

## 10. License

The complete code will be released under an open-source license (to be confirmed) after paper acceptance. Until then, the modules in this repository are provided **for peer-review verification only**; please do not redistribute without the authors' permission.

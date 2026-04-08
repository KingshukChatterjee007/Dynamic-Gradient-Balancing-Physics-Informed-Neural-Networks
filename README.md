# Dynamic Gradient Balancing Physics-Informed Neural Networks (DGB-PINN)

[![Paper](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of **"Resolving Gradient Pathologies in Physics-Informed Neural Networks via Dynamic Gradient Surgery and Adaptive Refinement for Robust Inverse Discovery Under Noise."** This framework resolves catastrophic gradient pathologies—both magnitude imbalance (Type-I) and directional conflict (Type-II)—in multi-objective PINN optimization. It is specifically engineered to recover hidden physical parameters (e.g., Reynolds number) from highly noisy sensor data ($10\%$ Gaussian noise) in stiff PDE systems where standard PINNs diverge.

---

## 🚀 Key Innovations

1. **Sinusoidal Representation Networks (SIREN):** Replaces standard ReLU/Tanh activations with periodic sine functions, providing analytically exact, non-vanishing higher-order derivatives essential for stiff PDEs.
2. **Forgetful EMA Dual-Balancer (DB-PINN):** Adaptively scales loss weights using an online Welford Exponential Moving Average (EMA) to prevent magnitude-based pathology.
3. **Gradient Surgery (PCGrad) + GTN:** Resolves the "Tug-of-War" directional conflict. If the gradients of two loss terms interfere ($\nabla_{\theta}\mathcal{L}_{i} \cdot \nabla_{\theta}\mathcal{L}_{j} < 0$), the interfering gradient is projected onto the normal plane of its competitor. Gradient Task Normalization (GTN) scales these vectors prior to surgery.
4. **Residual-based Adaptive Refinement (RAR):** Dynamically hunts down physics violations (e.g., fluid turbulence, phase interfaces) by probabilistically redistributing collocation points to regions with the highest PDE residual.
5. **Softplus Physics Anchor:** A bounded, differentiable constraint ($Re = \beta \cdot \ln(1 + e^k)$) that stabilizes inverse parameter discovery and prevents mathematically impossible physical states (e.g., negative viscosity) during noisy gradient updates.

---

## 🧮 Mathematical Architecture

### The Gradient Pathology Problem
Standard PINNs optimize a static scalarized multi-task objective:
$$\mathcal{L}_{total} = \lambda_{pde}\mathcal{L}_{pde} + \sum_{k=1}^{K} \lambda_{bc,k}\mathcal{L}_{bc,k} + \lambda_{data}\mathcal{L}_{data}$$
This formulation often collapses due to conflicting gradient directions. 

### The PCGrad Projection Solution
To prevent the data loss from overwriting the physics pathway, we apply projective surgery:
$$\nabla_{\theta}\mathcal{L}_{i}^{*} = \nabla_{\theta}\mathcal{L}_{i} - \frac{\nabla_{\theta}\mathcal{L}_{i} \cdot \nabla_{\theta}\mathcal{L}_{j}}{||\nabla_{\theta}\mathcal{L}_{j}||_{2}^{2} + \epsilon} \nabla_{\theta}\mathcal{L}_{j}$$

### Energy-Adaptive Sampling (RAR)
For stiff systems like Allen-Cahn, points are sampled dynamically proportional to the Ginzburg-Landau energy density:
$$e(u) = \frac{\epsilon}{2}|\nabla u|^{2} + \frac{1}{4\epsilon}(u^{2} - 1)^{2}$$

---

## 📊 Benchmarks & Results

The architecture is rigorously validated across two canonical environments:
1. **The Allen-Cahn Equation:** A heavily stiff phase-field transition ($\epsilon = 10^{-4}$).
2. **Navier-Stokes (Cylinder Flow):** Multi-variable forward and inverse fluid dynamics ($Re=100$).

### 🏆 Ablation Study: Inverse Discovery Under $10\%$ Noise
*Target: Discover $Re=100$ from heavily corrupted wake sensor data.*

| Configuration | DB Balancer | PCGrad | Softplus Anchor | Final $Re$ | $\%$ Error |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Baseline (Standard PINN)** | ❌ | ❌ | ❌ | Diverged (NaN) | $\infty$ |
| **Test A (EMA Only)** | ✅ | ❌ | ❌ | Flatlined ($\approx 50.0$) | $50.0\%$ |
| **SOTA (Full Pipeline)** | ✅ | ✅ | ✅ | **$98.5 \pm 1.2$** | **$< 1.5\%$** |

---

## ⚙️ Installation & Setup

Ensure you have a CUDA-capable GPU. The models rely on `torch.float64` to prevent numerical rounding from corrupting the gradient surgery inner products.

```bash
# Clone the repository
git clone [https://github.com/yourusername/Dynamic-Gradient-Balancing-PINNs.git](https://github.com/yourusername/Dynamic-Gradient-Balancing-PINNs.git)
cd Dynamic-Gradient-Balancing-PINNs

# Install dependencies
pip install -r requirements.txt

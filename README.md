# Dynamic Gradient Balancing & Directional Alignment for PINNs

[![SOTA 2026](https://img.shields.io/badge/Status-SOTA%202026-blue.svg)](https://github.com/KingshukChatterjee007/Dynamic-Gradient-Balancing-Physics-Informed-Neural-Networks)
[![Physics-Informed ML](https://img.shields.io/badge/Field-Physics--Informed%20ML-green.svg)](https://github.com/KingshukChatterjee007/Dynamic-Gradient-Balancing-Physics-Informed-Neural-Networks)

## Overview

This repository contains a highly optimized Physics-Informed Neural Network (PINN) framework engineered to solve stiff, non-linear Partial Differential Equations (PDEs). Standard PINNs frequently fail to converge on complex physical systems due to gradient pathology—where multi-objective loss functions (e.g., physical residuals vs. boundary conditions) create severely conflicting optimization trajectories.

This framework mitigates these optimization failures by implementing a tri-modular architecture: Dynamic Gradient Balancing (DB-PINN), Directional Gradient Alignment (PCGrad), and Residual-based Adaptive Refinement (RAR). The architecture is benchmarked against rigorous computational fluid dynamics (CFD) and phase-field problems, including 2D Navier-Stokes (Cylinder Flow at Re=100) and the Allen-Cahn equation.

## Key Architecture & Features

### 1. Dual-Balancing (DB-PINN)
Employs an online statistical tracking mechanism (`EMAWelford` algorithm) using Exponential Moving Averages to dynamically weight loss components, preventing boundary constraints from dominating the underlying physical laws.

### 2. Directional Alignment Module (DAM)
Integrates Gradient Surgery (PCGrad) and Gradient Task Normalization (GTN). This prevents "Tug-of-War" gradient conflicts by projecting interfering gradients onto each other's normal planes, ensuring smooth multi-task optimization.

### 3. SIREN-based Architecture
Utilizes Sinusoidal Representation Networks (`SineLayer`) to naturally capture high-frequency components and ensure stable, non-vanishing higher-order derivatives required for stiff PDEs. Maintained at `float64` precision natively to guard against precision-loss masking gradient pathology.

### 4. Residual-based Adaptive Refinement (RAR / EAS)
Dynamically focuses the network's capacity on physically demanding regions. Computes spatial PDE violation densities on a meshless continuous sample pool and utilizes probabilistic multinomial redistribution to concentrate points exponentially around complex turbulent structures.

### 5. Inverse Fluid Discovery Pipeline
Elevates the forward PINN solver into a robust parameter discovery machine capable of dynamically extracting unknown physical coefficients from synthetic, noisy experimental data. Incorporates a **Softplus Physics Anchor** to prevent convergence into physically impossible bounds.

## 📊 Benchmarks

### Allen-Cahn Equation
A "stiff" PDE benchmark known for sharp interfaces and gradient pathology. The framework successfully converges without manual weight tuning.

### Navier-Stokes (Flow Around a Cylinder)
Simulates steady flow at **$Re=100$**.
- Resolves the destructive conflict between **Pressure ($p$)** and **Velocity ($u, v$)** gradients.
- Ensures local mass conservation (Continuity) is satisfied early in training.
- **Inverse Discovery Extension**: Successfully isolates and predicts hidden fluid Reynolds metrics utilizing only wake measurements containing 10% realistic Gaussian noise levels.

## 📁 Repository Structure

```text
.
├── experiments/
│   ├── run_ac.py                  # Training script for 1D Allen-Cahn
│   ├── run_cylinder.py            # Training script for 2D Navier-Stokes
│   ├── run_inverse_cylinder.py    # Parameter discovery pipeline for 2D flow
│   └── run_ablation_study.py      # Automated benchmarking suite
├── pinn_engine/
│   ├── balancer.py                # DB-PINN and EMAWelford implementations
│   ├── model.py                   # Core neural network and SIREN architecture
│   ├── sampling.py                # Residual-Based Adaptive Refinement (RAR)
│   └── surgery.py                 # PCGrad and GTN implementations
├── problems/
│   ├── allen_cahn.py              # Equation, IC, and BC definitions for AC
│   ├── inverse_allen_cahn.py      # Setup for inverse coefficient discovery
│   └── navier_stokes.py           # 2D steady flow PDE and cylinder boundaries
└── results/                       # Generated models (.pth) and visualizations
```

## 🛠️ Usage

### Run Allen-Cahn Training
```bash
python experiments/run_ac.py --max_epochs 1000
```

### Run Navier-Stokes (Cylinder Flow) Training
```bash
python experiments/run_cylinder.py --max_epochs 1000
```

### Run Full Ablation Benchmarks (Forward & Inverse)
```bash
python experiments/run_ablation_study.py --fwd_epochs 2000 --inv_epochs 2000 --noise 0.1
```

---
*Developed for research into "Type II" PINN failures, advanced gradient balancing, and inverse problem discovery (2026).*

**Author:** Kingshuk Chatterjee

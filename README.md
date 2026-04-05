# Dynamic Gradient Balancing & Directional Alignment for PINNs

[![SOTA 2026](https://img.shields.io/badge/Status-SOTA%202026-blue.svg)](https://github.com/KingshukChatterjee007/Dynamic-Gradient-Balancing-Physics-Informed-Neural-Networks)
[![Physics-Informed ML](https://img.shields.io/badge/Field-Physics--Informed%20ML-green.svg)](https://github.com/KingshukChatterjee007/Dynamic-Gradient-Balancing-Physics-Informed-Neural-Networks)

A state-of-the-art framework for solving **Gradient Pathology** in Physics-Informed Neural Networks (PINNs). This repository implements solutions for both **Magnitude Imbalance (Type I)** and the "Unsolved" **Directional Gradient Conflict (Type II - The Tug-of-War)**.

## 🚀 Key Features

### 1. Dual-Balancing (DB-PINN)
Addresses the magnitude imbalance between PDE residuals and boundary conditions.
- **Inter-balancing**: Scalability between governed equations and external constraints.
- **Intra-balancing**: Granular weighting for complex boundary regions (e.g., turbulent wake vs. laminar inflow).
- **Stochastic Spike Filtering**: Uses Welford's online algorithm to stabilize weight updates against stochastic gradient noise.

### 2. Directional Alignment Module (DAM)
Resolves the "Tug-of-War" where satisfying one loss term actively moves the model away from another.
- **Gradient Surgery (PCGrad)**: Projects conflicting gradients onto each other's normal planes to find a synergistic descent path.
- **Conflict Resolution**: Specifically tuned for the hierarchical structure of PINN losses.

### 3. Gradient Task Normalization (GTN)
Optimized for multi-variable fluid dynamics (e.g., Navier-Stokes).
- Normalizes gradients across vastly different numerical scales (e.g., $Pressure$ vs. $Velocity$).
- Prevents high-magnitude pressure residuals from "drowning out" velocity directional information.

### 4. SIREN-based Architecture
- Utilizes **Sinusoidal Representation Networks** (SIREN) for superior capturing of high-frequency components and smooth derivatives in stiff PDEs.

## 📊 Benchmarks

### Allen-Cahn Equation
A "stiff" PDE benchmark known for sharp interfaces and gradient pathology. The framework successfully converges without manual weight tuning.

### Navier-Stokes (Flow Around a Cylinder)
Simulates steady flow at **$Re=100$**.
- Resolves the destructive conflict between **Pressure ($p$)** and **Velocity ($u, v$)** gradients.
- Ensures local mass conservation (Continuity) is satisfied early in training.

## 🛠️ Usage

### Installation
```bash
pip install torch numpy matplotlib
```

### Run Allen-Cahn Training
```bash
python train.py
```

### Run Navier-Stokes (Cylinder Flow) Training
```bash
python train_cylinder.py
```

## 📜 Repository Structure
- `pinn_model.py`: SIREN-based Multi-Output MLP.
- `db_pinn_balancer.py`: Dual-Balancing magnitude engine.
- `directional_alignment.py`: Gradient Surgery and GTN module.
- `navier_stokes_2d.py`: 2D Incompressible Fluid Residuals.
- `train_cylinder.py`: Benchmark script for aerodynamics.

---
*Developed for research into "Type II" PINN failures and advanced gradient balancing (2026).*

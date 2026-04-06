# Dynamic Gradient Balancing & Directional Alignment for PINNs

[![SOTA 2026](https://img.shields.io/badge/Status-SOTA%202026-blue.svg)](https://github.com/KingshukChatterjee007/Dynamic-Gradient-Balancing-Physics-Informed-Neural-Networks)
[![Physics-Informed ML](https://img.shields.io/badge/Field-Physics--Informed%20ML-green.svg)](https://github.com/KingshukChatterjee007/Dynamic-Gradient-Balancing-Physics-Informed-Neural-Networks)

# Dynamic Gradient Balancing in Physics-Informed Neural Networks (PINNs)

## Overview

This repository contains a highly optimized Physics-Informed Neural Network (PINN) framework engineered to solve stiff, non-linear Partial Differential Equations (PDEs). Standard PINNs frequently fail to converge on complex physical systems due to gradient pathology—where multi-objective loss functions (e.g., physical residuals vs. boundary conditions) create severely conflicting optimization trajectories.

This framework mitigates these optimization failures by implementing a tri-modular architecture: Dynamic Gradient Balancing (DB-PINN), Directional Gradient Alignment (PCGrad), and Energy-Adaptive Sampling (EAS). The architecture is benchmarked against rigorous computational fluid dynamics (CFD) and phase-field problems, including 2D Navier-Stokes (Cylinder Flow at Re=100) and the Allen-Cahn equation.

## Key Architecture & Features

* **Generic PINN with SIREN Integration:** Utilizes Sinusoidal Representation Networks (`SineLayer`) to naturally capture high-frequency components and ensure stable, non-vanishing higher-order derivatives required for stiff PDEs.
* **Dual-Balancing (DB-PINN):** Employs an online statistical tracking mechanism (`EMAWelford` algorithm) using Exponential Moving Averages to dynamically weight loss components, preventing boundary constraints from dominating the underlying physical laws.
* **Directional Alignment Module (DAM):** Integrates Gradient Surgery (PCGrad) and Gradient Task Normalization (GTN). This prevents "Tug-of-War" gradient conflicts by projecting interfering gradients onto each other's normal planes, ensuring smooth multi-task optimization.
* **Energy-Adaptive Sampling (EAS):** Instead of uniform collocation points, this module dynamically allocates training points proportionally to the pointwise energy density (residual error) of the PDE, focusing computational effort on turbulent or stiff regions.
* **Inverse Discovery Machine:** Includes an `InverseLearner` module capable of dynamically extracting unknown physical coefficients from synthetic, noisy experimental data.

## Repository Structure

```text
.
├── experiments/
│   ├── run_ac.py                  # Training script for 1D Allen-Cahn
│   ├── run_cylinder.py            # Training script for 2D Navier-Stokes
│   └── run_inverse_discovery.py   # Parameter discovery via noisy data
├── pinn_engine/
│   ├── balancer.py                # DB-PINN and EMAWelford implementations
│   ├── model.py                   # Core neural network and SIREN architecture
│   ├── sampling.py                # Energy-Adaptive Sampling (EAS) logic
│   └── surgery.py                 # PCGrad and GTN implementations
├── problems/
│   ├── allen_cahn.py              # Equation, IC, and BC definitions for AC
│   ├── inverse_allen_cahn.py      # Setup for inverse coefficient discovery
│   └── navier_stokes.py           # 2D steady flow PDE and cylinder boundaries
└── results/                       # Generated models (.pth) and visualizations
```

#Author
Kingshuk Chatterjee

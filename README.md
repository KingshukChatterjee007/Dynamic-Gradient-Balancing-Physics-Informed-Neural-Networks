# Resolving Gradient Pathologies in Physics-Informed Neural Networks


[![Paper](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This repository contains the official implementation of the paper: **Resolving Gradient Pathologies in Physics-Informed Neural Networks via Dynamic Gradient Surgery and Adaptive Refinement for Robust Inverse Discovery Under Noise**.

## Overview

Physics-Informed Neural Networks (PINNs) offer a mesh-free alternative to classical numerical solvers by embedding governing partial differential equations (PDEs) directly into the training objective. However, they routinely suffer from two forms of gradient pathology:
* **Magnitude Imbalance (Type-I):** One loss term dominates the optimization.
* **Directional Conflict (Type-II):** Satisfying one physical constraint actively degrades another (the "Tug-of-War" problem).

These pathologies become catastrophic in inverse parameter discovery, where sparse, noise-corrupted sensor data must coexist with strict PDE enforcement. This repository provides a unified framework to overcome these challenges.

## Key Components

The canonical PINN objective takes the form:
$$\mathcal{L}_{\text{total}}=\lambda_{\text{pde}}\mathcal{L}_{\text{pde}}+\sum_{k=1}^{K}\lambda_{k}\mathcal{L}_{\text{bc},k}+\lambda_{\text{data}}\mathcal{L}_{\text{data}}$$

Our framework couples four synergistic modules to stabilize training and enable robust inverse discovery:

### 1. Dynamic Gradient Balancing (DB-PINN)
Addresses Type-I magnitude imbalance by tracking the running gradient scale of each loss term using a Forgetful Exponential Moving Average (EMA). The EMA estimate is:
$$\hat{g}_i^{(t)}=\alpha\cdot\hat{g}_i^{(t-1)}+(1-\alpha)\cdot g_i^{(t)}$$
where $g_i^{(t)}=\|\nabla_\theta\mathcal{L}_i^{(t)}\|$ and $\alpha=0.999$. Loss weights are adaptively rebalanced based on these statistics.

### 2. Gradient Surgery (PCGrad) with Gradient Task Normalization (GTN)
Resolves Type-II directional conflicts. If gradients conflict ($\nabla_\theta\mathcal{L}_i\cdot\nabla_\theta\mathcal{L}_j<0$), the interfering gradient is projected onto the normal plane of its competitor:
$$\nabla_\theta\mathcal{L}_i^{*}=\nabla_\theta\mathcal{L}_i-\frac{\nabla_\theta\mathcal{L}_i\cdot\nabla_\theta\mathcal{L}_j}{\|\nabla_\theta\mathcal{L}_j\|_2^2+\epsilon}\nabla_\theta\mathcal{L}_j$$
Before surgery, GTN normalizes each gradient vector to $O(1)$ scale to prevent domination by high-magnitude terms like pressure gradients in fluid dynamics.

### 3. Residual-based Adaptive Refinement (RAR)
Dynamically redistributes collocation points based on the current residual landscape, sampling heavily in regions with high PDE residual magnitude $|\mathcal{R}(\bm{x}_j)|$.

### 4. Softplus Physics Anchor
Ensures physical validity during inverse problems. Rather than a raw parameter, unknown variables (like Reynolds number) are parameterized through a Softplus activation to guarantee positive values:
$$Re=\beta\cdot\ln(1+e^k)$$

## Architecture

The framework utilizes Sinusoidal Representation Networks (SIREN), where each hidden layer applies:
$$\bm{h}_{l+1}=\sin\!\left(\omega_0\cdot(\bm{W}_l\bm{h}_l+\bm{b}_l)\right)$$
This provides analytically exact, non-vanishing derivatives critical for computing PDE residuals involving second-order spatial derivatives. All computations are performed in `float64` precision.

## Benchmarks

The framework is validated on two canonical benchmarks:
* **Allen-Cahn Equation:** A stiff phase-field equation ($\epsilon=10^{-4}$) modeling phase separation dynamics.
* **2D Navier-Stokes (Cylinder Flow):** Steady incompressible flow around a cylinder at $Re=100$.

## Results Highlight

In inverse parameter estimation (discovering $Re$ from sensors with 10% Gaussian noise), baseline PINNs diverge or flatline. Our complete framework converges stably, recovering the target Reynolds number with approximately 1.5% relative error ($Re_{\text{pred}}\approx 98.5\pm 1.2$).

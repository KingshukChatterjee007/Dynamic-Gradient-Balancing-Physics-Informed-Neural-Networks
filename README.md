# Resolving Gradient Pathologies in Physics-Informed Neural Networks


[![Paper](https://img.shields.io/badge/arXiv-Pending-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


[cite_start]This repository contains the official implementation of the paper: **Resolving Gradient Pathologies in Physics-Informed Neural Networks via Dynamic Gradient Surgery and Adaptive Refinement for Robust Inverse Discovery Under Noise**[cite: 1].

## Overview

[cite_start]Physics-Informed Neural Networks (PINNs) offer a mesh-free alternative to classical numerical solvers by embedding governing partial differential equations (PDEs) directly into the training objective[cite: 2]. [cite_start]However, they routinely suffer from two forms of gradient pathology[cite: 3]:
* [cite_start]**Magnitude Imbalance (Type-I):** One loss term dominates the optimization[cite: 3].
* [cite_start]**Directional Conflict (Type-II):** Satisfying one physical constraint actively degrades another (the "Tug-of-War" problem)[cite: 3].

[cite_start]These pathologies become catastrophic in inverse parameter discovery, where sparse, noise-corrupted sensor data must coexist with strict PDE enforcement[cite: 4]. [cite_start]This repository provides a unified framework to overcome these challenges[cite: 5].

## Key Components

[cite_start]The canonical PINN objective takes the form[cite: 9]:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{pde}} \mathcal{L}_{\text{pde}} + \sum_{k=1}^{K} \lambda_{k} \mathcal{L}_{\text{bc},k} + \lambda_{\text{data}} \mathcal{L}_{\text{data}}$$

[cite_start]Our framework couples four synergistic modules to stabilize training and enable robust inverse discovery[cite: 5]:

### 1. Dynamic Gradient Balancing (DB-PINN)
[cite_start]Addresses Type-I magnitude imbalance by tracking the running gradient scale of each loss term using a Forgetful Exponential Moving Average (EMA)[cite: 34]. [cite_start]The EMA estimate is[cite: 35]:
$$\hat{g}_i^{(t)} = \alpha \cdot \hat{g}_i^{(t-1)} + (1 - \alpha) \cdot g_i^{(t)}$$
[cite_start]where $g_i^{(t)} = \|\nabla_\theta \mathcal{L}_i^{(t)}\|$ and $\alpha = 0.999$[cite: 34, 35]. [cite_start]Loss weights are adaptively rebalanced based on these statistics[cite: 35, 36].

### 2. Gradient Surgery (PCGrad) with Gradient Task Normalization (GTN)
[cite_start]Resolves Type-II directional conflicts[cite: 38]. [cite_start]If gradients conflict ($\nabla_\theta \mathcal{L}_i \cdot \nabla_\theta \mathcal{L}_j < 0$), the interfering gradient is projected onto the normal plane of its competitor[cite: 39]:
$$\nabla_\theta \mathcal{L}_i^{*} = \nabla_\theta \mathcal{L}_i - \frac{\nabla_\theta \mathcal{L}_i \cdot \nabla_\theta \mathcal{L}_j}{\|\nabla_\theta \mathcal{L}_j\|_2^2 + \epsilon} \nabla_\theta \mathcal{L}_j$$
[cite_start]Before surgery, GTN normalizes each gradient vector to $O(1)$ scale to prevent domination by high-magnitude terms like pressure gradients in fluid dynamics[cite: 40, 41, 42].

### 3. Residual-based Adaptive Refinement (RAR)
[cite_start]Dynamically redistributes collocation points based on the current residual landscape, sampling heavily in regions with high PDE residual magnitude $|\mathcal{R}(\bm{x}_j)|$[cite: 47, 48, 50].

### 4. Softplus Physics Anchor
[cite_start]Ensures physical validity during inverse problems[cite: 19]. [cite_start]Rather than a raw parameter, unknown variables (like Reynolds number) are parameterized through a Softplus activation to guarantee positive values[cite: 55, 56]:
$$Re = \beta \cdot \ln(1 + e^k)$$

## Architecture

[cite_start]The framework utilizes Sinusoidal Representation Networks (SIREN)[cite: 29], where each hidden layer applies:
$$\bm{h}_{l+1} = \sin\!\left(\omega_0 \cdot (\bm{W}_l \bm{h}_l + \bm{b}_l)\right)$$
[cite_start]This provides analytically exact, non-vanishing derivatives critical for computing PDE residuals involving second-order spatial derivatives[cite: 30]. [cite_start]All computations are performed in `float64` precision[cite: 32].

## Benchmarks

[cite_start]The framework is validated on two canonical benchmarks[cite: 6]:
* [cite_start]**Allen-Cahn Equation:** A stiff phase-field equation ($\epsilon = 10^{-4}$) modeling phase separation dynamics[cite: 58, 59].
* [cite_start]**2D Navier-Stokes (Cylinder Flow):** Steady incompressible flow around a cylinder at $Re=100$[cite: 6].

## Results Highlight

[cite_start]In inverse parameter estimation (discovering $Re$ from sensors with 10% Gaussian noise), baseline PINNs diverge or flatline[cite: 89]. [cite_start]Our complete framework converges stably, recovering the target Reynolds number with approximately 1.5% relative error ($Re_{\text{pred}} \approx 98.5 \pm 1.2$)[cite: 87, 88].

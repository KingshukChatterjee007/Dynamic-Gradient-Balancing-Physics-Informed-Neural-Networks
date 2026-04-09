"""
===========================================================================
Master Ablation Study: Dynamic Gradient Balancing PINN Framework
===========================================================================
Tests 4 configurations across Forward (Navier-Stokes) and Inverse
(Allen-Cahn epsilon discovery) benchmarks:

    Baseline:  Uniform Sampling, No EMA, No PCGrad
    Test A:    EMA Welford Balancer only
    Test B:    EMA + PCGrad Surgery
    SOTA:      EMA + PCGrad + RAR Sampler

Usage:
    python experiments/run_ablation_study.py --smoke-test   # 50 epochs
    python experiments/run_ablation_study.py                # 20000 epochs
===========================================================================
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless rendering before pyplot import
import matplotlib.pyplot as plt
import sys
import os
import argparse
import pandas as pd
import time
import gc

# --------------- Project Path Setup --------------- #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinn_engine.model import PINN
from pinn_engine.balancer import DBBalancer
from pinn_engine.surgery import PINNGradientSurgery
from pinn_engine.sampling import ResidualSampler, EnergyAdaptiveSampler
from problems.navier_stokes import (
    navier_stokes_residuals, cylinder_bc_loss,
    sample_domain_ns, cylinder_mask
)
from problems.inverse_allen_cahn import (
    inverse_ac_residual, generate_noisy_data
)

# --------------- Global Precision --------------- #
torch.set_default_dtype(torch.float64)
os.makedirs("results", exist_ok=True)

# ================================================================== #
#                   FORWARD BENCHMARK (Navier-Stokes)                 #
# ================================================================== #
def run_forward(config_name, epochs, use_ema, use_pcgrad, use_rar, device='cpu',
                n_pde=1500, n_bc=400, rar_candidates=15000, lr=5e-4):
    """
    Trains a forward PINN for cylinder flow at Re=100.
    Returns: final_loss, wall_time, loss_history
    """
    print(f"\n{'='*60}")
    print(f"  FORWARD [{config_name}]  |  Epochs: {epochs}")
    print(f"  EMA={use_ema}  PCGrad={use_pcgrad}  RAR={use_rar}  Device={device}")
    print(f"{'='*60}")

    model = PINN(in_features=2, hidden_features=128, hidden_layers=5, out_features=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -- Components --
    num_losses = 9
    balancer = DBBalancer(num_conditions=num_losses - 1, update_freq=50) if use_ema else None
    surgery  = PINNGradientSurgery(optimizer, use_gtn=True) if use_pcgrad else None

    if use_rar:
        bounds  = [(0.0, 1.1), (0.0, 0.41)]
        sampler = ResidualSampler(model, navier_stokes_residuals, bounds,
                                  mask_fn=cylinder_mask)

    # -- Initial uniform collocation --
    x_pde, y_pde = sample_domain_ns(n_pde=n_pde, device=device)

    loss_history = []
    t_start = time.time()

    for epoch in range(epochs):
        if use_rar and epoch > 0 and epoch % 50 == 0:
            sampled = sampler.sample(n_points=n_pde, n_candidate=rar_candidates)
            x_pde = sampled[:, 0:1]
            y_pde = sampled[:, 1:2]

        l_pde = navier_stokes_residuals(model, x_pde, y_pde, re=100)
        l_bc  = cylinder_bc_loss(model, n_bc=n_bc)
        all_losses = l_pde + l_bc
        sum_loss   = sum(l for l in all_losses)
        loss_history.append(sum_loss.item())

        if use_pcgrad and surgery is not None:
            weights   = balancer.get_weights() if use_ema else None
            grad_mags = surgery.step(all_losses, weights=weights)
        else:
            optimizer.zero_grad()
            if use_ema and balancer is not None:
                weights = balancer.get_weights()
                weighted = sum(w * l for w, l in zip(weights, all_losses))
                weighted.backward()
                grad_mags = [l.item() for l in all_losses]
            else:
                sum_loss.backward()
                grad_mags = None
            optimizer.step()

        if use_ema and balancer is not None and grad_mags is not None:
            balancer.update_gradient_stats(grad_mags)
            balancer.balance_weights()

        if epoch % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch:6d}/{epochs} | Loss: {sum_loss.item():.6f}")

    wall_time  = time.time() - t_start
    final_loss = loss_history[-1]
    return final_loss, wall_time, loss_history


# ================================================================== #
#              INVERSE BENCHMARK (Allen-Cahn ε Discovery)             #
# ================================================================== #
class InverseAllenCahnLearner(torch.nn.Module):
    """Wraps a PINN + learnable epsilon with optional Softplus anchor."""
    def __init__(self, use_softplus=True, device='cpu'):
        super().__init__()
        self.use_softplus = use_softplus
        self.k = torch.nn.Parameter(torch.tensor([-4.6], dtype=torch.float64, device=device))
        self.model = PINN(in_features=2, hidden_features=128,
                          hidden_layers=4, out_features=1).to(device)

    @property
    def epsilon(self):
        if self.use_softplus:
            return F.softplus(self.k)
        return self.k.abs() + 1e-10

    def forward(self, x):
        return self.model(x)


def run_inverse(config_name, epochs, use_ema, use_pcgrad, use_softplus, device='cpu',
                target_epsilon=1e-4, noise_lv=0.1, lr=1e-3):
    """
    Trains an inverse PINN to discover Allen-Cahn epsilon from noisy data.
    Returns: final_epsilon, epsilon_history, loss_history
    """
    print(f"\n{'='*60}")
    print(f"  INVERSE [{config_name}]  |  Epochs: {epochs}")
    print(f"  EMA={use_ema}  PCGrad={use_pcgrad}  Softplus={use_softplus}  Device={device}")
    print(f"{'='*60}")

    gt_model = PINN(in_features=2, hidden_features=128,
                    hidden_layers=4, out_features=1).to(device)
    snapshots = [0.1, 0.25, 0.5, 0.75, 1.0]
    noisy_data = generate_noisy_data(gt_model, target_epsilon,
                                     snapshots=snapshots, noise_lv=noise_lv, device=device)

    learner   = InverseAllenCahnLearner(use_softplus=use_softplus, device=device)
    optimizer = optim.Adam(learner.parameters(), lr=lr)

    num_losses = 1 + len(snapshots)
    balancer = DBBalancer(num_conditions=num_losses - 1,
                          alpha=0.999, update_freq=50) if use_ema else None
    surgery  = PINNGradientSurgery(optimizer, use_gtn=True) if use_pcgrad else None

    eps_history  = []
    loss_history = []

    for epoch in range(epochs):
        curr_eps = learner.epsilon

        x_pde = (torch.rand(1000, 1, dtype=torch.float64, device=device) * 2 - 1).requires_grad_(True)
        t_pde = torch.rand(1000, 1, dtype=torch.float64, device=device).requires_grad_(True)
        l_pde = inverse_ac_residual(learner.model, x_pde, t_pde, curr_eps)

        l_data_list = []
        for x_s, t_s, u_noisy in noisy_data:
            u_pred = learner.model(torch.cat([x_s, t_s], dim=1))
            l_data_list.append(torch.mean((u_pred - u_noisy)**2))

        all_losses = [l_pde] + l_data_list
        sum_loss   = sum(l for l in all_losses)
        loss_history.append(sum_loss.item())
        eps_history.append(curr_eps.item())

        if use_pcgrad and surgery is not None:
            weights   = balancer.get_weights() if use_ema else None
            grad_mags = surgery.step(all_losses, weights=weights)
        else:
            optimizer.zero_grad()
            if use_ema and balancer is not None:
                weights = balancer.get_weights()
                weighted = sum(w * l for w, l in zip(weights, all_losses))
                weighted.backward()
                grad_mags = [l.item() for l in all_losses]
            else:
                sum_loss.backward()
                grad_mags = None
            optimizer.step()

        if use_ema and balancer is not None and grad_mags is not None:
            balancer.update_gradient_stats(grad_mags)
            balancer.balance_weights()

        if epoch % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch:6d}/{epochs} | Loss: {sum_loss.item():.6f} "
                  f"| ε_pred: {curr_eps.item():.8f}")

    final_eps = eps_history[-1]
    return final_eps, eps_history, loss_history


# ================================================================== #
#                     MEMORY SAFETY UTILITIES                         #
# ================================================================== #
def flush_memory():
    """Aggressively release GPU/CPU memory between runs."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print("  [Memory flushed: gc.collect + cuda.empty_cache]")


# ================================================================== #
#                          MAIN EXECUTION                             #
# ================================================================== #
ABLATION_CONFIGS = [
    ("Baseline",       False,   False,      False),
    ("Test A (EMA)",   True,    False,      False),
    ("Test B (EMA+PC)",True,    True,       False),
    ("SOTA (Full)",    True,    True,       True ),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation Study for DGB-PINN Framework")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.smoke_test:
        EPOCHS         = 50
        RAR_CANDIDATES = 1000
    else:
        EPOCHS         = 20000
        RAR_CANDIDATES = 15000

    results       = []
    fwd_histories = {}
    inv_histories = {}

    print("\n" + "="*70)
    print("  PHASE 1: FORWARD PROBLEM — Navier-Stokes Cylinder Flow (Re=100)")
    print("="*70)

    for name, use_ema, use_pcgrad, use_rar in ABLATION_CONFIGS:
        final_loss, wall_time, loss_hist = run_forward(
            config_name    = name,
            epochs         = EPOCHS,
            use_ema        = use_ema,
            use_pcgrad     = use_pcgrad,
            use_rar        = use_rar,
            device         = device,
            rar_candidates = RAR_CANDIDATES,
        )
        fwd_histories[name] = loss_hist
        results.append({
            "Problem":    "Forward (NS)",
            "Config":     name,
            "Final Loss": f"{final_loss:.6f}",
            "Wall Time":  f"{wall_time:.1f}s",
            "ε_pred":     "N/A",
            "ε_error":    "N/A",
        })
        flush_memory()

    print("\n" + "="*70)
    print("  PHASE 2: INVERSE PROBLEM — Allen-Cahn ε Discovery (10% Noise)")
    print("="*70)

    TARGET_EPS = 1e-4

    for name, use_ema, use_pcgrad, use_softplus in ABLATION_CONFIGS:
        final_eps, eps_hist, loss_hist = run_inverse(
            config_name    = name,
            epochs         = EPOCHS,
            use_ema        = use_ema,
            use_pcgrad     = use_pcgrad,
            use_softplus   = use_softplus,
            device         = device,
            target_epsilon = TARGET_EPS,
            noise_lv       = args.noise,
        )
        inv_histories[name] = eps_hist
        eps_error = abs(final_eps - TARGET_EPS) / TARGET_EPS * 100
        results.append({
            "Problem":    "Inverse (AC)",
            "Config":     name,
            "Final Loss": f"{loss_hist[-1]:.6f}",
            "Wall Time":  "—",
            "ε_pred":     f"{final_eps:.8f}",
            "ε_error":    f"{eps_error:.2f}%",
        })
        flush_memory()

    df = pd.DataFrame(results)
    csv_path = "results/ablation_metrics.csv"
    df.to_csv(csv_path, index=False)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    for i, (name, _, _, _) in enumerate(ABLATION_CONFIGS):
        if name in fwd_histories:
            hist = np.array(fwd_histories[name])
            hist = np.clip(hist, 1e-12, None)
            ax.semilogy(hist, label=name, color=colors[i], alpha=0.85)
    ax.legend()
    
    ax = axes[1]
    for i, (name, _, _, _) in enumerate(ABLATION_CONFIGS):
        if name in inv_histories:
            ax.semilogy(inv_histories[name], label=name, color=colors[i], alpha=0.85)
    ax.axhline(y=TARGET_EPS, color='black', linestyle='--')
    ax.legend()

    plt.tight_layout()
    plt.savefig("results/ablation_loss_curve.png", dpi=200)
    plt.close()

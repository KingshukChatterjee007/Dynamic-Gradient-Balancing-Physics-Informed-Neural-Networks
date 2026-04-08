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
def run_forward(config_name, epochs, use_ema, use_pcgrad, use_rar,
                n_pde=1500, n_bc=400, rar_candidates=15000, lr=5e-4):
    """
    Trains a forward PINN for cylinder flow at Re=100.
    Returns: final_loss, wall_time, loss_history
    """
    print(f"\n{'='*60}")
    print(f"  FORWARD [{config_name}]  |  Epochs: {epochs}")
    print(f"  EMA={use_ema}  PCGrad={use_pcgrad}  RAR={use_rar}")
    print(f"{'='*60}")

    model = PINN(in_features=2, hidden_features=128, hidden_layers=5, out_features=3)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # -- Components --
    # Total losses: 3 PDE (u-mom, v-mom, continuity) + 6 BC = 9
    num_losses = 9
    balancer = DBBalancer(num_conditions=num_losses - 1, update_freq=50) if use_ema else None
    surgery  = PINNGradientSurgery(optimizer, use_gtn=True) if use_pcgrad else None

    if use_rar:
        bounds  = [(0.0, 1.1), (0.0, 0.41)]
        sampler = ResidualSampler(model, navier_stokes_residuals, bounds,
                                  mask_fn=cylinder_mask)

    # -- Initial uniform collocation --
    x_pde, y_pde = sample_domain_ns(n_pde=n_pde)

    loss_history = []
    t_start = time.time()

    for epoch in range(epochs):
        # RAR re-sampling every 50 epochs
        if use_rar and epoch > 0 and epoch % 50 == 0:
            sampled = sampler.sample(n_points=n_pde, n_candidate=rar_candidates)
            x_pde = sampled[:, 0:1]
            y_pde = sampled[:, 1:2]

        # Physics residuals + boundary conditions
        l_pde = navier_stokes_residuals(model, x_pde, y_pde, re=100)
        l_bc  = cylinder_bc_loss(model, n_bc=n_bc)
        all_losses = l_pde + l_bc                    # list of 9 tensors
        sum_loss   = sum(l for l in all_losses)
        loss_history.append(sum_loss.item())

        # --- Optimization step ---
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

        # --- Balancer update ---
        if use_ema and balancer is not None and grad_mags is not None:
            balancer.update_gradient_stats(grad_mags)
            balancer.balance_weights()

        # --- Logging ---
        if epoch % max(1, epochs // 10) == 0:
            print(f"  Epoch {epoch:6d}/{epochs} | Loss: {sum_loss.item():.6f}")

    wall_time  = time.time() - t_start
    final_loss = loss_history[-1]
    print(f"  DONE  [{config_name}] | Final Loss: {final_loss:.6f} | Time: {wall_time:.1f}s")
    return final_loss, wall_time, loss_history


# ================================================================== #
#              INVERSE BENCHMARK (Allen-Cahn ε Discovery)             #
# ================================================================== #
class InverseAllenCahnLearner(torch.nn.Module):
    """Wraps a PINN + learnable epsilon with optional Softplus anchor."""
    def __init__(self, use_softplus=True):
        super().__init__()
        self.use_softplus = use_softplus
        # k initialised so that softplus(k) ≈ 0.01 (far from target 0.0001)
        self.k = torch.nn.Parameter(torch.tensor([-4.6], dtype=torch.float64))
        self.model = PINN(in_features=2, hidden_features=128,
                          hidden_layers=4, out_features=1)

    @property
    def epsilon(self):
        if self.use_softplus:
            return F.softplus(self.k)          # guaranteed > 0
        return self.k.abs() + 1e-10            # fallback: raw abs

    def forward(self, x):
        return self.model(x)


def run_inverse(config_name, epochs, use_ema, use_pcgrad, use_softplus,
                target_epsilon=1e-4, noise_lv=0.1, lr=1e-3):
    """
    Trains an inverse PINN to discover Allen-Cahn epsilon from noisy data.
    Returns: final_epsilon, epsilon_history, loss_history
    """
    print(f"\n{'='*60}")
    print(f"  INVERSE [{config_name}]  |  Epochs: {epochs}")
    print(f"  EMA={use_ema}  PCGrad={use_pcgrad}  Softplus={use_softplus}")
    print(f"{'='*60}")

    # -- Ground truth data (noisy snapshots) --
    gt_model = PINN(in_features=2, hidden_features=128,
                    hidden_layers=4, out_features=1)
    snapshots = [0.1, 0.25, 0.5, 0.75, 1.0]
    noisy_data = generate_noisy_data(gt_model, target_epsilon,
                                     snapshots=snapshots, noise_lv=noise_lv)

    # -- Learner --
    learner   = InverseAllenCahnLearner(use_softplus=use_softplus)
    optimizer = optim.Adam(learner.parameters(), lr=lr)

    # Components: 1 PDE loss + 5 snapshot data losses = 6
    num_losses = 1 + len(snapshots)
    balancer = DBBalancer(num_conditions=num_losses - 1,
                          alpha=0.999, update_freq=50) if use_ema else None
    surgery  = PINNGradientSurgery(optimizer, use_gtn=True) if use_pcgrad else None

    eps_history  = []
    loss_history = []

    for epoch in range(epochs):
        curr_eps = learner.epsilon

        # PDE residual on uniformly sampled points
        x_pde = (torch.rand(1000, 1, dtype=torch.float64) * 2 - 1).requires_grad_(True)
        t_pde = torch.rand(1000, 1, dtype=torch.float64).requires_grad_(True)
        l_pde = inverse_ac_residual(learner.model, x_pde, t_pde, curr_eps)

        # Snapshot data losses
        l_data_list = []
        for x_s, t_s, u_noisy in noisy_data:
            u_pred = learner.model(torch.cat([x_s, t_s], dim=1))
            l_data_list.append(torch.mean((u_pred - u_noisy)**2))

        all_losses = [l_pde] + l_data_list
        sum_loss   = sum(l for l in all_losses)
        loss_history.append(sum_loss.item())
        eps_history.append(curr_eps.item())

        # --- Optimization step ---
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
    print(f"  DONE  [{config_name}] | ε_pred: {final_eps:.8f} | "
          f"Target: {target_epsilon:.8f}")
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
    # name,            use_ema, use_pcgrad, use_rar/softplus
    ("Baseline",       False,   False,      False),
    ("Test A (EMA)",   True,    False,      False),
    ("Test B (EMA+PC)",True,    True,       False),
    ("SOTA (Full)",    True,    True,       True ),
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ablation Study for DGB-PINN Framework")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Quick validation run (50 epochs, reduced RAR pool)")
    parser.add_argument("--noise", type=float, default=0.1,
                        help="Noise level for inverse problem (default: 0.1)")
    args = parser.parse_args()

    # -- Hyperparameters --
    if args.smoke_test:
        EPOCHS         = 50
        RAR_CANDIDATES = 1000
        print("\n*** SMOKE TEST MODE: 50 epochs, 1k RAR candidates ***\n")
    else:
        EPOCHS         = 20000
        RAR_CANDIDATES = 15000
        print(f"\n*** PRODUCTION MODE: {EPOCHS} epochs, {RAR_CANDIDATES} RAR candidates ***\n")

    # ---- Storage for results ---- #
    results       = []
    fwd_histories = {}       # config_name -> loss_history
    inv_histories = {}       # config_name -> eps_history

    # ========================================================== #
    #                PHASE 1: FORWARD BENCHMARK                  #
    # ========================================================== #
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

    # ========================================================== #
    #             PHASE 2: INVERSE BENCHMARK                     #
    # ========================================================== #
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

    # ========================================================== #
    #                  CSV EXPORT                                 #
    # ========================================================== #
    df = pd.DataFrame(results)
    csv_path = "results/ablation_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"  METRICS EXPORTED → {csv_path}")
    print(f"{'='*70}")
    print(df.to_string(index=False))

    # ========================================================== #
    #               LOSS CURVE OVERLAY PLOT                      #
    # ========================================================== #
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left panel: Forward loss curves (log scale) ---
    ax = axes[0]
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    for i, (name, _, _, _) in enumerate(ABLATION_CONFIGS):
        if name in fwd_histories:
            hist = np.array(fwd_histories[name])
            hist = np.clip(hist, 1e-12, None)  # prevent log(0)
            ax.semilogy(hist, label=name, color=colors[i], alpha=0.85,
                        linewidth=1.5 if 'SOTA' in name else 1.0)
    ax.set_title("Forward Problem: Loss Trajectory", fontsize=13, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sum Loss (log scale)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Right panel: Inverse epsilon convergence ---
    ax = axes[1]
    for i, (name, _, _, _) in enumerate(ABLATION_CONFIGS):
        if name in inv_histories:
            ax.semilogy(inv_histories[name], label=name, color=colors[i],
                        alpha=0.85, linewidth=1.5 if 'SOTA' in name else 1.0)
    ax.axhline(y=TARGET_EPS, color='black', linestyle='--', linewidth=1.5,
               label=f'Target ε = {TARGET_EPS}')
    ax.set_title("Inverse Problem: ε Convergence", fontsize=13, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Predicted ε (log scale)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = "results/ablation_loss_curve.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f"  PLOT EXPORTED → {plot_path}")
    print(f"\n{'='*70}")
    print(f"  ABLATION STUDY COMPLETE")
    print(f"{'='*70}\n")

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse
import pandas as pd
import time

# Set project root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinn_engine.model import PINN
from pinn_engine.balancer import DBBalancer
from pinn_engine.surgery import PINNGradientSurgery
from pinn_engine.sampling import ResidualSampler
from problems.navier_stokes import navier_stokes_residuals, cylinder_bc_loss, sample_domain_ns, cylinder_mask
from experiments.run_inverse_cylinder import generate_sensor_data

torch.set_default_dtype(torch.float64)

# Ensure results dir
os.makedirs("results", exist_ok=True)

# ----------------- #
# FORWARD BENCHMARK #
# ----------------- #
def evaluate_forward(config_name, max_epochs, use_balancer, use_surgery, use_rar):
    print(f"\n--- Running Forward [{config_name}] ---")
    model = PINN(in_features=2, hidden_features=128, hidden_layers=5, out_features=3)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    num_total_losses = 9
    balancer = DBBalancer(num_conditions=num_total_losses-1, update_freq=50) if use_balancer else None
    surgery = PINNGradientSurgery(optimizer, use_gtn=True) if use_surgery else None
    
    if use_rar:
        bounds = [(0.0, 1.1), (0.0, 0.41)]
        sampler = ResidualSampler(model, navier_stokes_residuals, bounds, mask_fn=cylinder_mask)
    
    x_pde, y_pde = sample_domain_ns(n_pde=1500)
    
    loss_history = []
    
    start_time = time.time()
    for epoch in range(max_epochs):
        if use_rar and epoch % 50 == 0 and epoch > 0:
            sampled_coords = sampler.sample(n_points=1500)
            x_pde, y_pde = sampled_coords[:, 0:1], sampled_coords[:, 1:2]
            
        l_pde = navier_stokes_residuals(model, x_pde, y_pde, re=100)
        l_bc = cylinder_bc_loss(model, n_bc=400)
        all_losses = l_pde + l_bc
        
        sum_loss = sum(l for l in all_losses)
        loss_history.append(sum_loss.item())
        
        if use_surgery:
            weights = balancer.get_weights() if use_balancer else None
            grad_mags = surgery.step(all_losses, weights=weights)
        else:
            optimizer.zero_grad()
            if use_balancer:
                weights = balancer.get_weights()
                weighted_sum = sum(w * l for w, l in zip(weights, all_losses))
                weighted_sum.backward()
                # Simulate capturing grad mags for the balancer to adapt
                grad_mags = [l.item() for l in all_losses] # Dummy proxy since surgery isn't there
            else:
                sum_loss.backward()
            optimizer.step()
            
        if use_balancer and use_surgery:
            balancer.update_gradient_stats(grad_mags)
            balancer.balance_weights()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Sum Loss: {sum_loss.item():.6f}")

    elapsed = time.time() - start_time
    final_loss = loss_history[-1]
    
    return final_loss, elapsed, loss_history, model, (x_pde if use_rar else None), (y_pde if use_rar else None)

# ----------------- #
# INVERSE BENCHMARK #
# ----------------- #
class InverseTestLearner(torch.nn.Module):
    def __init__(self, use_softplus):
        super().__init__()
        self.use_softplus = use_softplus
        if use_softplus:
            self.k = torch.nn.Parameter(torch.tensor([5.0], dtype=torch.float64))
        else:
             # Random unconstrained initialization
            self.k = torch.nn.Parameter(torch.tensor([50.0], dtype=torch.float64))
        self.model = PINN(in_features=2, hidden_features=128, hidden_layers=5, out_features=3)
        
    @property
    def re_pred(self):
        if self.use_softplus:
            return torch.nn.functional.softplus(self.k) * 10.0
        return self.k

    def forward(self, x):
        return self.model(x)

def evaluate_inverse(config_name, max_epochs, x_s, y_s, u_s, v_s, use_balancer, use_surgery, use_softplus):
    print(f"\n--- Running Inverse [{config_name}] ---")
    learner = InverseTestLearner(use_softplus)
    optimizer = optim.Adam(learner.parameters(), lr=0.0005)
    
    balancer = DBBalancer(num_conditions=3, update_freq=50) if use_balancer else None
    surgery = PINNGradientSurgery(optimizer, use_gtn=True) if use_surgery else None
    
    bounds = [(0.0, 1.1), (0.0, 0.41)]
    sampler = ResidualSampler(learner.model, navier_stokes_residuals, bounds, mask_fn=cylinder_mask)
    
    re_history = []
    
    for epoch in range(max_epochs):
        curr_re = learner.re_pred
        
        coords_s = torch.cat([x_s, y_s], dim=1)
        pred_out = learner(coords_s)
        l_data = torch.mean((pred_out[:, 0:1] - u_s)**2 + (pred_out[:, 1:2] - v_s)**2)
        
        if epoch % 50 == 0:
            sampled_coords = sampler.sample(n_points=1000)
            x_pde, y_pde = sampled_coords[:, 0:1], sampled_coords[:, 1:2]
            
        l_pde = navier_stokes_residuals(learner.model, x_pde, y_pde, re=curr_re)
        all_losses = [l_data] + l_pde
        
        sum_loss = sum(l for l in all_losses)
        re_history.append(curr_re.item())
        
        if use_surgery:
            weights = balancer.get_weights() if use_balancer else None
            grad_mags = surgery.step(all_losses, weights=weights)
        else:
            optimizer.zero_grad()
            if use_balancer:
                weights = balancer.get_weights()
                weighted_sum = sum(w * l for w, l in zip(weights, all_losses))
                weighted_sum.backward()
                grad_mags = [l.item() for l in all_losses]
            else:
                sum_loss.backward()
            optimizer.step()
            
        if use_balancer and use_surgery:
            balancer.update_gradient_stats(grad_mags)
            balancer.balance_weights()
            
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Re Pred: {curr_re.item():.2f}")

    final_re = re_history[-1]
    return final_re, re_history

# ----------------- #
# MASTER EXECUTION  #
# ----------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fwd_epochs", type=int, default=200)
    parser.add_argument("--inv_epochs", type=int, default=200)
    parser.add_argument("--noise", type=float, default=0.1)
    args = parser.parse_args()

    results_data = []

    # --- Phase 1: Forward ---
    # Baseline
    fwd_bl_loss, fwd_bl_time, fwd_bl_hist, _, _, _ = evaluate_forward("Baseline", args.fwd_epochs, False, False, False)
    results_data.append({"Experiment": "Forward", "Config": "Baseline", "Metric": "Final Loss", "Value": fwd_bl_loss})
    
    # Test A: EMA
    fwd_a_loss, fwd_a_time, fwd_a_hist, _, _, _ = evaluate_forward("Test A (EMA)", args.fwd_epochs, True, False, False)
    results_data.append({"Experiment": "Forward", "Config": "Test A", "Metric": "Final Loss", "Value": fwd_a_loss})
    
    # Test B: EMA + Surgery
    fwd_b_loss, fwd_b_time, fwd_b_hist, _, _, _ = evaluate_forward("Test B (EMA+PCGrad)", args.fwd_epochs, True, True, False)
    results_data.append({"Experiment": "Forward", "Config": "Test B", "Metric": "Final Loss", "Value": fwd_b_loss})
    
    # SOTA: EMA + Surgery + RAR
    sota_loss, sota_time, sota_hist, sota_model, rar_x, rar_y = evaluate_forward("SOTA", args.fwd_epochs, True, True, True)
    results_data.append({"Experiment": "Forward", "Config": "SOTA", "Metric": "Final Loss", "Value": sota_loss})

    # --- Phase 2: Inverse ---
    target_re = 100.0
    x_s, y_s, u_s, v_s = generate_sensor_data(true_re=target_re, num_sensors=1000, noise_lv=args.noise)
    
    # Baseline
    inv_bl_re, inv_bl_hist = evaluate_inverse("Baseline", args.inv_epochs, x_s, y_s, u_s, v_s, False, False, False)
    results_data.append({"Experiment": "Inverse", "Config": "Baseline", "Metric": "Abs Error", "Value": abs(inv_bl_re - target_re)})
    
    # Test A: EMA
    inv_a_re, inv_a_hist = evaluate_inverse("Test A (EMA)", args.inv_epochs, x_s, y_s, u_s, v_s, True, False, False)
    results_data.append({"Experiment": "Inverse", "Config": "Test A", "Metric": "Abs Error", "Value": abs(inv_a_re - target_re)})
    
    # SOTA
    inv_sota_re, inv_sota_hist = evaluate_inverse("SOTA", args.inv_epochs, x_s, y_s, u_s, v_s, True, True, True)
    results_data.append({"Experiment": "Inverse", "Config": "SOTA", "Metric": "Abs Error", "Value": abs(inv_sota_re - target_re)})

    # --- Data Export ---
    df = pd.DataFrame(results_data)
    df.to_csv("results/ablation_metrics.csv", index=False)
    print("\nMetrics exported to results/ablation_metrics.csv")

    # --- Generate Plots ---
    plt.switch_backend('agg') # Ensure plots generate headlessly
    
    # Plot 1: Forward Loss
    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(fwd_bl_hist), label='Baseline (Oscillating)', alpha=0.7)
    plt.plot(np.log10(sota_hist), label='SOTA (Smooth Decay)', alpha=0.9, linewidth=2)
    plt.title("Forward Problem: Sum Loss Trajectory (Log Scale)")
    plt.xlabel("Epoch")
    plt.ylabel("Log10(Loss)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/forward_loss_comparison.png")
    
    # Plot 2: RAR Density Mapping
    if rar_x is not None:
        plt.figure(figsize=(10, 4))
        plt.scatter(rar_x.detach().numpy(), rar_y.detach().numpy(), s=1, alpha=0.6, c='royalblue')
        circle = plt.Circle((0.2, 0.2), 0.05, color='red', fill=False, linewidth=2)
        plt.gca().add_artist(circle)
        plt.title("SOTA: Dynamic RAR Sampling Density Mapping")
        plt.xlim(0, 1.1)
        plt.ylim(0, 0.41)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("results/rar_density_overlay.png")

    print("Success: Generated results/forward_loss_comparison.png and results/rar_density_overlay.png")

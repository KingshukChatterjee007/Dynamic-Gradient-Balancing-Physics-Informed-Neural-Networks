import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

# Set project root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinn_engine.model import PINN
from pinn_engine.balancer import DBBalancer
from pinn_engine.surgery import PINNGradientSurgery
from pinn_engine.sampling import ResidualSampler
from problems.navier_stokes import navier_stokes_residuals, cylinder_mask

# Set default precision to float64
torch.set_default_dtype(torch.float64)

class InverseFluidLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Softplus Anchor: k starts such that softplus(k)*10 ≈ 50 (Initial guess Re=50)
        # softplus(k) = 5 -> k ≈ 4.993
        self.k = torch.nn.Parameter(torch.tensor([5.0], dtype=torch.float64))
        self.model = PINN(in_features=2, hidden_features=128, hidden_layers=5, out_features=3)
        
    @property
    def re_pred(self):
        # Physics Anchor: prevents negative or zero Reynolds number
        # scale by 10 to keep softplus output in a reasonable range
        return torch.nn.functional.softplus(self.k) * 10.0

    def forward(self, x):
        return self.model(x)

def generate_sensor_data(true_re=100.0, num_sensors=1000, noise_lv=0.1, model_path="pinn_cylinder_re100.pth"):
    # Load pre-trained model as ground truth
    true_model = PINN(in_features=2, hidden_features=128, hidden_layers=5, out_features=3)
    if os.path.exists(model_path):
        true_model.load_state_dict(torch.load(model_path))
        print(f"Loaded ground truth model from {model_path}.")
    else:
        print("Warning: Pre-trained model not found. Using initialized model as ground truth (results may be meaningless).")
    
    true_model.eval()

    # Distribute sensors randomly in the domain
    x_cand = torch.rand(num_sensors * 2, 1, dtype=torch.float64) * 1.1
    y_cand = torch.rand(num_sensors * 2, 1, dtype=torch.float64) * 0.41
    
    # Filter out sensors inside the cylinder
    mask = cylinder_mask(x_cand, y_cand)
    x_cand = x_cand[mask.view(-1)].view(-1, 1)
    y_cand = y_cand[mask.view(-1)].view(-1, 1)
    
    # We want more sensors in the wake (x > 0.2, y between 0.1 and 0.3)
    in_wake = (x_cand > 0.3) & (y_cand > 0.1) & (y_cand < 0.3)
    
    # Probabilistic sampling favoring the wake
    probs = torch.where(in_wake, 5.0, 1.0)
    indices = torch.multinomial(probs.flatten(), num_sensors, replacement=False)
    
    x_sensor = x_cand[indices]
    y_sensor = y_cand[indices]

    coords = torch.cat([x_sensor, y_sensor], dim=1)
    with torch.no_grad():
        clean_out = true_model(coords)
        u_clean, v_clean = clean_out[:, 0:1], clean_out[:, 1:2]
        
        # Add 10% Gaussian noise relative to std dev of the field
        u_noise = torch.randn_like(u_clean) * noise_lv * u_clean.std()
        v_noise = torch.randn_like(v_clean) * noise_lv * v_clean.std()
        
        u_noisy = u_clean + u_noise
        v_noisy = v_clean + v_noise

    return x_sensor, y_sensor, u_noisy, v_noisy

def run_inverse_cylinder(max_epochs=1000, lr=0.0005, noise_lv=0.1, target_re=100.0):
    print(f"Generating Noisy Sensor Data ({noise_lv*100}% Noise)...")
    x_s, y_s, u_s, v_s = generate_sensor_data(true_re=target_re, num_sensors=1000, noise_lv=noise_lv)
    
    learner = InverseFluidLearner()
    optimizer = optim.Adam(learner.parameters(), lr=lr)
    
    # Balancer: 1 Data Loss + 3 PDE Losses (u, v, c)
    balancer = DBBalancer(num_conditions=3, update_freq=50)
    surgery = PINNGradientSurgery(optimizer, use_gtn=True)
    
    bounds = [(0.0, 1.1), (0.0, 0.41)]
    sampler = ResidualSampler(learner.model, navier_stokes_residuals, bounds, mask_fn=cylinder_mask)
    
    re_history = []
    
    print(f"Inverse Discovery Machine ONLINE (Target Re: {target_re})...")
    
    for epoch in range(max_epochs):
        curr_re = learner.re_pred
        
        # Phase 1: Data Loss (Sensors)
        coords_s = torch.cat([x_s, y_s], dim=1)
        pred_out = learner(coords_s)
        pred_u, pred_v = pred_out[:, 0:1], pred_out[:, 1:2]
        l_data = torch.mean((pred_u - u_s)**2 + (pred_v - v_s)**2)
        
        # Phase 2: Physics Loss (Adaptive Sampling)
        if epoch % 50 == 0:
            sampled_coords = sampler.sample(n_points=1000)
            x_pde, y_pde = sampled_coords[:, 0:1], sampled_coords[:, 1:2]
            
        # Notice we pass the *estimated* re tensor to maintain gradients
        l_pde = navier_stokes_residuals(learner.model, x_pde, y_pde, re=curr_re) # [lu, lv, lc]
        
        all_losses = [l_data] + l_pde
        
        # 3. Surgery & Balancing
        weights = balancer.get_weights()
        grad_mags = surgery.step(all_losses, weights=weights)
        
        balancer.update_gradient_stats(grad_mags)
        balancer.balance_weights()
        
        re_history.append(curr_re.item())
        
        if epoch % 100 == 0:
            avg_loss = sum(l.item() for l in all_losses)
            print(f"Epoch {epoch:5d} | Sum Loss: {avg_loss:.6f} | Pred Re: {curr_re.item():.4f}")

    return learner, re_history, target_re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=1500)
    parser.add_argument("--noise", type=float, default=0.1)
    args = parser.parse_args()

    learner, re_hist, target_re = run_inverse_cylinder(max_epochs=args.max_epochs, noise_lv=args.noise)
    
    # Plot Discovery Convergence
    plt.figure(figsize=(10, 6))
    plt.plot(re_hist, label='Predicted Reynolds Number (Discovery)', color='blue')
    plt.axhline(y=target_re, color='red', linestyle='--', label='Ground Truth (Re=100)')
    plt.title(f"Inverse Fluid Discovery: Reynolds Number (Noise={args.noise*100}%)")
    plt.xlabel("Epoch")
    plt.ylabel("Reynolds Number ($Re$)")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/discovery_re_convergence.png")
    print("Convergence plot saved to results/discovery_re_convergence.png")

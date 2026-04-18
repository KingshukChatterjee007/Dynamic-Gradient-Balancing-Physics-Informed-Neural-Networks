import torch
import torch.optim as optim
import numpy as np
import sys
import os
import argparse

# Set project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinn_engine.model import PINN
from pinn_engine.balancer import DBBalancer, gll_loss
from pinn_engine.surgery import PINNGradientSurgery
from pinn_engine.sampling import ResidualSampler
from pinn_engine.diagnostics import PINNDiagnostics
from problems.navier_stokes_3d import navier_stokes_3d_residuals, sphere_bc_loss, sphere_mask

# Set precison
torch.set_default_dtype(torch.float64)

def train_robust_3d(max_epochs=500, lr=0.0001, n_pde=10000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Robust 3D PINN (Probabilistic Mode) on {device}")

    # 1. Initialize Probabilistic Model
    # 4 inputs (x,y,z,t) -> 4 outputs (u,v,w,p) MEAN + 4 outputs log(VAR) = 8 total outputs
    model = PINN(in_features=4, hidden_features=128, hidden_layers=5, 
                 out_features=4, probabilistic=True, dropout_rate=0.05).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 2. Balancer, Surgery, Diagnostics
    num_total_losses = 8
    balancer = DBBalancer(num_conditions=num_total_losses-1, update_freq=100)
    surgery = PINNGradientSurgery(optimizer, use_gtn=True)
    diagnostics = PINNDiagnostics(model)
    
    # Define a wrapper to extract only the MEAN heads for physics/BCs
    class MeanHeadWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.pinn = model
        def forward(self, x):
            return self.pinn(x)[:, :self.pinn.out_features]
            
    mean_model = MeanHeadWrapper(model)
    
    # 3. Sampler
    # Note: Sampler needs the mean model to compute physical residuals
    bounds = [[0.0, 1.1], [0.0, 0.41], [0.0, 0.41], [0.0, 1.0]]
    sampler = ResidualSampler(mean_model, navier_stokes_3d_residuals, bounds, mask_fn=sphere_mask)
    
    all_coords = torch.rand(n_pde, 4, device=device) * (torch.tensor([b[1]-b[0] for b in bounds], device=device)) + torch.tensor([b[0] for b in bounds], device=device)
    all_coords.requires_grad_(True)
    
    for epoch in range(max_epochs):
        if epoch % 50 == 0 and epoch > 0:
            all_coords = sampler.sample(n_points=n_pde)

        # 4. Probabilistic Loss Calculation
        # Forward pass: model outputs [mean_u, mean_v, mean_w, mean_p, log_var_u, ...]
        out = model(all_coords)
        log_var = out[:, 4:] # shape [N, 4]
        
        x, y, z, t = all_coords[:, 0:1], all_coords[:, 1:2], all_coords[:, 2:3], all_coords[:, 3:4]
        # Use mean_model for residuals and BCs
        pde_res = navier_stokes_3d_residuals(mean_model, x, y, z, t, re=100) # [4 tensors]
        
        # Compute GLL Loss for each PDE term
        l_pde = []
        for i in range(4):
            l_pde.append(gll_loss(pde_res[i], torch.zeros_like(pde_res[i]), log_var[:, i]))
            
        # BC losses (Standard MSE for now, or GLL if we have data)
        l_bc_list = sphere_bc_loss(mean_model, n_bc=400, bounds=bounds)
        
        total_losses = l_pde + l_bc_list
        
        # 5. Optimize
        weights = balancer.get_weights()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_mags = surgery.step(total_losses, weights=weights)
        
        balancer.update_gradient_stats(grad_mags)
        balancer.balance_weights()
        
        if epoch % 10 == 0:
            avg_log_var = log_var.mean().item()
            print(f"Epoch {epoch:4d} | PDE L: {sum(l.item() for l in l_pde):.4f} | BC L: {sum(l.item() for l in l_bc_list):.4f} | Avg LogVar: {avg_log_var:.4f}")

        # Active Learning: Suggest sensors at the end of training
        if epoch == max_epochs - 1:
            print("\nCalculating Optimal Sensor Locations for experimental validation...")
            # Combine residuals for saliency
            total_res = torch.abs(torch.stack(pde_res).sum(dim=0))
            sensors = diagnostics.get_optimal_sensor_locations(all_coords, total_res, n_sensors=5)
            print("Suggested Sensor Locations (X, Y, Z, t):")
            for i, s in enumerate(sensors):
                print(f"  Sensor {i+1}: {s}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    
    model = train_robust_3d(max_epochs=args.epochs)
    torch.save(model.state_dict(), "pinn_robust_3d.pth")
    print("Robust Model saved.")

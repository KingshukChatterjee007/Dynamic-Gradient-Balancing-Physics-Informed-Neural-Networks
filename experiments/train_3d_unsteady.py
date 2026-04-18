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
from problems.navier_stokes_3d import navier_stokes_3d_residuals, sphere_bc_loss, sphere_mask

# Set default precision to float64
torch.set_default_dtype(torch.float64)

def train_3d_unsteady(max_epochs=1000, lr=0.0005, re=100, n_pde=5000, n_batches=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # 1. Initialize Model (4 inputs: x, y, z, t | 4 outputs: u, v, w, p)
    model = PINN(in_features=4, hidden_features=128, hidden_layers=5, out_features=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 2. Balancers & Surgery
    # PDE losses (4: u, v, w, c) + BC losses (4: in, out, wall, sphere) = 8
    num_total_losses = 8
    balancer = DBBalancer(num_conditions=num_total_losses-1, update_freq=50)
    surgery = PINNGradientSurgery(optimizer, use_gtn=True)
    
    # 3. Sampler Setup
    bounds = [[0.0, 1.1], [0.0, 0.41], [0.0, 0.41], [0.0, 1.0]]
    sampler = ResidualSampler(model, navier_stokes_3d_residuals, bounds, mask_fn=sphere_mask)
    
    # Initial uniform sampling
    def get_uniform(n):
        coords = []
        for b in bounds:
            coords.append(torch.rand(n, 1, device=device) * (b[1] - b[0]) + b[0])
        return torch.cat(coords, dim=1).requires_grad_(True)

    all_coords = get_uniform(n_pde)
    
    for epoch in range(max_epochs):
        # Adaptive Refinement
        if epoch % 50 == 0 and epoch > 0:
            all_coords = sampler.sample(n_points=n_pde)

        # 4. Sub-batching for Gradient Accumulation
        batch_size = n_pde // n_batches
        
        # Initialize flat gradient buffers for each of the 8 losses
        num_params = sum(p.numel() for p in model.parameters())
        accumulated_grads = [torch.zeros(num_params, device=device) for _ in range(num_total_losses)]
        
        # We also need boundary points for each epoch
        # (For simplicity here, we sample BCs once per epoch)
        # In a more advanced version, BCs could also be sub-batched if n_bc is huge.
        l_bc_list = sphere_bc_loss(model, n_bc=400, bounds=bounds)
        
        # Process PDE Residuals in mini-batches
        for b_idx in range(n_batches):
            start, end = b_idx * batch_size, (b_idx + 1) * batch_size
            coords_batch = all_coords[start:end]
            
            x, y, z, t = coords_batch[:, 0:1], coords_batch[:, 1:2], coords_batch[:, 2:3], coords_batch[:, 3:4]
            
            # Compute PDE losses for this batch
            l_pde_batch = navier_stokes_3d_residuals(model, x, y, z, t, re=re)
            
            # For each PDE loss term
            for i in range(4):
                optimizer.zero_grad()
                # retain_graph=True if we have more losses to go for this batch
                # though actually each loss call generates its own graph here? 
                # navier_stokes_3d_residuals calls model(coords) once.
                # So we need retain_graph=True for the first 3 losses of the batch.
                l_pde_batch[i].backward(retain_graph=(i < 3))
                
                # Accumulate normalized gradient
                accumulated_grads[i] += surgery._get_flat_grad() / n_batches
        
        # Compute BC Gradients (Processed once per epoch as they are fewer points)
        for i in range(4):
            optimizer.zero_grad()
            l_bc_list[i].backward(retain_graph=(i < 3))
            accumulated_grads[i+4] = surgery._get_flat_grad()
            
        # 5. Apply Surgery and Optimize
        weights = balancer.get_weights()
        grad_mags = surgery.step_with_grads(accumulated_grads, weights=weights)
        
        # 6. Update Stats
        balancer.update_gradient_stats(grad_mags)
        balancer.balance_weights()
        
        if epoch % 10 == 0:
            pde_val = sum(l.item() for l in l_pde_batch) if 'l_pde_batch' in locals() else 0
            bc_val = sum(l.item() for l in l_bc_list)
            print(f"Epoch {epoch:4d} | PDE L: {pde_val:.6f} | BC L: {bc_val:.6f} | Weights: {weights[:3].numpy()}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--re", type=float, default=100.0)
    args = parser.parse_args()

    model = train_3d_unsteady(max_epochs=args.epochs, re=args.re)
    
    torch.save(model.state_dict(), "pinn_3d_ns_sphere.pth")
    print("3D Unsteady Model saved as 'pinn_3d_ns_sphere.pth'")

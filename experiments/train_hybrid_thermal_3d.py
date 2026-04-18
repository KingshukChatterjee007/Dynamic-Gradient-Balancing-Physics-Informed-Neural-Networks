import torch
import torch.optim as optim
import sys
import os
import argparse

# Set project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinn_engine.model import PINN
from pinn_engine.balancer import DBBalancer
from pinn_engine.surgery import PINNGradientSurgery
from pinn_engine.sampling import ResidualSampler
from problems.navier_stokes_3d import navier_stokes_3d_residuals, sphere_bc_loss, sphere_mask

# Set precison
torch.set_default_dtype(torch.float64)

def train_hybrid_thermal_3d(max_epochs=100, lr=0.0001, n_pde=5000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Hybrid GNN-PINN with Boussinesq Thermal Coupling on {device}")

    # 1. Initialize Hybrid Model
    # 5 Output channels: (u,v,w,p,T)
    model = PINN(in_features=4, hidden_features=128, hidden_layers=5, 
                 out_features=5, use_gnn=True, dropout_rate=0.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 2. Balancer, Surgery
    # 5 PDE losses (u,v,w,c,T) + 4 BC losses = 9 total
    balancer = DBBalancer(num_conditions=8, update_freq=50)
    surgery = PINNGradientSurgery(optimizer, use_gtn=True)
    
    # 3. Sampler
    bounds = [[0.0, 1.1], [0.0, 0.41], [0.0, 0.41], [0.0, 1.0]]
    sampler = ResidualSampler(model, navier_stokes_3d_residuals, bounds, mask_fn=sphere_mask)
    
    all_coords = torch.rand(n_pde, 4, device=device) * (torch.tensor([b[1]-b[0] for b in bounds], device=device)) + torch.tensor([b[0] for b in bounds], device=device)
    all_coords.requires_grad_(True)
    
    for epoch in range(max_epochs):
        if epoch % 50 == 0 and epoch > 0:
            # RAR Refinement
            all_coords = sampler.sample(n_points=n_pde)

        # 4. Residual Calculation
        x, y, z, t = all_coords[:, 0:1], all_coords[:, 1:2], all_coords[:, 2:3], all_coords[:, 3:4]
        # Boussinesq residuals: [res_u, res_v, res_w, res_c, res_T]
        pde_res_list = navier_stokes_3d_residuals(model, x, y, z, t, re=100, pr=0.71, ri=1.0)
        
        # 5. BC Calculation
        # BC losses: [l_in, l_out_p, l_wall, l_sphere]
        bc_res_list = sphere_bc_loss(model, n_bc=400, bounds=bounds)
        
        total_losses = pde_res_list + bc_res_list
        
        # 6. Optimization Step
        weights = balancer.get_weights()
        grad_mags = surgery.step(total_losses, weights=weights)
        
        balancer.update_gradient_stats(grad_mags)
        balancer.balance_weights()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | PDE L: {sum(l.item() for l in pde_res_list):.4f} | BC L: {sum(l.item() for l in bc_res_list):.4f}")

    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--n_pde", type=int, default=2000) # Small for Mac testing
    args = parser.parse_args()
    
    model = train_hybrid_thermal_3d(max_epochs=args.epochs, n_pde=args.n_pde)
    torch.save(model.state_dict(), "pinn_hybrid_thermal_3d.pth")
    print("Hybrid Thermal Model saved.")

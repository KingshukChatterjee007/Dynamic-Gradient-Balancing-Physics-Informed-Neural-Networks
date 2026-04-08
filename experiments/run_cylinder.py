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
from problems.navier_stokes import navier_stokes_residuals, cylinder_bc_loss, sample_domain_ns, cylinder_mask

# Set default precision to float64 (Required for SIREN/surgery stability)
torch.set_default_dtype(torch.float64)

def train_cylinder(max_epochs=2000, lr=0.0005, re=100, use_gtn=True):
    # 1. Initialize Model (3 outputs: u, v, p)
    model = PINN(in_features=2, hidden_features=128, hidden_layers=5, out_features=3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 2. Advanced Balancers
    # Total loss terms: 3 (PDE: u, v, c) + 6 (BC: in_u, in_v, out_p, wall_v, cyl_u, cyl_v) = 9
    num_total_losses = 9
    balancer = DBBalancer(num_conditions=num_total_losses-1, update_freq=50)
    surgery = PINNGradientSurgery(optimizer, use_gtn=use_gtn)
    
    print(f"Fluid Dynamics Training: Flow over Cylinder (Re={re})")
    print(f"GTN Enabled: {use_gtn}")
    
    # 3. RAR Setup
    bounds = [(0.0, 1.1), (0.0, 0.41)]
    sampler = ResidualSampler(model, navier_stokes_residuals, bounds, mask_fn=cylinder_mask)
    x_pde, y_pde = sample_domain_ns(n_pde=1500) # Start with uniform
    
    for epoch in range(max_epochs):
        # Adaptive Refinement every 50 epochs
        if epoch % 50 == 0 and epoch > 0:
            sampled_coords = sampler.sample(n_points=1500)
            x_pde, y_pde = sampled_coords[:, 0:1], sampled_coords[:, 1:2]
        
        # Calculate Losses
        l_pde = navier_stokes_residuals(model, x_pde, y_pde, re=re) # [lu, lv, lc]
        l_bc = cylinder_bc_loss(model, n_bc=400) # [in_u, in_v, out_p, wall_v, cyl_u, cyl_v]
        
        all_losses = l_pde + l_bc # List of 9 loss tensors
        
        # 3. Surgery and Balancing
        weights = balancer.get_weights()
        grad_mags = surgery.step(all_losses, weights=weights)
        
        # 4. Update Balancer
        balancer.update_gradient_stats(grad_mags)
        balancer.balance_weights()
        
        if epoch % 100 == 0:
            avg_loss = sum(l.item() for l in all_losses)
            # Collect names for logging
            pde_total = sum(l.item() for l in l_pde)
            bc_total = sum(l.item() for l in l_bc)
            print(f"Epoch {epoch:5d} | Sum Loss: {avg_loss:.6f} | PDE: {pde_total:.4f} | BC: {bc_total:.4f}")

    return model, x_pde, y_pde

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_epochs", type=int, default=1000)
    args = parser.parse_args()

    # Run the fluid training
    model, x_pde, y_pde = train_cylinder(max_epochs=args.max_epochs, use_gtn=True)
    
    # Save the model
    torch.save(model.state_dict(), "pinn_cylinder_re100.pth")
    print("Training complete. Model saved.")
    
    # Visualization: Velocity Magnitude
    nx, ny = 100, 50
    x = np.linspace(0, 1.1, nx)
    y = np.linspace(0, 0.41, ny)
    X, Y = np.meshgrid(x, y)
    coords_vis = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float64)
    
    with torch.no_grad():
        out = model(coords_vis)
        u, v, p = out[:, 0].reshape(ny, nx), out[:, 1].reshape(ny, nx), out[:, 2].reshape(ny, nx)
        vel_mag = np.sqrt(u**2 + v**2)

    # Plot
    plt.figure(figsize=(12, 4))
    plt.contourf(X, Y, vel_mag, levels=50, cmap='jet')
    plt.colorbar(label='Velocity Magnitude')
    # Draw cylinder
    circle = plt.Circle((0.2, 0.2), 0.05, color='white', fill=True)
    plt.gca().add_artist(circle)
    plt.title(f"2D Flow Around Cylinder ($Re=100$) - GTN PINN")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("cylinder_velocity.png")
    print("Velocity field saved as 'cylinder_velocity.png'.")
    
    # 2. Plot Sampling Density
    plt.figure(figsize=(10, 4))
    # Regenerate current sampling for visualization
    # (Assuming we use the final sampled x_pde, y_pde)
    plt.scatter(x_pde.detach().numpy(), y_pde.detach().numpy(), s=1, alpha=0.5, c='blue')
    circle = plt.Circle((0.2, 0.2), 0.05, color='red', fill=False)
    plt.gca().add_artist(circle)
    plt.title("RAR Sampling Density (Final Refinement)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, 1.1)
    plt.ylim(0, 0.41)
    plt.savefig("sampling_density.png")
    print("Sampling density saved as 'sampling_density.png'.")

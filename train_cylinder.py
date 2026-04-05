import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pinn_model import PINN
from db_pinn_balancer import DBBalancer
from directional_alignment import PINNGradientSurgery
from navier_stokes_2d import navier_stokes_residuals, cylinder_bc_loss, sample_domain_ns

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
    
    for epoch in range(max_epochs):
        # Sample Domain
        x_pde, y_pde = sample_domain_ns(n_pde=1500)
        
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

    return model

if __name__ == "__main__":
    # Run the fluid training
    model = train_cylinder(max_epochs=1000, use_gtn=True)
    
    # Save the model
    torch.save(model.state_dict(), "pinn_cylinder_re100.pth")
    print("Training complete. Model saved.")
    
    # Visualization: Velocity Magnitude
    nx, ny = 100, 50
    x = np.linspace(0, 1.1, nx)
    y = np.linspace(0, 0.41, ny)
    X, Y = np.meshgrid(x, y)
    coords_vis = torch.tensor(np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32)
    
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

import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Set project root in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinn_engine.model import PINN
from pinn_engine.balancer import DBBalancer
from pinn_engine.surgery import PINNGradientSurgery
from problems.allen_cahn import allen_cahn_residual, initial_condition_loss, boundary_condition_loss, sample_domain

def train(max_epochs=5000, lr=0.001, use_db=True, use_surgery=True):
    # 1. Initialize Model
    model = PINN(in_features=2, hidden_features=128, hidden_layers=4, out_features=1)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 2. Components
    num_conditions = 3 # (IC, BC_neg, BC_pos)
    balancer = DBBalancer(num_conditions=num_conditions, update_freq=50)
    surgery = PINNGradientSurgery(optimizer) if use_surgery else None
    
    # Tracking
    loss_history = []
    
    print(f"Training started (DB: {use_db}, Surgery: {use_surgery})...")
    
    for epoch in range(max_epochs):
        # Sample points
        (x_pde, t_pde), x_ic, t_bc = sample_domain(n_pde=1000, n_ic=200, n_bc=100)
        
        # Define Losses
        # u(x, t) for PDE points
        u_pde = model(torch.cat([x_pde, t_pde], dim=1))
        l_pde = allen_cahn_residual(u_pde, x_pde, t_pde)
        
        # u(x, 0) for IC
        l_ic = initial_condition_loss(model, x_ic)
        
        # u(-1, t), u(1, t) for BCs
        l_bc_neg, l_bc_pos = boundary_condition_loss(model, t_bc)
        
        losses = [l_pde, l_ic, l_bc_neg, l_bc_pos]
        
        # 3. Optimization Step
        if use_surgery:
            # Surgery handles both weighting and gradient projection
            weights = balancer.get_weights() if use_db else torch.ones(len(losses))
            grad_mags = surgery.step(losses, weights=weights)
        else:
            # Standard weighted sum
            optimizer.zero_grad()
            weights = balancer.get_weights() if use_db else torch.ones(len(losses))
            total_loss = sum(w * l for w, l in zip(weights, losses))
            total_loss.backward()
            optimizer.step()
            
            # Get gradient magnitudes manually for the balancer
            with torch.no_grad():
                grad_mags = [torch.norm(torch.autograd.grad(l, model.parameters(), retain_graph=True)[0]) for l in losses]
                # Note: Above is inefficient, but for testing purposes.
        
        # 4. Update Balancer
        if use_db:
            balancer.update_gradient_stats(grad_mags)
            balancer.balance_weights()
            
        # Logging
        if epoch % 100 == 0:
            avg_loss = sum(l.item() for l in losses)
            loss_history.append(avg_loss)
            w = [f"{v:.2f}" for v in weights]
            print(f"Epoch {epoch:5d} | Mean Loss: {avg_loss:.6f} | Weights: {w}")

    return model, loss_history

if __name__ == "__main__":
    # Run the advanced training
    model, history = train(max_epochs=1000, use_db=True, use_surgery=True)
    
    # Save the model
    torch.save(model.state_dict(), "pinn_ac_model.pth")
    print("Training complete and model saved.")
    
    # Basic Visualization of Results (Snapshot at t=1.0)
    x_test = torch.linspace(-1, 1, 100).view(-1, 1)
    t_test = torch.ones_like(x_test)
    coords = torch.cat([x_test, t_test], dim=1)
    u_pred = model(coords).detach().numpy()
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_test.numpy(), u_pred, label='PINN Prediction (t=1.0)')
    plt.title("Allen-Cahn Solution at Final Time")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.legend()
    plt.grid(True)
    plt.savefig("solution_snapshot.png")
    print("Result visualization saved as 'solution_snapshot.png'.")

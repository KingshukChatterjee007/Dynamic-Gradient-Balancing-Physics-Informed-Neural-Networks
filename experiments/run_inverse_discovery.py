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
from pinn_engine.sampling import EnergyAdaptiveSampler
from problems.inverse_allen_cahn import softplus_epsilon, inverse_ac_residual, snapshot_loss, generate_noisy_data

# Global Precision Shift
torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(True)

class InverseLearner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Starts softplus(k) around 0.01
        self.k = torch.nn.Parameter(torch.tensor([-4.6], dtype=torch.float64))
        self.model = PINN(in_features=2, hidden_features=128, hidden_layers=4, out_features=1)
        
    @property
    def epsilon(self):
        # The Physics Anchor: prevents negative epsilon explosions
        return torch.nn.functional.softplus(self.k)

    def forward(self, x):
        return self.model(x)

def run_inverse_discovery(max_epochs=2000, lr=0.001, noise_lv=0.1, target_epsilon=0.0001):
    # 1. Generate 'Ground Truth' Data
    print("Generating Noisy Ground Truth Data (10% Noise)...")
    model_true = PINN(in_features=2, hidden_features=128, hidden_layers=4, out_features=1)
    
    snapshots = [0.1, 0.25, 0.5, 0.75, 1.0]
    noisy_data = generate_noisy_data(model_true, target_epsilon, snapshots=snapshots, noise_lv=noise_lv)
    
    # 2. Initialize Discovery Machine
    learner = InverseLearner()
    
    # Optimizer includes model parameters and the unknown anchor k
    optimizer = optim.Adam(learner.parameters(), lr=lr)
    
    # 3. BALANCERS & SAMPLERS
    balancer = DBBalancer(num_conditions=5, alpha=0.999, update_freq=50) # Forgetful EMA
    surgery = PINNGradientSurgery(optimizer, use_gtn=True) # Scalar-GTN
    
    epsilon_history = []
    
    print(f"Inverse Discovery Machine ONLINE (Target Epsilon: {target_epsilon})...")
    
    for epoch in range(max_epochs):
        curr_epsilon = learner.epsilon
        
        # Phase 2: Energy-Adaptive Sampling
        sampler = EnergyAdaptiveSampler(learner.model, curr_epsilon)
        x_pde, t_pde = sampler.sample(n_points=1000)
        
        # Calculate Losses
        l_pde = inverse_ac_residual(learner.model, x_pde, t_pde, curr_epsilon)
        
        l_data_list = []
        for x, t, u_noisy in noisy_data:
             u_pred = learner.model(torch.cat([x, t], dim=1))
             l_data_list.append(torch.mean((u_pred - u_noisy)**2))
             
        all_losses = [l_pde] + l_data_list
        
        # 4. Gradient Surgery & Balancing (GTN + PCGrad)
        weights = balancer.get_weights()
        grad_mags = surgery.step(all_losses, weights=weights)
        
        # 5. Update Balancer (Forgetful EMA)
        balancer.update_gradient_stats(grad_mags)
        balancer.balance_weights()
        
        # Track Discovery
        epsilon_history.append(curr_epsilon.item())
        
        if epoch % 100 == 0:
            avg_loss = sum(l.item() for l in all_losses)
            print(f"Epoch {epoch:5d} | Sum Loss: {avg_loss:.6f} | Pred Epsilon: {curr_epsilon.item():.8f}")

    return learner, epsilon_history, target_epsilon

if __name__ == "__main__":
    model, eps_history, target_eps = run_inverse_discovery(max_epochs=1000)
    
    # Plot Discovery Convergence
    plt.figure(figsize=(10, 6))
    plt.plot(eps_history, label='Predicted Epsilon (Discovery)')
    plt.axhline(y=target_eps, color='r', linestyle='--', label='Ground Truth')
    plt.title("Inverse Discovery Convergence: Allen-Cahn Coefficient")
    plt.xlabel("Epoch")
    plt.ylabel("Epsilon")
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig("results/discovery_convergence.png")
    print("Discovery convergence plot saved to results/discovery_convergence.png")

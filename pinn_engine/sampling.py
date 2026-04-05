import torch
import numpy as np

def compute_energy_density(model, x, t, epsilon):
    """
    Computes pointwise energy density for Allen-Cahn:
    e(u) = (epsilon/2) * |u_x|^2 + (1/(4*epsilon)) * (u^2 - 1)^2
    """
    x = x.clone().detach().requires_grad_(True)
    u = model(torch.cat([x, t], dim=1))
    
    u_x = torch.autograd.grad(u.sum(), x, create_graph=False)[0]
    
    term1 = (epsilon / 2.0) * (u_x**2)
    term2 = (1.0 / (4.0 * epsilon)) * (u**2 - 1.0)**2
    
    return (term1 + term2).detach()

class EnergyAdaptiveSampler:
    """
    SOTA 2026 Energy-Adaptive Sampler (EAS).
    Samples points in proportion to the Allen-Cahn energy density.
    """
    def __init__(self, model, epsilon):
        self.model = model
        self.epsilon = epsilon  # This can be the 'current' softplus estimate

    def sample(self, n_points, n_candidate=10000):
        # 1. Sample candidate points uniformly
        x_cand = torch.rand(n_candidate, 1, dtype=torch.float64) * 2 - 1
        t_cand = torch.rand(n_candidate, 1, dtype=torch.float64)
        
        # 2. Compute energy density at candidates
        with torch.no_grad():
            # Since we need gradients for energy, we temporarily enable grad
            with torch.enable_grad():
                 e_vals = compute_energy_density(self.model, x_cand, t_cand, self.epsilon)
        
        # 3. Weighted Random Sampling
        e_vals = e_vals.flatten()
        # Add a small epsilon to avoid zero probability in flat regions
        probs = (e_vals + 1e-6) / (e_vals.sum() + 1e-6)
        
        indices = torch.multinomial(probs, n_points, replacement=True)
        
        x_sampled = x_cand[indices].clone().detach().requires_grad_(True)
        t_sampled = t_cand[indices].clone().detach().requires_grad_(True)
        
        return x_sampled, t_sampled

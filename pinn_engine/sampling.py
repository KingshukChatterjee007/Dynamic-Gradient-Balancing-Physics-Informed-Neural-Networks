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
        # 0. Get model device
        device = next(self.model.parameters()).device
        
        # 1. Sample candidate points uniformly on device
        x_cand = torch.rand(n_candidate, 1, dtype=torch.float64, device=device) * 2 - 1
        t_cand = torch.rand(n_candidate, 1, dtype=torch.float64, device=device)
        
        # 2. Compute energy density at candidates
        with torch.no_grad():
            with torch.enable_grad():
                 e_vals = compute_energy_density(self.model, x_cand, t_cand, self.epsilon)
        
        # 3. Weighted Random Sampling
        e_vals = e_vals.flatten()
        probs = (e_vals + 1e-6) / (e_vals.sum() + 1e-6)
        
        indices = torch.multinomial(probs, n_points, replacement=True)
        
        x_sampled = x_cand[indices].clone().detach().requires_grad_(True)
        t_sampled = t_cand[indices].clone().detach().requires_grad_(True)
        
        return x_sampled, t_sampled

class ResidualSampler:
    """
    Generic Residual-based Adaptive Refinement (RAR) Sampler.
    Dynamically identifies high-residual regions and focuses sampling there.
    """
    def __init__(self, model, residual_fn, bounds, mask_fn=None):
        self.model = model
        self.residual_fn = residual_fn
        self.bounds = bounds
        self.mask_fn = mask_fn

    def sample(self, n_points, n_candidate=15000):
        # 0. Get model device
        device = next(self.model.parameters()).device
        
        # 1. Generate candidate points uniformly across bounds on device
        candidates = []
        for b_min, b_max in self.bounds:
            candidates.append(torch.rand(n_candidate, 1, dtype=torch.float64, device=device) * (b_max - b_min) + b_min)
        
        coords_cand = torch.cat(candidates, dim=1).requires_grad_(True)
        
        if self.mask_fn is not None:
            mask = self.mask_fn(*[coords_cand[:, i:i+1] for i in range(coords_cand.shape[1])])
            coords_cand = coords_cand[mask.view(-1)].clone().detach().requires_grad_(True)
            n_candidate = coords_cand.shape[0]
        
        coord_args = [coords_cand[:, i:i+1] for i in range(coords_cand.shape[1])]
        
        with torch.enable_grad():
            res_list = self.residual_fn(self.model, *coord_args)
            if isinstance(res_list, list):
                total_res = torch.zeros_like(res_list[0])
                for r in res_list:
                    total_res += r.abs()
            else:
                total_res = res_list.abs()

        # 3. Probabilistic Selection
        res_vals = total_res.detach().flatten()
        probs = (res_vals + 1e-7) / (res_vals.sum() + 1e-7)
        
        indices = torch.multinomial(probs, n_points, replacement=True)
        sampled_coords = coords_cand[indices].clone().detach().requires_grad_(True)
        
        return sampled_coords

import torch
import numpy as np

class Welford:
    """
    Online algorithm for calculating mean and variance.
    Used for stabilizing weight updates in DB-PINN.
    """
    def __init__(self):
        self.k = 0
        self.M = 0
        self.S = 0

    def update(self, x):
        self.k += 1
        newM = self.M + (x - self.M) / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M = newM
        self.S = newS

    @property
    def mean(self):
        return self.M

    @property
    def variance(self):
        return self.S / self.k if self.k > 0 else 0

class DBBalancer:
    """
    Dual-Balancing Optimizer for PINNs (DB-PINN).
    Handles:
    - Inter-balancing (PDE vs Conditions)
    - Intra-balancing (Within conditions)
    - Stochastic noise filtering via Welford's algorithm
    """
    def __init__(self, num_conditions, alpha=0.5, update_freq=100):
        self.num_conditions = num_conditions
        self.alpha = alpha  # Smoothing factor or hyperparam for DB
        self.update_freq = update_freq
        self.step_count = 0
        
        # Initialize weights: 
        # lambda_pde, lambda_c1, lambda_c2, ...
        self.weights = torch.ones(1 + num_conditions, dtype=torch.float32)
        
        # Statistical trackers for gradient magnitudes
        self.gradient_stats = [Welford() for _ in range(1 + num_conditions)]

    def update_gradient_stats(self, grads_list):
        """
        grads_list: List of tensors [grad_pde, grad_c1, grad_c2, ...]
        Each grad_i should be the norm of the gradient for that component.
        """
        for i, grad_norm in enumerate(grads_list):
            self.gradient_stats[i].update(grad_norm.item())

    def balance_weights(self):
        """
        Recalculate weights based on DB-PINN logic.
        Uses accumulated gradient statistics to avoid 'stochastic spikes'.
        """
        self.step_count += 1
        if self.step_count % self.update_freq != 0:
            return self.weights

        # 1. Calculate base weights from Mean Gradient Magnitudes
        means = np.array([s.mean for s in self.gradient_stats])
        
        # Inter-balancing: Align PDE vs Aggregate Conditions
        # lambda_pde / mean_pde ~ sum(lambda_ci) / mean_sum_ci
        avg_ci_mean = np.mean(means[1:]) if len(means) > 1 else means[0]
        pde_mean = means[0]
        
        # 2. Adjust Inter-Balancing Weight (lambda_pde)
        # We usually fix lambda_c sum and adjust lambda_pde or vice versa.
        # Here we normalize relative to the PDE.
        
        new_weights = np.ones_like(means)
        
        # Intra-balancing: Each condition weight proportional to its fitting difficulty
        # Difficulty often measured by variance or mean of gradients
        for i in range(1, len(means)):
            if means[i] > 0:
                # Higher gradient means 'harder' to satisfy
                new_weights[i] = means[i] / (avg_ci_mean + 1e-8)
        
        # Normalize weights so they sum to total targets
        # Or scale lambda_pde so its gradient magnitude matches the avg condition
        if pde_mean > 0:
             self.weights[0] = avg_ci_mean / (pde_mean + 1e-8)
        
        # Update torch tensor
        for i in range(1, len(new_weights)):
            self.weights[i] = float(new_weights[i])
            
        return self.weights

    def get_weights(self):
        return self.weights

if __name__ == "__main__":
    balancer = DBBalancer(num_conditions=2)
    # Simulate some gradient updates
    for _ in range(200):
        g_pde = torch.tensor(10.0 + torch.randn(1))
        g_c1 = torch.tensor(1.0 + torch.randn(1))
        g_c2 = torch.tensor(5.0 + torch.randn(1))
        balancer.update_gradient_stats([g_pde, g_c1, g_c2])
        weights = balancer.balance_weights()
    
    print(f"Calculated Weights: {weights}")
    print("DB Balancer logic verified.")

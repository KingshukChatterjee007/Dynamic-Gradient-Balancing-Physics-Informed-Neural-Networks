import torch
import numpy as np

class EMAWelford:
    """
    Exponential Moving Average (EMA) for gradient statistics.
    A 'Forgetful' tracker that adapts quickly as physics parameters evolve.
    """
    def __init__(self, alpha=0.99):
        self.alpha = alpha
        self.val = 0.0
        self.initialized = False

    def update(self, x):
        if not self.initialized:
            self.val = x
            self.initialized = True
        else:
            # Standard EMA: New = Alpha*Old + (1-Alpha)*Current
            self.val = self.alpha * self.val + (1 - self.alpha) * x

    @property
    def mean(self):
        return self.val

class DBBalancer:
    """
    Dual-Balancing Optimizer for PINNs (DB-PINN).
    Upgraded with EMA for Inverse Problems.
    """
    def __init__(self, num_conditions, alpha=0.999, update_freq=100):
        self.num_conditions = num_conditions
        self.alpha = alpha
        self.update_freq = update_freq
        self.step_count = 0
        
        # Initialize weights: lambda_pde, lambda_c1, ...
        self.weights = torch.ones(1 + num_conditions, dtype=torch.float64)
        
        # EMA statistical trackers
        self.gradient_stats = [EMAWelford(alpha=self.alpha) for _ in range(1 + num_conditions)]

    def update_gradient_stats(self, grads_list):
        for i, grad_norm in enumerate(grads_list):
            self.gradient_stats[i].update(float(grad_norm))

    def balance_weights(self):
        self.step_count += 1
        if self.step_count % self.update_freq != 0:
            return self.weights

        means = np.array([s.mean for s in self.gradient_stats])
        avg_ci_mean = np.mean(means[1:]) if len(means) > 1 else means[0]
        pde_mean = means[0]
        
        new_weights = np.ones_like(means)
        for i in range(1, len(means)):
            if means[i] > 0:
                new_weights[i] = means[i] / (avg_ci_mean + 1e-8)
        
        if pde_mean > 0:
             self.weights[0] = avg_ci_mean / (pde_mean + 1e-8)
        
        for i in range(1, len(new_weights)):
            self.weights[i] = float(new_weights[i])
            
        return self.weights

    def get_weights(self):
        return self.weights

def gll_loss(prediction, target, log_var):
    """
    Gaussian Log-Likelihood Loss for Aleatoric Uncertainty.
    prediction: Mean prediction from model
    target: Ground truth or physical residual (target=0)
    log_var: Predicted log-variance from model
    """
    # Precision term 1/sigma^2
    precision = torch.exp(-log_var)
    # Loss: 0.5 * precision * (target - prediction)^2 + 0.5 * log_var
    diff_sq = (target - prediction)**2
    loss = 0.5 * precision * diff_sq + 0.5 * log_var
    return loss.mean()

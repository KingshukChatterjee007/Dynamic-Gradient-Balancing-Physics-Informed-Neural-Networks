import torch
import copy
import random

class PINNGradientSurgery:
    """
    Directional Alignment Module (DAM) using Gradient Surgery (PCGrad).
    This fixes 'The Tug-of-War' in PINNs by projecting conflicting gradients 
    onto each other's normal planes.
    """
    def __init__(self, optimizer, use_gtn=False):
        self._optim = optimizer
        self.use_gtn = use_gtn
        self.task_norms = None # To be initialized as Welford objects if GTN is used

    def step(self, losses, weights=None):
        """
        Calculates gradients for each loss, applies surgery, and updates parameters.
        losses: List of loss components [L_pde, L_c1, L_c2, ...]
        weights: Weights for each component (e.g. from DBBalancer).
        """
        num_losses = len(losses)
        
        # Initialize GTN stats if needed
        if self.use_gtn and self.task_norms is None:
            from db_pinn_balancer import Welford
            self.task_norms = [Welford() for _ in range(num_losses)]

        grads = []
        # 1. Compute gradients for each loss individually
        for i, loss in enumerate(losses):
            self._optim.zero_grad()
            loss.backward(retain_graph=(i < num_losses - 1))
            
            grad = self._get_flat_grad()
            
            # Apply GTN: Normalize by running mean of gradient magnitude
            if self.use_gtn:
                gnorm = torch.norm(grad).item()
                self.task_norms[i].update(gnorm)
                # Rescale gradient to common magnitude (e.g. 1.0)
                scale = 1.0 / (self.task_norms[i].mean + 1e-8)
                grad *= scale
                
            if weights is not None:
                grad *= weights[i]
            grads.append(grad)

        # 2. Apply Gradient Surgery (PCGrad)
        indices = list(range(num_losses))
        random.shuffle(indices)
        
        reduced_grads = copy.deepcopy(grads)
        for i in indices:
            for j in indices:
                if i == j: continue
                dot_prod = torch.dot(reduced_grads[i], grads[j])
                if dot_prod < 0:
                    norm_sq = torch.dot(grads[j], grads[j]) + 1e-8
                    reduced_grads[i] -= (dot_prod / norm_sq) * grads[j]

        # 3. Aggregate
        final_grad = torch.stack(reduced_grads).sum(dim=0)
        
        # 4. Update
        self._set_flat_grad(final_grad)
        self._optim.step()

        with torch.no_grad():
             grad_magnitudes = [torch.norm(g) for g in grads]
             
        return grad_magnitudes

    def _get_flat_grad(self):
        """Helper to flatten all gradients into a single vector."""
        grads = []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    grads.append(torch.zeros_like(p.data).view(-1))
                else:
                    grads.append(p.grad.data.clone().view(-1))
        return torch.cat(grads)

    def _set_flat_grad(self, flat_grad):
        """Helper to restore flattened gradient back to parameters."""
        offset = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                num_el = p.numel()
                p.grad.data.copy_(flat_grad[offset:offset + num_el].view_as(p.grad.data))
                offset += num_el

if __name__ == "__main__":
    # Small sanity check
    # Let's say we have 2 parameters and 2 conflicting losses
    params = [torch.randn(2, requires_grad=True)]
    optimizer = torch.optim.SGD(params, lr=0.1)
    surgery = PINNGradientSurgery(optimizer)
    
    # Loss 1: wants p to be [1, 1]
    # Loss 2: wants p to be [-1, -1]
    # Standard sum: gradients cancel out.
    # Surgery: should project gradients to avoid complete cancellation.
    
    p = params[0]
    l1 = (p - torch.tensor([1.0, 1.0])).pow(2).sum()
    l2 = (p + torch.tensor([1.0, 1.0])).pow(2).sum()
    
    mag = surgery.step([l1, l2])
    print(f"Gradient Magnitudes: {mag}")
    print("Surgery executed.")

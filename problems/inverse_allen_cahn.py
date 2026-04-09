import torch
import torch.nn.functional as F

def softplus_epsilon(k):
    """
    Ensures epsilon is always positive using Softplus.
    k: Learnable parameter.
    """
    return F.softplus(k)

def inverse_ac_residual(model, x, t, epsilon):
    """
    Computes PDE residual for Allen-Cahn with trainable epsilon.
    u_t - epsilon * u_xx - (u - u^3) = 0
    """
    out = model(torch.cat([x, t], dim=1))
    u = out[:, 0:1] 
    # Safe Gradient Helper
    def grad_safe(q, var):
        g = torch.autograd.grad(q.sum(), var, create_graph=True, allow_unused=True)[0]
        return g if g is not None else torch.zeros_like(var)

    u_t = grad_safe(u, t)
    u_x = grad_safe(u, x)
    u_xx = grad_safe(u_x, x)
    
    residual = u_t - epsilon * u_xx - (u - u**3)
    return residual.pow(2).mean()

def snapshot_loss(model, snapshot_data):
    """
    L_data (Data Loss) for multiple temporal snapshots.
    snapshot_data: List of (x, t, u_noisy) tuples.
    """
    total_loss = 0.0
    for x, t, u_noisy in snapshot_data:
        u_pred = model(torch.cat([x, t], dim=1))
        total_loss += torch.mean((u_pred - u_noisy)**2)
    return total_loss / len(snapshot_data)

def generate_noisy_data(model_true, epsilon_true, snapshots=[0.1, 0.25, 0.5, 0.75, 1.0], n_pts=200, noise_lv=0.1, device='cpu'):
    """
    Generates synthetic 'experimental' data with Gaussian noise.
    """
    data = []
    for t_val in snapshots:
        x = torch.linspace(-1, 1, n_pts, dtype=torch.float64, device=device).view(-1, 1)
        t = torch.ones_like(x) * t_val
        
        with torch.no_grad():
            u_clean = model_true(torch.cat([x, t], dim=1))
            # Scale noise relative to the standard deviation of the current field snapshot
            noise = torch.randn_like(u_clean) * noise_lv * u_clean.std()
            u_noisy = u_clean + noise
            
        data.append((x.to(device), t.to(device), u_noisy.to(device)))
    return data

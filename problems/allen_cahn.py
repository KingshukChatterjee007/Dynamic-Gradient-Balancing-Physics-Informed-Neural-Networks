import torch

def allen_cahn_residual(u, x, t, epsilon=0.0001):
    """
    Computes the PDE residual for the Allen-Cahn equation.
    u_t - epsilon * u_xx - (u - u^3) = 0
    """
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    
    residual = u_t - epsilon * u_xx - (u - u**3)
    return residual.pow(2).mean()

def initial_condition_loss(model, x_points):
    """
    Initial condition: u(x, 0) = x^2 * cos(pi * x)
    (A typical benchmark IC for Allen-Cahn)
    """
    t_zeros = torch.zeros_like(x_points, requires_grad=True)
    coords = torch.cat([x_points, t_zeros], dim=1)
    u_pred = model(coords)
    
    u_exact = x_points**2 * torch.cos(torch.pi * x_points)
    return (u_pred - u_exact).pow(2).mean()

def boundary_condition_loss(model, t_points):
    """
    Boundary conditions: u(-1, t) = -1, u(1, t) = -1
    Periodic or Dirichlet boundaries. Here let's use Dirichlet.
    """
    # Boundary at x = -1
    x_neg = -torch.ones_like(t_points, requires_grad=True)
    coords_neg = torch.cat([x_neg, t_points], dim=1)
    u_neg = model(coords_neg)
    loss_neg = (u_neg + 1.0).pow(2).mean()
    
    # Boundary at x = 1
    x_pos = torch.ones_like(t_points, requires_grad=True)
    coords_pos = torch.cat([x_pos, t_points], dim=1)
    u_pos = model(coords_pos)
    loss_pos = (u_pos + 1.0).pow(2).mean()
    
    return loss_neg, loss_pos

def sample_domain(n_pde, n_ic, n_bc):
    """
    Samples points for training:
    - PDE residuals in the domain [ -1, 1 ] x [ 0, 1 ]
    - Initial conditions at t=0
    - Boundary conditions at x=-1 and x=1
    """
    # PDE Domain points
    x_pde = torch.rand(n_pde, 1) * 2 - 1
    t_pde = torch.rand(n_pde, 1)
    x_pde.requires_grad = True
    t_pde.requires_grad = True
    
    # IC points
    x_ic = torch.rand(n_ic, 1) * 2 - 1
    
    # BC points
    t_bc = torch.rand(n_bc, 1)
    
    return (x_pde, t_pde), x_ic, t_bc

import torch

def navier_stokes_residuals(model, x, y, re=100):
    """
    Computes 2D steady Navier-Stokes residuals.
    x, y: tensors of shape (N, 1) with requires_grad=True
    re: Reynolds number (can be a scalar or a differentiable tensor for inverse discovery)
    """
    coords = torch.cat([x, y], dim=1)
    out = model(coords)
    u, v, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    
    # Gradients
    def grad(q, input_var):
        g = torch.autograd.grad(q.sum(), input_var, create_graph=True, allow_unused=True)[0]
        return g if g is not None else torch.zeros_like(input_var)
    
    u_x = grad(u, x)
    u_y = grad(u, y)
    v_x = grad(v, x)
    v_y = grad(v, y)
    p_x = grad(p, x)
    p_y = grad(p, y)
    
    u_xx = grad(u_x, x)
    u_yy = grad(u_y, y)
    v_xx = grad(v_x, x)
    v_yy = grad(v_y, y)
    
    # Continuity
    res_c = u_x + v_y
    
    # Momentum
    re_inv = 1.0 / re
    res_u = (u * u_x + v * u_y) + p_x - re_inv * (u_xx + u_yy)
    res_v = (u * v_x + v * v_y) + p_y - re_inv * (v_xx + v_yy)
    
    return [res_u.pow(2).mean(), res_v.pow(2).mean(), res_c.pow(2).mean()]

def cylinder_bc_loss(model, n_bc=200):
    """
    Defines boundary losses for Flow Around a Cylinder.
    Returns a list of individual loss terms for surgery.
    """
    # 1. Inlet (x=0)
    y_in = torch.rand(n_bc // 4, 1) * 0.41
    x_in = torch.zeros_like(y_in)
    in_coords = torch.cat([x_in, y_in], dim=1)
    in_out = model(in_coords)
    l_in_u = (in_out[:, 0:1] - 1.0).pow(2).mean() # u=1
    l_in_v = (in_out[:, 1:2] - 0.0).pow(2).mean() # v=0
    
    # 2. Outlet (x=1.1)
    y_out = torch.rand(n_bc // 4, 1) * 0.41
    x_out = torch.ones_like(y_out) * 1.1
    out_coords = torch.cat([x_out, y_out], dim=1)
    out_out = model(out_coords)
    l_out_p = (out_out[:, 2:3] - 0.0).pow(2).mean() # p=0
    
    # 3. Walls (top/bottom)
    x_wall = torch.rand(n_bc // 4, 1) * 1.1
    y_wall = torch.cat([torch.zeros(n_bc // 8, 1), torch.ones(n_bc // 8, 1) * 0.41])
    wall_coords = torch.cat([x_wall, y_wall], dim=1)
    wall_out = model(wall_coords)
    l_wall_v = (wall_out[:, 1:2] - 0.0).pow(2).mean() # v=0 (No penetration)
    
    # 4. Cylinder No-Slip
    theta = torch.rand(n_bc // 2, 1) * 2 * torch.pi
    cx, cy, r = 0.2, 0.2, 0.05
    x_cyl = cx + r * torch.cos(theta)
    y_cyl = cy + r * torch.sin(theta)
    cyl_coords = torch.cat([x_cyl, y_cyl], dim=1)
    cyl_out = model(cyl_coords)
    l_cyl_u = (cyl_out[:, 0:1] - 0.0).pow(2).mean()
    l_cyl_v = (cyl_out[:, 1:2] - 0.0).pow(2).mean()
    
    return [l_in_u, l_in_v, l_out_p, l_wall_v, l_cyl_u, l_cyl_v]

def cylinder_mask(x, y):
    """Returns a mask where True means point is OUTSIDE the cylinder."""
    cx, cy, r_sq = 0.2, 0.2, 0.05**2
    dist_sq = (x - cx)**2 + (y - cy)**2
    return dist_sq > r_sq

def sample_domain_ns(n_pde=2000):
    # Sample within rectangle [0, 1.1] x [0, 0.41]
    x = torch.rand(n_pde, 1) * 1.1
    y = torch.rand(n_pde, 1) * 0.41
    
    mask = cylinder_mask(x, y)
    
    x = x[mask.view(-1)].clone().detach().requires_grad_(True)
    y = y[mask.view(-1)].clone().detach().requires_grad_(True)
    return x, y

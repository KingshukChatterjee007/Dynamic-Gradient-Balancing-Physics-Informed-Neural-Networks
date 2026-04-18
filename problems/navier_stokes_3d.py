import torch

def navier_stokes_3d_residuals(model, x, y, z, t, re=100, pr=0.71, ri=1.0):
    """
    Computes 3D Unsteady Navier-Stokes residuals with Boussinesq Thermal Coupling.
    x, y, z, t: tensors of shape (N, 1) with requires_grad=True
    re: Reynolds number
    pr: Prandtl number (Heat/Momentum diff ratio)
    ri: Richardson number (Buoyancy strength)
    """
    coords = torch.cat([x, y, z, t], dim=1)
    out = model(coords)
    
    # We expect 5 outputs: u, v, w, p, T
    u, v, w, p, T = out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4], out[:, 4:5]
    
    # Gradient helper to handle graph creation for 2nd order derivatives
    def grad(q, input_var):
        g = torch.autograd.grad(q.sum(), input_var, create_graph=True, allow_unused=True)[0]
        return g if g is not None else torch.zeros_like(input_var)
    
    # First order derivatives
    u_t, u_x, u_y, u_z = grad(u, t), grad(u, x), grad(u, y), grad(u, z)
    v_t, v_x, v_y, v_z = grad(v, t), grad(v, x), grad(v, y), grad(v, z)
    w_t, w_x, w_y, w_z = grad(w, t), grad(w, x), grad(w, y), grad(w, z)
    T_t, T_x, T_y, T_z = grad(T, t), grad(T, x), grad(T, y), grad(T, z)
    
    p_x, p_y, p_z = grad(p, x), grad(p, y), grad(p, z)
    
    # Second order spatial derivatives
    u_xx, u_yy, u_zz = grad(u_x, x), grad(u_y, y), grad(u_z, z)
    v_xx, v_yy, v_zz = grad(v_x, x), grad(v_y, y), grad(v_z, z)
    w_xx, w_yy, w_zz = grad(w_x, x), grad(w_y, y), grad(w_z, z)
    T_xx, T_yy, T_zz = grad(T_x, x), grad(T_y, y), grad(T_z, z)
    
    # 1. Continuity Equation: div(U) = 0
    res_c = u_x + v_y + w_z
    
    # 2. Momentum Equations (with Buoyancy Term)
    re_inv = 1.0 / re
    
    res_u = u_t + (u * u_x + v * u_y + w * u_z) + p_x - re_inv * (u_xx + u_yy + u_zz)
    res_v = v_t + (u * v_x + v * v_y + w * v_z) + p_y - re_inv * (v_xx + v_yy + v_zz)
    # Buoyancy acts in Z direction (w)
    res_w = w_t + (u * w_x + v * w_y + w * w_z) + p_z - re_inv * (w_xx + w_yy + w_zz) - ri * T
    
    # 3. Energy Equation
    alpha = re_inv / pr
    res_T = T_t + (u * T_x + v * T_y + w * T_z) - alpha * (T_xx + T_yy + T_zz)
    
    return [
        res_u.pow(2).mean(), 
        res_v.pow(2).mean(), 
        res_w.pow(2).mean(), 
        res_c.pow(2).mean(),
        res_T.pow(2).mean()
    ]

def sphere_bc_loss(model, n_bc=500, bounds=None):
    """
    Defines 3D boundary losses for Flow Around a Sphere.
    bounds: [[x_min, x_max], [y_min, y_max], [z_min, z_max], [t_min, t_max]]
    """
    if bounds is None:
        # Default domain: [0, 1.1] x [0, 0.41] x [0, 0.41] x [0, 1.0]
        bounds = [[0.0, 1.1], [0.0, 0.41], [0.0, 0.41], [0.0, 1.0]]
        
    device = next(model.parameters()).device
    
    def get_rand(n, b):
        return torch.rand(n, 1, device=device, dtype=torch.float64) * (b[1] - b[0]) + b[0]

    # 1. Inlet (x = x_min, u=1, v=0, w=0, T=0)
    t_in = get_rand(n_bc // 5, bounds[3])
    y_in = get_rand(n_bc // 5, bounds[1])
    z_in = get_rand(n_bc // 5, bounds[2])
    x_in = torch.ones_like(t_in) * bounds[0][0]
    in_coords = torch.cat([x_in, y_in, z_in, t_in], dim=1)
    in_out = model(in_coords)
    # Velocity (u=1, v=0, w=0) + Temperature (T=0)
    l_in = (in_out[:, 0:1] - 1.0).pow(2).mean() + in_out[:, 1:3].pow(2).mean() + in_out[:, 4:5].pow(2).mean()
    
    # 2. Outlet (x = x_max, p=0)
    t_out = get_rand(n_bc // 5, bounds[3])
    y_out = get_rand(n_bc // 5, bounds[1])
    z_out = get_rand(n_bc // 5, bounds[2])
    x_out = torch.ones_like(t_out) * bounds[0][1]
    out_coords = torch.cat([x_out, y_out, z_out, t_out], dim=1)
    out_out = model(out_coords)
    l_out_p = out_out[:, 3:4].pow(2).mean()
    
    # 3. Walls (y=min/max, z=min/max, v=0, w=0, T=0)
    t_wall = get_rand(n_bc // 5, bounds[3])
    x_wall = get_rand(n_bc // 5, bounds[0])
    y_wall = get_rand(n_bc // 5, bounds[1])
    z_wall = get_rand(n_bc // 5, bounds[2])
    wall_coords = torch.cat([x_wall, y_wall, z_wall, t_wall], dim=1)
    wall_out = model(wall_coords)
    l_wall = wall_out[:, 1:3].pow(2).mean() + wall_out[:, 4:5].pow(2).mean()
    
    # 4. Sphere No-Slip & Heated (at (0.2, 0.2, 0.2) with r=0.05, u=v=w=0, T=1.0)
    phi = torch.rand(n_bc // 2, 1, device=device, dtype=torch.float64) * 2 * torch.pi
    theta = torch.rand(n_bc // 2, 1, device=device, dtype=torch.float64) * torch.pi
    t_sphere = get_rand(n_bc // 2, bounds[3])
    
    cx, cy, cz, r = 0.2, 0.2, 0.2, 0.05
    x_s = cx + r * torch.sin(theta) * torch.cos(phi)
    y_s = cy + r * torch.sin(theta) * torch.sin(phi)
    z_s = cz + r * torch.cos(theta)
    
    sphere_coords = torch.cat([x_s, y_s, z_s, t_sphere], dim=1)
    sphere_out = model(sphere_coords)
    # Velocity (u=v=w=0) + Heated Source (T=1.0)
    l_sphere = sphere_out[:, 0:3].pow(2).mean() + (sphere_out[:, 4:5] - 1.0).pow(2).mean()
    
    return [l_in, l_out_p, l_wall, l_sphere]

def sphere_mask(x, y, z, t=None):
    """Returns a mask where True means point is OUTSIDE the sphere."""
    cx, cy, cz, r_sq = 0.2, 0.2, 0.2, 0.05**2
    dist_sq = (x - cx)**2 + (y - cy)**2 + (z - cz)**2
    return dist_sq > r_sq

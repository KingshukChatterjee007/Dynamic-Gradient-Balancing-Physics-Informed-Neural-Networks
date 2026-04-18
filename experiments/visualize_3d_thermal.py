import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Set project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinn_engine.model import PINN

def visualize_hybrid_thermal(model_path="pinn_hybrid_thermal_3d.pth"):
    device = torch.device('cpu')
    print(f"Loading model from {model_path}...")
    
    # 1. Reconstruct Model (Must match training params)
    model = PINN(in_features=4, hidden_features=128, hidden_layers=5, 
                 out_features=5, use_gnn=True)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # 2. Create high-resolution slice (XY plane at Z=0.2, T=1.0)
    nx, ny = 100, 100
    x = np.linspace(0.0, 1.1, nx)
    y = np.linspace(0.0, 0.41, ny)
    X, Y = np.meshgrid(x, y)
    
    # Flatten grid to coordinates [X, Y, Z=0.2, T=1.0]
    x_flat = X.flatten()[:, None]
    y_flat = Y.flatten()[:, None]
    z_flat = np.ones_like(x_flat) * 0.2
    t_flat = np.ones_like(x_flat) * 1.0 # Snapshot at end of simulation
    
    coords = torch.tensor(np.hstack([x_flat, y_flat, z_flat, t_flat]), dtype=torch.float64)
    
    # 3. Model Inference
    print("Running inference on 3D slice...")
    with torch.no_grad():
        out = model(coords)
    
    u, v, w, p, T = out[:, 0], out[:, 1], out[:, 2], out[:, 3], out[:, 4]
    vel_mag = torch.sqrt(u**2 + v**2 + w**2).reshape(nx, ny).numpy()
    T_map = T.reshape(nx, ny).numpy()
    
    # 4. Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot Velocity Magnitude
    cp1 = ax[0].contourf(X, Y, vel_mag, levels=50, cmap='viridis')
    fig.colorbar(cp1, ax=ax[0])
    ax[0].set_title("Velocity Magnitude (Slice @ Z=0.2)")
    ax[0].set_ylabel("Y")
    
    # Add sphere representation
    circle = plt.Circle((0.2, 0.2), 0.05, color='white', fill=True, label="Sphere")
    ax[0].add_patch(circle)
    
    # Plot Temperature Plume
    cp2 = ax[1].contourf(X, Y, T_map, levels=50, cmap='hot')
    fig.colorbar(cp2, ax=ax[1])
    ax[1].set_title("Temperature Plume (Thermal Buoyancy)")
    ax[1].set_xlabel("X (Streamwise)")
    ax[1].set_ylabel("Y")
    
    # Add sphere representation
    circle2 = plt.Circle((0.2, 0.2), 0.05, color='black', fill=True)
    ax[1].add_patch(circle2)
    
    plt.tight_layout()
    output_img = "thermal_simulation_result.png"
    plt.savefig(output_img)
    print(f"Visualization saved to {output_img}")
    plt.show()

if __name__ == "__main__":
    visualize_hybrid_thermal()

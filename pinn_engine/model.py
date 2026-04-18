import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """
    SIREN activation layer as proposed in Sitzmann et al., 2020.
    Excellent for capturing high-frequency details and stable derivatives.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30.):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

from .gnn_layers import PointGNNLayer

class PINN(nn.Module):
    """
    Generic PINN Architecture with SIREN layers, Uncertainty Support, and GNN Option.
    Args:
        in_features: Input dimensions (e.g. 4 for x, y, z, t)
        hidden_features: Neurons per hidden layer
        hidden_layers: Number of hidden layers
        out_features: Output dimensions (e.g. 5 for u, v, w, p, T)
        outer_omega_0: SIREN fundamental frequency parameter
        probabilistic: If True, model outputs 2*out_features (mean + log_var)
        dropout_rate: Rate for MC-Dropout (0.0 to disable)
        use_gnn: If True, integrates a spatial GNN layer for local awareness
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, 
                 outer_omega_0=30., probabilistic=False, dropout_rate=0.0, use_gnn=False):
        super().__init__()
        self.probabilistic = probabilistic
        self.out_features = out_features
        self.dropout_rate = dropout_rate
        self.use_gnn = use_gnn
        
        # SIREN Backbone
        self.layers = nn.ModuleList()
        # Input Layer
        self.layers.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=outer_omega_0))
        
        # Hidden Layers
        for i in range(hidden_layers):
            self.layers.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=outer_omega_0))

        # Optional GNN Branch
        if use_gnn:
            self.gnn = PointGNNLayer(hidden_features, hidden_features, k=8)
        
        # Output Head
        actual_out = out_features if not probabilistic else 2 * out_features
        self.final_linear = nn.Linear(hidden_features, actual_out)
        
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / outer_omega_0, 
                                             np.sqrt(6 / hidden_features) / outer_omega_0)

    def forward(self, coords):
        # coords shape: (batch, in_features)
        x = coords
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.dropout_rate > 0:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Inject GNN context halfway through the network
            if self.use_gnn and i == len(self.layers) // 2:
                x = self.gnn(x, coords)
                
        return self.final_linear(x)

if __name__ == "__main__":
    # Quick test
    model = PINN(in_features=2, hidden_features=64, hidden_layers=3, out_features=3)
    x = torch.randn(10, 2)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape} (u, v, p)")
    print("Multi-output Model initialized successfully.")

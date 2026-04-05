import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """
    SIREN activation layer as proposed in Sitzmann et al., 2020.
    Excellent for capturing high-frequency details and stable derivatives.
    """
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
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

class PINN(nn.Module):
    """
    Generic PINN Architecture with SIREN layers.
    Args:
        in_features: Input dimensions (e.g. 2 for x, t)
        hidden_features: Neurons per hidden layer
        hidden_layers: Number of hidden layers
        out_features: Output dimensions (e.g. 1 for u)
        omega_0: SIREN fundamental frequency parameter
    """
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outer_omega_0=30):
        super().__init__()
        
        self.net = []
        # Input Layer
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=outer_omega_0))

        # Hidden Layers
        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=outer_omega_0))

        # Output Layer (Linear for regression)
        final_linear = nn.Linear(hidden_features, out_features)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / outer_omega_0, 
                                         np.sqrt(6 / hidden_features) / outer_omega_0)
            
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        # coords shape: (batch, in_features)
        return self.net(coords)

if __name__ == "__main__":
    # Quick test
    model = PINN(in_features=2, hidden_features=64, hidden_layers=3, out_features=3)
    x = torch.randn(10, 2)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape} (u, v, p)")
    print("Multi-output Model initialized successfully.")

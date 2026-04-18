import torch
import torch.nn as nn
import torch.nn.functional as F

class PointGNNLayer(nn.Module):
    """
    Lightweight, vectorized Graph Layer for PINNs.
    Aggregates information from K-Nearest Neighbors to provide local spatial context.
    """
    def __init__(self, in_features, out_features, k=8):
        super().__init__()
        self.k = k
        self.in_features = in_features
        self.out_features = out_features
        
        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(in_features * 2 + 4, out_features), # [self_feat, neigh_feat, rel_pos]
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        
        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(in_features + out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )

    def forward(self, x, coords):
        """
        x: Feature tensor (N, in_features)
        coords: Spatial coordinates (N, 4) [x, y, z, t]
        """
        n = x.size(0)
        
        # 1. Faster KNN using cdist (Spatial distance only, ignoring time for local structure)
        # We use xyz for distance
        dist = torch.cdist(coords[:, :3], coords[:, :3]) 
        
        # Get indices of top k neighbors
        _, indices = torch.topk(dist, self.k, largest=False) # (N, k)
        
        # 2. Gather neighbor features and relative coordinates
        # neigh_x: (N, k, in_features)
        neigh_x = x[indices] 
        # neigh_coords: (N, k, 4)
        neigh_coords = coords[indices] 
        # rel_coords: (N, k, 4)
        rel_coords = neigh_coords - coords.unsqueeze(1)
        
        # Repeat self features: (N, k, in_features)
        self_x_rep = x.unsqueeze(1).repeat(1, self.k, 1)
        
        # 3. Message Passing
        # Concatenate: [self, neighbor, relative_pos] -> (N, k, 2*in + 4)
        msg_input = torch.cat([self_x_rep, neigh_x, rel_coords], dim=-1)
        messages = self.message_mlp(msg_input) # (N, k, out)
        
        # Aggregate (Max pooling) -> (N, out)
        aggr = torch.max(messages, dim=1)[0]
        
        # 4. Update
        updated = self.update_mlp(torch.cat([x, aggr], dim=-1))
        
        return updated

class PINNGraphBackbone(nn.Module):
    """
    Wraps the GNN layers for integration into the PINN model.
    """
    def __init__(self, hidden_features, k=8):
        super().__init__()
        self.gnn = PointGNNLayer(hidden_features, hidden_features, k=k)

    def forward(self, features, coords):
        return self.gnn(features, coords)

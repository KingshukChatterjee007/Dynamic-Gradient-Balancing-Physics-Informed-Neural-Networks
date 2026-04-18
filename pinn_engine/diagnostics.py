import torch
import numpy as np
from sklearn.cluster import KMeans

class PINNDiagnostics:
    """
    Scientific Diagnostic Tools for PINNs.
    Includes Active Learning (Sensor Placement) and Uncertainty Mapping.
    """
    def __init__(self, model):
        self.model = model

    def get_optimal_sensor_locations(self, coords, residuals, n_sensors=10):
        """
        Active Learning: Suggests optimal sensor placement indices by clustering 
        regions with highest physical residuals.
        """
        # 1. Identify "Hard" regions (top 20% of residuals)
        res_values = residuals.detach().cpu().numpy()
        threshold = np.percentile(res_values, 80)
        high_res_mask = res_values >= threshold
        
        high_res_coords = coords.detach().cpu().numpy()[high_res_mask]
        
        # 2. Cluster these regions to find $N$ distinct locations
        if len(high_res_coords) < n_sensors:
             return high_res_coords
             
        kmeans = KMeans(n_clusters=n_sensors, random_state=42, n_init=10)
        kmeans.fit(high_res_coords)
        
        # Returns the centroid coordinates of the higher-residual clusters
        return kmeans.cluster_centers_

    def predict_with_uncertainty(self, sub_coords, n_samples=20):
        """
        Epistemic Uncertainty via MC-Dropout.
        Runs multiple forward passes with dropout enabled.
        """
        preds = []
        
        # Enable Dropout during inference
        self.model.train() 
        
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.model(sub_coords)
                
                # If model is probabilistic, we only take the mean head for epigstemic UQ
                if self.model.probabilistic:
                    out = out[:, :self.model.out_features]
                    
                preds.append(out.unsqueeze(0))
        
        preds = torch.cat(preds, dim=0) # [Samples, N, Dims]
        
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        
        return mean, std

    def get_confidence_metrics(self, coords):
        """
        Returns combined Aleatoric (Data) and Epistemic (Model) uncertainty.
        Only works if model was initialized with probabilistic=True.
        """
        if not self.model.probabilistic:
             raise ValueError("Model must be probabilistic to compute Confidence Metrics.")
             
        self.model.eval()
        with torch.no_grad():
            out = self.model(coords)
            mean = out[:, :self.model.out_features]
            log_var = out[:, self.model.out_features:]
            aleatoric_std = torch.exp(0.5 * log_var)
            
        return mean, aleatoric_std

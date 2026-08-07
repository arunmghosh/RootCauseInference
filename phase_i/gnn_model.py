import torch
import torch.nn as nn
import torch.nn.functional as F

class BipartiteConv(nn.Module):
    """
    Custom message passing convolution layer for Bipartite Graphs (Factories <-> Stores).
    """
    def __init__(self, in_fac_dim, in_store_dim, in_edge_dim, out_fac_dim, out_store_dim):
        super().__init__()
        # Factory -> Store messaging
        self.fac_proj = nn.Linear(in_fac_dim, out_store_dim)
        self.edge_proj1 = nn.Linear(in_edge_dim, out_store_dim)
        self.store_update = nn.Linear(in_store_dim + out_store_dim, out_store_dim)
        
        # Store -> Factory messaging
        self.store_proj = nn.Linear(out_store_dim, out_fac_dim)
        self.edge_proj2 = nn.Linear(in_edge_dim, out_fac_dim)
        self.fac_update = nn.Linear(in_fac_dim + out_fac_dim, out_fac_dim)
        
    def forward(self, h_fac, h_store, edge_index_F, edge_index_S, edge_features):
        """
        Parameters:
        - h_fac: factory node embeddings, shape (N_total, in_fac_dim)
        - h_store: store node embeddings, shape (M_total, in_store_dim)
        - edge_index_F: indices of factories in edges, shape (E_total,)
        - edge_index_S: indices of stores in edges, shape (E_total,)
        - edge_features: edge properties, shape (E_total, in_edge_dim)
        """
        N = h_fac.size(0)
        M = h_store.size(0)
        
        # --- 1. Factory to Store Message Passing ---
        proj_fac = self.fac_proj(h_fac)  # (N, out_store_dim)
        proj_edge1 = self.edge_proj1(edge_features)  # (E, out_store_dim)
        
        # Gather factory features for each edge
        edge_msg1 = F.relu(proj_fac[edge_index_F] + proj_edge1)  # (E, out_store_dim)
        
        # Aggregate messages at store nodes using index_add
        agg_store_msg = torch.zeros(M, proj_fac.size(1), device=h_fac.device)
        agg_store_msg.index_add_(0, edge_index_S, edge_msg1)  # (M, out_store_dim)
        
        # Update store representations
        h_store_new = F.relu(self.store_update(torch.cat([h_store, agg_store_msg], dim=-1)))
        
        # --- 2. Store to Factory Message Passing ---
        proj_store = self.store_proj(h_store_new)  # (M, out_fac_dim)
        proj_edge2 = self.edge_proj2(edge_features)  # (E, out_fac_dim)
        
        # Gather store features for each edge
        edge_msg2 = F.relu(proj_store[edge_index_S] + proj_edge2)  # (E, out_fac_dim)
        
        # Aggregate messages at factory nodes using index_add
        agg_fac_msg = torch.zeros(N, proj_store.size(1), device=h_fac.device)
        agg_fac_msg.index_add_(0, edge_index_F, edge_msg2)  # (N, out_fac_dim)
        
        # Update factory representations
        h_fac_new = F.relu(self.fac_update(torch.cat([h_fac, agg_fac_msg], dim=-1)))
        
        return h_fac_new, h_store_new

class BipartiteGNN(nn.Module):
    """
    Stackable bipartite Graph Neural Network for factory quality control inference.
    """
    def __init__(self, fac_in_dim=3, store_in_dim=3, edge_in_dim=2, hidden_dim=64, num_layers=2):
        super().__init__()
        # Input projects
        self.fac_embed = nn.Linear(fac_in_dim, hidden_dim)
        self.store_embed = nn.Linear(store_in_dim, hidden_dim)
        
        # Message passing layers
        self.layers = nn.ModuleList([
            BipartiteConv(hidden_dim, hidden_dim, edge_in_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        # Regression head (predicts continuous quality in [0, 1])
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Classification head (predicts probability of failure in [0, 1])
        self.fail_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_fac, x_store, edge_index_F, edge_index_S, edge_features):
        """
        Forward pass.
        Returns:
        - est_quality: tensor of shape (N_total,)
        - fail_prob: tensor of shape (N_total,)
        """
        # Embed initial features
        h_fac = F.relu(self.fac_embed(x_fac))
        h_store = F.relu(self.store_embed(x_store))
        
        # Message passing
        for layer in self.layers:
            h_fac, h_store = layer(h_fac, h_store, edge_index_F, edge_index_S, edge_features)
            
        # Predict factory properties
        est_quality = self.quality_head(h_fac).squeeze(-1)
        fail_prob = self.fail_head(h_fac).squeeze(-1)
        
        return est_quality, fail_prob

class FactoryMLP(nn.Module):
    """
    Multi-Layer Perceptron baseline using local factory-level features.
    """
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.fail_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        h = self.net(x)
        est_quality = self.quality_head(h).squeeze(-1)
        fail_prob = self.fail_head(h).squeeze(-1)
        return est_quality, fail_prob

if __name__ == '__main__':
    # Simple test of GNN and MLP shapes
    model = BipartiteGNN()
    print("GNN instantiated successfully:")
    print(model)
    
    mlp = FactoryMLP()
    print("MLP instantiated successfully:")
    print(mlp)
    
    # Generate dummy batch
    N, M, E = 10, 5, 15
    dummy_x_fac = torch.randn(N, 3)
    dummy_x_store = torch.randn(M, 3)
    dummy_edge_index_F = torch.randint(0, N, (E,))
    dummy_edge_index_S = torch.randint(0, M, (E,))
    dummy_edge_features = torch.randn(E, 2)
    
    est_q, fail_p = model(dummy_x_fac, dummy_x_store, dummy_edge_index_F, dummy_edge_index_S, dummy_edge_features)
    print("GNN Output Shapes:")
    print("Estimated Quality:", est_q.shape)
    print("Failure Probability:", fail_p.shape)
    
    dummy_mlp_in = torch.randn(N, 4)
    est_q_mlp, fail_p_mlp = mlp(dummy_mlp_in)
    print("MLP Output Shapes:")
    print("Estimated Quality:", est_q_mlp.shape)
    print("Failure Probability:", fail_p_mlp.shape)

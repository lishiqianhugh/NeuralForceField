import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """Graph Convolutional Layer implementation"""
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.self_transform = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        
    def forward(self, x, adj):
        batch_size, num_nodes = adj.shape[0], adj.shape[1]
        
        adj_sum = torch.sum(adj, dim=-1, keepdim=True) + 1e-8
        norm_adj = adj / adj_sum
        
        # Process self features and neighbor aggregated features separately
        self_features = self.self_transform(x)
        neighbor_features = torch.bmm(norm_adj, self.linear(x))
        
        # Combine both feature parts
        output = self_features + neighbor_features
        return self.norm(output)
    

class GCNetwork(nn.Module):
    """Multi-layer GCN network"""
    def __init__(self, feature_dim, hidden_dim, num_layers):
        super(GCNetwork, self).__init__()
        
        # Build network using improved GCN layers
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(feature_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        self.layers.append(GCNLayer(hidden_dim, hidden_dim))
        
        # Final output layer, similar to IN's node MLP
        self.output_layer = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
        
    def forward(self, states, adj):
        bs, body_num, _ = states.shape
        
        x = states
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x, adj))
        x = self.layers[-1](x, adj)
        
        combined = torch.cat([states, x], dim=-1)
        update = self.output_layer(combined)
        
        return update

class GCN(nn.Module):
    def __init__(self, feature_dim, hidden_dim, num_layers):
        super(GCN, self).__init__()
        self.gcn = GCNetwork(feature_dim, hidden_dim, num_layers)
        
        # Add node update network, similar to IN's node MLP
        self.node_update = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )
    
    def compute_adjacency(self, states):
        """Improved adjacency matrix computation, closer to IN's interaction pattern"""
        bs, body_num, _ = states.shape
        
        # Extract position information
        positions = states[:, :, :3]
        
        # Calculate pairwise distances
        pos_i = positions.unsqueeze(2)  # [bs, body_num, 1, 3]
        pos_j = positions.unsqueeze(1)  # [bs, 1, body_num, 3]
        r_ij = pos_i - pos_j  # [bs, body_num, body_num, 3]
        dist = torch.sqrt(torch.sum(r_ij ** 2, dim=-1) + 1e-8)  # [bs, body_num, body_num]
        
        adj = torch.exp(-dist / 5.0)
        
        # Remove self-connections
        eye = torch.eye(body_num, device=states.device).unsqueeze(0).expand(bs, -1, -1)
        adj = adj * (1 - eye)
        
        return adj

    def forward(self, initial_state, time_points):
        bs, body_num, feature_dim = initial_state.shape
        steps = len(time_points)
        
        trajectory = torch.zeros(bs, steps, body_num, feature_dim, device=initial_state.device)
        trajectory[:, 0] = initial_state
        
        for t in range(1, steps):
            current_state = trajectory[:, t - 1]
            
            adj = self.compute_adjacency(current_state)
            interaction_features = self.gcn(current_state, adj)
            
            combined = torch.cat([current_state, interaction_features], dim=-1)
            update = self.node_update(combined)
            
            trajectory[:, t] = current_state + update / 10.0
        
        return trajectory
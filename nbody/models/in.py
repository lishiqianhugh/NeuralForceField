import torch
import torch.nn as nn

class GNN(nn.Module):
    def __init__(self, interaction_feature_dim, hidden_dim, num_layers):
        super(GNN, self).__init__()
        
        # Construct edge MLP
        edge_layers = []
        input_dim = 2 * 4
        for i in range(num_layers-1):
            edge_layers.append(nn.Linear(input_dim, hidden_dim))
            edge_layers.append(nn.ReLU())
            input_dim = hidden_dim
        edge_layers.append(nn.Linear(input_dim, interaction_feature_dim)) # fx, fy, fz
        self.edge_mlp = nn.Sequential(*edge_layers)
    
        # Construct node MLP
        node_layers = []
        input_dim = 7 + interaction_feature_dim
        for _ in range(num_layers):
            node_layers.append(nn.Linear(input_dim, hidden_dim))
            node_layers.append(nn.ReLU())
            input_dim = hidden_dim
        node_layers.append(nn.Linear(input_dim, 7))  # Final layer to output feature_dim
        self.node_mlp = nn.Sequential(*node_layers)
    
    def forward(self, states):
        bs, body_num, feature_dim = states.shape
        
        # Compute pairwise interactions
        states_exp = states.unsqueeze(2).repeat(1, 1, body_num, 1)  # Expand for pairwise # [bs, body_num, body_num, feature_dim]
        states_pair = torch.cat([states_exp[:,:,:,:4], states_exp.transpose(1, 2)[:,:,:,:4]], dim=-1)
        
        # Process interactions through edge MLP
        interactions = self.edge_mlp(states_pair)
        interactions = interactions.sum(dim=2)  # Aggregate interactions per body
        
        # Update body features using node MLP
        updated_states = self.node_mlp(torch.cat([states, interactions], dim=-1))
        
        return updated_states

class InteractionNetwork(nn.Module):
    def __init__(self, interaction_feature_dim, hidden_dim, num_layers):
        super(InteractionNetwork, self).__init__()
        self.interaction_net = GNN(interaction_feature_dim, hidden_dim, num_layers)
    
    def forward(self, initial_state, time_points):
        bs, body_num, feature_dim = initial_state.shape
        steps = len(time_points)
        
        # Initialize trajectory tensor
        trajectory = torch.zeros(bs, steps, body_num, feature_dim, device=initial_state.device)
        trajectory[:, 0] = initial_state
        
        # Predict future states
        for t in range(1, steps):
            current_state = trajectory[:, t - 1]
            res = self.interaction_net(current_state) / 10
            trajectory[:, t] = current_state + res
        
        return trajectory
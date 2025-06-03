import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        """
        Args:
            x: [B, N, F] node features
            adj: [B, N, N] adjacency matrix
        Returns:
            output: [B, N, out_features] updated node features
        """
        adj_sum = adj.sum(dim=-1, keepdim=True) + 1e-8  # [B, N, 1]
        adj_norm = adj / adj_sum

        support = torch.bmm(adj_norm, x)  # [B, N, F]
        output = self.linear(support)
        
        return output


class GCNEncoder(nn.Module):
    def __init__(self, object_dim=12, effect_dim=200, num_layers=2, hidden_dim=256, 
                 use_dist_mask=True, history_len=1, angle_scale=1e1):
        super(GCNEncoder, self).__init__()
        self.use_dist_mask = use_dist_mask
        self.history_len = history_len
        self.object_dim = object_dim
        self.angle_scale = angle_scale
        self.num_layers = num_layers
        
        self.node_embedding = nn.Linear((5+3+2) * history_len, hidden_dim)
        
        self.gcn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gcn_layers.append(GraphConvLayer(
                hidden_dim if i == 0 else hidden_dim,
                hidden_dim
            ))
        
        self.output_linear = nn.Linear(hidden_dim, effect_dim)
    
    def compute_dist_mask(self, init_x_exp, query_x_exp):

        A1 = torch.cat([init_x_exp[...,0:1]-init_x_exp[...,2:3]/2*torch.cos(init_x_exp[...,3:4]*self.angle_scale), 
                        init_x_exp[...,1:2]-init_x_exp[...,2:3]/2*torch.sin(init_x_exp[...,3:4]*self.angle_scale)], dim=-1)
        B1 = torch.cat([init_x_exp[...,0:1]+init_x_exp[...,2:3]/2*torch.cos(init_x_exp[...,3:4]*self.angle_scale),
                        init_x_exp[...,1:2]+init_x_exp[...,2:3]/2*torch.sin(init_x_exp[...,3:4]*self.angle_scale)], dim=-1)
        r1 = init_x_exp[..., 4].unsqueeze(-1)

        A2 = torch.cat([query_x_exp[...,0:1]-query_x_exp[...,2:3]/2*torch.cos(query_x_exp[...,3:4]*self.angle_scale),
                        query_x_exp[...,1:2]-query_x_exp[...,2:3]/2*torch.sin(query_x_exp[...,3:4]*self.angle_scale)], dim=-1)

        B2 = torch.cat([query_x_exp[...,0:1]+query_x_exp[...,2:3]/2*torch.cos(query_x_exp[...,3:4]*self.angle_scale),
                        query_x_exp[...,1:2]+query_x_exp[...,2:3]/2*torch.sin(query_x_exp[...,3:4]*self.angle_scale)], dim=-1)
        
        r2 = query_x_exp[..., 4].unsqueeze(-1)
        A1B1 = B1 - A1
        A2B2 = B2 - A2
        A1B1_squared = (A1B1 ** 2).sum(dim=-1, keepdim=True) + 1e-8
        A2B2_squared = (A2B2 ** 2).sum(dim=-1, keepdim=True) + 1e-8

        A1A2 = A2 - A1
        t_A2A1B1 = (A1A2 * A1B1).sum(dim=-1, keepdim=True) / A1B1_squared
        t_A2A1B1 = torch.clamp(t_A2A1B1, 0.0, 1.0)
        Q_A2A1B1 = A1 + t_A2A1B1 * A1B1
        distance_A2A1B1 = torch.norm(A2 - Q_A2A1B1, dim=-1, keepdim=True)

        A1B2 = B2 - A1
        t_B2A1B1 = (A1B2 * A1B1).sum(dim=-1, keepdim=True) / A1B1_squared
        t_B2A1B1 = torch.clamp(t_B2A1B1, 0.0, 1.0)
        Q_B2A1B1 = A1 + t_B2A1B1 * A1B1
        distance_B2A1B1 = torch.norm(B2 - Q_B2A1B1, dim=-1, keepdim=True)

        A2A1 = A1 - A2
        t_A1A2B2 = (A2A1 * A2B2).sum(dim=-1, keepdim=True) / A2B2_squared
        t_A1A2B2 = torch.clamp(t_A1A2B2, 0.0, 1.0)
        Q_A1A2B2 = A2 + t_A1A2B2 * A2B2
        distance_A1A2B2 = torch.norm(A1 - Q_A1A2B2, dim=-1, keepdim=True)

        A2B1 = B1 - A2
        t_B1A2B2 = (A2B1 * A2B2).sum(dim=-1, keepdim=True) / A2B2_squared
        t_B1A2B2 = torch.clamp(t_B1A2B2, 0.0, 1.0)
        Q_B1A2B2 = A2 + t_B1A2B2 * A2B2
        distance_B1A2B2 = torch.norm(B1 - Q_B1A2B2, dim=-1, keepdim=True)

        min_distance, min_index = torch.min(torch.stack([distance_A2A1B1, distance_B2A1B1, distance_A1A2B2, distance_B1A2B2], dim=-1), dim=-1)

        supporting_direction = torch.gather(torch.stack([A2 - Q_A2A1B1, B2 - Q_B2A1B1, A1 - Q_A1A2B2, B1 - Q_B1A2B2], dim=-2), dim=-2, index=min_index.unsqueeze(-1).repeat(1, 1, 1,1, 2))
        supporting_direction = supporting_direction.squeeze(-2)
        supporting_direction = supporting_direction / (torch.norm(supporting_direction, dim=-1, keepdim=True) + 1e-8)

        assert min_distance.min() >= 0, "Minimum distance must be non-negative"
        distance_to_capsule = min_distance - r1 - r2
        return distance_to_capsule, supporting_direction

    def forward(self, objects):
        """
        Args:
            objects: [B, obj_num, history_len, object_dim] Object states.
        Returns:
            node_features: [B, obj_num, effect_dim] 节点特征
        """
        batch_size, obj_num, history_len, object_dim = objects.shape
        device = objects.device
        
        # consturct adjacency matrix
        if self.use_dist_mask:
            obj_i = objects.unsqueeze(2).repeat(1, 1, obj_num, 1, 1)  # [B, obj_num, obj_num, history_len, object_dim]
            obj_j = objects.unsqueeze(1).repeat(1, obj_num, 1, 1, 1)  # [B, obj_num, obj_num, history_len, object_dim]
            
            distance_to_capsule, _ = self.compute_dist_mask(obj_i[:,:,:,-1,:], obj_j[:,:,:,-1,:])
            adj_matrix = (distance_to_capsule < 0)  # [B, obj_num, obj_num, 1]
            
            # add spring and joint
            have_spring = (obj_i[:,:,:,-1,8:9] == obj_j[:,:,:,-1,8:9]) & (obj_j[:,:,:,-1,8:9] > 0)
            have_joint = ((obj_i[:,:,:,-1,7:8].long() == obj_j[:,:,:,-1,7:8].long()) & (obj_i[:,:,:,-1,7:8] > 0))
            adj_matrix = adj_matrix | have_spring | have_joint
            adj_matrix = adj_matrix.squeeze(-1)  # [B, obj_num, obj_num]
            
            eye_mask = 1 - torch.eye(obj_num, device=device).unsqueeze(0)
            adj_matrix = adj_matrix * eye_mask
        else:
            adj_matrix = torch.ones(batch_size, obj_num, obj_num, device=device)
            adj_matrix = adj_matrix - torch.eye(obj_num, device=device).unsqueeze(0)
        
        # extract node features
        node_features = objects[:, :, :, [0,1,2,3,4,7,8,9,10,11]]
        node_features = node_features.reshape(batch_size, obj_num, history_len * (5+3+2))  # [B, obj_num, history_len * 10]
        
        node_features = self.node_embedding(node_features)  # [B, obj_num, hidden_dim]
        node_features = F.relu(node_features)
        
        for i in range(self.num_layers):
            node_features = self.gcn_layers[i](node_features, adj_matrix)
            node_features = F.relu(node_features)
        
        node_features = self.output_linear(node_features)  # [B, obj_num, effect_dim]
        
        return node_features


class GCNRollouter(nn.Module):
    def __init__(
        self,
        object_dim=12,
        num_layers=4,
        hidden_dim=256,
        history_len=1,
        use_dist_mask=True,
        angle_scale=1e1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_dist_mask = use_dist_mask
        self.object_dim = object_dim
        self.history_len = history_len

        self.gcn_encoder = GCNEncoder(
            object_dim=object_dim,
            effect_dim=hidden_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            history_len=history_len,
            use_dist_mask=use_dist_mask,
            angle_scale=angle_scale,
        )
        
        self.prediction_network = nn.Sequential(
            nn.Linear((5+3) + hidden_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers)],
            nn.Linear(hidden_dim, 6)
        )

    def forward(self, z0, pred_len, t):
        """Predict trajectories.
        Args:
            z0: [B, history_len, obj_num, object_dim+1] Initial state of all objects.
            pred_len: int Number of timesteps to predict.
            t: [B, pred_len] Time indices for each timestep.

        Returns:
            predicted_object: [B, pred_len, obj_num, object_dim+1]
        """
        B, history_len, obj_num, _ = z0.shape
        device = z0.device
        rollout_result = z0.clone()
        
        for i in range(pred_len):
            t_i = t[:, i].unsqueeze(-1).unsqueeze(-1).repeat(1, obj_num, self.object_dim+1)
            disappear_time = z0[:,0,:,12:13]
            disappear_or_not = (t_i < disappear_time).float()

            dynamic_mask = z0[:,0,:,6:7]

            input_objects = rollout_result[:,-history_len:,:,:]
            input_objects = input_objects.permute(0,2,1,3)  # [B, obj_num, history_len, object_dim]

            node_features = self.gcn_encoder(input_objects)  # [B, obj_num, hidden_dim]
            
            input_object_feature = input_objects[:,:,-1,:]  # [B, obj_num, object_dim]
            input_object_feature = input_object_feature[:,:,[0,1,2,3,4,9,10,11]]  # [B, obj_num, 8]

            x = torch.cat([input_object_feature, node_features], dim=-1)
            next_objects_delta = self.prediction_network(x)  # [B, obj_num, 6]
            
            next_objects = torch.cat([
                next_objects_delta[:,:,:2],                # x, y
                input_objects[:,:,-1,2:3],                # length
                next_objects_delta[:,:,2:3],              # theta
                input_objects[:,:,-1,4:9],                # radius, is_dynamic, joint_idx, spring_idx, mass
                next_objects_delta[:,:,3:6],              # vx, vy, angular_velocity
                input_objects[:,:,-1,12:13]               # disappear_time
            ], dim=-1)
            
            next_objects = next_objects * dynamic_mask + rollout_result[:,0,:,:] * (1 - dynamic_mask)
            next_objects = next_objects * disappear_or_not
            
            rollout_result = torch.cat([rollout_result, next_objects.unsqueeze(1)], dim=1)
        
        return rollout_result[:,history_len:,:,:]


class GCN(nn.Module):
    def __init__(
        self,
        history_len,
        num_layers=4,
        hidden_dim=512,
        use_dist_mask=True,
        angle_scale=1e1,
    ):
        super().__init__()
        self.history_len = history_len

        self.rollouter = GCNRollouter(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            use_dist_mask=use_dist_mask,
            angle_scale=angle_scale,
            history_len=history_len,
        )

    def forward(self, z0, disappear_time, t):
        """Predict trajectories.

        Args:
            z0: [B, obj_num, object_dim] Initial state of all objects.
            disappear_time: [B, obj_num, 1] Time for object disppeaar.
            t: [B, timesteps] Time indices for each timestep.

        Returns:
            predicted_trajectories: [B, timesteps, num_slots, 13]
        """
        B, timesteps = t.shape
        z0 = torch.cat([z0, disappear_time], dim=-1)  # [B, obj_num, object_dim+1]

        # Expand z0 to match timesteps for burn-in input
        z0_expanded = z0.unsqueeze(1).repeat(1, self.history_len, 1, 1)  # [B, history_len, obj_num, object_dim+1]

        # Prepare burn-in input based on initial state
        burn_in_input = z0_expanded

        # Predict for the remaining timesteps autoregressively
        pred_len = timesteps - 1
        predicted_slots = self.rollouter(burn_in_input, pred_len, t)  # [B, pred_len, obj_num, object_dim+1]

        # Concatenate burn-in input and predicted slots
        predicted_trajectories = torch.cat([burn_in_input[:,-1:], predicted_slots], dim=1)  # [B, timesteps, obj_num, object_dim+1]
        
        # Transform disappear_time to disappear_mask [B, timesteps, obj_num]
        disappear_time = disappear_time.squeeze(-1).unsqueeze(1) # [B, 1, obj_num]
        t = t.unsqueeze(-1)  # [B, timesteps, 1]
        disappear_mask = (t < disappear_time).float().unsqueeze(-1)  # [B, timesteps, obj_num, 1]
        predicted_trajectories = predicted_trajectories * disappear_mask

        return predicted_trajectories
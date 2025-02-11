
import torch
import torch.nn as nn

class InteractionEncoder(nn.Module):
    def __init__(self, object_dim=12, effect_dim=200, num_layers=4, hidden_dim=256,use_dist_mask=False,history_len=1,angle_scale=1e1):
        super(InteractionEncoder, self).__init__()
        self.mlp = []
        self.mlp.append(nn.Linear((5+3+2) * history_len * 2, hidden_dim))
        self.mlp.append(nn.ReLU())
        for _ in range(num_layers):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(hidden_dim, effect_dim))
        self.mlp = nn.Sequential(*self.mlp)

        self.use_dist_mask = use_dist_mask
        self.history_len = history_len
        self.object_dim = object_dim
        self.angle_scale = angle_scale
    
    def compute_dist_mask(self, init_x_exp, query_x_exp):
        """
        Args:
            init_x_exp: [batch, obj_num, 5] Initial object states.
            query_x_exp: [batch, obj_num, target_obj_num, 5] Query object states.
        
        Returns:
            distance_to_capsule: [batch, obj_num, target_obj_num, 1] Distance to the capsule.
            supporting_direction: [batch, obj_num, target_obj_num, 2] Supporting direction.
        """
        A1 = torch.cat([init_x_exp[...,0:1]-init_x_exp[...,2:3]/2*torch.cos(init_x_exp[...,3:4]*self.angle_scale), 
                        init_x_exp[...,1:2]-init_x_exp[...,2:3]/2*torch.sin(init_x_exp[...,3:4]*self.angle_scale)], dim=-1)
        B1 = torch.cat([init_x_exp[...,0:1]+init_x_exp[...,2:3]/2*torch.cos(init_x_exp[...,3:4]*self.angle_scale),
                        init_x_exp[...,1:2]+init_x_exp[...,2:3]/2*torch.sin(init_x_exp[...,3:4]*self.angle_scale)], dim=-1)
        r1 = init_x_exp[..., 4].unsqueeze(-1)  # [batch, obj_num, target_obj_num, 1]


        A2 = torch.cat([query_x_exp[...,0:1]-query_x_exp[...,2:3]/2*torch.cos(query_x_exp[...,3:4]*self.angle_scale),
                        query_x_exp[...,1:2]-query_x_exp[...,2:3]/2*torch.sin(query_x_exp[...,3:4]*self.angle_scale)], dim=-1)

        B2 = torch.cat([query_x_exp[...,0:1]+query_x_exp[...,2:3]/2*torch.cos(query_x_exp[...,3:4]*self.angle_scale),
                        query_x_exp[...,1:2]+query_x_exp[...,2:3]/2*torch.sin(query_x_exp[...,3:4]*self.angle_scale)], dim=-1)
        
        r2 = query_x_exp[..., 4].unsqueeze(-1)  # [batch, obj_num, target_obj_num, 1]

        A1B1 = B1 - A1 
        A2B2 = B2 - A2 
        A1B1_squared = (A1B1 ** 2).sum(dim=-1, keepdim=True) + 1e-8
        A2B2_squared = (A2B2 ** 2).sum(dim=-1, keepdim=True) + 1e-8

        # min((A2,A1B1),(B2,A1B1),(A1,A2B2),(B1,A2B2))
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
            objects: [B, obj_num,history_len, object_dim] Object states.
        Returns:
            effect_vectors: [B, obj_num, obj_num, effect_dim] Interaction effects.
        """
        batch_size, obj_num, history_len, object_dim = objects.size()
        device = objects.device
        
        # Repeat the objects tensor to create pairs
        current_objects = objects # [B, obj_num, history_len, object_dim]
        obj_i = current_objects.unsqueeze(2).repeat(1, 1, obj_num,1, 1) # [B, obj_num, obj_num, history_len, object_dim]
        obj_j = current_objects.unsqueeze(1).repeat(1, obj_num, 1,1, 1) # [B, obj_num, obj_num, history_len, object_dim]
        
        # Concatenate along the last dimension to form pairs
        obj_i_feature = obj_i[:,:,:,:,[0,1,2,3,4,7,8,9,10,11]].reshape(batch_size, obj_num, obj_num, history_len * (5+3+2)) # [B, obj_num, obj_num, history_len * 6]
        obj_j_feature = obj_j[:,:,:,:,[0,1,2,3,4,7,8,9,10,11]].reshape(batch_size, obj_num, obj_num, history_len * (5+3+2)) # [B, obj_num, obj_num, history_len * 6]

        pairs = torch.cat([obj_i_feature, obj_j_feature], dim=-1)
        
        # Reshape pairs to apply MLP
        pairs = pairs.reshape(-1, (5+3+2) * history_len * 2) # [B * obj_num * obj_num, history_len * 6 * 2]
        interactions = self.mlp(pairs)
        
        # Reshape interactions back to the batch structure
        interactions = interactions.reshape(batch_size, obj_num, obj_num, -1) # [B, obj_num, obj_num, effect_dim]
        
        # Zero out self-interactions
        mask = (torch.eye(obj_num, device=device).unsqueeze(0).unsqueeze(-1) == 0)
        interactions = interactions * mask
        
        if self.use_dist_mask:
            # Compute distance mask
            distance_to_capsule, _ = self.compute_dist_mask(obj_i[:,:,:,-1,:], obj_j[:,:,:,-1,:]) 
            distance_mask = (distance_to_capsule < 0) # [B, obj_num, obj_num, 1]
            have_spring = (obj_i[:,:,:,-1,8:9] == obj_j[:,:,:,-1,8:9]) & (obj_j[:,:,:,-1,8:9] > 0)
            have_joint = ((obj_i[:,:,:,-1,7:8].long() == obj_j[:,:,:,-1,7:8].long()) & (obj_i[:,:,:,-1,7:8] > 0))
            distance_mask = distance_mask | have_spring | have_joint
            interactions = interactions * distance_mask
        
        interactions = interactions.sum(dim=2)  # [B, obj_num, effect_dim]

        return interactions


class PredictionNetwork(nn.Module):
    def __init__(self, effect_dim=200, num_layers=4,hidden_dim=256):
        super(PredictionNetwork, self).__init__()
        # The input dimension now considers concatenated effects from all other objects
        self.mlp = []
        self.mlp.append(nn.Linear((5+3) + effect_dim, hidden_dim))
        self.mlp.append(nn.ReLU())
        for _ in range(num_layers):
            self.mlp.append(nn.Linear(hidden_dim, hidden_dim))
            self.mlp.append(nn.ReLU())
        self.mlp.append(nn.Linear(hidden_dim, 6))
        self.mlp = nn.Sequential(*self.mlp)
    
    def forward(self, objects, effects):
        """
        Args:
            objects: [B, obj_num, 6] Object states.
            effects: [B, obj_num, effect_dim] Interaction effects.
        Returns:
            next_objects: [B, obj_num, object_dim] Predicted object states.
        """
        x = torch.cat([objects, effects], dim=-1)
        x = self.mlp(x)
        return x


class INRollouter(nn.Module):
    def __init__(
        self,
        object_dim=12,
        num_layers=4,
        hidden_dim=256,
        history_len=1,
        use_dist_mask=False,
        angle_scale=1e1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_dist_mask = use_dist_mask
        self.object_dim = object_dim
        self.history_len = history_len

        self.interaction_encoder = InteractionEncoder(
            object_dim=object_dim,
            effect_dim=hidden_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            history_len=history_len,
            use_dist_mask=use_dist_mask,
            angle_scale=angle_scale,
        )
        self.prediction_network = PredictionNetwork(
            effect_dim=hidden_dim,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
        )


    def forward(self, z0, pred_len, t):
        """Predict trajectories.
        Args:
            z0: [B, history_len, obj_num, FEATURE_DIM+3+1] Initial state of all objects.
            pred_len: int Number of timesteps to predict.
            t: [B, pred_len] Time indices for each timestep.

        Returns:
            predicted_object: [B, pred_len, obj_num, object_dim+1]
        """
        B, history_len, obj_num, _ = z0.shape
        device = z0.device

        rollout_result = z0.clone()
        for i in range(pred_len):
            t_i = t[:, i].unsqueeze(-1).unsqueeze(-1).repeat(1, obj_num, self.object_dim+1)  # [B, obj_num, object_dim]
            disappear_time = z0[:,0,:,12:13] # [B, obj_num, 1]
            disappear_or_not = (t_i < disappear_time).float()  # [B, obj_num, object_dim]

            dynamic_mask = z0[:,0,:,6:7] # [B, obj_num, 1]

            input_objects = rollout_result[:,-history_len:,:,:]  # [B, history_len, obj_num, object_dim]
            input_objects = input_objects.permute(0,2,1,3) # [B, obj_num, history_len, object_dim]

            effect_vectors = self.interaction_encoder(input_objects) # [B, obj_num, effect_dim]
            input_object_feature = input_objects[:,:,-1,:] # [B, obj_num, object_dim]
            input_object_feature = input_object_feature[:,:,[0,1,2,3,4,9,10,11]] # [B, obj_num, 6]

            next_objects = self.prediction_network(input_object_feature, effect_vectors) # [B, obj_num, 6]
            next_objects = torch.cat([next_objects[:,:,:2],input_objects[:,:,-1,2:3],next_objects[:,:,2:3], \
                                      input_objects[:,:,-1,4:9],next_objects[:,:,3:6],input_objects[:,:,-1,12:13]],dim=-1) # [B, obj_num, object_dim+1]
            
            next_objects = next_objects * dynamic_mask + rollout_result[:,0,:,:] * (1 - dynamic_mask) # [B, obj_num, object_dim]
            next_objects = next_objects * disappear_or_not # [B, obj_num, object_dim+1]
            rollout_result = torch.cat([rollout_result, next_objects.unsqueeze(1)], dim=1)  # [B, i+1+history_len, obj_num, object_dim+1]
        
        return rollout_result[:,history_len:,:,:] # [B, pred_len, obj_num, object_dim+1]




class InteractionNetwork(nn.Module):
    def __init__(
        self,
        history_len,  # burn-in steps
        num_layers=4,
        hidden_dim=512,
        use_dist_mask=False,
        angle_scale=1e1,
    ):
        super().__init__()
        self.history_len = history_len

        self.rollouter = INRollouter(
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            use_dist_mask=use_dist_mask,
            angle_scale=angle_scale,
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
        predicted_slots =  self.rollouter(burn_in_input, pred_len, t)  # [B, pred_len, obj_num, object_dim+1]

        # Concatenate burn-in input and predicted slots
        predicted_trajectories = torch.cat([burn_in_input[:,-1:], predicted_slots], dim=1)  # [B, timesteps, obj_num, object_dim+1]
        
        # Transform disappear_time to disappear_mask [B, timesteps, obj_num]
        disappear_time = disappear_time.squeeze(-1).unsqueeze(1) # [B, 1, obj_num]
        t = t.unsqueeze(-1)  # [B, timesteps, 1]
        disappear_mask = (t < disappear_time).float().unsqueeze(-1)  # [B, timesteps, obj_num, 1]
        predicted_trajectories = predicted_trajectories * disappear_mask

        return predicted_trajectories
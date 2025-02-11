import torch
import torch.nn as nn
from torchdiffeq import odeint , odeint_adjoint

from utils.util import *
from configs.iphyre_configs import *

class ForceFieldPredictor(nn.Module):
    """
    Predict the interaction force between objects
    Input: Initial feature and current position of objects
    Output: Predicted force vector
    """
    def __init__(self, hidden_dim, output_layer, use_dist_mask=True, dist_boundary=0, use_dist_input=True, dist_input_scale=1e2, angle_scale=1e2):
        super(ForceFieldPredictor, self).__init__()
        self.use_dist_mask = use_dist_mask
        self.use_dist_input = use_dist_input
        self.dist_input_scale = dist_input_scale
        self.angle_scale = angle_scale
        self.dist_boundary = dist_boundary
        if self.use_dist_input:
            trunk_input_dim = 5+2+1+1
        else:
            trunk_input_dim = 5+2+1
        branch_input_dim = 5
        trunk_layers = []
        trunk_layers.append(nn.Linear(trunk_input_dim, hidden_dim)) # x_2,y_2,l_2,theta_2,r_2  v_x,v_y,angular_v
        for _ in range(output_layer):
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.trunk_net = nn.Sequential(*trunk_layers)

        branch_layers = []
        branch_layers.append(nn.Linear(branch_input_dim, hidden_dim)) # x_1,y_1,l_1,theta_1,r_1
        for _ in range(output_layer):
            branch_layers.append(nn.ReLU())
            branch_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net = nn.Sequential(*branch_layers)

        self.output_layer = nn.Linear(hidden_dim,3)

        self.spring_mlp = nn.Sequential(nn.Linear(1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)) 
        self.joint_mlp = nn.Sequential(nn.Linear(5+5+2+1+1+1, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)) 
    
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

        # Minimum distance min((A2,A1B1),(B2,A1B1),(A1,A2B2),(B1,A2B2))
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

        min_distance = torch.min(torch.min(distance_A2A1B1, distance_B2A1B1), torch.min(distance_A1A2B2, distance_B1A2B2))
        assert min_distance.min() >= 0, "Minimum distance must be non-negative"
        distance_to_capsule = min_distance - r1 - r2
        return distance_to_capsule
    
    def forward(self, init_x,query_x,init_v,query_v,init_angular_v,query_angular_v):
        """
        init_x: [batch, obj_num, 9] - objects applying force (x,y,l,theta,r)
        query_x: [batch, target_obj_num, 9] - objects receiving force (x,y,l,theta,r)
        init_v: [batch, target_obj_num, 2] - velocities of objects applying force
        query_v: [batch, obj_num, 2] - velocities of objects receiving force
        init_angular_v: [batch, target_obj_num, 1] - angular velocities of objects applying force
        query_angular_v: [batch, obj_num, 1] - angular velocities of objects receiving force

        return: [batch, obj_num, target_obj_num, 3] - forces applied to objects receiving force (fx,fy,ftheta)
        """
        batch_size, obj_num, _ = init_x.shape
        target_obj_num = query_x.shape[1]

        # expand to object pairs
        init_x_exp = init_x.unsqueeze(2).expand(-1, -1, target_obj_num, -1)  # [batch, obj_num, target_obj_num, 9]
        query_x_exp = query_x.unsqueeze(1).expand(-1, obj_num, -1, -1)  # [batch, obj_num, target_obj_num, 9]
        init_v_exp = init_v.unsqueeze(2).expand(-1, -1, target_obj_num, -1) # [batch, obj_num, target_obj_num, 2]
        query_v_exp = query_v.unsqueeze(1).expand(-1, obj_num, -1, -1) # [batch, obj_num, target_obj_num, 2]
        init_angular_v_exp = init_angular_v.unsqueeze(2).expand(-1, -1, target_obj_num, -1) # [batch, obj_num, target_obj_num, 1]
        query_angular_v_exp = query_angular_v.unsqueeze(1).expand(-1, obj_num, -1, -1) # [batch, obj_num, target_obj_num, 1]

        # apply relative position
        relative_x = (query_x_exp[...,:2] - init_x_exp[...,:2]).clone() # [batch, obj_num, target_obj_num, 2]
        query_x_exp = query_x_exp.clone()
        query_x_exp[..., :2] = relative_x

        init_x_exp = init_x_exp.clone()
        init_x_exp[..., :2] = 0

        # apply relative velocity
        query_v_exp = query_v_exp.clone()
        query_v_exp -= init_v_exp

        # apply relative angular velocity
        query_angular_v_exp = query_angular_v_exp.clone()
        query_angular_v_exp -= init_angular_v_exp

        # calculate distance mask
        if self.use_dist_mask:
            distance_to_capsule = self.compute_dist_mask(init_x_exp.clone(), query_x_exp.clone())
            dist_mask = (distance_to_capsule <= self.dist_boundary).float()
            dist_input = distance_to_capsule * dist_mask
            dist_input *= self.dist_input_scale
        # predict force
        branch_input =  torch.cat([init_x_exp[...,:5]], dim=-1)  # [batch, obj_num, target_obj_num, 5]
        if self.use_dist_input:
            trunk_input =  torch.cat([query_x_exp[...,:5], query_v_exp, query_angular_v_exp, dist_input], dim=-1)  # [batch, obj_num, target_obj_num, 5+2+1]
        else:
            trunk_input =  torch.cat([query_x_exp[...,:5], query_v_exp, query_angular_v_exp], dim=-1)  # [batch, obj_num, target_obj_num, 5+2+1]
        branch_input_flat = branch_input.reshape(batch_size * obj_num * target_obj_num, branch_input.shape[-1])  # [batch * obj_num * target_obj_num, 5]
        trunk_input_flat = trunk_input.reshape(batch_size * obj_num * target_obj_num, trunk_input.shape[-1])  # [batch * obj_num * target_obj_num, 5+2+1+1]
 
        branch_output = self.branch_net(branch_input_flat)
        trunk_output = self.trunk_net(trunk_input_flat)
        
        # force_flat = self.output_layer(nn.ReLU()(branch_output * trunk_output))
        force_flat = self.output_layer(branch_output * trunk_output)
        force = force_flat.reshape(batch_size, obj_num, target_obj_num, 3)  # [batch, obj_num, target_obj_num, 3]

        # spring force
        have_spring = ((init_x_exp[...,8:9] == query_x_exp[...,8:9]) & (init_x_exp[...,8:9] > 0)).float() # [batch, obj_num, target_obj_num, 1]
        direction_vector = (init_x_exp[...,:2] - query_x_exp[...,:2]) # [batch, obj_num, target_obj_num, 2] 
        direction_vector = direction_vector / (torch.norm(direction_vector, dim=-1, keepdim=True) + 1e-8)
        length = torch.norm(init_x_exp[...,:2] - query_x_exp[...,:2], dim=-1, keepdim=True) # [batch, obj_num, target_obj_num, 1]
        spring_force = self.spring_mlp(length)
        spring_force = spring_force.reshape(batch_size, obj_num, target_obj_num, 1)  # [batch, obj_num, target_obj_num, 3]
        # spring force direction
        spring_force = spring_force * direction_vector # [batch, obj_num, target_obj_num, 2]
        spring_force = torch.cat([spring_force,torch.zeros_like(spring_force[:,:,:,:1])],dim=-1) # [batch, obj_num, target_obj_num, 3]
        spring_force = spring_force * have_spring

        # joint force
        have_joint = ((init_x_exp[...,7:8].long() == query_x_exp[...,7:8].long()) & (init_x_exp[...,7:8] > 0)).float() # [batch, obj_num, target_obj_num, 1]
        joint_force = self.joint_mlp(torch.cat([init_x_exp[...,:5],query_x_exp[...,:5],query_v_exp,init_angular_v_exp,query_angular_v_exp,(init_x_exp[...,7:8]-init_x_exp[...,7:8].long().float())],dim=-1))
        joint_force = joint_force.reshape(batch_size, obj_num, target_obj_num, 2)  # [batch, obj_num, target_obj_num, 3]
        joint_force = joint_force * have_joint
        joint_force = torch.cat([joint_force,torch.zeros_like(joint_force[:,:,:,:1])],dim=-1) # [batch, obj_num, target_obj_num, 3]

        if self.use_dist_mask:
            force = force*dist_mask

        force = force + spring_force + joint_force

        return force, 0

class ODEFunc(nn.Module):
    """
    Define the dynamic equation of the system
    - Force field prediction
    - Acceleration calculation
    - State derivative calculation
    """

    def __init__(self, force_predictor, mass=0.1, dtheta_scale=1e2,acceleration_clip=0):
        super(ODEFunc, self).__init__()
        self.force_predictor = force_predictor
        self.mass = mass
        self.dtheta_scale = dtheta_scale
        self.acceleration_clip = acceleration_clip
        self.gravity = torch.tensor(1/60, device=self.force_predictor.spring_mlp[0].weight.device) # g=1/60  1/60 * 150 * 15 * mass = 3.75
        
    def forward(self, t, z):
        """
        Input:
            t: current time
            z: state [batch_size, obj_num, FEATURE_DIM + 3 + 1] 9 features + 2 velocities + 1 angular velocity + 1 disappear time
        Output:
            dzdt: state derivative [batch_size, obj_num, FEATURE_DIM + 3 + 1]
        """
        batch_size, obj_num, _ = z.shape 

        velocities = z[:, :, FEATURE_DIM:FEATURE_DIM+2]
        angular_v = z[:, :, FEATURE_DIM+2:FEATURE_DIM+3]
        dynamic_mask = z[:, :, FEATURE_DIM-3:FEATURE_DIM-2]

        disappear_time = z[:,:,FEATURE_DIM+3:]  # [batch_size, obj_num, 1]
        disappear_or_not = t < disappear_time.squeeze(-1) # [batch_size, obj_num]

        all_features = z[:,:,:FEATURE_DIM] # [batch_size, obj_num, FEATURE_DIM]

        pairwise_force, _ = self.force_predictor(
            init_x=all_features,
            query_x=all_features,
            init_v=velocities,
            query_v=velocities,
            init_angular_v=angular_v,
            query_angular_v=angular_v
        ) # [batch_size, obj_num, obj_num, 3] [i,j] means the force of object j received from object i

        mask = 1- torch.eye(obj_num, device=z.device).unsqueeze(0) # [1, obj_num, obj_num]
        mask = mask.unsqueeze(-1) # [1, obj_num, obj_num, 1]
        pairwise_force = pairwise_force * mask # [batch_size, obj_num, obj_num, 3]

        disappear_or_not = disappear_or_not.unsqueeze(-1).repeat(1, 1, obj_num) # [batch_size, obj_num, obj_num]
        pairwise_force = pairwise_force * disappear_or_not.unsqueeze(-1) # [batch_size, obj_num, obj_num, 3]
        pairwise_force /= self.mass

        dxdt = velocities * dynamic_mask
        dthetadt = angular_v * dynamic_mask

        acceleration = pairwise_force[:,:,:,:2].sum(dim=1)
        acceleration[:,:,1] += self.gravity / self.mass
        acceleration = acceleration * dynamic_mask

        if self.acceleration_clip > 0:
            acceleration = acceleration * (acceleration.abs() > self.acceleration_clip).float()

        dangvdt = pairwise_force[:,:,:,2:3].sum(dim=1) * self.dtheta_scale # [batch_size, obj_num, 1]
        # dangvdt = torch.clamp(dangvdt, -1000, 1000)
        dangvdt = dangvdt * dynamic_mask

        # set dangvdt = 0 if the object is a ball
        not_ball_mask = (all_features[:,:,2:3].abs() > 1e-5).float()
        dangvdt *= not_ball_mask

        dzdt = torch.cat([
            dxdt,  # derivative of position
            torch.zeros_like(z[:,:,2:3]),  # keep length unchanged
            dthetadt,  # derivative of angle
            torch.zeros_like(z[:,:,4:FEATURE_DIM]),  # keep other features unchanged
            acceleration,
            dangvdt, # angular velocity
            torch.zeros_like(disappear_time)  # keep disappear time unchanged
        ], dim=-1)

        return dzdt

class NeuralODEModel(nn.Module):
    def __init__(self, ode_func, use_adjoint=False, step_size=1/1200):
        super(NeuralODEModel, self).__init__()
        self.ode_func = ode_func
        self.use_adjoint = use_adjoint
        self.step_size = step_size
    
    def forward(self, z0, disappear_time, t):
        # z0: [batch_size, obj_num, FEATURE_DIM+3]
        # disappear_time: [batch_size, obj_num, 1]
        # t: [batch_size,time_steps]
        # return: [time_steps, batch_size, obj_num, FEATURE_DIM]

        t = t[0,:] # [time_steps]

        ts = t.shape[0]
        z0 = torch.cat([z0, disappear_time], dim=-1)  # [batch_size, obj_num, FEATURE_DIM + 3 + 1]
        if self.use_adjoint:
            res = odeint_adjoint(self.ode_func, z0, t,method='euler',options={'step_size':self.step_size})
        else:
            res = odeint(self.ode_func, z0, t,method='euler',options={'step_size':self.step_size})  # [time_steps, batch_size, obj_num, FEATURE_DIM+3+1]
        res = res.permute(1,0,2,3) # [batch_size, time_steps, obj_num, FEATURE_DIM+3+1]

        # apply disappear mask to trajectories
        range_tensor = torch.arange(ts, device=z0.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float() # [1, time_steps, 1, 1]
        range_tensor = range_tensor / 10
        disappear_mask = (range_tensor < disappear_time.unsqueeze(1)) # [batch_size, time_steps, obj_num, 1]
        res = res * disappear_mask # [batch_size, time_steps, obj_num, FEATURE_DIM+3+1]
        return res
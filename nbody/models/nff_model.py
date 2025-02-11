from torchdiffeq import odeint
import torch.nn as nn
import torch

class ForceFieldPredictor(nn.Module):
    def __init__(self, layer_num, feature_dim, hidden_dim):
        super(ForceFieldPredictor, self).__init__()
        trunk_layers = []
        trunk_layers.append(nn.Linear(4, hidden_dim)) # m1, x1, y1, z1, vx1, vy1, vz1
        for _ in range(layer_num):
            trunk_layers.append(nn.ReLU())
            trunk_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.trunk_net = nn.Sequential(*trunk_layers)
        branch_layers = []
        branch_layers.append(nn.Linear(1, hidden_dim)) # m2, x2, y2, z2, vx2, vy2, vz2
        for _ in range(layer_num):
            branch_layers.append(nn.ReLU())
            branch_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.branch_net = nn.Sequential(*branch_layers)

        self.output_layer = nn.Linear(hidden_dim,3)

    def forward(self, init, query):
        batch_size, obj_num, _ = init.shape
        target_obj_num = query.shape[1]
        
        # expand to object pairs
        init_x_exp = init.unsqueeze(2).expand(-1, -1, target_obj_num, -1)[...,:4]
        query_exp = query.unsqueeze(1).expand(-1, obj_num, -1, -1)[...,:4]
        # # relative position
        relative_pos = query_exp[...,1:4] - init_x_exp[...,1:4]
        # branch_input_flat = init_x_exp.reshape(batch_size * obj_num * target_obj_num, init_x_exp.shape[-1])
        # trunk_input_flat = query_exp.reshape(batch_size * obj_num * target_obj_num, query_exp.shape[-1])
        branch_output = self.branch_net(init_x_exp[...,:1])
        trunk_output = self.trunk_net(torch.cat([query_exp[...,:1], relative_pos], dim=-1))
        

        # branch_output = self.branch_net(branch_input_flat)
        # trunk_output = self.trunk_net(trunk_input_flat)

        force_flat = self.output_layer(trunk_output * branch_output)
        force = force_flat.reshape(batch_size, obj_num, target_obj_num, -1)
        return force

# Define the ODE function
class ODEFunc(nn.Module):
    def __init__(self, force_predictor):
        super(ODEFunc, self).__init__()
        self.force_predictor = force_predictor

    def forward(self, t, state):
        '''
        state:
        0: mass
        1: x
        2: y
        3: z
        4: vx
        5: vy
        6: vz
        '''
        dmassdt = torch.zeros_like(state[...,0:1])
        dpdt = state[...,4:7]
        pairwise_force = self.force_predictor(state, state)
        # mask the self force
        mask = torch.eye(state.shape[1], device=state.device).unsqueeze(0).unsqueeze(-1)
        pairwise_force = pairwise_force * (1 - mask)
        # if mass is zero, set mass to 1000
        zero_mass_mask = state[...,0:1] == 0
        mass = state[...,0:1].clone()
        mass[zero_mass_mask] = 1000
        # only keep the pair with non-zero mass
        # pairwise_force [2250, 4, 4, 3] zero_mass_mask [2250, 4, 1]
        pairwise_force = pairwise_force * ~zero_mass_mask.unsqueeze(-1)
        pairwise_force = pairwise_force * ~zero_mass_mask.unsqueeze(1)
        dvdt = pairwise_force.sum(dim=1) / mass

        dzdt = torch.cat([
            dmassdt,  
            dpdt, 
            dvdt], dim=-1)

        return dzdt

# Neural ODE Model
class NeuralODE(nn.Module):
    def __init__(self, odefunc, step_size=1/200):
        super(NeuralODE, self).__init__()
        self.odefunc = odefunc
        self.step_size = step_size

    def forward(self, initial_state, time_points):
        return odeint(self.odefunc, 
                      initial_state, 
                      time_points, 
                    #   method='rk4'
                    # atol=1e-6, rtol=1e-6
                      method='euler',options={'step_size':self.step_size}
                      ).permute(1,0,2,3)
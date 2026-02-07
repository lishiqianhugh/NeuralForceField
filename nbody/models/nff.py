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

        branch_output = self.branch_net(init_x_exp[...,:1])
        trunk_output = self.trunk_net(torch.cat([query_exp[...,:1], relative_pos], dim=-1))
        
        force_flat = self.output_layer(trunk_output * branch_output)
        force = force_flat.reshape(batch_size, obj_num, target_obj_num, -1)
        return force

# Define the ODE function
class ODEFunc(nn.Module):
    def __init__(self, force_predictor):
        super(ODEFunc, self).__init__()
        self.force_predictor = force_predictor
        self.called_times = 0

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
        self.called_times += 1

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
    def __init__(self, odefunc, step_size=1/200,method='rk4',tol=1e-3):
        super(NeuralODE, self).__init__()
        self.odefunc = odefunc
        self.step_size = step_size
        self.method = method
        self.tol = tol

    def forward(self, initial_state, time_points):

        self.odefunc.called_times = 0

        if self.method == 'adaptive':
            res = odeint(self.odefunc, 
                      initial_state, 
                      time_points, 
                      atol=self.tol, rtol=self.tol
                      ).permute(1,0,2,3)
            # print(f"Integration method called times: {self.odefunc.called_times}")
            return res
        else:
            res = odeint(self.odefunc, 
                      initial_state, 
                      time_points, 
                      method=self.method,
                      options={'step_size':self.step_size}
                      ).permute(1,0,2,3)
            # print(f"Integration method called times: {self.odefunc.called_times}")
            return res
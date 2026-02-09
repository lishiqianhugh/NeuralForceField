# === eg nn.py (drop-in replacement) ==================================
import torch
import torch.nn as nn
FEATURE_DIM = 9
# ============================================================
#  SEGNO-style EGCL: layer-wise velocity-position integration
#  (core idea: replace direct coordinate update with a->v->x step-wise integration)
#  — following segno_model.py
# ============================================================
class SEGNO_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(),
                 coords_weight=1.0, n_layers=4, attention=False, tanh=False):
        super().__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.attention = attention
        self.tanh = tanh
        self.n_layers = n_layers

        # edge MLP (uses x_i, x_j, ||x_i-x_j||^2 [+ edge_attr])
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + 1 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        # node MLP (updates hidden features)
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        # generate coordinate direction coefficient (with bias, same as segno_model.py)
        layer = nn.Linear(hidden_nf, 1, bias=True)
        nn.init.xavier_uniform_(layer.weight, gain=0.001)
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            out = out * self.att_mlp(out)
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr=None):
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        agg = torch.cat([x, agg], dim=1) if node_attr is None else torch.cat([x, agg, node_attr], dim=1)
        out = self.node_mlp(agg)
        # optional residual; SEGNO original uses x ← x + f(x, agg), kept here
        out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """Do NOT directly add to coord here — return an acceleration-like quantity."""
        row, _ = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)           # (E,3)
        trans = torch.clamp(trans, min=-100, max=100)            # explosion prevention
        a_like = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        return a_like * self.coords_weight

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)
        return radial, coord_diff

    def forward(self, h, edge_index, coord, vel, disappear_or_not, edge_attr=None, node_attr=None, dt=1.0):
        """SEGNO core: perform explicit v/x integration inside the layer."""
        step = float(dt) / float(self.n_layers)                  # per-layer timestep
        radial, coord_diff = self.coord2radial(edge_index, coord)
        
        edge_feat = self.edge_model(h[edge_index[0]], h[edge_index[1]], radial, edge_attr)

        # 1) obtain acceleration-like quantity from messages
        a_like = self.coord_model(coord, edge_index, coord_diff, edge_feat) * disappear_or_not.reshape(-1, 1)

        # 2) explicit integration (simple Euler here)
        vel = vel + a_like * step
        coord = coord + vel * step

        # 3) node feature update
        h = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, vel


# ============================================================
#  Backbone network using SEGNO_GCL (replacing original EGNN_network)
# ============================================================
class SEGNO_network(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf=None, n_layers=4,
                 coords_weight=1.0, act_fn=nn.SiLU(), attention=False, tanh=False):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.in_node_nf = in_node_nf 
        self.embedding = nn.Linear(in_node_nf, hidden_nf)

        # single SEGNO layer reused in forward loop (same as segno_model.py)
        self.segno_layer = SEGNO_GCL(
            input_nf=hidden_nf, output_nf=hidden_nf, hidden_nf=hidden_nf,
            edges_in_d=0, act_fn=act_fn, coords_weight=coords_weight,
            n_layers=n_layers, attention=attention, tanh=tanh
        )

    def forward(self, states, t, dt=1.0):
        """
        states: (B, N, 7) -> [mass, x, y, z, vx, vy, vz]
        returns: residual of current state (δmass, δx, δv)
        """
        bs, n_nodes, _ = states.shape
        coords0 = states[:, :, :2]
        angle = states[:, :, 3:4]  
        velocities = states[:, :, FEATURE_DIM:FEATURE_DIM+2]
        angular_v = states[:, :, FEATURE_DIM+2:FEATURE_DIM+3]
        dynamic_mask = states[:, :, FEATURE_DIM-3:FEATURE_DIM-2].clone()

        disappear_time = states[:,:,FEATURE_DIM+3:]  
        disappear_or_not = t < disappear_time.squeeze(-1) 

        all_features = states[:,:,:FEATURE_DIM]  

        # initial node features
        h = torch.cat([all_features, velocities, angular_v], dim=-1).to(torch.float32)
        h = self.embedding(h)

        # flatten for graph computation
        h      = h.reshape(bs * n_nodes, -1)
        coords = torch.cat([coords0, angle], dim=-1).reshape(bs * n_nodes, -1)
        vel    = torch.cat([velocities, angular_v], dim=-1).reshape(bs * n_nodes, -1)

        # fully-connected graph (no self loops)
        edges = self.create_edges(n_nodes, bs, states.device)

        # SEGNO micro-step integration inside layer
        for _ in range(self.n_layers):
            h, coords, vel = self.segno_layer(
                h, edges, coords, vel,
                disappear_or_not=disappear_or_not,
                edge_attr=None, node_attr=None, dt=dt
            )

        # reshape back to batch
        coords = coords.view(bs, n_nodes, -1)
        vel    = vel.view(bs, n_nodes, -1)

        delta_coord = (coords[...,:2] - coords0) * dynamic_mask
        dthetadt = (coords[...,2:3] - angle) * dynamic_mask
        delta_vel   = (vel[...,:2] - velocities) * dynamic_mask
        dangvdt = (vel[...,2:3] - angular_v) * dynamic_mask * 1e1

        not_ball_mask = (all_features[:,:,2:3].abs() > 1e-5).float()
        dangvdt *= not_ball_mask

        res = torch.cat([
            delta_coord, 
            torch.zeros_like(states[..., 0:1]),  
            dthetadt,
            torch.zeros_like(states[:,:,4:FEATURE_DIM]),  
            delta_vel,
            dangvdt,
            torch.zeros_like(disappear_time)  
        ], dim=-1)

        return res

    def create_edges(self, n_nodes, batch_size, device):
        idx = torch.arange(n_nodes, device=device)
        row, col = torch.meshgrid(idx, idx, indexing='ij')
        mask = row != col
        base_rows = row[mask]
        base_cols = col[mask]
        batch_offset = torch.arange(batch_size, device=device) * n_nodes
        batch_rows = (base_rows.unsqueeze(0) + batch_offset.unsqueeze(1)).reshape(-1)
        batch_cols = (base_cols.unsqueeze(0) + batch_offset.unsqueeze(1)).reshape(-1)
        return [batch_rows, batch_cols]


class SEGNO(nn.Module):
    def __init__(self, hidden_dim, num_layers, coords_weight=1.0):
        super(SEGNO, self).__init__()
        # in_node_nf = mass(1) + vel(3) = 4
        self.segno = SEGNO_network(
            in_node_nf=12,
            hidden_nf=hidden_dim,
            out_node_nf=12,
            n_layers=num_layers,
            coords_weight=coords_weight
        )

    def forward(self, z0, disappear_time, t):
        """
        z0: (B, N, 12)
        t:   (B, T,) or list-like
        """
        z0 = torch.cat([z0, disappear_time], dim=-1)  
        bs, body_num, feature_dim = z0.shape
        steps = t.shape[-1]

        traj = torch.zeros(bs, steps, body_num, feature_dim,
                           device=z0.device, dtype=z0.dtype)
        traj[:, 0] = z0

        # time stepping, dt from t differences (fallback to 1.0 if needed)
        for st in range(1, steps):
            current = traj[:, st - 1]
            dt_val = float(
                (t[0, st] - t[0, st - 1]).item()
                if torch.is_tensor(t)
                else (t[0, st] - t[0, st - 1])
            )
            res = self.segno(current, t=t[:,st-1:st], dt=dt_val)
            traj[:, st] = current + res

        range_tensor = torch.arange(steps, device=z0.device)\
            .unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
        range_tensor = range_tensor / 10
        disappear_mask = (range_tensor < disappear_time.unsqueeze(1))
        traj = traj * disappear_mask

        return traj


# ============================================================
def unsorted_segment_sum(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result

def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)
    count  = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

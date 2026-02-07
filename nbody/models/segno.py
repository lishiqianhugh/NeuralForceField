import torch
import torch.nn as nn

class SEGNO_GCL(nn.Module):
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(),
                 coords_weight=1.0, n_layers=4, attention=False, tanh=False):
        super().__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.attention = attention
        self.tanh = tanh
        self.n_layers = n_layers

        # edge MLP ( x_i, x_j, ||x_i-x_j||^2 [+ edge_attr])
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + 1 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        # node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

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
        out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        """不在这里直接加到 coord！返回“加速度方向量” a~ """
        row, _ = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)           # (E,3)
        trans = torch.clamp(trans, min=-100, max=100)
        a_like = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        return a_like * self.coords_weight

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)
        return radial, coord_diff

    def forward(self, h, edge_index, coord, vel, edge_attr=None, node_attr=None, dt=1.0):
        step = float(dt) / float(self.n_layers)
        radial, coord_diff = self.coord2radial(edge_index, coord)
        
        edge_feat = self.edge_model(h[edge_index[0]], h[edge_index[1]], radial, edge_attr)

        # 1) get acceleration-like message
        a_like = self.coord_model(coord, edge_index, coord_diff, edge_feat)

        # 2) explicit euler
        vel = vel + a_like * step
        coord = coord + vel * step

        # 3) update node features
        h = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, vel


class SEGNO_network(nn.Module):
    def __init__(self, in_node_nf, hidden_nf, out_node_nf=None, n_layers=4,
                 coords_weight=1.0, act_fn=nn.SiLU(), attention=False, tanh=False):
        super().__init__()
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers

        self.in_node_nf = in_node_nf
        self.embedding = nn.Linear(in_node_nf, hidden_nf)

        self.segno_layer = SEGNO_GCL(
            input_nf=hidden_nf, output_nf=hidden_nf, hidden_nf=hidden_nf,
            edges_in_d=0, act_fn=act_fn, coords_weight=coords_weight,
            n_layers=n_layers, attention=attention, tanh=tanh
        )

    def forward(self, states, dt=1.0):
        """
        states: (B, N, 7) -> [mass, x, y, z, vx, vy, vz]
        return residual (δmass, δx, δv)
        """
        bs, n_nodes, _ = states.shape
        mass   = states[..., 0:1]
        coords0 = states[..., 1:4]
        vel0    = states[..., 4:7]

        h = torch.cat([mass, vel0], dim=-1).to(torch.float32)
        h = self.embedding(h)

        h      = h.reshape(bs * n_nodes, -1)
        coords = coords0.reshape(bs * n_nodes, -1)
        vel    = vel0.reshape(bs * n_nodes, -1)

        edges = self.create_edges(n_nodes, bs, states.device)

        for _ in range(self.n_layers):
            h, coords, vel = self.segno_layer(h, edges, coords, vel, edge_attr=None, node_attr=None, dt=dt)

        coords = coords.view(bs, n_nodes, -1)
        vel    = vel.view(bs, n_nodes, -1)

        delta_mass  = torch.zeros_like(mass)
        delta_coord = coords - coords0
        delta_vel   = vel - vel0

        res = torch.cat([delta_mass, delta_coord, delta_vel], dim=-1)  # (B, N, 7)
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
        self.segno = SEGNO_network(in_node_nf=4, hidden_nf=hidden_dim, out_node_nf=4, n_layers=num_layers,coords_weight=coords_weight)

    def forward(self, initial_state, time_points):
        """
        initial_state: (B, N, 7)  [m, x, y, z, vx, vy, vz]
        time_points:   (T,) or list-like
        """
        bs, body_num, feature_dim = initial_state.shape
        steps = len(time_points)

        traj = torch.zeros(bs, steps, body_num, feature_dim,
                           device=initial_state.device, dtype=initial_state.dtype)
        traj[:, 0] = initial_state

        for t in range(1, steps):
            current = traj[:, t - 1]
            dt_val = float((time_points[t] - time_points[t - 1]).item()
                            if torch.is_tensor(time_points) else (time_points[t] - time_points[t - 1]))
            res = self.segno(current, dt=dt_val)
            traj[:, t] = current + res

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
import torch
import torch.nn as nn

def get_sin_pos_enc(seq_len, d_model):
    """Sinusoid absolute positional encoding."""
    inv_freq = 1. / (10000**(torch.arange(0.0, d_model, 2.0) / d_model))
    pos_seq = torch.arange(seq_len - 1, -1, -1).type_as(inv_freq)
    sinusoid_inp = torch.outer(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb.unsqueeze(0)  # [1, L, C]


def build_pos_enc(pos_enc, input_len, d_model):
    """Positional Encoding of shape [1, L, D]."""
    if not pos_enc:
        return None
    # ViT, BEiT etc. all use zero-init learnable pos enc
    if pos_enc == 'learnable':
        pos_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
    # in SlotFormer, we find out that sine P.E. is already good enough
    elif 'sin' in pos_enc:  # 'sin', 'sine'
        pos_embedding = nn.Parameter(
            get_sin_pos_enc(input_len, d_model), requires_grad=False)
    else:
        raise NotImplementedError(f'unsupported pos enc {pos_enc}')
    return pos_embedding

class Rollouter(nn.Module):
    """Base class for a predictor based on slot_embs."""

    def forward(self, x):
        raise NotImplementedError

    def burnin(self, x):
        pass

    def reset(self):
        pass

class SlotRollouter(Rollouter):
    """Transformer encoder only."""

    def __init__(
        self,
        num_slots,
        slot_size,
        history_len,  # burn-in steps
        t_pe='sin',  # temporal P.E.
        slots_pe='',  # slots P.E., None in SlotFormer
        # Transformer-related configs
        d_model=128,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        norm_first=True,
        slotres_scale=1e2,
        use_dist_mask=False,
    ):
        super().__init__()

        self.num_slots = num_slots
        self.history_len = history_len
        self.in_proj = nn.Linear(7, d_model)
        self.slotres_scale = slotres_scale
        self.use_dist_mask = use_dist_mask

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            norm_first=norm_first,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=enc_layer, num_layers=num_layers)
        self.enc_t_pe = build_pos_enc(t_pe, history_len, d_model)
        self.enc_slots_pe = build_pos_enc(slots_pe, num_slots, d_model)
        self.out_proj = nn.Linear(d_model, slot_size)

    def forward(self, x, pred_len, t):
        """Forward function.

        Args:
            x: [B, history_len, num_slots, slot_size]
            pred_len: int

        Returns:
            [B, pred_len, num_slots, slot_size]
        """
        assert x.shape[1] == self.history_len, 'wrong burn-in steps'

        # Dynamic mask: some objects are always static
        dynamic_mask = x[:, 0, :, 9-3:9-2] # [B, num_slots, 1]
        # only 0 1 3 -4 -3 -2 dim will change
        feature_mask = torch.zeros((1, 1, x.shape[-1]), device=x.device) # [1, 1, slot_size]
        feature_mask[...,0] = 1
        feature_mask[...,1] = 1
        feature_mask[...,3] = 1
        # feature_mask[...,-4] = 1
        # feature_mask[...,-3] = 1
        # feature_mask[...,-2] = 1
        disappear_time = x[:, 0, :, -1:]  # [B, num_slots, 1]
        disappear_time = disappear_time.squeeze(-1).unsqueeze(1) # [B, 1, num_slots]
        t = t.unsqueeze(-1)  # [B, timesteps, 1]
        disappear_mask = (t < disappear_time).float().unsqueeze(-1)  # [B, timesteps, num_slots, 1]

        B = x.shape[0]
        x = x.flatten(1, 2)  # [B, T * N, slot_size]
        in_x = x

        # temporal_pe repeat for each slot, shouldn't be None
        # [1, T, D] --> [B, T, N, D] --> [B, T * N, D]
        enc_pe = self.enc_t_pe.unsqueeze(2).\
            repeat(B, 1, self.num_slots, 1).flatten(1, 2)
        # slots_pe repeat for each timestep
        if self.enc_slots_pe is not None:
            slots_pe = self.enc_slots_pe.unsqueeze(1).\
                repeat(B, self.history_len, 1, 1).flatten(1, 2)
            enc_pe = slots_pe + enc_pe

        # generate future slots autoregressively
        pred_out = []  # [B, N, D]
        for i in range(pred_len):
            # project to latent space
            # x = self.in_proj(torch.cat([in_x[...,:5], in_x[...,7:9], in_x[...,9:12]], dim=-1)) # add velocity
            x = self.in_proj(torch.cat([in_x[...,:5], in_x[...,7:9]], dim=-1))
            # encoder positional encoding
            x = x + enc_pe
            # spatio-temporal interaction via transformer
            x = self.transformer_encoder(x)
            # take the last N output tokens to predict slots
            pred_slots_res = self.out_proj(x[:, -self.num_slots:]) / self.slotres_scale # [B, N, D]
            # apply dynamic mask
            pred_slots_res = pred_slots_res * dynamic_mask * feature_mask
                
            pred_slots = pred_slots_res + in_x[:, -self.num_slots:]
            # apply disappear mask
            pred_slots = pred_slots * disappear_mask[:,i+1]
            pred_out.append(pred_slots)
            # feed the predicted slots autoregressively
            in_x = torch.cat([in_x[:, self.num_slots:], pred_out[-1]], dim=1)

        return torch.stack(pred_out, dim=1)

    @property
    def dtype(self):
        return self.in_proj.weight.dtype

    @property
    def device(self):
        return self.in_proj.weight.device


class DynamicsSlotFormer(nn.Module):
    def __init__(
        self,
        num_slots,
        slot_size,
        history_len,  # burn-in steps
        t_pe='sin',  # temporal P.E.
        slots_pe='',  # slots P.E.
        d_model=128,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        norm_first=True,
        slotres_scale=1e2,
        use_dist_mask=False,
    ):
        super().__init__()
        self.history_len = history_len
        self.num_slots = num_slots

        self.rollouter = SlotRollouter(
            num_slots=num_slots,
            slot_size=slot_size,
            history_len=history_len,
            t_pe=t_pe,
            slots_pe=slots_pe,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            norm_first=norm_first,
            slotres_scale=slotres_scale,
            use_dist_mask=use_dist_mask,
        )

    def forward(self, z0, disappear_time, t):
        """Predict trajectories.

        Args:
            z0: [B, num_slots, 12] Initial state of all objects.
            disappear_time: [B, num_slots, 1] Time for object disppeaar.
            t: [B, timesteps] Time indices for each timestep.

        Returns:
            predicted_trajectories: [B, timesteps, num_slots, 13]
        """
        B, timesteps = t.shape
        z0 = torch.cat([z0, disappear_time], dim=-1)  # [B, num_slots, slot_size]

        # Expand z0 to match timesteps for burn-in input
        z0_expanded = z0.unsqueeze(1).repeat(1, self.history_len, 1, 1)  # [B, history_len, num_slots, slot_size]

        # Prepare burn-in input based on initial state
        burn_in_input = z0_expanded

        # Predict for the remaining timesteps autoregressively
        pred_len = timesteps - 1
        predicted_slots =  self.rollouter(burn_in_input, pred_len, t)  # [B, pred_len, num_slots, slot_size]

        # Concatenate burn-in input and predicted slots
        predicted_trajectories = torch.cat([burn_in_input[:,-1:], predicted_slots], dim=1)  # [B, timesteps, num_slots, slot_size]
        
        # Transform disappear_time to disappear_mask [B, timesteps, num_slots]
        disappear_time = disappear_time.squeeze(-1).unsqueeze(1) # [B, 1, num_slots]
        t = t.unsqueeze(-1)  # [B, timesteps, 1]
        disappear_mask = (t < disappear_time).float().unsqueeze(-1)  # [B, timesteps, num_slots, 1]
        predicted_trajectories = predicted_trajectories * disappear_mask

        return predicted_trajectories

# Example usage
if __name__ == "__main__":
    batch_size = 4
    num_slots = 5
    feature_dim = 16
    timesteps = 20
    history_len = 5

    z0 = torch.randn(batch_size, num_slots, feature_dim)
    disappear_time = torch.randint(0, 2, (batch_size, timesteps, num_slots)).bool()
    t = torch.arange(timesteps).unsqueeze(0).repeat(batch_size, 1)  # Time indices

    model = DynamicsSlotFormer(
        num_slots=num_slots,
        slot_size=feature_dim,
        history_len=history_len,
        d_model=64,
        num_layers=3,
        num_heads=4,
        ffn_dim=256,
        norm_first=True,
    )

    predicted_trajectories = model(z0, disappear_time, t)
    print(predicted_trajectories.shape)  # Should be [batch_size, timesteps, num_slots, feature_dim]

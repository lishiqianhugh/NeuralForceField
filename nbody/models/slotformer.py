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
    if pos_enc == 'learnable':
        pos_embedding = nn.Parameter(torch.zeros(1, input_len, d_model))
    elif 'sin' in pos_enc:
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
    """Transformer encoder only, now with masking support."""

    def __init__(
        self,
        num_slots,
        slot_size,
        history_len,
        t_pe='sin',
        slots_pe='',
        d_model=128,
        num_layers=4,
        num_heads=8,
        ffn_dim=512,
        norm_first=True,
    ):
        super().__init__()
        
        self.num_slots = num_slots
        self.history_len = history_len

        self.in_proj = nn.Linear(slot_size, d_model)

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
        self.enc_slots_pe = build_pos_enc(slots_pe, self.num_slots, d_model)
        self.out_proj = nn.Linear(d_model, slot_size)

    def forward(self, x, pred_len):
        """
        Forward function with attention masking.

        Args:
            x: [B, history_len, max_slots, slot_size] (PADDED tensor)
            pred_len: int

        Returns:
            [B, pred_len, max_slots, slot_size] (PADDED tensor)
        """
        assert x.shape[1] == self.history_len, 'wrong burn-in steps'
        assert x.shape[2] == self.num_slots, 'input slots dim mismatch with model max_slots'

        B = x.shape[0]

        is_padding_mask = (x[:, -1, :, 0] == 0)
        
        attn_mask = is_padding_mask.unsqueeze(1).repeat(1, self.history_len, 1).flatten(1, 2)

        x = x.flatten(1, 2)  # [B, T * N, slot_size]
        in_x = x

        enc_pe = self.enc_t_pe.unsqueeze(2).\
            repeat(B, 1, self.num_slots, 1).flatten(1, 2)
        if self.enc_slots_pe is not None:
            slots_pe = self.enc_slots_pe.unsqueeze(1).\
                repeat(B, self.history_len, 1, 1).flatten(1, 2)
            enc_pe = slots_pe + enc_pe

        pred_out = []
        for _ in range(pred_len):
            x_latent = self.in_proj(in_x)
            x_latent = x_latent + enc_pe
            
            # add attn mask to prevent attention to padding slots
            x_out_latent = self.transformer_encoder(
                x_latent, src_key_padding_mask=attn_mask)
            
            res = self.out_proj(x_out_latent[:, -self.num_slots:]) / 10
            pred_slots = res + in_x[:, -self.num_slots:]

            pred_slots[is_padding_mask] = 0.0
            
            pred_out.append(pred_slots)
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
        history_len,
        t_pe='sin',
        slots_pe='',
        d_model=256,
        num_layers=4,
        num_heads=4,
        ffn_dim=256,
        norm_first=True,
    ):
        super().__init__()
        self.history_len = history_len
        self.num_slots = num_slots
        self.slot_size = slot_size

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
        )

    def forward(self, initial_state, time_points):
        """
        Predict future states from variable-length input.

        Args:
            initial_state: [B, actual_num_slots, slot_size]
            time_points: [T]

        Returns:
            [B, T, actual_num_slots, slot_size] (PADDED tensor)
        """
        
        B, actual_num_slots, _ = initial_state.shape

        if actual_num_slots > self.num_slots:
            raise ValueError(
                f"Input has {actual_num_slots} slots, but model is configured "
                f"for a maximum of {self.num_slots} slots."
            )

        pad_size = self.num_slots - actual_num_slots
        
        if pad_size > 0:
            padding = torch.zeros(
                B, pad_size, self.slot_size,
                device=initial_state.device, dtype=initial_state.dtype
            )
            padded_initial_state = torch.cat([initial_state, padding], dim=1)
        else:
            padded_initial_state = initial_state
        
        burnin_state = padded_initial_state.unsqueeze(1).repeat(1, self.history_len, 1, 1)
        pred_len = len(time_points) - 1
        pred_out = self.rollouter(burnin_state, pred_len)
        padded_final_prediction = torch.cat([burnin_state[:, -1:], pred_out], dim=1)

        if pad_size > 0:
            unpadded_final_prediction = padded_final_prediction[:, :, :-pad_size, :]
        else:
            unpadded_final_prediction = padded_final_prediction
        
        return unpadded_final_prediction
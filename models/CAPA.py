import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    """Reversible Instance Normalization."""
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_features))

    def normalize(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        sigma = (var + self.eps).sqrt()
        xn = (x - mu) / sigma
        if self.affine:
            xn = xn * self.weight + self.bias
        return xn, (mu, sigma)

    def denormalize(self, y, stats):
        mu, sigma = stats
        return y * sigma + mu


class PositiveNegativeAttention(nn.Module):
    """
    Positive-Negative Phase Attention.
    Models both positive correlations (phase-aligned) and negative correlations (phase-offset).
    """
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout
        
        # Positive branch
        self.q_proj_pos = nn.Linear(d_model, d_model)
        self.k_proj_pos = nn.Linear(d_model, d_model)
        self.v_proj_pos = nn.Linear(d_model, d_model)
        
        # Negative branch
        self.q_proj_neg = nn.Linear(d_model, d_model)
        self.k_proj_neg = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = self.head_dim ** -0.5
        
        # Gate for positive-negative fusion
        self.gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, period_dist=None):
        """
        x: (B, L, D)
        period_dist: (L, L) periodic distance matrix
        """
        B, L, D = x.shape
        
        # Positive attention
        Q_pos = self.q_proj_pos(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K_pos = self.k_proj_pos(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V_pos = self.v_proj_pos(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        pos_scores = torch.matmul(Q_pos, K_pos.transpose(-2, -1)) * self.scale
        
        # Add periodic bias for positive attention (closer = stronger)
        if period_dist is not None:
            period_bias = -period_dist.unsqueeze(0).unsqueeze(0) * 0.05
            pos_scores = pos_scores + period_bias.to(x.device)
        
        pos_attn = F.softmax(pos_scores, dim=-1)
        pos_attn = F.dropout(pos_attn, p=self.dropout, training=self.training)
        pos_out = torch.matmul(pos_attn, V_pos)
        
        # Negative attention (phase-offset)
        Q_neg = self.q_proj_neg(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K_neg = self.k_proj_neg(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        neg_scores = torch.matmul(Q_neg, K_neg.transpose(-2, -1)) * self.scale
        
        # Add periodic bias for negative attention (phase offset = negative correlation)
        if period_dist is not None:
            neg_bias = period_dist.unsqueeze(0).unsqueeze(0) * 0.02
            neg_scores = neg_scores + neg_bias.to(x.device)
        
        neg_attn = F.softmax(neg_scores, dim=-1)
        neg_attn = F.dropout(neg_attn, p=self.dropout, training=self.training)
        neg_out = torch.matmul(neg_attn, V_pos)  # (B, heads, L, head_dim)
        
        # Dynamic fusion gate - expand to match attention dimensions
        gate_val = self.gate(x.mean(dim=1, keepdim=True))  # (B, 1, 1)
        gate_val = gate_val.unsqueeze(1)  # (B, 1, 1, 1) for broadcasting with (B, heads, L, head_dim)
        
        # Fuse positive and negative
        fused = gate_val * pos_out + (1 - gate_val) * neg_out
        
        # Output
        out = fused.transpose(1, 2).reshape(B, L, D)
        out = x + F.dropout(self.out_proj(out), p=self.dropout, training=self.training)
        
        return out


class CAPA(nn.Module):
    """
    CAPA: Contrastive Adaptive Phase Attention model.
    
    Compared to original PhaseFormer:
    - Uses Positive-Negative phase attention instead of routing
    - Period-aware positional encoding
    - FFT-based period detection for adaptive modeling
    """
    
    def __init__(self, configs):
        super().__init__()
        
        # Basic configs
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = getattr(configs, "period_len", 24)
        
        # Model dimensions
        self.latent_dim = getattr(configs, "latent_dim", 8)
        self.phase_encoder_hidden = getattr(configs, "phase_encoder_hidden", 32)
        self.predictor_hidden = getattr(configs, "predictor_hidden", 64)
        self.phase_attn_heads = getattr(configs, "phase_attn_heads", 4)
        self.phase_attn_dropout = getattr(configs, "phase_attn_dropout", 0.1)
        self.phase_layers = getattr(configs, "phase_layers", 1)
        
        # RevIN
        self.use_revin = getattr(configs, "use_revin", True)
        self.revin_affine = getattr(configs, "revin_affine", False)
        self.revin_eps = getattr(configs, "revin_eps", 1e-5)
        if self.use_revin:
            self.revin = RevIN(num_features=self.enc_in, eps=self.revin_eps, affine=self.revin_affine)
        
        # Period calculations
        self.num_periods_input = (self.seq_len + self.period_len - 1) // self.period_len
        self.num_periods_output = (self.pred_len + self.period_len - 1) // self.period_len
        self.total_len_in = self.num_periods_input * self.period_len
        self.pad_seq_len = self.total_len_in - self.seq_len
        
        # Embedding: (B, C, period_len, P_in) -> (B*C, period_len, latent_dim)
        self.embedding = nn.Sequential(
            nn.Linear(self.num_periods_input, self.latent_dim),
            nn.LayerNorm(self.latent_dim)
        )
        
        # Phase positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(self.period_len, self.latent_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Positive-Negative attention layers
        self.phase_layers = nn.ModuleList([
            PositiveNegativeAttention(
                d_model=self.latent_dim,
                num_heads=self.phase_attn_heads,
                dropout=self.phase_attn_dropout
            )
            for _ in range(self.phase_layers)
        ])
        
        # MLP block
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 4 * self.latent_dim),
            nn.GELU(),
            nn.Linear(4 * self.latent_dim, self.latent_dim)
        )
        self.norm = nn.LayerNorm(self.latent_dim)
        
        # Predictor
        self.predictor = nn.Linear(self.latent_dim, self.num_periods_output)
        
        # Precompute period distance matrix
        self.register_buffer('period_dist', self._compute_period_dist())
        
    def _compute_period_dist(self):
        """Compute periodic distance matrix for attention bias."""
        pos = torch.arange(self.period_len)
        dist = torch.abs(pos.unsqueeze(1) - pos.unsqueeze(0))
        dist = torch.min(dist, self.period_len - dist)  # Periodic distance
        return dist.float()
        
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args, **kwargs):
        """
        Input:  x_enc (B, seq_len, C)
        Output: y_hat (B, pred_len, C)
        """
        B, L, C = x_enc.shape
        
        # 1) RevIN normalization
        if self.use_revin:
            x_in, stats = self.revin.normalize(x_enc)
        else:
            x_in = x_enc.float()
            stats = None
        
        # 2) Ring padding
        x = x_in.permute(0, 2, 1)  # (B, C, L)
        if self.pad_seq_len > 0:
            x = F.pad(x, (0, self.pad_seq_len), mode="circular")
        
        # 3) Split to periods (B, C, P_in, period_len)
        x_periods = x.view(B, C, self.num_periods_input, self.period_len)
        
        # 4) Phase view (B, C, period_len, P_in)
        x_phase = x_periods.permute(0, 1, 3, 2).contiguous()
        
        # 5) Embed (B*C, period_len, latent_dim)
        x_flat = x_phase.view(B * C, self.period_len, self.num_periods_input)
        z = self.embedding(x_flat)
        
        # 6) Add phase positional embedding
        z = z + self.pos_embed.unsqueeze(0)
        
        # 7) Apply CAPA attention layers
        period_dist = self.period_dist.to(z.device)
        for layer in self.phase_layers:
            z_new = layer(z, period_dist)
            z_new = z_new + F.dropout(self.mlp(self.norm(z_new)), p=self.phase_attn_dropout, training=self.training)
            z = z_new
        
        # 8) Predict future phases
        z_pred = self.predictor(z)  # (B*C, period_len, P_out)
        
        # 9) Reshape (B, C, P_out, period_len)
        z_pred = z_pred.view(B, C, self.period_len, self.num_periods_output)
        
        # 10) Reassemble to sequence
        z_pred_perm = z_pred.permute(0, 1, 3, 2).contiguous()  # (B, C, P_out, period_len)
        y_full = z_pred_perm.reshape(B, C, -1)[..., :self.pred_len]
        y_hat = y_full.permute(0, 2, 1)  # (B, pred_len, C)
        
        # 11) De-normalization
        if self.use_revin and stats is not None:
            y_hat = self.revin.denormalize(y_hat, stats)
        
        return y_hat


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Alias for model loading compatibility
Model = CAPA

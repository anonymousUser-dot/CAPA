# CAPA: Complementary Adaptive Phase Attention for Periodic Time-Series Forecasting

Implementation of **CAPA** in `models/CAPA.py`, compatible with the [Time-Series Library](https://github.com/thuml/Time-Series-Library) framework.

---

## 1. Overview

CAPA is a phase-view forecasting model designed for periodic heterogeneity in long-term multivariate time-series forecasting. Unlike standard sequence models that treat time tokens as an ordinary chain, CAPA reorganizes the input into **phase tokens** and explicitly models two complementary types of phase-to-phase dependencies:

- **Positive (aligned) phase attention**: captures co-varying phase positions across periods, e.g., morning peaks repeating across days.
- **Negative (offset) phase attention**: captures complementary offset relations, e.g., peak-valley alternation or day-night compensation.

A **dynamic fusion gate** adaptively combines the two branches for each sample, rather than using a fixed weight. This allows CAPA to handle both regularly periodic segments (where aligned attention dominates) and transitional segments (where offset relations are more informative).

The phase view also brings a significant efficiency advantage: when the configured period length \(P\) is much smaller than the input length \(L\), the attention cost drops from \(O(L^2)\) to \(O(P^2)\). CAPA remains compact — only **0.002 M parameters** — while achieving competitive accuracy across seven benchmarks.

---

## 2. Architecture

### 2.1 Reversible Instance Normalization (RevIN)

CAPA applies RevIN normalization along the temporal dimension for each sample and variable independently:

\[
\tilde{X} = (X - \mu) / \sigma, \quad \mu, \sigma \in \mathbb{R}^{B \times 1 \times C}
\]

RevIN reduces distribution shift between training and test windows. It is reversible, so the forecast is denormalized at the end of the forward pass. Affine parameters are disabled by default (`affine=False`).

### 2.2 Ring Padding and Phase View Reshaping

The normalized sequence is circularly padded so its length is divisible by the period length \(P\):

```python
x = F.pad(x, (0, pad_seq_len), mode="circular")
x_periods = x.view(B, C, num_periods_input, period_len)      # (B, C, N, P)
x_phase    = x_periods.permute(0, 1, 3, 2).contiguous()      # (B, C, P, N)
```

Each phase token at position \(p\) collects the values observed at the same within-period position across all historical periods \(N\).

### 2.3 Phase Embedding

Phase tokens are flattened across batch and variable dimensions and projected by a linear embedding with layer normalization:

```python
z = embedding(x_flat)   # (B*C, P, latent_dim)
z = z + pos_embed        # add learnable phase positional embedding (size P x latent_dim)
```

The default `latent_dim = 8`.

### 2.4 Positive-Negative Phase Attention

The core operator is `PositiveNegativeAttention` (`models/CAPA.py`, line 30). It has two parallel branches:

**Positive branch (aligned):**

```python
Q_pos = q_proj_pos(x);  K_pos = k_proj_pos(x);  V_pos = v_proj_pos(x)
pos_scores = Q_pos @ K_pos^T / sqrt(d_h) - λ⁺ * D_φ
pos_attn   = softmax(pos_scores)
pos_out    = pos_attn @ V_pos
```

The negative bias \(-λ^+ D_\phi\) encourages closer phase positions to receive stronger attention. Here \(D_\phi(i,j) = \min(|i-j|, P - |i-j|)\) is the circular phase distance matrix.

**Negative branch (offset):**

```python
Q_neg = q_proj_neg(x);  K_neg = k_proj_neg(x)
neg_scores = Q_neg @ K_neg^T / sqrt(d_h) + λ⁻ * D_φ
neg_attn   = softmax(neg_scores)
neg_out    = neg_attn @ V_pos    # shares value projection with positive branch
```

The positive bias \(+λ^- D_\phi\) promotes offset phase positions as candidate sources of predictive evidence.

**Dynamic fusion gate:**

```python
gate_val = sigmoid(W_g · mean_phi(Z) + b_g)   # (B, 1, 1, 1)
fused    = gate_val * pos_out + (1 - gate_val) * neg_out
```

The gate value is conditioned on the global phase representation, so it adapts per sample.

### 2.5 MLP Refinement

After attention, a standard MLP block with GELU activation and residual connection refines the phase representation:

```python
z_new = layer_norm(z_new + dropout(mlp(z_new)))
```

### 2.6 Phase Predictor

A linear head maps from the latent phase dimension to the number of output periods \(M = \lceil H / P \rceil\):

```python
z_pred = predictor(z)        # (B*C, P, M)
z_pred = reshape(z_pred)     # (B, C, P, M)
y_full = permute + flatten    # -> (B, H, C)
y_hat  = y_full[..., :pred_len]
```

Finally, RevIN denormalization restores the original scale.

---

## 3. Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `seq_len` | from config | Input look-back window length |
| `pred_len` | from config | Prediction horizon length |
| `enc_in` | from config | Number of input variables |
| `period_len` | `24` | Configured period length in samples (`24` for hourly data; `96` for 15-min ETTm) |
| `latent_dim` | `8` | Hidden dimension of each phase token |
| `phase_encoder_hidden` | `32` | Hidden size of the phase embedding MLP |
| `predictor_hidden` | `64` | Hidden size of the predictor MLP |
| `phase_attn_heads` | `4` | Number of attention heads |
| `phase_attn_dropout` | `0.1` | Dropout rate |
| `phase_layers` | `1` | Number of CAPA attention layers |
| `use_revin` | `True` | Whether to apply RevIN normalization |
| `revin_affine` | `False` | Whether RevIN has learnable affine parameters |
| `revin_eps` | `1e-5` | RevIN epsilon for numerical stability |

**Period selection:** CAPA uses frequency-domain analysis to identify the dominant spectral component. For the benchmark datasets: `P=24` for ETTh, Weather, Electricity, and Traffic; `P=96` for the 15-minute ETTm datasets.

---

## 4. Usage

CAPA follows the standard Time-Series Library model interface:

```python
from models.CAPA import CAPA, count_parameters

# configs is a Config object with seq_len, pred_len, enc_in, period_len, etc.
model = CAPA(configs)
print(f"Parameters: {count_parameters(model):,}")
```

Example script to run on ETTh1:

```bash
bash ./scripts/run_ett_hour.py
```

---

## 5. Key Design Insights

| Design Choice | Rationale |
|---|---|
| Phase tokenization (instead of time tokens) | Directly models within-cycle position interactions; avoids spending capacity rediscovering the periodic layout |
| Positive + negative branch attention | Separates aligned recurrence from offset candidate evidence; both are useful but have different semantic roles |
| Circular phase distance \(D_\phi\) | Prevents the model from treating the first and last phase as unrelated across the period boundary |
| Dynamic fusion gate | Some windows are dominated by regular aligned repetition; others by offset relations — the gate adapts per sample |
| RevIN normalization | Reduces distribution shift between training and test windows without learnable parameters |

---

## 6. Experiments Summary

CAPA was evaluated on seven datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather, Electricity, Traffic) at horizons \(H \in \{96, 192, 336, 720\}\) with input length \(L=720\). It is competitive with PhaseFormer, TimeXer, TQNet, iTransformer, and PatchTST.

| Metric | CAPA | Notes |
|---|---|---|
| Params | 0.002 M | Lightweight compared to transformer baselines |
| FLOPs | 0.224 G | Efficient due to phase-view attention over \(P\) tokens |
| Ablation | Positive branch, negative branch, and dynamic gate each contribute positively | Removing any component degrades averaged MSE across datasets |

**Ablation highlights:**

- Removing the positive branch increases averaged MSE by ~7% on ETTh1, ~8% on Weather.
- Removing the negative branch increases averaged MSE by ~5% on Electricity and ~7% on Traffic, where peak-valley alternations are frequent.
- Replacing the dynamic gate with fixed 0.5/0.5 averaging also degrades performance, confirming sample-adaptive fusion is beneficial.

---

## 7. Citation

If CAPA is useful in your research, please cite:

```bibtex
@article{capa2026,
  title   = {CAPA: Complementary Adaptive Phase Attention for Periodic Time-Series Forecasting},
  author  = {Anonymous Submission},
  year    = {2026}
}
```


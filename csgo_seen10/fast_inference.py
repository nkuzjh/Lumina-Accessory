"""Static, masked Accessory denoiser with invariant text/radar work cached per batch.

The real caption mask remains false at padding positions. Fixed slots change only
where padding lives; true-token RoPE coordinates and the Accessory condition time
modulation are the same as the official packed forward.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models_accessory.model import modulate

CAP_WIDTH = 256
COND_TOKENS = 14 * 14
IMAGE_TOKENS = 28 * 28


def _attention(module, x, mask, freqs):
    batch, sequence, _ = x.shape
    qh, kh, hd = module.n_local_heads, module.n_local_kv_heads, module.head_dim
    q, k, v = torch.split(module.qkv(x), [qh * hd, kh * hd, kh * hd], dim=-1)
    q = module.q_norm(q.reshape(batch, sequence, qh, hd))
    k = module.k_norm(k.reshape(batch, sequence, kh, hd))
    v = v.reshape(batch, sequence, kh, hd)

    def rotary(tensor):
        pairs = tensor.float().reshape(*tensor.shape[:-1], -1, 2)
        cosine = freqs[..., 0].unsqueeze(2)
        sine = freqs[..., 1].unsqueeze(2)
        return torch.stack((pairs[..., 0] * cosine - pairs[..., 1] * sine,
                            pairs[..., 0] * sine + pairs[..., 1] * cosine), dim=-1).flatten(-2).to(tensor.dtype)

    q, k = rotary(q), rotary(k)
    if module.n_rep > 1:
        k = k.repeat_interleave(module.n_rep, dim=2)
        v = v.repeat_interleave(module.n_rep, dim=2)
    output = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
        attn_mask=mask[:, None, None, :], dropout_p=0.0, is_causal=False,
        scale=module.head_dim ** -0.5,
    ).transpose(1, 2).reshape(batch, sequence, qh * hd)
    return module.out(output)


def _static_block(block, x, mask, freqs, t, t_cond=None, cond_start=None, cond_end=None):
    if not block.modulation:
        x = x + block.attention_norm2(_attention(block.attention, block.attention_norm1(x), mask, freqs))
        return x + block.ffn_norm2(block.feed_forward(block.ffn_norm1(x)))
    scale_msa, gate_msa, scale_mlp, gate_mlp = block.adaLN_modulation(t).chunk(4, dim=1)
    if t_cond is not None:
        scale_msa_cond, gate_msa_cond, scale_mlp_cond, gate_mlp_cond = block.adaLN_modulation(t_cond).chunk(4, dim=1)

    def with_condition(normal, alternate):
        if cond_start is None:
            return normal
        return torch.cat((normal[:, :cond_start], alternate[:, cond_start:cond_end], normal[:, cond_end:]), dim=1)

    norm = block.attention_norm1(x)
    altered = with_condition(modulate(norm, scale_msa), modulate(norm, scale_msa_cond) if t_cond is not None else norm)
    attended = block.attention_norm2(_attention(block.attention, altered, mask, freqs))
    gate = with_condition(gate_msa[:, None].tanh() * attended,
                          gate_msa_cond[:, None].tanh() * attended if t_cond is not None else attended)
    x = x + gate
    norm = block.ffn_norm1(x)
    altered = with_condition(modulate(norm, scale_mlp), modulate(norm, scale_mlp_cond) if t_cond is not None else norm)
    fed = block.ffn_norm2(block.feed_forward(altered))
    gate = with_condition(gate_mlp[:, None].tanh() * fed,
                          gate_mlp_cond[:, None].tanh() * fed if t_cond is not None else fed)
    return x + gate


def _patchify(latent: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    batch, channels, height, width = latent.shape
    return latent.reshape(batch, channels, height // patch_size, patch_size, width // patch_size, patch_size).permute(0, 2, 4, 3, 5, 1).reshape(batch, -1, patch_size * patch_size * channels)


def _unpatchify(tokens: torch.Tensor, patch_size: int = 2) -> torch.Tensor:
    batch, _, dim = tokens.shape
    channels = dim // (patch_size * patch_size)
    height = width = 28
    return tokens.reshape(batch, height, width, patch_size, patch_size, channels).permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, height * patch_size, width * patch_size)


@dataclass(frozen=True)
class PreparedCondition:
    context: torch.Tensor
    cond: torch.Tensor
    full_mask: torch.Tensor
    image_freqs: torch.Tensor
    full_freqs: torch.Tensor
    t_one: torch.Tensor


def build_position_ids(lengths: torch.Tensor) -> torch.Tensor:
    """Official offset coordinates in fixed slots for 196 radar and 784 FPV tokens."""
    batch = len(lengths)
    device = lengths.device
    positions = torch.zeros(batch, CAP_WIDTH + COND_TOKENS + IMAGE_TOKENS, 3, dtype=torch.int32, device=device)
    positions[:, :CAP_WIDTH, 0] = torch.arange(CAP_WIDTH, device=device, dtype=torch.int32)
    positions[:, CAP_WIDTH:CAP_WIDTH + COND_TOKENS, 0] = lengths[:, None]
    positions[:, CAP_WIDTH + COND_TOKENS:, 0] = lengths[:, None] + 1
    condition_grid = torch.arange(COND_TOKENS, device=device)
    positions[:, CAP_WIDTH:CAP_WIDTH + COND_TOKENS, 1] = (condition_grid // 14 + 28)[None]
    positions[:, CAP_WIDTH:CAP_WIDTH + COND_TOKENS, 2] = (condition_grid % 14 + 28)[None]
    image_grid = torch.arange(IMAGE_TOKENS, device=device)
    positions[:, CAP_WIDTH + COND_TOKENS:, 1] = (image_grid // 28)[None]
    positions[:, CAP_WIDTH + COND_TOKENS:, 2] = (image_grid % 28)[None]
    return positions


@torch.no_grad()
def prepare_condition(model, cap_feats, cap_mask, radar_latent) -> PreparedCondition:
    batch = cap_feats.shape[0]
    if cap_feats.shape[1] > CAP_WIDTH or radar_latent.shape != (batch, 16, 28, 28):
        raise ValueError("Static path requires <=256 caption tokens and radar latent [N,16,28,28]")
    cap_mask = cap_mask.bool()
    lengths = cap_mask.sum(dim=1)
    if not torch.all(cap_mask == (torch.arange(cap_mask.shape[1], device=cap_mask.device)[None] < lengths[:, None])):
        raise ValueError("Caption mask must be right padded")
    device = cap_feats.device
    pad = CAP_WIDTH - cap_feats.shape[1]
    cap_feats = F.pad(cap_feats, (0, 0, 0, pad))
    cap_mask = F.pad(cap_mask, (0, pad), value=False)
    positions = build_position_ids(lengths)
    freqs = model.rope_embedder(positions)
    context = model.cap_embedder(cap_feats)
    for layer in model.context_refiner:
        context = layer(context, cap_mask, freqs[:, :CAP_WIDTH])
    cond = model.cond_embedder(_patchify(radar_latent))
    t_one = model.t_embedder(torch.ones(batch, device=device))
    all_valid = torch.ones(batch, COND_TOKENS, dtype=torch.bool, device=device)
    for layer in model.cond_refiner:
        cond = layer(cond, all_valid, freqs[:, CAP_WIDTH:CAP_WIDTH + COND_TOKENS], t_one)
    full_mask = torch.cat((cap_mask, torch.ones(batch, COND_TOKENS + IMAGE_TOKENS, dtype=torch.bool, device=device)), dim=1)
    # Real pairs avoid complex operations inside the compiled denoiser graph.
    real_freqs = torch.view_as_real(freqs)
    return PreparedCondition(context, cond, full_mask,
                             real_freqs[:, CAP_WIDTH + COND_TOKENS:], real_freqs, t_one)


def static_denoise(model, x: torch.Tensor, t: torch.Tensor, prepared: PreparedCondition) -> torch.Tensor:
    """One native Accessory velocity call using static slots and true masks."""
    time = model.t_embedder(t)
    image = model.x_embedder(_patchify(x))
    image_mask = torch.ones(x.shape[0], IMAGE_TOKENS, dtype=torch.bool, device=x.device)
    for layer in model.noise_refiner:
        image = _static_block(layer, image, image_mask, prepared.image_freqs, time)
    full = torch.cat((prepared.context, prepared.cond, image), dim=1)
    for layer in model.layers:
        full = _static_block(layer, full, prepared.full_mask, prepared.full_freqs, time, prepared.t_one,
                             CAP_WIDTH, CAP_WIDTH + COND_TOKENS)
    output = model.final_layer(full, time)
    return _unpatchify(output[:, CAP_WIDTH + COND_TOKENS:])


def build_engine(model, engine: str, compile_mode: str = "default"):
    if engine == "eager":
        return lambda x, t, prepared: static_denoise(model, x, t, prepared)
    if engine != "compiled" or compile_mode not in {"default", "reduce-overhead"}:
        raise ValueError("engine must be eager or compiled with default/reduce-overhead")
    def one_step(x, t, prepared):
        return static_denoise(model, x, t, prepared)
    if compile_mode == "default":
        return torch.compile(one_step, options={"triton.cudagraphs": False}, fullgraph=True)
    return torch.compile(one_step, mode="reduce-overhead", fullgraph=True)

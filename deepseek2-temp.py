import math
import time

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

print(time.strftime("start_time: %Y-%m-%d %H:%M:%S", time.localtime()))
# ----------------------------------------------------------------------------------------------------------------------
# model parameters.
hidden_size = 2048
heads = 16
kv_heads = 16
kv_lora_rank = 512

head_dim = hidden_size // heads
GQA = heads // kv_heads

nope_head_dim = 128
rope_head_dim = 64

norm_eps = 1e-06
rope_theta = 10000

shared_experts = 2
routed_experts = 64
experts_per_tok = 6

moe_hidden_size = 1408
moe_layer_freq = 1

vocab_size = 102400
layers = 27

# ----------------------------------------------------------------------------------------------------------------------
tokenizer = Tokenizer.from_file("/stores/llm_models/deepseek/DeepSeek-V2-Lite/tokenizer.json")

# ----------------------------------------------------------------------------------------------------------------------
model = {}
safetensors = 4
for i in range(1, safetensors + 1):
    safetensor = "/stores/llm_models/deepseek/DeepSeek-V2-Lite/model-000%02d-of-000%03d.safetensors" % (i, safetensors)
    with safe_open(safetensor, framework="pt") as f:
        for k in f.keys():
            model[k] = f.get_tensor(k)

# ----------------------------------------------------------------------------------------------------------------------
# input.
input_sentence = "I believe the meaning of life is to be"
print(f"input_sentence = {input_sentence}")
tokens = tokenizer.encode(input_sentence).ids
tokens = torch.tensor(tokens)

# embedding.
embedding = torch.nn.Embedding(vocab_size, hidden_size, dtype=torch.float)
embedding.weight.data.copy_(model["model.embed_tokens.weight"].to(torch.float))
embedding_output = embedding(tokens)


# rms_norm.
def rms_norm(x, norm_weights):
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + torch.tensor(norm_eps))) * norm_weights


# rope.
def rope(x):
    token_length = x.shape[0]
    zero_to_one_split_into_64_parts = torch.tensor(range(64)) / 64
    freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
    freqs_for_each_token = torch.outer(torch.arange(token_length), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)
    # rope.
    pairs = x.float().view(x.shape[0], -1, 2)
    complex_numbers = torch.view_as_complex(pairs)
    pairs_rotated = torch.view_as_real(complex_numbers * freqs_cis)
    x_rotated = pairs_rotated.view(x.shape)
    return x_rotated


# ----------------------------------------------------------------------------------------------------------------------
factor = 40
mscale = 0.707
mscale_all_dim = 0.707
beta_fast = 32
beta_slow = 1
original_max_position_embeddings = 4096


def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_get_mscale(scale=1, mscale=1.0):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


max_len = 100
freq_extra = 1.0 / (
        rope_theta ** (torch.arange(0, rope_head_dim, 2, dtype=torch.float32, device='cpu') / rope_head_dim))
freq_inter = 1.0 / (factor * rope_theta ** (
        torch.arange(0, rope_head_dim, 2, dtype=torch.float32, device='cpu') / rope_head_dim))

low, high = yarn_find_correction_range(beta_fast, beta_slow, rope_head_dim, rope_theta,
                                       original_max_position_embeddings)
inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, rope_head_dim // 2).to(device='cpu', dtype=torch.float32)
inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

t = torch.arange(max_len, device='cpu', dtype=torch.float32)

freqs = torch.outer(t, inv_freq)

_mscale = float(
    yarn_get_mscale(factor, mscale)
    / yarn_get_mscale(factor, mscale_all_dim))

emb = torch.cat((freqs, freqs), dim=-1)

cos_cached = (emb.cos() * _mscale).to(torch.float)
sin_cached = (emb.sin() * _mscale).to(torch.float)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


mscale = 0.1 * mscale_all_dim * math.log(factor) + 1.0
softmax_scale = (nope_head_dim + rope_head_dim) ** 0.5 * mscale * mscale

# ----------------------------------------------------------------------------------------------------------------------
# --------------------
# transformer layers.
# --------------------
transformer_output = embedding_output
for layer_index in range(layers):
    # -------
    # MHA.
    # -------
    mha_rms_norm_weight = model[f"model.layers.{layer_index}.input_layernorm.weight"].to(torch.float)
    mha_rms_norm_output = rms_norm(transformer_output, mha_rms_norm_weight)

    wq = model[f"model.layers.{layer_index}.self_attn.q_proj.weight"].to(torch.float)
    wq = wq.reshape(heads, nope_head_dim + rope_head_dim, -1)

    w_k_v = model[f"model.layers.{layer_index}.self_attn.kv_a_proj_with_mqa.weight"].to(torch.float)
    w_k_v_compressed, w_k_v_rope = torch.split(w_k_v, [kv_lora_rank, rope_head_dim], dim=0)

    kv_per_token_compressed = torch.matmul(mha_rms_norm_output, w_k_v_compressed.T)
    kv_per_token_rope = torch.matmul(mha_rms_norm_output, w_k_v_rope.T)

    kv_layer_norm = model[f"model.layers.{layer_index}.self_attn.kv_a_layernorm.weight"].to(torch.float)
    kv_layer_b = model[f"model.layers.{layer_index}.self_attn.kv_b_proj.weight"].to(torch.float)
    kv_per_token_b = torch.matmul(rms_norm(kv_per_token_compressed, kv_layer_norm), kv_layer_b.T)
    kv_per_token_compressed = kv_per_token_b.reshape(len(tokens), heads, nope_head_dim + head_dim).transpose(0, 1)
    k_per_token_nope, v_states = torch.split(kv_per_token_compressed, [nope_head_dim, head_dim], dim=-1)

    kv_seq_len = v_states.shape[-2]
    cos, sin = cos_cached[:kv_seq_len], sin_cached[:kv_seq_len]

    s, d = kv_per_token_rope.shape
    kv_per_token_rope = kv_per_token_rope.view(s, d // 2, 2).transpose(2, 1).reshape(s, d)
    k_per_token_pe = (kv_per_token_rope * cos) + (rotate_half(kv_per_token_rope) * sin)

    qkv_attention_list = []
    for head in range(heads):
        wq_head = wq[head]
        q_layer_head_nope, q_layer_head_pe = torch.split(wq_head, [nope_head_dim, rope_head_dim], dim=0)
        q_per_token_nope = torch.matmul(mha_rms_norm_output, q_layer_head_nope.T)
        q_per_token_pe = torch.matmul(mha_rms_norm_output, q_layer_head_pe.T)

        k_layer_head_nope = k_per_token_nope[head]
        v_per_token = v_states[head]

        kv_seq_len = v_per_token.shape[0]
        cos, sin = cos_cached[:kv_seq_len], sin_cached[:kv_seq_len]

        s, d = q_per_token_pe.shape
        q_per_token_pe = q_per_token_pe.view(s, d // 2, 2).transpose(2, 1).reshape(s, d)
        q_per_token_pe = (q_per_token_pe * cos) + (rotate_half(q_per_token_pe) * sin)
        
        query_states = k_per_token_pe.new_empty(q_per_token_pe.shape[0], nope_head_dim + rope_head_dim)
        query_states[:, : nope_head_dim] = q_per_token_nope
        query_states[:, nope_head_dim:] = q_per_token_pe

        key_states = k_per_token_pe.new_empty(k_per_token_pe.shape[0], nope_head_dim + rope_head_dim)
        key_states[:, : nope_head_dim] = k_layer_head_nope
        key_states[:, nope_head_dim:] = k_per_token_pe

        # dot production attention.
        qk_per_token = torch.matmul(query_states, key_states.T) / softmax_scale
        mask = torch.full((len(tokens), len(tokens)), float("-inf"), device=tokens.device)
        mask = torch.triu(mask, diagonal=1)
        qk_per_token_after_masking = qk_per_token + mask

        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking, dim=1,
                                                                               dtype=torch.float32).to(torch.float)

        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)

        # append.
        qkv_attention_list.append(qkv_attention)

    qkv_attention_all = torch.cat(qkv_attention_list, dim=-1)

    wo = model[f"model.layers.{layer_index}.self_attn.o_proj.weight"].T.to(torch.float)
    mha_output = torch.matmul(qkv_attention_all, wo)

    mha_output_with_residual = mha_output + transformer_output

    # -------
    # FFN.
    # -------
    ffn_rms_norm_weight = model[f"model.layers.{layer_index}.post_attention_layernorm.weight"].to(torch.float)
    ffn_rms_norm_output = rms_norm(mha_output_with_residual, ffn_rms_norm_weight)

    if layer_index == 0:
        w_up = model[f"model.layers.{layer_index}.mlp.up_proj.weight"].T.to(torch.float)
        w_gate = model[f"model.layers.{layer_index}.mlp.gate_proj.weight"].T.to(torch.float)
        w_down = model[f"model.layers.{layer_index}.mlp.down_proj.weight"].T.to(torch.float)

        up = torch.matmul(ffn_rms_norm_output, w_up)
        gate = torch.functional.F.silu(torch.matmul(ffn_rms_norm_output, w_gate))
        ffn_output = torch.matmul(up * gate, w_down)

        transformer_output = ffn_output + mha_output_with_residual
    else:
        # shared exports
        w1 = model[f"model.layers.{layer_index}.mlp.shared_experts.gate_proj.weight"].to(torch.float)
        w2 = model[f"model.layers.{layer_index}.mlp.shared_experts.down_proj.weight"].to(torch.float)
        w3 = model[f"model.layers.{layer_index}.mlp.shared_experts.up_proj.weight"].to(torch.float)
        output_after_shared_exports = torch.matmul(
            torch.functional.F.silu(torch.matmul(ffn_rms_norm_output, w1.T)) * torch.matmul(ffn_rms_norm_output, w3.T),
            w2.T)

        # route exports
        # gate
        gate = model[f'model.layers.{layer_index}.mlp.gate.weight'].to(torch.float)
        gate_logits = torch.matmul(ffn_rms_norm_output, gate.T)
        gate_scores = gate_logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(gate_scores, k=experts_per_tok, dim=-1, sorted=False)
        # route
        cnts = topk_idx.new_zeros((topk_idx.shape[0], routed_experts))
        cnts.scatter_(1, topk_idx, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_idx.view(-1).argsort()
        sorted_tokens = ffn_rms_norm_output[idxs // topk_idx.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            w1 = model[f"model.layers.{layer_index}.mlp.experts.{i}.gate_proj.weight"].to(torch.float)
            w2 = model[f"model.layers.{layer_index}.mlp.experts.{i}.down_proj.weight"].to(torch.float)
            w3 = model[f"model.layers.{layer_index}.mlp.experts.{i}.up_proj.weight"].to(torch.float)
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = torch.matmul(
                torch.functional.F.silu(torch.matmul(tokens_for_this_expert, w1.T)) * torch.matmul(
                    tokens_for_this_expert, w3.T), w2.T)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)

        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        output_after_routeexports = new_x.view(*topk_idx.shape, -1).type(topk_weight.dtype).mul_(
            topk_weight.unsqueeze(dim=-1)).sum(dim=1).type(new_x.dtype)
        ffn_output = output_after_shared_exports + output_after_routeexports

        transformer_output = ffn_output + mha_output_with_residual

# --------------------
# Post Process
# --------------------
output_rms_norm = rms_norm(transformer_output, model["model.norm.weight"]).to(torch.float)
output_logits = torch.matmul(output_rms_norm[-1], model["lm_head.weight"].T.to(torch.float))

# decode last token.
next_token = torch.argmax(output_logits, dim=-1)
next_word = tokenizer.decode([next_token.item()])
print(f"next_word = '{next_word}'")

print(time.strftime("stop_time : %Y-%m-%d %H:%M:%S", time.localtime()))

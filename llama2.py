import time

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

print(time.strftime("start_time: %Y-%m-%d %H:%M:%S", time.localtime()))
# ----------------------------------------------------------------------------------------------------------------------
# model parameters.
hidden_size = 4096
heads = 32
kv_heads = 32
head_dim = hidden_size // heads
GQA = heads // kv_heads
norm_eps = 1e-05
rope_theta = 10000
vocab_size = 32000
layers = 32

# ----------------------------------------------------------------------------------------------------------------------
tokenizer = Tokenizer.from_file("/stores/llm_models/llama/Llama-2-7b-hf/tokenizer.json")

# ----------------------------------------------------------------------------------------------------------------------
model = {}
safetensors = 2
for i in range(1, safetensors + 1):
    safetensor = "/stores/llm_models/llama/Llama-2-7b-hf/model-000%02d-of-000%02d.safetensors" % (i, safetensors)
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
    wk = model[f"model.layers.{layer_index}.self_attn.k_proj.weight"].to(torch.float)
    wv = model[f"model.layers.{layer_index}.self_attn.v_proj.weight"].to(torch.float)

    wq = wq.view(heads, head_dim, hidden_size)
    wk = wk.view(kv_heads, head_dim, hidden_size)
    wv = wv.view(kv_heads, head_dim, hidden_size)

    qkv_attention_list = []
    for head in range(heads):
        wq_head = wq[head].T
        wk_head = wk[head // GQA].T
        wv_head = wv[head // GQA].T

        q = torch.matmul(mha_rms_norm_output, wq_head)
        k = torch.matmul(mha_rms_norm_output, wk_head)
        v = torch.matmul(mha_rms_norm_output, wv_head)

        # rope.
        q_rope = rope(q)
        k_rope = rope(k)

        # dot production attention.
        qk = torch.matmul(q_rope, k_rope.T) / (head_dim ** 0.5)
        mask = torch.full(qk.shape, float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        qk_masked = qk + mask
        qk_masked_softmax = torch.nn.functional.softmax(qk_masked, dim=1)
        qkv_attention = torch.matmul(qk_masked_softmax, v)

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

    w_up = model[f"model.layers.{layer_index}.mlp.up_proj.weight"].T.to(torch.float)
    w_gate = model[f"model.layers.{layer_index}.mlp.gate_proj.weight"].T.to(torch.float)
    w_down = model[f"model.layers.{layer_index}.mlp.down_proj.weight"].T.to(torch.float)

    up = torch.matmul(ffn_rms_norm_output, w_up)
    gate = torch.functional.F.silu(torch.matmul(ffn_rms_norm_output, w_gate))
    ffn_output = torch.matmul(up * gate, w_down)

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

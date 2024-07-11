import torch
from safetensors import safe_open

# ----------------------------------------------------------------------------------------------------------------------
# model parameters.
hidden_size = 4096
heads = 32
kv_heads = 2
head_dim = hidden_size // heads
GQA = heads // kv_heads
norm_eps = 0.00000015625
rope_theta = 500000
vocab_size = 151552
layers = 40


# ----------------------------------------------------------------------------------------------------------------------
def create_tokenizer():
    import base64
    import tiktoken
    pat_str = "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[" \
              "\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"
    mergeable_ranks = {}
    with open("/stores/llm_models/glm/glm-4-9b/tokenizer.model") as f:
        for line in f:
            token, rank = line.strip().split()
            rank = int(rank)
            token = base64.b64decode(token)
            mergeable_ranks[token] = rank

    return tiktoken.Encoding(
        name="tokenizer",
        pat_str=pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens={}
    )
tokenizer = create_tokenizer()

# ----------------------------------------------------------------------------------------------------------------------
model = {}
safetensors = 10
for i in range(1, safetensors + 1):
    safetensor = "/stores/llm_models/glm/glm-4-9b/model-000%02d-of-000%02d.safetensors" % (i, safetensors)
    with safe_open(safetensor, framework="pt") as f:
        for k in f.keys():
            model[k] = f.get_tensor(k)

# ----------------------------------------------------------------------------------------------------------------------
# input.
input_sentence = "I believe the meaning of life is to be"
print(f"input_sentence = {input_sentence}")
tokens = tokenizer.encode(input_sentence)
tokens = torch.tensor(tokens)

# embedding.
embedding = torch.nn.Embedding(vocab_size, hidden_size, dtype=torch.float)
embedding.weight.data.copy_(model["transformer.embedding.word_embeddings.weight"].to(torch.float))
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
    mha_rms_norm_weight = model[f"transformer.encoder.layers.{layer_index}.input_layernorm.weight"].to(torch.float)
    mha_rms_norm_output = rms_norm(transformer_output, mha_rms_norm_weight)

    w_q_k_v = model[f"transformer.encoder.layers.{layer_index}.self_attention.query_key_value.weight"].to(torch.float)
    wq, wk, wv = w_q_k_v.split([heads * head_dim, kv_heads * head_dim, kv_heads * head_dim],
                               dim=0)

    b_q_k_v = model[f"transformer.encoder.layers.{layer_index}.self_attention.query_key_value.bias"].to(torch.float)
    bq, bk, bv = b_q_k_v.split([heads * head_dim, kv_heads * head_dim, kv_heads * head_dim])

    wq = wq.view(heads, head_dim, hidden_size)
    wk = wk.view(kv_heads, head_dim, hidden_size)
    wv = wv.view(kv_heads, head_dim, hidden_size)

    bq = bq.reshape(heads, -1)
    bk = bk.reshape(kv_heads, -1)
    bv = bv.reshape(kv_heads, -1)

    qkv_attention_list = []
    for head in range(heads):
        wq_head = wq[head].T
        wk_head = wk[head // GQA].T
        wv_head = wv[head // GQA].T

        bq_head = bq[head]
        bk_head = bk[head // GQA]
        bv_head = bv[head // GQA]

        q = torch.matmul(mha_rms_norm_output, wq_head) + bq_head
        k = torch.matmul(mha_rms_norm_output, wk_head) + bk_head
        v = torch.matmul(mha_rms_norm_output, wv_head) + bv_head

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

    wo = model[f"transformer.encoder.layers.{layer_index}.self_attention.dense.weight"].T.to(torch.float)
    mha_output = torch.matmul(qkv_attention_all, wo)

    mha_output_with_residual = mha_output + transformer_output

    # -------
    # FFN.
    # -------
    ffn_rms_norm_weight = model[f"transformer.encoder.layers.{layer_index}.post_attention_layernorm.weight"].to(
        torch.float)
    ffn_rms_norm_output = rms_norm(mha_output_with_residual, ffn_rms_norm_weight)

    w_up = model[f"transformer.encoder.layers.{layer_index}.mlp.dense_h_to_4h.weight"].T.to(torch.float)
    w_down = model[f"transformer.encoder.layers.{layer_index}.mlp.dense_4h_to_h.weight"].T.to(torch.float)

    up = torch.matmul(ffn_rms_norm_output, w_up)
    up = torch.chunk(up, 2, dim=-1)
    ffn_output = torch.matmul(torch.functional.F.silu(up[0]) * up[1], w_down)

    transformer_output = ffn_output + mha_output_with_residual

# --------------------
# Post Process
# --------------------
output_rms_norm = rms_norm(transformer_output, model["transformer.encoder.final_layernorm.weight"]).to(torch.float)
output_logits = torch.matmul(output_rms_norm[-1], model["transformer.output_layer.weight"].T.to(torch.float))

# decode last token.
next_token = torch.argmax(output_logits, dim=-1)
next_word = tokenizer.decode([next_token.item()])
print(f"next_word = '{next_word}'")

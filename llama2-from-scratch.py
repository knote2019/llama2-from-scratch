import torch

from sentencepiece import SentencePieceProcessor

# ----------------------------------------------------------------------------------------------------------------------
# llama2 7b parameters.
dim = 4096
vocab_size = 32000
norm_eps = 1e-05
n_heads = 32
n_kv_heads = 32
n_layers = 32
rope_theta = 10000.0

# ----------------------------------------------------------------------------------------------------------------------
tokenizer = SentencePieceProcessor("/stores/llm_models/llama/Llama-2-7b/tokenizer.model")
model = torch.load("/stores/llm_models/llama/Llama-2-7b/consolidated.00.pth")


def rms_norm(x, norm_weights):
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + torch.tensor(norm_eps))) * norm_weights


# ----------------------------------------------------------------------------------------------------------------------
# --------------------
# Input
# --------------------
input_sentence = "I believe the meaning of life is to be"
print(f"input_sentence = {input_sentence}")

tokens = tokenizer.encode(input_sentence)
tokens = [tokenizer.bos_id()] + tokens
tokens = torch.tensor(tokens)

# --------------------
# Embedding Layer
# --------------------
embedding_layer = torch.nn.Embedding(vocab_size, dim)
embedding_layer.weight.data.copy_(model["tok_embeddings.weight"])
token_embeddings_unnormalized = embedding_layer(tokens).to(torch.bfloat16)

# --------------------
# prepare freqs_cis
# --------------------
# freqs = model["rope.freqs"].to(torch.float)
zero_to_one_split_into_64_parts = torch.tensor(range(64)) / 64
freqs = 1.0 / (rope_theta ** zero_to_one_split_into_64_parts)
freqs_for_each_token = torch.outer(torch.arange(len(tokens)), freqs)
freqs_cis = torch.polar(torch.ones_like(freqs_for_each_token), freqs_for_each_token)

# --------------------
# Transformer Layers
# --------------------
final_embedding = token_embeddings_unnormalized
for layer in range(n_layers):
    # -------
    # MHA
    # -------
    qkv_attention_store = []

    layer_embedding_norm = rms_norm(final_embedding, model[f"layers.{layer}.attention_norm.weight"]).to(torch.bfloat16)

    q_layer_weight = model[f"layers.{layer}.attention.wq.weight"]
    k_layer_weight = model[f"layers.{layer}.attention.wk.weight"]
    v_layer_weight = model[f"layers.{layer}.attention.wv.weight"]

    q_layer = q_layer_weight.view(n_heads, q_layer_weight.shape[0] // n_heads, dim)
    k_layer = k_layer_weight.view(n_kv_heads, k_layer_weight.shape[0] // n_kv_heads, dim)
    v_layer = v_layer_weight.view(n_kv_heads, v_layer_weight.shape[0] // n_kv_heads, dim)

    for head in range(n_heads):
        q_layer_head = q_layer[head]
        k_layer_head = k_layer[head]
        v_layer_head = v_layer[head]

        q_per_token = torch.matmul(layer_embedding_norm, q_layer_head.T)
        k_per_token = torch.matmul(layer_embedding_norm, k_layer_head.T)
        v_per_token = torch.matmul(layer_embedding_norm, v_layer_head.T)

        # q rope.
        q_per_token_split_into_pairs = q_per_token.float().view(q_per_token.shape[0], -1, 2)
        q_per_token_as_complex_numbers = torch.view_as_complex(q_per_token_split_into_pairs)
        q_per_token_split_into_pairs_rotated = torch.view_as_real(q_per_token_as_complex_numbers * freqs_cis)
        q_per_token_rotated = q_per_token_split_into_pairs_rotated.view(q_per_token.shape)

        # k rope.
        k_per_token_split_into_pairs = k_per_token.float().view(k_per_token.shape[0], -1, 2)
        k_per_token_as_complex_numbers = torch.view_as_complex(k_per_token_split_into_pairs)
        k_per_token_split_into_pairs_rotated = torch.view_as_real(k_per_token_as_complex_numbers * freqs_cis)
        k_per_token_rotated = k_per_token_split_into_pairs_rotated.view(k_per_token.shape)

        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (128 ** 0.5)
        mask = torch.full((len(token_embeddings_unnormalized), len(token_embeddings_unnormalized)), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        qk_per_token_after_masking = qk_per_token + mask
        qk_per_token_after_masking_after_softmax = torch.nn.functional.softmax(qk_per_token_after_masking,
                                                                               dim=1).to(torch.bfloat16)
        qkv_attention = torch.matmul(qk_per_token_after_masking_after_softmax, v_per_token)
        qkv_attention_store.append(qkv_attention)

    stacked_qkv_attention = torch.cat(qkv_attention_store, dim=-1)

    w_layer = model[f"layers.{layer}.attention.wo.weight"]
    embedding_delta = torch.matmul(stacked_qkv_attention, w_layer.T)

    embedding_after_edit = final_embedding + embedding_delta

    # -------
    # FFN
    # -------
    embedding_after_edit_normalized = rms_norm(embedding_after_edit,
                                               model[f"layers.{layer}.ffn_norm.weight"]).to(torch.bfloat16)
    w1 = model[f"layers.{layer}.feed_forward.w1.weight"]
    w2 = model[f"layers.{layer}.feed_forward.w2.weight"]
    w3 = model[f"layers.{layer}.feed_forward.w3.weight"]
    output_after_feedforward = torch.matmul(
        torch.functional.F.silu(torch.matmul(embedding_after_edit_normalized, w1.T)) *
        torch.matmul(embedding_after_edit_normalized, w3.T), w2.T)
    final_embedding = embedding_after_edit + output_after_feedforward

# --------------------
# Post Process
# --------------------
final_embedding = rms_norm(final_embedding, model["norm.weight"]).to(torch.bfloat16)
logits = torch.matmul(final_embedding[-1], model["output.weight"].T)

# decode last token.
next_token = torch.argmax(logits, dim=-1)
next_word = tokenizer.decode([next_token.item()])
print(f"next_word = {next_word}")

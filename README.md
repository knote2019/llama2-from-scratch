# 1. Introduction

this part will introduce structure of llama2 7b model, and also the links to download model weight.

### 1.1 model structure.

![image](images/llama2-structure.png)
NOTE:
above picture edited from images/llama2-structure.dia with dia image draw tool. you can download dia image draw tool from here: https://sourceforge.net/projects/dia-installer

### 1.2 download model.

from below two site you can download model weight for inference.

| site       | download link                                          |
|------------|--------------------------------------------------------|
| gitee      | https://ai.gitee.com/hf-models/meta-llama/Llama-2-7b   |
| modelscope | https://www.modelscope.cn/models/shakechen/Llama-2-7b  |

### 1.3 model parameters.
```python
import json
import torch

model = torch.load("/stores/llm_models/llama/Llama-2-7b/consolidated.00.pth")
print(json.dumps(list(model.keys()), indent=4))
```

    [
        "tok_embeddings.weight",
        "norm.weight",
        "output.weight",
        "layers.0.attention.wq.weight",
        "layers.0.attention.wk.weight",
        "layers.0.attention.wv.weight",
        "layers.0.attention.wo.weight",
        "layers.0.feed_forward.w1.weight",
        "layers.0.feed_forward.w2.weight",
        "layers.0.feed_forward.w3.weight",
        "layers.0.attention_norm.weight",
        "layers.0.ffn_norm.weight",
        "layers.1.attention.wq.weight",
        "layers.1.attention.wk.weight",
        "layers.1.attention.wv.weight",
        "layers.1.attention.wo.weight",
        "layers.1.feed_forward.w1.weight",
        "layers.1.feed_forward.w2.weight",
        "layers.1.feed_forward.w3.weight",
        "layers.1.attention_norm.weight",
        "layers.1.ffn_norm.weight",
        "layers.2.attention.wq.weight",
        "layers.2.attention.wk.weight",
        "layers.2.attention.wv.weight",
        "layers.2.attention.wo.weight",
        "layers.2.feed_forward.w1.weight",
        "layers.2.feed_forward.w2.weight",
        "layers.2.feed_forward.w3.weight",
        "layers.2.attention_norm.weight",
        "layers.2.ffn_norm.weight",
        "layers.3.attention.wq.weight",
        "layers.3.attention.wk.weight",
        "layers.3.attention.wv.weight",
        "layers.3.attention.wo.weight",
        "layers.3.feed_forward.w1.weight",
        "layers.3.feed_forward.w2.weight",
        "layers.3.feed_forward.w3.weight",
        "layers.3.attention_norm.weight",
        "layers.3.ffn_norm.weight",
        "layers.4.attention.wq.weight",
        "layers.4.attention.wk.weight",
        "layers.4.attention.wv.weight",
        "layers.4.attention.wo.weight",
        "layers.4.feed_forward.w1.weight",
        "layers.4.feed_forward.w2.weight",
        "layers.4.feed_forward.w3.weight",
        "layers.4.attention_norm.weight",
        "layers.4.ffn_norm.weight",
        "layers.5.attention.wq.weight",
        "layers.5.attention.wk.weight",
        "layers.5.attention.wv.weight",
        "layers.5.attention.wo.weight",
        "layers.5.feed_forward.w1.weight",
        "layers.5.feed_forward.w2.weight",
        "layers.5.feed_forward.w3.weight",
        "layers.5.attention_norm.weight",
        "layers.5.ffn_norm.weight",
        "layers.6.attention.wq.weight",
        "layers.6.attention.wk.weight",
        "layers.6.attention.wv.weight",
        "layers.6.attention.wo.weight",
        "layers.6.feed_forward.w1.weight",
        "layers.6.feed_forward.w2.weight",
        "layers.6.feed_forward.w3.weight",
        "layers.6.attention_norm.weight",
        "layers.6.ffn_norm.weight",
        "layers.7.attention.wq.weight",
        "layers.7.attention.wk.weight",
        "layers.7.attention.wv.weight",
        "layers.7.attention.wo.weight",
        "layers.7.feed_forward.w1.weight",
        "layers.7.feed_forward.w2.weight",
        "layers.7.feed_forward.w3.weight",
        "layers.7.attention_norm.weight",
        "layers.7.ffn_norm.weight",
        "layers.8.attention.wq.weight",
        "layers.8.attention.wk.weight",
        "layers.8.attention.wv.weight",
        "layers.8.attention.wo.weight",
        "layers.8.feed_forward.w1.weight",
        "layers.8.feed_forward.w2.weight",
        "layers.8.feed_forward.w3.weight",
        "layers.8.attention_norm.weight",
        "layers.8.ffn_norm.weight",
        "layers.9.attention.wq.weight",
        "layers.9.attention.wk.weight",
        "layers.9.attention.wv.weight",
        "layers.9.attention.wo.weight",
        "layers.9.feed_forward.w1.weight",
        "layers.9.feed_forward.w2.weight",
        "layers.9.feed_forward.w3.weight",
        "layers.9.attention_norm.weight",
        "layers.9.ffn_norm.weight",
        "layers.10.attention.wq.weight",
        "layers.10.attention.wk.weight",
        "layers.10.attention.wv.weight",
        "layers.10.attention.wo.weight",
        "layers.10.feed_forward.w1.weight",
        "layers.10.feed_forward.w2.weight",
        "layers.10.feed_forward.w3.weight",
        "layers.10.attention_norm.weight",
        "layers.10.ffn_norm.weight",
        "layers.11.attention.wq.weight",
        "layers.11.attention.wk.weight",
        "layers.11.attention.wv.weight",
        "layers.11.attention.wo.weight",
        "layers.11.feed_forward.w1.weight",
        "layers.11.feed_forward.w2.weight",
        "layers.11.feed_forward.w3.weight",
        "layers.11.attention_norm.weight",
        "layers.11.ffn_norm.weight",
        "layers.12.attention.wq.weight",
        "layers.12.attention.wk.weight",
        "layers.12.attention.wv.weight",
        "layers.12.attention.wo.weight",
        "layers.12.feed_forward.w1.weight",
        "layers.12.feed_forward.w2.weight",
        "layers.12.feed_forward.w3.weight",
        "layers.12.attention_norm.weight",
        "layers.12.ffn_norm.weight",
        "layers.13.attention.wq.weight",
        "layers.13.attention.wk.weight",
        "layers.13.attention.wv.weight",
        "layers.13.attention.wo.weight",
        "layers.13.feed_forward.w1.weight",
        "layers.13.feed_forward.w2.weight",
        "layers.13.feed_forward.w3.weight",
        "layers.13.attention_norm.weight",
        "layers.13.ffn_norm.weight",
        "layers.14.attention.wq.weight",
        "layers.14.attention.wk.weight",
        "layers.14.attention.wv.weight",
        "layers.14.attention.wo.weight",
        "layers.14.feed_forward.w1.weight",
        "layers.14.feed_forward.w2.weight",
        "layers.14.feed_forward.w3.weight",
        "layers.14.attention_norm.weight",
        "layers.14.ffn_norm.weight",
        "layers.15.attention.wq.weight",
        "layers.15.attention.wk.weight",
        "layers.15.attention.wv.weight",
        "layers.15.attention.wo.weight",
        "layers.15.feed_forward.w1.weight",
        "layers.15.feed_forward.w2.weight",
        "layers.15.feed_forward.w3.weight",
        "layers.15.attention_norm.weight",
        "layers.15.ffn_norm.weight",
        "layers.16.attention.wq.weight",
        "layers.16.attention.wk.weight",
        "layers.16.attention.wv.weight",
        "layers.16.attention.wo.weight",
        "layers.16.feed_forward.w1.weight",
        "layers.16.feed_forward.w2.weight",
        "layers.16.feed_forward.w3.weight",
        "layers.16.attention_norm.weight",
        "layers.16.ffn_norm.weight",
        "layers.17.attention.wq.weight",
        "layers.17.attention.wk.weight",
        "layers.17.attention.wv.weight",
        "layers.17.attention.wo.weight",
        "layers.17.feed_forward.w1.weight",
        "layers.17.feed_forward.w2.weight",
        "layers.17.feed_forward.w3.weight",
        "layers.17.attention_norm.weight",
        "layers.17.ffn_norm.weight",
        "layers.18.attention.wq.weight",
        "layers.18.attention.wk.weight",
        "layers.18.attention.wv.weight",
        "layers.18.attention.wo.weight",
        "layers.18.feed_forward.w1.weight",
        "layers.18.feed_forward.w2.weight",
        "layers.18.feed_forward.w3.weight",
        "layers.18.attention_norm.weight",
        "layers.18.ffn_norm.weight",
        "layers.19.attention.wq.weight",
        "layers.19.attention.wk.weight",
        "layers.19.attention.wv.weight",
        "layers.19.attention.wo.weight",
        "layers.19.feed_forward.w1.weight",
        "layers.19.feed_forward.w2.weight",
        "layers.19.feed_forward.w3.weight",
        "layers.19.attention_norm.weight",
        "layers.19.ffn_norm.weight",
        "layers.20.attention.wq.weight",
        "layers.20.attention.wk.weight",
        "layers.20.attention.wv.weight",
        "layers.20.attention.wo.weight",
        "layers.20.feed_forward.w1.weight",
        "layers.20.feed_forward.w2.weight",
        "layers.20.feed_forward.w3.weight",
        "layers.20.attention_norm.weight",
        "layers.20.ffn_norm.weight",
        "layers.21.attention.wq.weight",
        "layers.21.attention.wk.weight",
        "layers.21.attention.wv.weight",
        "layers.21.attention.wo.weight",
        "layers.21.feed_forward.w1.weight",
        "layers.21.feed_forward.w2.weight",
        "layers.21.feed_forward.w3.weight",
        "layers.21.attention_norm.weight",
        "layers.21.ffn_norm.weight",
        "layers.22.attention.wq.weight",
        "layers.22.attention.wk.weight",
        "layers.22.attention.wv.weight",
        "layers.22.attention.wo.weight",
        "layers.22.feed_forward.w1.weight",
        "layers.22.feed_forward.w2.weight",
        "layers.22.feed_forward.w3.weight",
        "layers.22.attention_norm.weight",
        "layers.22.ffn_norm.weight",
        "layers.23.attention.wq.weight",
        "layers.23.attention.wk.weight",
        "layers.23.attention.wv.weight",
        "layers.23.attention.wo.weight",
        "layers.23.feed_forward.w1.weight",
        "layers.23.feed_forward.w2.weight",
        "layers.23.feed_forward.w3.weight",
        "layers.23.attention_norm.weight",
        "layers.23.ffn_norm.weight",
        "layers.24.attention.wq.weight",
        "layers.24.attention.wk.weight",
        "layers.24.attention.wv.weight",
        "layers.24.attention.wo.weight",
        "layers.24.feed_forward.w1.weight",
        "layers.24.feed_forward.w2.weight",
        "layers.24.feed_forward.w3.weight",
        "layers.24.attention_norm.weight",
        "layers.24.ffn_norm.weight",
        "layers.25.attention.wq.weight",
        "layers.25.attention.wk.weight",
        "layers.25.attention.wv.weight",
        "layers.25.attention.wo.weight",
        "layers.25.feed_forward.w1.weight",
        "layers.25.feed_forward.w2.weight",
        "layers.25.feed_forward.w3.weight",
        "layers.25.attention_norm.weight",
        "layers.25.ffn_norm.weight",
        "layers.26.attention.wq.weight",
        "layers.26.attention.wk.weight",
        "layers.26.attention.wv.weight",
        "layers.26.attention.wo.weight",
        "layers.26.feed_forward.w1.weight",
        "layers.26.feed_forward.w2.weight",
        "layers.26.feed_forward.w3.weight",
        "layers.26.attention_norm.weight",
        "layers.26.ffn_norm.weight",
        "layers.27.attention.wq.weight",
        "layers.27.attention.wk.weight",
        "layers.27.attention.wv.weight",
        "layers.27.attention.wo.weight",
        "layers.27.feed_forward.w1.weight",
        "layers.27.feed_forward.w2.weight",
        "layers.27.feed_forward.w3.weight",
        "layers.27.attention_norm.weight",
        "layers.27.ffn_norm.weight",
        "layers.28.attention.wq.weight",
        "layers.28.attention.wk.weight",
        "layers.28.attention.wv.weight",
        "layers.28.attention.wo.weight",
        "layers.28.feed_forward.w1.weight",
        "layers.28.feed_forward.w2.weight",
        "layers.28.feed_forward.w3.weight",
        "layers.28.attention_norm.weight",
        "layers.28.ffn_norm.weight",
        "layers.29.attention.wq.weight",
        "layers.29.attention.wk.weight",
        "layers.29.attention.wv.weight",
        "layers.29.attention.wo.weight",
        "layers.29.feed_forward.w1.weight",
        "layers.29.feed_forward.w2.weight",
        "layers.29.feed_forward.w3.weight",
        "layers.29.attention_norm.weight",
        "layers.29.ffn_norm.weight",
        "layers.30.attention.wq.weight",
        "layers.30.attention.wk.weight",
        "layers.30.attention.wv.weight",
        "layers.30.attention.wo.weight",
        "layers.30.feed_forward.w1.weight",
        "layers.30.feed_forward.w2.weight",
        "layers.30.feed_forward.w3.weight",
        "layers.30.attention_norm.weight",
        "layers.30.ffn_norm.weight",
        "layers.31.attention.wq.weight",
        "layers.31.attention.wk.weight",
        "layers.31.attention.wv.weight",
        "layers.31.attention.wo.weight",
        "layers.31.feed_forward.w1.weight",
        "layers.31.feed_forward.w2.weight",
        "layers.31.feed_forward.w3.weight",
        "layers.31.attention_norm.weight",
        "layers.31.ffn_norm.weight",
        "rope.freqs"
    ]

# 2. Operators.

this part will introduce operators that used in llama2 7b model.

### 2.1 tokenizer.

### 2.2 embedding.

### 2.3 RMS (Root Mean Square Normalization).

### 2.3 RoPE (Rotary Position Embedding).

### 2.4 MHA (Multi-Headed Attention).

### 2.5 FFN (Multi-Headed Attention).

### 2.6 LM head.

# 3. Inference code.

this part shows the python code of llama2 7b inference.

```python
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


def rms_norm(tensor, norm_weights):
    return (tensor * torch.rsqrt(tensor.pow(2).mean(-1, keepdim=True) + torch.tensor(norm_eps))) * norm_weights


# ----------------------------------------------------------------------------------------------------------------------
# set embedding_layer.

# set freqs

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
# zero_to_one_split_into_64_parts = torch.tensor(range(64)) / 64
# freqs = 1.0 / (rope_theta**zero_to_one_split_into_64_parts)
freqs = model["rope.freqs"].to(torch.float)
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
    q_layer = model[f"layers.{layer}.attention.wq.weight"]
    q_layer = q_layer.view(n_heads, q_layer.shape[0] // n_heads, dim)
    k_layer = model[f"layers.{layer}.attention.wk.weight"]
    k_layer = k_layer.view(n_kv_heads, k_layer.shape[0] // n_kv_heads, dim)
    v_layer = model[f"layers.{layer}.attention.wv.weight"]
    v_layer = v_layer.view(n_kv_heads, v_layer.shape[0] // n_kv_heads, dim)
    w_layer = model[f"layers.{layer}.attention.wo.weight"]
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

        qk_per_token = torch.matmul(q_per_token_rotated, k_per_token_rotated.T) / (128**0.5)
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
```

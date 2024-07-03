import os

import fire
import torch as t
import requests
from safetensors.torch import load_file
from transformers import AutoTokenizer
from tqdm import tqdm
from torch.nn.functional import gelu, softmax, layer_norm, linear
from termcolor import colored as c


if not os.path.exists("model.safetensors"):
    print("Downloading model.safetensors...")
    url = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with open("model.safetensors", "wb") as f, tqdm(
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                size = f.write(chunk)
                pbar.update(size)


tensors = load_file("model.safetensors")
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
N_CTX = 1024
N_HEAD = 12
N_LAYER = 12


def feed_forward_network(x, i):
    return linear(
        gelu(
            linear(
                x, tensors[f"h.{i}.mlp.c_fc.weight"].T, tensors[f"h.{i}.mlp.c_fc.bias"]
            )
        ),
        tensors[f"h.{i}.mlp.c_proj.weight"].T,
        tensors[f"h.{i}.mlp.c_proj.bias"],
    )


def attention(q, k, v, mask):
    return (
        softmax(
            (q @ k.transpose(-2, -1)) / t.sqrt(t.tensor(q.shape[-1], dtype=q.dtype))
            + mask,
            dim=-1,
        )
        @ v
    )


def multihead_attention(x, i):
    x = linear(
        x, tensors[f"h.{i}.attn.c_attn.weight"].T, tensors[f"h.{i}.attn.c_attn.bias"]
    )
    qkv = t.split(x, x.shape[-1] // 3, dim=-1)
    qkv_heads = [t.split(a, a.shape[-1] // N_HEAD, dim=-1) for a in qkv]
    causal_mask = (1 - t.tril(t.ones(x.shape[0], x.shape[0]))) * -1e10
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    x = linear(
        t.cat(out_heads, dim=-1),
        tensors[f"h.{i}.attn.c_proj.weight"].T,
        tensors[f"h.{i}.attn.c_proj.bias"],
    )
    return x


def transformer_block(x, i):
    x += multihead_attention(
        layer_norm(
            x, x.shape[-1:], tensors[f"h.{i}.ln_1.weight"], tensors[f"h.{i}.ln_1.bias"]
        ),
        i,
    )
    x += feed_forward_network(
        layer_norm(
            x, x.shape[-1:], tensors[f"h.{i}.ln_2.weight"], tensors[f"h.{i}.ln_2.bias"]
        ),
        i,
    )
    return x


def gpt2(inputs):
    x = tensors["wte.weight"][inputs] + tensors["wpe.weight"][t.arange(len(inputs))]
    for i in range(N_LAYER):
        x = transformer_block(x, i)
    return (
        layer_norm(x, x.shape[-1:], tensors["ln_f.weight"], tensors["ln_f.bias"])
        @ tensors["wte.weight"].T
    )


def generate(inputs, n_tokens_to_generate, temperature=1.0):
    for _ in tqdm(range(n_tokens_to_generate), "Generating"):
        logits = gpt2(inputs)
        logits = logits[-1] / temperature
        probs = softmax(logits, dim=-1)
        next_id = t.multinomial(probs, num_samples=1).item()
        inputs = t.cat([inputs, t.tensor([next_id])])

    return inputs[-n_tokens_to_generate:].tolist()


def main(
    prompt: str,
    n_tokens_to_generate: int = 50,
    temperature: float = 0.7,
    seed: int = 0,
):
    if seed != 0:
        t.manual_seed(seed)
    temperature = max(temperature, 1e-10)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").view(-1)
    assert len(input_ids) + n_tokens_to_generate < N_CTX
    output_ids = generate(input_ids, n_tokens_to_generate, temperature)
    print()
    print(c(prompt, "red"), end="")
    print(c(tokenizer.decode(output_ids, skip_special_tokens=True), "blue"))
    print()


if __name__ == "__main__":
    fire.Fire(main)

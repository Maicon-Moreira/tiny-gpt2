# Tiny GPT-2

Minimal GPT-2 implementation in PyTorch (~100 lines).

Uses [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) tokenizer and weights.

## Install

```bash
pip install torch transformers safetensors fire tqdm termcolor requests
```

## Usage

```bash
python main.py --prompt "Your text" \
               --n_tokens_to_generate 50 \  # optional
               --temperature 0.7            # optional
               --seed 0                     # optional, random if 0
```

Example:
```bash
python main.py --prompt "Once upon a time"
```

## Example Outputs

```
Once upon a time it seemed as if the world would burn itself out.
The world doesn't move. But when it does, it does so in such a way 
that it is hard to resist the pressure of moving. When the world 
does move, it moves
```

```
Once upon a time, the Raccoon Express arrived at the station and 
explained that the Raccoon found the bundle of Raccoon artifacts, 
which contained the Red and Blue Raccoon artifacts. "It's a good 
time to be a member of
```

```
Once upon a time, understanding the beauty of the human mind, we 
can fully understand the essence of the beauty of the human mind. 
The same is true of the two senses of sight, which are modalities 
of the senses, and of the senses of smell and taste
```
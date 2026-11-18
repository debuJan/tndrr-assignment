"""Task 1 (part 4) — GPU memory needed to fine-tune a 7B model.

Computes exact LoRA trainable-parameter counts from the real Qwen2.5-7B
config (fetched from Hugging Face), then works out memory for:
  (a) full fine-tuning,  (b) LoRA,  (c) QLoRA
with assumptions stated explicitly. No weights are downloaded.
"""
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_URL = ("https://huggingface.co/Qwen/Qwen2.5-7B/raw/main/config.json")
HEADERS = {"User-Agent": "tndrr-assignment/1.0"}


def fetch_config():
    req = urllib.request.Request(CONFIG_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    cfg = fetch_config()
    n_layers = cfg["num_hidden_layers"]
    hidden = cfg["hidden_size"]
    inter = cfg["intermediate_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = cfg.get("head_dim", hidden // heads)
    vocab = cfg["vocab_size"]
    tie = cfg.get("tie_word_embeddings", False)

    # total params (approx, embeddings + per-layer linears + norms)
    embed = vocab * hidden * (1 if tie else 2)
    attn = (heads * head_dim * hidden + kv_heads * head_dim * hidden * 2
            + hidden * heads * head_dim)
    mlp = hidden * inter * 3
    per_layer = attn + mlp + 2 * hidden  # + rmsnorm weights
    total = embed + n_layers * per_layer
    print(f"Qwen2.5-7B config: layers={n_layers} hidden={hidden} "
          f"intermediate={inter} heads={heads} kv_heads={kv_heads} "
          f"head_dim={head_dim} vocab={vocab} tie={tie}")
    print(f"approx total params: {total / 1e9:.2f}B")

    # ---- LoRA trainable params, r=8, all linear modules ----
    r = 8
    per_layer_lora = (
        r * (hidden + heads * head_dim)          # q_proj
        + r * (hidden + kv_heads * head_dim)     # k_proj
        + r * (hidden + kv_heads * head_dim)     # v_proj
        + r * (heads * head_dim + hidden)        # o_proj
        + r * (hidden + inter)                   # gate_proj
        + r * (hidden + inter)                   # up_proj
        + r * (inter + hidden)                   # down_proj
    )
    lora_params = per_layer_lora * n_layers
    print(f"LoRA r={r} trainable params: {lora_params:,} "
          f"({100 * lora_params / total:.3f}% of model)")

    GB = 1024 ** 3
    B = float(total)  # use the real parameter count from the config

    # (a) Full fine-tuning, mixed precision AdamW:
    #   bf16 weights 2B/param, fp32 master weights 4B/param,
    #   Adam m+v fp32 8B/param, grads bf16 2B/param (fp32 grads: 4B/param)
    full = {
        "weights_bf16_GB": B * 2 / GB,
        "master_weights_fp32_GB": B * 4 / GB,
        "adam_states_fp32_GB": B * 8 / GB,
        "grads_bf16_GB": B * 2 / GB,
        "grads_fp32_GB": B * 4 / GB,
    }
    full_total_bf16grads = (full["weights_bf16_GB"] + full["master_weights_fp32_GB"]
                            + full["adam_states_fp32_GB"] + full["grads_bf16_GB"])
    full_total_fp32grads = (full["weights_bf16_GB"] + full["master_weights_fp32_GB"]
                            + full["adam_states_fp32_GB"] + full["grads_fp32_GB"])

    # (b) LoRA: frozen bf16 base + fp32 LoRA weights + Adam on LoRA only
    #     + activations (batch 1, seq 2048, no grad checkpointing ~ 2-4 GB)
    lora = {
        "base_weights_bf16_GB": B * 2 / GB,
        "lora_weights_fp32_GB": lora_params * 4 / GB,
        "adam_states_fp32_GB": lora_params * 8 / GB,
        "lora_grads_fp32_GB": lora_params * 4 / GB,
        "activations_est_GB": 3.0,
    }
    lora_total = sum(lora.values())

    # (c) QLoRA: NF4 base (0.5 B/param) + same LoRA bits, activations ~1-2 GB
    #     (gradient checkpointing on)
    qlora = {
        "base_weights_NF4_GB": B * 0.5 / GB,
        "lora_weights_fp32_GB": lora_params * 4 / GB,
        "adam_states_fp32_GB": lora_params * 8 / GB,
        "lora_grads_fp32_GB": lora_params * 4 / GB,
        "activations_est_GB": 1.5,
    }
    qlora_total = sum(qlora.values())

    out = {
        "model": "Qwen2.5-7B",
        "config": {"layers": n_layers, "hidden": hidden,
                   "intermediate": inter, "heads": heads,
                   "kv_heads": kv_heads, "head_dim": head_dim},
        "total_params_B": round(total / 1e9, 2),
        "lora_rank": r,
        "lora_trainable_params": lora_params,
        "lora_trainable_pct": round(100 * lora_params / total, 3),
        "full_finetune": {**{k: round(v, 1) for k, v in full.items()},
                          "total_bf16_grads_GB": round(full_total_bf16grads, 1),
                          "total_fp32_grads_GB": round(full_total_fp32grads, 1)},
        "lora": {k: round(v, 2) for k, v in lora.items()}
                | {"total_GB": round(lora_total, 1)},
        "qlora": {k: round(v, 2) for k, v in qlora.items()}
                 | {"total_GB": round(qlora_total, 1)},
        "assumptions": [
            "Mixed-precision AdamW: bf16 weights + fp32 master weights + fp32 Adam m/v.",
            "LoRA r=8 on all 7 linear modules per layer (q,k,v,o,gate,up,down).",
            "Activations estimated for batch 1, seq 2048; LoRA without grad checkpointing, QLoRA with it.",
            "Excludes CUDA context / fragmentation overhead (~1 GB).",
        ],
    }
    with open(os.path.join(HERE, "memory_calc.json"), "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n(a) FULL fine-tune:  ~{full_total_bf16grads:.0f} GB "
          f"(bf16 grads) / ~{full_total_fp32grads:.0f} GB (fp32 grads)")
    print(f"    weights {full['weights_bf16_GB']:.0f} + master "
          f"{full['master_weights_fp32_GB']:.0f} + Adam "
          f"{full['adam_states_fp32_GB']:.0f} + grads "
          f"{full['grads_bf16_GB']:.0f} GB")
    print(f"(b) LoRA r=8:        ~{lora_total:.1f} GB "
          f"(base {lora['base_weights_bf16_GB']:.0f} + LoRA+optim "
          f"{lora['lora_weights_fp32_GB'] + lora['adam_states_fp32_GB'] + lora['lora_grads_fp32_GB']:.2f} "
          f"+ activations ~{lora['activations_est_GB']:.0f})")
    print(f"(c) QLoRA (NF4):     ~{qlora_total:.1f} GB "
          f"(base {qlora['base_weights_NF4_GB']:.1f} + LoRA+optim "
          f"{qlora['lora_weights_fp32_GB'] + qlora['adam_states_fp32_GB'] + qlora['lora_grads_fp32_GB']:.2f} "
          f"+ activations ~{qlora['activations_est_GB']:.1f})")
    print("wrote task1/memory_calc.json")


if __name__ == "__main__":
    main()

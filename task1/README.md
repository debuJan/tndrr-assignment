# Task 1 — LoRA / QLoRA fine-tuning

Fine-tune a small open-weight model on a narrow task, and compare base vs
fine-tuned on a held-out test set.

## The task

Plain-English reminder → fixed JSON:

```json
{"action": "remind", "task": "call mom", "date": "2026-11-18",
 "time": "18:00", "priority": "high"}
```

The data is synthetic (240 train / 60 held-out test, seed 42, no overlap):
reminders are built from a small set of tasks, dates and time phrases
("6pm" → "18:00", "noon" → "12:00"), so the real learning is **time
normalisation + exact JSON format**, not world knowledge. Training uses
completion-only loss (loss is computed on the JSON answer only, not on
the prompt).

## Model and settings

- **Base model:** Qwen/Qwen2.5-0.5B, bfloat16 (0.5B params, fits on CPU)
- **LoRA:** rank 8, alpha 16, dropout 0.05, target modules q/k/v/o proj
- **Training:** LR 2e-4, 3 epochs, batch size 4, AdamW
- **Trainable params:** 1,081,344 (0.22% of the model)
- **Run:** 180 steps in 949 s (~5.3 s/step, CPU, 2 threads), peak RSS 2.7 GB

**Why these settings:** rank 8 is plenty for a format task — the model
only has to learn a fixed output shape and a time mapping, not new
knowledge. Alpha 16 (2x rank) is the standard scaling. Targeting only
the four attention projections keeps trainable params tiny while covering
the part of the model that has to learn the extraction mapping. LR 2e-4
is the usual LoRA range; 3 epochs because the loss had already collapsed
by epoch 2 (see below) — a 4th epoch would add nothing.

## Results (60 held-out examples, greedy decoding)

| model | exact match | valid JSON | action | task | date | time | priority |
|-------|------------|------------|--------|------|------|------|----------|
| base Qwen2.5-0.5B | 0.000 | 0.983 | — | — | — | — | — |
| + LoRA (this run) | **1.000** | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

Epoch losses: 0.0919 → 0.0003 → 0.0002. The loss collapses after epoch
1 — the model memorises the template almost immediately.

**Read this honestly:** 100% exact match on this held-out set means the
model learned the format completely, *not* that it is generally good at
reminders. The held-out examples use the same templates as training, so
this measures format learning. The edge cases below are the real test of
robustness, and that is where it breaks.

## Failure examples

**Fine-tuned model on the 60 held-out examples: zero failures.** There
are no FT failure examples to show from the test set, so I probed the
model with 3 out-of-template inputs instead (see `probe.py`,
`probe_results.json`):

1. **Missing time** — "Remind me to call mom tomorrow"
   → `{"instruction": "Remind me to call mom tomorrow"}`. The model
   falls back to the *base* behaviour (echoing the input) when the time
   slot is empty. It never learned what to do with missing information.
2. **Ambiguous hour** — "Team meeting at 6 on Friday"
   → `{"action": "remind", "task": "team meeting", "date": "Friday",
   "time": "18:00", "priority": "medium"}`. It silently guessed 18:00
   (6 could equally be 06:00) and left the date un-normalised
   ("Friday" instead of a real date). No uncertainty is expressed.
3. **Two tasks in one sentence** — "Remind me to buy milk and call
   dad at 5pm tomorrow" → `{"task": "buy milk", ...}`. It kept only the
   first task and silently dropped "call dad". The single-task schema
   cannot represent this, and the model does not say so.

**Base model failures (what LoRA fixed):** on all 3 probed test prompts
the base model produced chatty prose + a markdown fence and echoed the
input as `{"instruction": "..."}` instead of extracting the fields —
e.g. "Set a reminder for call the bank on on Sunday at 9pm" →
`{"instruction": "Set a reminder for call the bank on on Sunday at
9pm"}`. That output *is* valid JSON (hence the 0.983 valid-JSON rate)
but has the wrong schema and no extracted fields. On longer generations
it also degenerates into repeating "You are a helpful assistant".

## How much GPU memory to fine-tune a 7B model? (worked calculation)

Real Qwen2.5-7B config from Hugging Face: 28 layers, hidden 3584,
intermediate 18944, 28 heads, 4 KV heads, vocab 152064, ≈ 7.62B params.
Assumptions: mixed-precision AdamW (bf16 weights + fp32 master weights
+ fp32 Adam m/v states). LoRA r=8 on all 7 linear modules per layer =
20,185,088 trainable params (0.265%). Full numbers in `memory_calc.py`
/ `memory_calc.json`.

| method | memory | breakdown |
|--------|--------|-----------|
| (a) Full fine-tune (bf16 grads) | **≈ 113 GB** | weights 14.2 + master 28.4 + Adam 56.7 + grads 14.2 |
| (a) Full fine-tune (fp32 grads) | **≈ 128 GB** | grads 28.4 instead of 14.2 |
| (b) LoRA (base in bf16) | **≈ 17.5 GB** | base 14.2 + LoRA weights/optim 0.3 + activations ~3 |
| (c) QLoRA (base in NF4) | **≈ 5.3 GB** | base 3.5 + LoRA weights/optim 0.3 + activations ~1.5 |

**Why the gap is so large:** full fine-tuning must keep a gradient and
two Adam states for *every* one of the 7.6B weights (Adam alone is ~57
GB), while LoRA only trains 20M adapter weights — the 7.6B base weights
are frozen and need no gradients or optimizer states. QLoRA goes further
by storing the frozen base in 4-bit (NF4), cutting the 14 GB base to
3.5 GB, which is what makes 7B fine-tuning fit on a single consumer GPU.

## Files

- `train.py` — data generation, LoRA training, base vs FT evaluation
  (`--pilot` runs 5 steps; `--skip-base` reuses a completed base eval)
- `memory_calc.py` / `memory_calc.json` — the 7B memory calculation
- `probe.py` / `probe_results.json` — base failures + FT edge cases
- `results.json` — full metrics; `lora_adapter/` — trained adapter

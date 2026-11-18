# Task 3 — Running a model locally & optimisation

Qwen2.5-0.5B-Instruct run locally in two quantisation settings with
llama.cpp (`llama-cpp-python`): **F16** vs **Q4_K_M**. Same 10 prompts,
greedy decoding (temperature 0), 60 tokens max each.

## How to run

```bash
python bench.py   # downloads both GGUF files, benchmarks, writes results.json
```

## Results (from real runs, see `results.json`)

| setting | tokens/sec | peak RSS* | model file | load time |
|---------|-----------|----------|------------|-----------|
| F16     | **3.3**   | 1044 MB  | 949 MB     | 3.2 s     |
| Q4_K_M  | **4.2**   | 1044 MB  | 380 MB     | 2.5 s     |

\* Peak RSS is the process peak (both models ran in one process, so the
number is the same for both — the F16 load set it). The real memory
difference is the file on disk: **949 MB vs 380 MB (2.5x smaller)**.

**Speed:** Q4_K_M was ~27% faster (4.2 vs 3.3 tok/s). Both numbers were
measured while the Task 1 LoRA training job ran at the same time on the
same 1.5 shared CPUs, so the absolute tok/s values are lower bounds — the
relative gap is the meaningful result. Generation with small batches is
memory-bandwidth bound, so the smaller weights are read faster.

Output quality on the same 10 prompts: compared side by side in
`results.json`. **No visible quality loss from Q4_K_M on these prompts.**
Both answered every factual question correctly (Paris, Jupiter, primary
colours, "Buenos días"). Two small differences worth noting:

- **17 × 6:** F16 rambled ("add 17 to itself 6 times...") while Q4_K_M
  went straight to multiplying — the quantised model was *more* direct.
- **Python `reverse a string`:** both models started by explaining
  instead of giving code, and both hit the 60-token cap — a prompt/limit
  effect, not a quantisation effect.
- **"The quick brown fox":** both completed it correctly with "jumps
  over the lazy dog".

## In my own words

**What the KV cache is:** when a transformer writes text one token at a
time, every new token needs attention over *all* previous tokens. Without
a cache, the model would recompute the Key and Value vectors for every past
token at every step — work that grows quadratically with length. The KV
cache simply stores the K and V vectors computed for past tokens, so each
new token only computes its own Q, K, V and attends against the stored
ones. It trades memory for speed. Its size is
2 (K and V) × layers × hidden size × sequence length × batch size × bytes
per value — so it grows with longer conversations and with more users at
the same time.

**Why quantisation saves memory:** a model's weights are just numbers. At
16-bit precision each weight takes 2 bytes; at 4-bit it takes half a byte,
so the same model is ~4x smaller on disk and in RAM. Generation with small
batches is limited by how fast weights can be read from memory, not by
arithmetic — smaller weights mean less reading per token, so it also runs
faster. The cost is rounding: 4-bit weights are approximations, which can
slightly change the model's answers. The KV cache can be quantised the same
way. That trade-off — memory and speed vs a little quality — is exactly
what the benchmark above measures.

## Back-of-envelope plan: 20 users at once on one 24 GB GPU

Assumptions: Qwen2.5-0.5B-Instruct, Q4_K_M weights (~0.4 GB), served with
vLLM (continuous batching), average conversation ~256 tokens of context,
~60 tokens generated per reply.

- **Weights:** ~0.4 GB (Q4_K_M), loaded once and shared by all users.
- **KV cache per user:** 2 × 24 layers × 896 hidden × 2 bytes (fp16 KV)
  × 256 tokens ≈ 22 MB. For 20 users: ≈ 0.44 GB.
- **Total:** weights 0.4 + KV 0.44 + activations/overhead ~0.5 ≈ **~1.5 GB
  — a small fraction of 24 GB.**
- The same 24 GB card could instead serve a 7B model in Q4 (~4 GB
  weights): KV per user ≈ 2 × 28 × 3584 × 2 × 256 ≈ 100 MB, so 20 users
  ≈ 2 GB, total ≈ 7 GB — still fits comfortably.
- **Throughput:** generation is memory-bandwidth bound. Batching 20 users
  together means the weights are read once per step for all 20 sequences,
  so total tokens/sec rises with batch size until the GPU's memory
  bandwidth saturates; per-user latency rises a little because each step
  does more work. If latency matters, cap concurrent sequences (e.g. 16)
  and queue the rest, and cap max context length.
- **If users grow beyond 20:** the KV cache is the first thing to
  squeeze — shorter max context, fp8/int8 KV cache, or prefix caching for
  shared system prompts. After that, a second GPU or a smaller model.

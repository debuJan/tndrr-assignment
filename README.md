# Tndrr AI/ML Intern — Assignment

Three tasks, all run locally on CPU. Every number in the report comes from
my own runs; see `Tndrr_Assignment_Report.pdf` for the 4-page summary.

- **Task 1 — LoRA fine-tuning** (`task1/`): Qwen2.5-0.5B fine-tuned with
  LoRA (r=8) on a reminder-to-JSON task. Base exact match 0.000 vs
  fine-tuned 1.000 on 60 held-out examples, plus a worked 7B GPU-memory
  calculation (full FT ~113 GB, LoRA ~17.5 GB, QLoRA ~5.3 GB).
- **Task 2 — RAG from scratch** (`task2/`): QA over 35 Wikipedia pages
  (135,915 words), no LangChain/LlamaIndex — all-MiniLM-L6-v2 embeddings
  + NumPy cosine retrieval. Recall@5: 0.667 at 400-char chunks vs 0.933
  at 1600-char chunks, with failure analysis and own-words BM25 /
  re-ranker explanations.
- **Task 3 — Local model & optimisation** (`task3/`): Qwen2.5-0.5B-Instruct
  in F16 vs Q4_K_M via llama.cpp. F16: 3.3 tok/s, 949 MB; Q4_K_M: 4.2
  tok/s, 380 MB — no visible quality loss on the same 10 prompts.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python task1/train.py        # Task 1 (downloads Qwen2.5-0.5B, ~1 GB)
python task2/rag.py          # Task 2 (uses cached docs + embeddings)
python task3/bench.py        # Task 3 (downloads 2 GGUF files, ~1.3 GB)
```

Task 3 needs `llama-cpp-python`; building it from source takes ~20 min
on a small CPU — see `task3/README.md`. The GGUF model files in
`task3/models/` are git-ignored (downloaded by `bench.py`).

## Environment used for the reported numbers

Python 3.12, torch 2.6.0+cpu, transformers, peft, sentence-transformers,
llama-cpp-python 0.3.37. 4 CPUs (1.5 shared), ~8 GB RAM. Task 3 tok/s
was measured while the Task 1 training job ran concurrently, so absolute
tok/s values are lower bounds.

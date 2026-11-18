"""Task 3 — Run a small open model locally in two quantisation settings.

Model: Qwen2.5-0.5B-Instruct (GGUF), F16 vs Q4_K_M, via llama-cpp-python on CPU.
Measures tokens/sec and peak memory for the same 10 prompts, and saves the
raw outputs so quality can be compared side by side.
"""
import json
import os
import resource
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

REPO = "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
FILES = {
    "F16": "Qwen2.5-0.5B-Instruct-f16.gguf",
    "Q4_K_M": "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf",
}

PROMPTS = [
    "What is the capital of France?",
    "Explain photosynthesis in one sentence.",
    "Write a haiku about rain.",
    "What is 17 multiplied by 6?",
    "Name the three primary colours.",
    "Write a Python function to reverse a string.",
    "Translate 'good morning' into Spanish.",
    "What is the largest planet in our solar system?",
    "Give one practical tip for healthy sleep.",
    "Complete the sentence: The quick brown fox",
]

MAX_TOKENS = 60


def peak_rss_mb():
    # ru_maxrss is in KB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def download():
    """Download both GGUF files with curl (resumable); skip if already present."""
    import subprocess
    paths = {}
    for label, fname in FILES.items():
        dest = os.path.join(MODELS_DIR, fname)
        if os.path.exists(dest) and os.path.getsize(dest) > 10e6:
            print(f"{label} already present ({os.path.getsize(dest) / 1e6:.0f} MB)")
        else:
            url = f"https://huggingface.co/{REPO}/resolve/main/{fname}"
            print(f"downloading {label} ({fname}) via curl ...")
            subprocess.run(["curl", "-L", "-C", "-", "--retry", "20",
                            "--retry-all-errors", "--retry-delay", "5",
                            "--max-time", "900", "-o", dest, url],
                           check=True)
            print(f"  -> {dest} ({os.path.getsize(dest) / 1e6:.0f} MB)")
        paths[label] = dest
    return paths


def bench(label, path):
    from llama_cpp import Llama
    print(f"\n=== {label}: {os.path.basename(path)} ===")
    t0 = time.time()
    llm = Llama(model_path=path, n_ctx=512, n_threads=4, verbose=False)
    load_s = time.time() - t0
    print(f"loaded in {load_s:.1f}s, peak RSS after load: {peak_rss_mb():.0f} MB")

    outs = []
    total_tokens = 0
    t0 = time.time()
    for p in PROMPTS:
        r = llm.create_completion(p, max_tokens=MAX_TOKENS, temperature=0.0)
        text = r["choices"][0]["text"]
        ntok = r["usage"]["completion_tokens"]
        total_tokens += ntok
        outs.append({"prompt": p, "output": text.strip(), "tokens": ntok})
        print(f"  [{ntok:3d} tok] {p!r} -> {text.strip()[:80]!r}")
    gen_s = time.time() - t0
    tps = total_tokens / gen_s
    print(f"{label}: {total_tokens} tokens in {gen_s:.1f}s = {tps:.1f} tok/s, "
          f"peak RSS {peak_rss_mb():.0f} MB")
    del llm
    return {"label": label, "load_s": round(load_s, 1), "gen_s": round(gen_s, 1),
            "total_tokens": total_tokens, "tokens_per_sec": round(tps, 1),
            "peak_rss_mb": round(peak_rss_mb(), 0), "outputs": outs}


def main():
    paths = download()
    results = {}
    for label in ["F16", "Q4_K_M"]:
        results[label] = bench(label, paths[label])
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote task3/results.json")
    a, b = results["F16"], results["Q4_K_M"]
    print(f"SUMMARY: F16 {a['tokens_per_sec']} tok/s, {a['peak_rss_mb']} MB | "
          f"Q4_K_M {b['tokens_per_sec']} tok/s, {b['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()

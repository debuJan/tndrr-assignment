"""Inspect what the retriever actually returned for a question.

Loads cached embeddings (produced by rag.py) so this is instant.
Usage: python inspect.py "question text" small_400|large_1600 [gold_keywords...]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    question = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else "large_1600"
    gold_kw = sys.argv[3:] if len(sys.argv) > 3 else []

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb = np.load(os.path.join(HERE, f"embeddings_{label}.npy"))
    with open(os.path.join(HERE, f"chunks_{label}.json")) as f:
        chunks = json.load(f)
    q_emb = model.encode([question], normalize_embeddings=True)[0]
    sims = emb @ q_emb
    top = np.argsort(-sims)[:5]
    print(f"Q: {question}\nindex: {label} ({len(chunks)} chunks)\n")
    for rank, idx in enumerate(top, 1):
        c = chunks[idx]
        is_gold = gold_kw and all(k.lower() in c["text"].lower() for k in gold_kw)
        print(f"#{rank} sim={sims[idx]:.3f} [{c['title']} chunk {c['chunk']}]"
              f"{'  <-- GOLD' if is_gold else ''}")
        print(f"    {c['text'][:280]}\n")


if __name__ == "__main__":
    main()

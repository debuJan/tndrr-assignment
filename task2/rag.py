"""Task 2 — RAG from scratch (no LangChain/LlamaIndex).

Pipeline: Wikipedia docs (35 pages, task2/docs/) -> sentence chunks ->
all-MiniLM-L6-v2 embeddings -> NumPy cosine similarity retrieval.
Evaluates recall@5 for two chunk sizes on 15 hand-written questions,
each with a verified gold passage.
"""
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(HERE, "docs")

# ---------------------------------------------------------------- questions
# Each question names the article that holds the answer (gold_article = file
# stem) plus distinctive keywords that must appear in the gold chunk.
QUESTIONS = [
    {"q": "What year was the paper 'Attention Is All You Need' published?",
     "gold_article": "attention_machine_learning",
     "keywords": ["Attention Is All You Need", "2017"]},
    {"q": "What does BERT stand for?",
     "gold_article": "bert_language_model",
     "keywords": ["Bidirectional Encoder Representations from Transformers"]},
    {"q": "What problem do LSTMs address in recurrent neural networks?",
     "gold_article": "long_short_term_memory",
     "keywords": ["vanishing gradient"]},
    {"q": "In a GAN, what are the two neural networks called?",
     "gold_article": "generative_adversarial_network",
     "keywords": ["generator", "discriminator"]},
    {"q": "What does LSTM stand for?",
     "gold_article": "long_short_term_memory",
     "keywords": ["Long Short-Term Memory"]},
    {"q": "What is the basic idea of gradient descent?",
     "gold_article": "gradient_descent",
     "keywords": ["minimizing", "loss function"]},
    {"q": "What is backpropagation used to compute in a neural network?",
     "gold_article": "backpropagation",
     "keywords": ["gradient", "weights"]},
    {"q": "In reinforcement learning, what does the agent try to maximise?",
     "gold_article": "reinforcement_learning",
     "keywords": ["cumulative reward"]},
    {"q": "How does supervised learning differ from unsupervised learning?",
     "gold_article": "supervised_learning",
     "keywords": ["labeled data"]},
    {"q": "What is overfitting in machine learning?",
     "gold_article": "overfitting",
     "keywords": ["training data", "generalize"]},
    {"q": "What do neurons in a convolutional layer learn to activate for?",
     "gold_article": "convolutional_neural_network",
     "keywords": ["activate for different features"]},
    {"q": "What is a word embedding?",
     "gold_article": "word_embedding",
     "keywords": ["vector", "words"]},
    {"q": "What is transfer learning?",
     "gold_article": "transfer_learning",
     "keywords": ["knowledge", "related task"]},
    {"q": "What is cross-validation used for?",
     "gold_article": "cross_validation_statistics",
     "keywords": ["generalize to an independent data set"]},
    {"q": "What is a random forest made of?",
     "gold_article": "random_forest",
     "keywords": ["decision trees"]},
]

# ---------------------------------------------------------------- chunking
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text, max_chars):
    """Split text into chunks of <= max_chars, never splitting a sentence."""
    sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > max_chars and cur:
            chunks.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip() if cur else s
    if cur:
        chunks.append(cur)
    return chunks


def load_docs():
    docs = {}
    for fn in sorted(os.listdir(DOCS_DIR)):
        if not fn.endswith(".txt"):
            continue
        with open(os.path.join(DOCS_DIR, fn), encoding="utf-8") as f:
            text = f.read()
        title, _, body = text.partition("\n\n")
        docs[fn[:-4]] = {"title": title.strip(), "body": body.strip()}
    return docs


def build_index(docs, max_chars):
    """Return (chunk_texts, chunk_meta) for one chunk size."""
    texts, meta = [], []
    for slug, d in docs.items():
        for i, ch in enumerate(chunk_text(d["body"], max_chars)):
            texts.append(ch)
            meta.append({"article": slug, "title": d["title"], "chunk": i})
    return texts, meta


def find_gold_chunk(texts, meta, gold_article, keywords):
    """First chunk of the gold article containing all keywords (case-insensitive)."""
    for idx, m in enumerate(meta):
        if m["article"] != gold_article:
            continue
        low = texts[idx].lower()
        if all(k.lower() in low for k in keywords):
            return idx
    return None


def main():
    from sentence_transformers import SentenceTransformer

    docs = load_docs()
    print(f"loaded {len(docs)} pages")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    results = {}
    for label, max_chars in [("small_400", 400), ("large_1600", 1600)]:
        texts, meta = build_index(docs, max_chars)
        print(f"\n[{label}] {len(texts)} chunks")
        emb = model.encode(texts, batch_size=64, show_progress_bar=False,
                           normalize_embeddings=True)
        emb = np.asarray(emb, dtype=np.float32)
        # cache embeddings + chunk texts so later inspection is instant
        np.save(os.path.join(HERE, f"embeddings_{label}.npy"), emb)
        with open(os.path.join(HERE, f"chunks_{label}.json"), "w") as f:
            json.dump([{"text": t, **m} for t, m in zip(texts, meta)], f)

        per_q = []
        for qi, item in enumerate(QUESTIONS, 1):
            gold = find_gold_chunk(texts, meta, item["gold_article"], item["keywords"])
            q_emb = model.encode([item["q"]], normalize_embeddings=True)[0]
            sims = emb @ q_emb
            top5 = np.argsort(-sims)[:5].tolist()
            hit = gold is not None and gold in top5
            rank = top5.index(gold) + 1 if gold is not None and gold in top5 else None
            per_q.append({"q": item["q"], "gold_chunk": gold, "hit@5": bool(hit),
                          "rank": rank,
                          "gold_text": texts[gold][:300] if gold is not None else None})
            print(f"  Q{qi:02d} hit={hit} rank={rank} gold={'OK' if gold is not None else 'NOT FOUND'}")
        recall = sum(p["hit@5"] for p in per_q) / len(per_q)
        results[label] = {"chunk_size_chars": max_chars, "n_chunks": len(texts),
                          "recall@5": recall, "per_question": per_q}
        print(f"[{label}] recall@5 = {recall:.3f} ({sum(p['hit@5'] for p in per_q)}/{len(per_q)})")

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote task2/results.json")


if __name__ == "__main__":
    main()

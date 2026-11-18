"""Fetch ~35 Wikipedia articles (ML/AI topics) as the Task 2 document set.
Stdlib only. Each article = one 'page' (>= 30 pages required).
Saves plain text to task2/docs/<slug>.txt
"""
import json
import os
import urllib.parse
import urllib.request

TOPICS = [
    "Machine learning", "Deep learning", "Artificial neural network",
    "Gradient descent", "Backpropagation",
    "Transformer (deep learning architecture)", "Attention (machine learning)",
    "Large language model", "Natural language processing", "Computer vision",
    "Reinforcement learning", "Supervised learning", "Unsupervised learning",
    "Self-supervised learning", "Convolutional neural network",
    "Recurrent neural network", "Long short-term memory",
    "Generative artificial intelligence", "Generative adversarial network",
    "Variational autoencoder", "Diffusion model", "Word embedding",
    "BERT (language model)", "GPT (language model)", "Prompt engineering",
    "Fine-tuning (deep learning)", "Transfer learning", "Overfitting",
    "Regularization (mathematics)", "Cross-validation (statistics)",
    "Feature engineering", "Decision tree", "Random forest",
    "Support vector machine", "K-nearest neighbors algorithm", "Clustering",
]

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
os.makedirs(OUT, exist_ok=True)

API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "tndrr-assignment/1.0 (research; contact: debanshudevgoswami@gmail.com)"}

def fetch(title):
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "titles": title, "format": "json", "formatversion": 2,
    })
    req = urllib.request.Request(API + "?" + params, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    pages = data["query"]["pages"]
    page = pages[0]
    if "missing" in page:
        return None
    return page.get("extract", "")

def slug(title):
    keep = "".join(c if c.isalnum() or c in " -" else "" for c in title)
    return keep.strip().lower().replace(" ", "_").replace("-", "_")

manifest = []
for t in TOPICS:
    text = fetch(t)
    if not text:
        print("MISSING:", t)
        continue
    fn = slug(t) + ".txt"
    with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
        f.write(t + "\n\n" + text)
    words = len(text.split())
    manifest.append({"title": t, "file": fn, "words": words})
    print(f"{t}: {words} words")

with open(os.path.join(OUT, "_manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)
total_words = sum(m["words"] for m in manifest)
print(f"\nDONE: {len(manifest)} pages, {total_words} words total")

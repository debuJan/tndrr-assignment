# Task 2 — Retrieval (RAG) from scratch

Question-answering over 35 Wikipedia pages (ML/AI topics, ~136k words),
built with **only** an embedding model + NumPy. No LangChain, no LlamaIndex.

## How to run

```bash
python fetch_docs.py   # downloads the 35 pages into task2/docs/
python rag.py          # chunks, embeds, evaluates recall@5, writes results.json
python retrieve_inspect.py "<question>" small_400|large_1600 [gold keywords...]
```

## Method

- **Documents:** 35 Wikipedia articles (see `docs/_manifest.json`), each one
  counts as a page. Fetched with the Wikipedia API.
- **Chunking:** text is split on sentence boundaries into chunks of at most
  N characters. Two sizes compared: **400** and **1600** characters.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (384-dim),
  normalised; retrieval is plain cosine similarity with NumPy.
- **Evaluation:** 15 questions I wrote myself. Each has one gold chunk — the
  chunk of a named article that contains the answer. Metric is **recall@5**:
  is the gold chunk in the top 5 retrieved?

## Results (from real runs, see `results.json`)

| chunk size | chunks | recall@5 |
|------------|--------|----------|
| 400 chars  | 2,855  | **0.667** (10/15) |
| 1,600 chars| 619    | **0.933** (14/15) |

**Larger chunks won clearly.** A 400-char chunk often cuts a fact away from
the sentence that gives it meaning, so its embedding matches the question
worse. A 1,600-char chunk keeps the definition together with its context.

## 2 cases where retrieval failed

**1. "How does supervised learning differ from unsupervised learning?"
(large chunks — gold not in top 5).** All top-5 chunks came from the
*unsupervised learning* article. The question is a comparison: its answer
spans two articles, but recall@5 is measured against a single gold chunk
(the supervised-learning lead). Notably the #1 chunk *does* answer the
question — "in contrast to supervised learning, algorithms learn patterns
exclusively from unlabeled data" — so the metric understates the system
here. **Fix:** label comparison questions with two gold chunks, or add a
re-ranker that rewards chunks covering both sides of the question.

**2. "What does BERT stand for?" (small chunks — gold not in top 5).**
All top-5 chunks were about BERT's modules, history and training — none
contained the expansion "Bidirectional Encoder Representations from
Transformers". At 1,600 chars the lead chunk (with the expansion) ranked
#3. The 400-char chunk holding the definition is mostly other lead text,
so its embedding drifts away from the question. **Fix:** hybrid BM25
(exact-term match catches acronyms and names) or larger chunks.

## In my own words

**What BM25 adds to embedding search:** BM25 is a keyword scorer. It looks
at the actual words in the question — how often each word appears in a chunk
(term frequency), how rare the word is across all chunks (inverse document
frequency), and it corrects for chunk length so long chunks don't win just
for being long. Embeddings understand meaning, so they handle paraphrases
("stands for" ≈ "is short for"). BM25 handles exactness: names, acronyms,
numbers, rare terms. Used together (hybrid search), you get both — the
embedding finds "means the same thing", BM25 finds "says the exact word".

**What a re-ranker does:** retrieval with embeddings is fast but shallow —
it compares two separate vectors and never actually reads the question and
the chunk together. A re-ranker (usually a cross-encoder) is a second,
slower model that reads the question *and* each candidate chunk together
and scores how well the chunk answers the question. You retrieve the top
20–50 cheaply, then re-rank down to the top 5. It fixes exactly the
failures above: chunks that look similar but don't contain the answer get
pushed down, and chunks that contain the answer get pushed up.

## Answers

The system is extractive: the answer shown for a question is the top
retrieved passage(s). A generation step (an LLM writing a final answer from
the retrieved passages) can be added on top without changing retrieval.

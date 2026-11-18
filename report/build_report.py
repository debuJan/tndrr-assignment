"""Build the max-4-page PDF report from the real run numbers.
Run:  python report/build_report.py   ->  Tndrr_Assignment_Report.pdf
Uses only ASCII text (fpdf2 core fonts are Latin-1).
"""
import os
from fpdf import FPDF

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Report(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "Tndrr AI/ML Intern Assignment - Debanshu Devgoswami", align="R")
        self.ln(10)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")
        self.set_text_color(0, 0, 0)

    def h1(self, text):
        self.set_font("Helvetica", "B", 13)
        self.cell(0, 9, text, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 0, 0)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(3)

    def h2(self, text):
        self.set_font("Helvetica", "B", 11)
        self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def p(self, text):
        self.set_font("Helvetica", "", 9.5)
        self.multi_cell(0, 5, text)
        self.ln(1.5)

    def bullet(self, text):
        self.set_font("Helvetica", "", 9.5)
        x = self.get_x()
        self.cell(5, 5, "-")
        self.multi_cell(0, 5, text)
        self.ln(0.5)

    def table(self, headers, rows, widths):
        self.set_font("Helvetica", "B", 9.5)
        for h, w in zip(headers, widths):
            self.cell(w, 6, h, border=1)
        self.ln()
        self.set_font("Helvetica", "", 9.5)
        for row in rows:
            for val, w in zip(row, widths):
                self.cell(w, 6, str(val), border=1)
            self.ln()
        self.ln(2)


def build():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(15, 15, 15)

    # ---------------- Page 1: title + Task 1 ----------------
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "Tndrr AI/ML Intern - Assignment Report", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Debanshu Devgoswami  |  November 2026", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.p("Three tasks, all run locally on CPU. Every number in this report "
          "comes from my own runs; the code and full results are in the "
          "submitted repository (one folder per task).")

    pdf.h1("Task 1 - LoRA / QLoRA fine-tuning")
    pdf.p("Task: plain-English reminder -> fixed JSON {action, task, date, "
          "time, priority}. Data is synthetic: 240 train / 60 held-out test "
          "examples (seed 42, no overlap). The real learning is time "
          "normalisation (\"6pm\" -> \"18:00\") and exact JSON format. Loss "
          "is computed on the JSON answer only.")
    pdf.h2("Model and settings")
    pdf.p("Base model Qwen2.5-0.5B in bfloat16. LoRA rank 8, alpha 16, "
          "dropout 0.05, target modules q/k/v/o projections. LR 2e-4, "
          "3 epochs, batch 4, AdamW. Trainable params 1,081,344 (0.22%). "
          "180 steps in 949 s (about 5.3 s/step, CPU), peak RSS 2.7 GB. Rank 8 "
          "is enough for a format task; alpha 16 (2x rank) is the standard "
          "scaling; 3 epochs because the loss had already collapsed by "
          "epoch 2 (0.0919 -> 0.0003).")
    pdf.h2("Results - 60 held-out examples, greedy decoding")
    pdf.table(["model", "exact match", "valid JSON"],
              [["base Qwen2.5-0.5B", "0.000", "0.983"],
               ["+ LoRA (this run)", "1.000", "1.000"]],
              [70, 50, 50])
    pdf.p("Per-field accuracy of the fine-tuned model: action, task, date, "
          "time and priority all 1.000. Read the 100% honestly: the held-out "
          "examples use the same templates as training, so this measures "
          "format learning, not robustness. The edge cases below are the "
          "real test, and that is where it breaks.")
    pdf.h2("Failure examples")
    pdf.bullet("Fine-tuned model on the 60 held-out examples: ZERO failures. "
               "So I probed it with 3 out-of-template inputs instead.")
    pdf.bullet("Missing time - \"Remind me to call mom tomorrow\" -> it "
               "echoed {\"instruction\": ...} (the base behaviour). It never "
               "learned what to do with missing information.")
    pdf.bullet("Ambiguous hour - \"Team meeting at 6 on Friday\" -> it "
               "silently guessed 18:00 and left the date as \"Friday\" "
               "instead of a real date. No uncertainty expressed.")
    pdf.bullet("Two tasks - \"buy milk and call dad at 5pm tomorrow\" -> it "
               "kept only \"buy milk\" and silently dropped \"call dad\". "
               "The single-task schema cannot represent this.")
    pdf.bullet("Base model failures (what LoRA fixed): on every probed "
               "prompt the base model wrapped its answer in chatty prose + "
               "a markdown fence and echoed the input as "
               "{\"instruction\": ...} - valid JSON (hence 0.983) but the "
               "wrong schema and no extracted fields.")
    pdf.h2("GPU memory to fine-tune a 7B model (worked calculation)")
    pdf.p("Real Qwen2.5-7B config: 28 layers, hidden 3584, about 7.62B params. "
          "Mixed-precision AdamW assumed (bf16 weights + fp32 master "
          "weights + fp32 Adam states). LoRA r=8 on all 7 linear modules "
          "= 20,185,088 trainable params (0.265%).")
    pdf.table(["method", "memory", "why"],
              [["(a) Full fine-tune", "about 113 GB", "Adam states alone about 57 GB"],
               ["(a) Full FT, fp32 grads", "about 128 GB", "grads 28.4 GB"],
               ["(b) LoRA, base bf16", "about 17.5 GB", "base 14.2 + adapt. 0.3"],
               ["(c) QLoRA, base NF4", "about 5.3 GB", "base 3.5 + adapt. 0.3"]],
              [55, 30, 85])
    pdf.p("Full fine-tuning keeps a gradient and two Adam states for every "
          "one of the 7.6B weights; LoRA trains only 20M adapter weights. "
          "QLoRA stores the frozen base in 4-bit, which is what makes 7B "
          "fine-tuning fit on a single consumer GPU.")

    # ---------------- Page 2: Task 2 ----------------
    pdf.add_page()
    pdf.h1("Task 2 - RAG from scratch (no LangChain / LlamaIndex)")
    pdf.p("Document set: 35 Wikipedia pages on ML/AI topics (135,915 "
          "words), fetched via the Wikipedia API. Method: sentence-boundary "
          "chunking, all-MiniLM-L6-v2 embeddings, NumPy cosine retrieval. "
          "15 self-written test questions, each with a verified gold source "
          "passage (every gold chunk was read and checked to actually "
          "answer its question).")
    pdf.h2("Recall@5 for two chunk sizes")
    pdf.table(["chunk size", "chunks", "recall@5"],
              [["400 chars", "2,855", "0.667 (10/15)"],
               ["1,600 chars", "619", "0.933 (14/15)"]],
              [50, 40, 80])
    pdf.p("Larger chunks won. Small chunks split facts from their context: "
          "e.g. \"What does BERT stand for?\" failed at 400 chars because "
          "the top-5 chunks were all about BERT modules/history/training "
          "and none contained the expansion; at 1,600 chars the chunk "
          "containing the expansion ranked #3. Bigger chunks carry more "
          "context per retrieval, at the cost of more tokens per chunk.")
    pdf.h2("Retrieval failures and fixes")
    pdf.bullet("Q09 \"How does supervised learning differ from unsupervised "
               "learning?\" (large chunks): top-5 all came from the "
               "unsupervised article. But the top-1 chunk (\"in contrast to "
               "supervised learning, algorithms learn patterns exclusively "
               "from unlabeled data\") actually answers the question - "
               "single-gold recall understates comparison questions. Fix: "
               "multi-gold labels, hybrid BM25, or a re-ranker.")
    pdf.bullet("Q02 \"What does BERT stand for?\" (small chunks): exact-term "
               "match missing from the top-5. Fix: BM25 exact-term retrieval "
               "or larger chunks (verified: large chunks retrieve it).")
    pdf.h2("In my own words")
    pdf.p("BM25: a scoring rule that ranks documents by the query words "
          "they contain, but smarter than counting. A rare word counts more "
          "than a common one, a word appearing many times counts more than "
          "once - but with diminishing returns - and long documents are "
          "penalised so they do not win just by being long. It is exact "
          "word matching, so it finds \"BERT\" when embeddings drift to "
          "related-but-wrong chunks.")
    pdf.p("Re-ranker: a second, slower model that reads the query together "
          "with each candidate passage (a cross-encoder) and scores how "
          "well the passage actually answers the question - not just how "
          "similar the vectors are. Retrieval first fetches 20-50 "
          "candidates cheaply, then the re-ranker re-orders them.")

    # ---------------- Page 3: Task 3 ----------------
    pdf.add_page()
    pdf.h1("Task 3 - Local model and optimisation")
    pdf.p("Qwen2.5-0.5B-Instruct run locally with llama.cpp "
          "(llama-cpp-python) in two settings: F16 (16-bit) vs Q4_K_M "
          "(4-bit). Same 10 prompts, greedy decoding (temperature 0), 60 "
          "tokens max each.")
    pdf.h2("Results")
    pdf.table(["setting", "tokens/sec", "peak RSS", "file size", "load time"],
              [["F16", "3.3", "1,044 MB", "949 MB", "3.2 s"],
               ["Q4_K_M", "4.2", "1,044 MB", "380 MB", "2.5 s"]],
              [30, 32, 32, 32, 32])
    pdf.p("Peak RSS is the process peak (both models ran in one process, so "
          "the number is the same for both). The real memory difference is "
          "the file: 949 MB vs 380 MB (2.5x smaller). Q4_K_M was about 27% "
          "faster. Both numbers were measured while the Task 1 training job "
          "ran at the same time on the same 1.5 shared CPUs, so absolute "
          "tok/s values are lower bounds; the relative gap is the "
          "meaningful result.")
    pdf.h2("Output quality on the same 10 prompts")
    pdf.p("No visible quality loss from Q4_K_M. Both answered every factual "
          "question correctly (Paris, Jupiter, primary colours, "
          "\"Buenos dias\"). Two differences worth noting: on \"17 x 6\" the "
          "F16 model rambled (\"add 17 to itself 6 times...\") while Q4_K_M "
          "went straight to multiplying - the quantised model was more "
          "direct. On \"write a Python function to reverse a string\" both "
          "models started explaining instead of giving code and hit the "
          "60-token cap - a prompt/limit effect, not a quantisation effect.")
    pdf.h2("In my own words")
    pdf.p("KV cache: when a transformer writes text one token at a time, "
          "every new token needs attention over all previous tokens. "
          "Without a cache the model would recompute the Key and Value "
          "vectors for every past token at every step. The KV cache stores "
          "the K and V vectors computed for past tokens, so each new token "
          "only computes its own Q, K, V and attends against the stored "
          "ones. It trades memory for speed. Its size is 2 (K and V) x "
          "layers x hidden size x sequence length x batch size x bytes per "
          "value - so it grows with longer conversations and more users.")
    pdf.p("Why quantisation saves memory: a model's weights are just "
          "numbers. At 16-bit precision each weight takes 2 bytes; at 4-bit "
          "it takes half a byte, so the same model is about 4x smaller on disk "
          "and in RAM. Generation with small batches is limited by how fast "
          "weights can be read from memory, not by arithmetic - smaller "
          "weights mean less reading per token, so it also runs faster. The "
          "cost is rounding: 4-bit weights are approximations, which can "
          "slightly change the model's answers.")
    pdf.h2("Back-of-envelope plan: 20 users on one 24 GB GPU")
    pdf.p("Assumptions: Qwen2.5-0.5B-Instruct, Q4_K_M weights (about 0.4 GB), "
          "vLLM with continuous batching, about 256 tokens of context per "
          "conversation, about 60 tokens generated per reply.")
    pdf.bullet("Weights: about 0.4 GB (Q4_K_M), loaded once, shared by all users.")
    pdf.bullet("KV cache per user: 2 x 24 layers x 896 hidden x 2 bytes x "
               "256 tokens = about 22 MB. For 20 users: about 0.44 GB.")
    pdf.bullet("Total: weights 0.4 + KV 0.44 + activations/overhead about 0.5 = "
               "about 1.5 GB - a small fraction of 24 GB. The same card could "
               "instead serve a 7B model in Q4 (about 4 GB weights): KV per user "
               "about 100 MB, 20 users about 2 GB, total about 7 GB - still fits.")
    pdf.bullet("Throughput: generation is memory-bandwidth bound. Batching "
               "20 users together means the weights are read once per step "
               "for all 20 sequences, so total tokens/sec rises with batch "
               "size until bandwidth saturates. If latency matters, cap "
               "concurrent sequences (e.g. 16) and queue the rest.")
    pdf.bullet("Beyond 20 users: squeeze the KV cache first - shorter max "
               "context, fp8/int8 KV cache, or prefix caching for shared "
               "system prompts. After that, a second GPU or a smaller model.")

    # ---------------- Page 4: reproducibility ----------------
    pdf.add_page()
    pdf.h1("Reproducibility and honest limitations")
    pdf.h2("How to reproduce")
    pdf.bullet("Task 1: python task1/train.py (downloads Qwen2.5-0.5B, "
               "about 1 GB). --pilot runs 5 steps; --skip-base reuses a "
               "completed base evaluation. Adapter saved to "
               "task1/lora_adapter/.")
    pdf.bullet("Task 2: python task2/rag.py (uses cached docs + "
               "embeddings; fetch_docs.py re-downloads the 35 pages).")
    pdf.bullet("Task 3: python task3/bench.py (downloads 2 GGUF files, "
               "about 1.3 GB). llama-cpp-python must be built first; on this "
               "machine the build took about 19 min from source.")
    pdf.h2("Environment")
    pdf.p("Python 3.12, torch 2.6.0+cpu, transformers, peft, "
          "sentence-transformers, llama-cpp-python 0.3.37. 4 CPUs (1.5 "
          "shared), about 8 GB RAM. All heavy jobs were run one at a time "
          "except where noted (Task 3 tok/s measured during Task 1 "
          "training).")
    pdf.h2("Limitations I want to be upfront about")
    pdf.bullet("Task 1's 100% exact match is format learning on a narrow "
               "synthetic template, not general reminder understanding. "
               "The edge-case probes (missing time, ambiguous hour, two "
               "tasks) show where it actually breaks.")
    pdf.bullet("Task 2's recall@5 uses a single gold chunk per question. "
               "Comparison questions (Q09) are understated by this metric; "
               "multi-gold labels would be fairer.")
    pdf.bullet("Task 3's peak RSS is a process peak, not a per-model "
               "measurement, and tok/s was measured under CPU contention. "
               "The file-size and relative-speed results are the robust "
               "ones.")
    pdf.bullet("No GPU was used anywhere. A GPU would change absolute "
               "tok/s and training time, but not the relative comparisons.")
    pdf.h2("Files in the repository")
    pdf.bullet("task1/ - train.py, memory_calc.py, probe.py, README.md, "
               "results.json, lora_adapter/")
    pdf.bullet("task2/ - rag.py, fetch_docs.py, retrieve_inspect.py, "
               "README.md, results.json, docs/ (35 pages), embeddings")
    pdf.bullet("task3/ - bench.py, README.md, results.json "
               "(models/ is git-ignored; bench.py downloads them)")
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, "All results above are from my own runs on 17 November 2026.",
            new_x="LMARGIN", new_y="NEXT")

    out = os.path.join(BASE, "Tndrr_Assignment_Report.pdf")
    pdf.output(out)
    print(f"wrote {out} ({pdf.pages_count} pages)")


if __name__ == "__main__":
    build()

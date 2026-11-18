"""Task 1 — Fine-tune a small open-weight model with LoRA.

Narrow task: plain-English reminder -> fixed JSON
  {"action": "remind", "task": ..., "date": ..., "time": "HH:MM",
   "priority": "high|medium|low"}

Data is synthetic (template-generated, fixed seed): 240 train / 60 held-out
test examples. Compares base vs LoRA fine-tuned model on the test set.

Usage:
  python train.py --pilot     # 5 training steps only, prints step time + RSS
  python train.py             # full train + eval, writes results.json
  python train.py --model HuggingFaceTB/SmolLM2-135M --dtype float32
"""
import argparse
import json
import os
import random
import re
import resource
import time

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- data
TASKS = ["call mom", "buy groceries", "submit the report", "water the plants",
         "book a cab", "pay the electricity bill", "email the professor",
         "finish the presentation", "take medicine", "feed the dog",
         "charge the laptop", "clean the kitchen", "read chapter five",
         "call the bank", "book a doctor appointment", "do the laundry",
         "practice guitar", "send the invoice", "water the garden",
         "prepare for the interview"]
DATES = ["today", "tomorrow", "on Monday", "on Friday", "on Saturday",
         "on Sunday", "next Monday", "on Wednesday"]
TIMES = [("6pm", "18:00"), ("9 in the morning", "09:00"), ("at noon", "12:00"),
         ("8:30pm", "20:30"), ("7am", "07:00"), ("10:30am", "10:30"),
         ("5pm", "17:00"), ("11am", "11:00"), ("9pm", "21:00"),
         ("8 in the morning", "08:00")]
TEMPLATES = [
    "Remind me to {task} {date} at {time}",
    "Please remind me to {task} {date} {time}, it's important",
    "Set a reminder for {task} on {date} at {time}",
    "Remind me to {task} {date} at {time}, no rush",
    "Can you remind me to {task} {date} {time}? It's urgent",
    "Remind me to {task} {date} at {time} please",
]


def make_example(rng):
    task = rng.choice(TASKS)
    date = rng.choice(DATES)
    time_phrase, time_24 = rng.choice(TIMES)
    tmpl = rng.choice(TEMPLATES)
    instr = tmpl.format(task=task, date=date, time=time_phrase)
    if "important" in tmpl or "urgent" in tmpl:
        prio = "high"
    elif "no rush" in tmpl:
        prio = "low"
    else:
        prio = "medium"
    target = {"action": "remind", "task": task, "date": date.replace("on ", ""),
              "time": time_24, "priority": prio}
    return {"instruction": instr, "target": target}


def build_data(seed=42, n_train=240, n_test=60):
    rng = random.Random(seed)
    seen = set()
    out = []
    while len(out) < n_train + n_test:
        ex = make_example(rng)
        key = (ex["instruction"], json.dumps(ex["target"], sort_keys=True))
        if key in seen:
            continue
        seen.add(key)
        out.append(ex)
    return out[:n_train], out[n_train:]


PROMPT = "Convert this reminder into JSON:\n{instruction}\nJSON:"
FIELDS = ["action", "task", "date", "time", "priority"]


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--skip-base", action="store_true",
                    help="skip base eval; use completed run's numbers "
                         "(exact-match 0.000, valid JSON 0.983, run1.log)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    args = ap.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(2)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    train, test = build_data()
    print(f"train={len(train)} test={len(test)}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"loaded tokenizer for {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=None)
    model.config.use_cache = False
    print(f"loaded {args.model} in {args.dtype}, RSS {rss_mb():.0f} MB")

    def tokenize(ex):
        prompt = PROMPT.format(instruction=ex["instruction"])
        full = prompt + " " + json.dumps(ex["target"]) + tok.eos_token
        p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
        f_ids = tok(full, add_special_tokens=True)["input_ids"]
        labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
        return {"input_ids": f_ids, "labels": labels}

    # ---- base model eval (before LoRA; skipped in pilot mode) ----
    def evaluate(m, n=None):
        m.eval()
        exact, field_ok, valid = 0, {f: 0 for f in FIELDS}, 0
        examples = []
        for ex in (test if n is None else test[:n]):
            prompt = PROMPT.format(instruction=ex["instruction"])
            ids = tok(prompt, return_tensors="pt")["input_ids"]
            with torch.no_grad():
                out = m.generate(ids, max_new_tokens=80, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
            pred = parse_json(text)
            if pred is not None:
                valid += 1
                for f in FIELDS:
                    if str(pred.get(f, "")).strip() == str(ex["target"][f]):
                        field_ok[f] += 1
                if all(str(pred.get(f, "")).strip() == str(ex["target"][f])
                       for f in FIELDS):
                    exact += 1
            examples.append({"instruction": ex["instruction"],
                             "expected": ex["target"],
                             "predicted": pred, "raw": text.strip()[:300]})
        n = len(test) if n is None else n
        return {"exact_match": exact / n, "valid_json_rate": valid / n,
                "field_accuracy": {f: field_ok[f] / n for f in FIELDS},
                "examples": examples}

    print("\n--- evaluating BASE model ---")
    if args.skip_base:
        base = {"exact_match": 0.0, "valid_json_rate": 0.983,
                "field_accuracy": {}, "examples": []}
        print("base eval skipped (--skip-base); using completed run's numbers: "
              "exact-match 0.000, valid JSON 0.983 (task1/run1.log)")
    else:
        base = evaluate(model) if not args.pilot else {"exact_match": None,
                                                       "valid_json_rate": None,
                                                       "field_accuracy": {},
                                                       "examples": []}
    if not args.pilot and not args.skip_base:
        print(f"base exact-match: {base['exact_match']:.3f}, "
              f"valid JSON: {base['valid_json_rate']:.3f}, RSS {rss_mb():.0f} MB")

    # ---- LoRA fine-tune ----
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                     task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    data = [tokenize(ex) for ex in train]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    steps = 0
    t0 = time.time()
    for ep in range(args.epochs):
        random.Random(42 + ep).shuffle(data)
        tot_loss, nb = 0.0, 0
        for i in range(0, len(data), 4):
            batch = data[i:i + 4]
            maxlen = max(len(b["input_ids"]) for b in batch)
            ids, labs, att = [], [], []
            for b in batch:
                pad = maxlen - len(b["input_ids"])
                ids.append(b["input_ids"] + [tok.pad_token_id] * pad)
                labs.append(b["labels"] + [-100] * pad)
                att.append([1] * len(b["input_ids"]) + [0] * pad)
            out = model(input_ids=torch.tensor(ids),
                        attention_mask=torch.tensor(att),
                        labels=torch.tensor(labs))
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            tot_loss += out.loss.item()
            nb += 1
            steps += 1
            if args.pilot and steps >= 5:
                print(f"PILOT: 5 steps in {time.time() - t0:.1f}s "
                      f"({(time.time() - t0) / 5:.1f}s/step), RSS {rss_mb():.0f} MB")
                return
        print(f"epoch {ep + 1}: avg loss {tot_loss / nb:.4f}, RSS {rss_mb():.0f} MB")
    print(f"trained {steps} steps in {time.time() - t0:.1f}s")

    print("\n--- evaluating FINE-TUNED model ---")
    ft = evaluate(model)
    print(f"ft exact-match: {ft['exact_match']:.3f}, "
          f"valid JSON: {ft['valid_json_rate']:.3f}")

    # 3 failure examples
    fails = [e for e in ft["examples"] if e["predicted"] != e["expected"]][:3]

    results = {
        "model": args.model, "dtype": args.dtype,
        "lora": {"rank": args.rank, "alpha": args.alpha,
                 "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                 "lr": args.lr, "epochs": args.epochs,
                 "train_examples": len(train)},
        "test_examples": len(test),
        "base": {k: v for k, v in base.items() if k != "examples"},
        "fine_tuned": {k: v for k, v in ft.items() if k != "examples"},
        "failure_examples": fails,
        "base_examples": base["examples"][:3],
    }
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    model.save_pretrained(os.path.join(HERE, "lora_adapter"))
    print("\nwrote task1/results.json + lora_adapter/")


if __name__ == "__main__":
    main()

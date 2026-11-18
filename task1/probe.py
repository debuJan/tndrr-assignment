"""Probe failures honestly: base model on 3 regular test prompts, and the
fine-tuned model on 3 out-of-template edge cases (the held-out set had
zero FT failures, so edge cases are where real failure modes live).
Writes task1/probe_results.json."""
import json, os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from train import build_data, PROMPT  # noqa: E402

torch.set_num_threads(2)
MODEL = "Qwen/Qwen2.5-0.5B"

_, test = build_data()
regular = test[:3]  # 3 ordinary held-out prompts (base model fails these)
edge = [
    # no time given -> model must invent one
    {"text": "Remind me to call mom tomorrow",
     "gold": {"action": "remind", "task": "call mom", "date": "2026-11-18",
              "time": "??", "priority": "medium"}},
    # ambiguous hour, no am/pm
    {"text": "Team meeting at 6 on Friday",
     "gold": {"action": "remind", "task": "team meeting", "date": "2026-11-20",
              "time": "??", "priority": "medium"}},
    # two tasks in one sentence; schema holds only one
    {"text": "Remind me to buy milk and call dad at 5pm tomorrow",
     "gold": {"action": "remind", "task": "buy milk AND call dad",
              "date": "2026-11-18", "time": "17:00", "priority": "medium"}},
]


def gen(model, tok, text, max_new=80):
    msgs = [{"role": "user", "content": PROMPT + text}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(s, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new,
                             do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cpu")
    model.eval()

    print("--- BASE model on 3 regular test prompts ---")
    base_failures = []
    for ex in regular:
        out = gen(model, tok, ex["instruction"])
        print(f"IN : {ex['instruction']}\nOUT: {out[:200]}\n")
        base_failures.append({"input": ex["instruction"], "gold": ex["target"],
                              "output": out})

    print("--- FINE-TUNED model on 3 edge cases ---")
    ft = PeftModel.from_pretrained(model, os.path.join(BASE, "lora_adapter"))
    ft.eval()
    edge_out = []
    for ex in edge:
        out = gen(ft, tok, ex["text"])
        print(f"IN : {ex['text']}\nOUT: {out[:300]}\n")
        edge_out.append({"input": ex["text"], "gold": ex["gold"],
                         "output": out})

    with open(os.path.join(BASE, "probe_results.json"), "w") as f:
        json.dump({"base_failures": base_failures,
                   "ft_edge_cases": edge_out}, f, indent=2)
    print("wrote task1/probe_results.json")


if __name__ == "__main__":
    main()

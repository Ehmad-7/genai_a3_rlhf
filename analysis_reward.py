import sys, json
sys.path.insert(0, "starter")
from models import load_tokenizer
from data import load_preference_splits

tok = load_tokenizer()
_, val = load_preference_splits(tok)
sc = json.load(open("results/reward_val_scores.json"))
assert len(sc) == len(val), (len(sc), len(val))

ok = [s["r_chosen"] > s["r_rejected"] for s in sc]
print(f"overall accuracy: {sum(ok)/len(ok):.3f} on {len(ok)} pairs")

buckets = {"chosen LONGER than rejected": [], "same length": [], "chosen SHORTER than rejected": []}
for v, o in zip(val, ok):
    d = len(v["chosen_ids"]) - len(v["rejected_ids"])
    key = "chosen LONGER than rejected" if d > 0 else "same length" if d == 0 else "chosen SHORTER than rejected"
    buckets[key].append(o)
for k, b in buckets.items():
    if b:
        print(f"{k:32s}: n={len(b):4d}  accuracy={sum(b)/len(b):.3f}")

errs = json.load(open("results/reward_val_errors.json"))
print("\n=== 3 confident mistakes (reward model preferred the REJECTED summary) ===")
for e in errs[:3]:
    print("-" * 70)
    print("POST (tail):", e["prompt_tail"][-250:].replace("\n", " "))
    print(f"CHOSEN   [{e['r_chosen']:.2f}]: {e['chosen']}")
    print(f"REJECTED [{e['r_rejected']:.2f}]: {e['rejected']}")
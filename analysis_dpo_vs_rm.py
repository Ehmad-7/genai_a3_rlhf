# analysis_dpo_vs_rm.py
import sys, json
sys.path.insert(0, "starter")
from models import load_tokenizer
from data import load_preference_splits

tok = load_tokenizer()
_, val = load_preference_splits(tok)
rm = json.load(open("results/reward_val_scores.json"))
dpo = json.load(open("results/dpo_val_scores.json"))      # implicit margins, same order
assert len(rm) == len(dpo) == len(val), (len(rm), len(dpo), len(val))

rm_ok = [s["r_chosen"] > s["r_rejected"] for s in rm]
dpo_ok = [m > 0 for m in dpo]
n = len(val)
print(f"RM accuracy {sum(rm_ok)/n:.3f} | DPO implicit accuracy {sum(dpo_ok)/n:.3f}")
both = sum(a and b for a, b in zip(rm_ok, dpo_ok))
neither = sum((not a) and (not b) for a, b in zip(rm_ok, dpo_ok))
agree = sum(a == b for a, b in zip(rm_ok, dpo_ok))
print(f"both right {both} | both wrong {neither} | same verdict on {agree}/{n} pairs")

for name, ok in (("RM", rm_ok), ("DPO implicit", dpo_ok)):
    b = {"chosen longer": [], "same length": [], "chosen shorter": []}
    for v, o in zip(val, ok):
        d = len(v["chosen_ids"]) - len(v["rejected_ids"])
        b["chosen longer" if d > 0 else "same length" if d == 0 else "chosen shorter"].append(o)
    print(name, {k: f"n={len(x)} acc={sum(x)/len(x):.3f}" for k, x in b.items() if x})
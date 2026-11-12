import sys, json, random, csv
sys.path.insert(0, "starter")
from models import load_tokenizer
from data import load_eval_posts

rows = [json.loads(l) for l in open("results/eval_summaries.jsonl")]
posts = load_eval_posts(load_tokenizer())
assert len(posts) == len(rows) == 300
for p, r in zip(posts, rows):
    assert p["prompt"][-500:] == r["prompt_tail"], "post order mismatch"

rng = random.Random(2026)                       # fixed seed
ids = sorted(rng.sample(range(300), 30))
key, md = {}, ["# Blind human evaluation sheet\n",
               "Har post ke liye A, B ya C mein se **sab se behtar** summary chuno "
               "(sahi baat, poori baat, saaf zubaan). Apna jawab sirf apni CSV mein likho, "
               "doosre member se puchna ya key kholna mana hai.\n"]
for n, i in enumerate(ids, 1):
    order = ["sft", "ppo", "dpo"]
    rng.shuffle(order)
    key[str(i)] = dict(zip("ABC", order))
    md.append(f"\n---\n## Post {n}  (id {i})\n\n{posts[i]['prompt']}\n")
    for letter, m in zip("ABC", order):
        md.append(f"- **{letter}:** {rows[i][m]['summary'].strip()}")

open("results/human_eval_sheet.md", "w").write("\n".join(md) + "\n")
json.dump(key, open("results/human_eval_key.json", "w"), indent=2)
for who in ("member1", "member2"):
    with open(f"results/human_eval_{who}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "pick"])
        for i in ids:
            w.writerow([i, ""])
print("done: 30 posts")
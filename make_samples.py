import csv, json, random, collections

rows = [json.loads(l) for l in open("results/eval_summaries.jsonl")]
H = {int(h["id"]): h for h in csv.DictReader(open("results/human_eval.csv"))}


def has_loop(text):
    w = text.lower().split()
    return any(v >= 3 for v in collections.Counter(zip(w, w[1:])).values())


def flags(r, m):
    f = []
    if has_loop(r[m]["summary"]):
        f.append("repeats itself")
    if r[m]["length"] >= 48:
        f.append("hits 48-token cap")
    return f


def gap(r, m):
    return r[m]["reward"] - r["sft"]["reward"]


better, worse = [], []
for r in rows:
    for m in ("ppo", "dpo"):
        sft_bad = bool(flags(r, "sft"))
        if gap(r, m) > 2 and not flags(r, m) and sft_bad:
            better.append((gap(r, m), r["idx"], m))
        if gap(r, m) < -2 and flags(r, m) and not sft_bad:
            worse.append((gap(r, m), r["idx"], m))
print("better candidates:", len(better), "| worse candidates:", len(worse))

chosen, used = [], set()


def take(cands, quotas, tag):
    for m, q in quotas:
        n = 0
        for g, i, mm in cands:
            if mm == m and i not in used and n < q:
                chosen.append((tag.format(m=m.upper()), i))
                used.add(i)
                n += 1


take(sorted(better, reverse=True), [("ppo", 2), ("dpo", 1)], "{m} clearly better than SFT")
take(sorted(worse), [("ppo", 2), ("dpo", 1)], "{m} clearly worse than SFT")
rest = [r["idx"] for r in rows if r["idx"] not in used]
for i in random.Random(0).sample(rest, 10 - len(chosen)):
    chosen.append(("Other (random)", i))

md = ["# Qualitative samples (10 evaluation posts)\n",
      "Reward = frozen reward-model score r_phi. 'Auto-notes' are computed by script; "
      "the 'Why' lines are written by hand after reading the summaries.\n"]
for n, (tag, i) in enumerate(chosen, 1):
    r = rows[i]
    md.append(f"\n---\n## Sample {n}: {tag} (post id {i})\n")
    md.append(f"**Post (tail):** {r['prompt_tail'][-400:].strip()}\n")
    md.append(f"- **Reference:** {r['reference']}")
    for m in ("sft", "ppo", "dpo"):
        md.append(f"- **{m.upper()}** [r={r[m]['reward']:.2f}, {r[m]['length']} tok]: {r[m]['summary']}")
    if i in H:
        md.append(f"- Human picks: member1={H[i]['member1_model']}, member2={H[i]['member2_model']}")
    notes = [f"{m.upper()} {', '.join(flags(r, m))}" for m in ("sft", "ppo", "dpo") if flags(r, m)]
    md.append(f"- Auto-notes: {'; '.join(notes) if notes else 'none'}")
    md.append("\n**Why:** TODO\n")
open("results/samples.md", "w", encoding="utf-8").write("\n".join(md) + "\n")
print([(t, i) for t, i in chosen])
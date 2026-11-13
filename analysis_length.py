import json, statistics as st

rows = [json.loads(l) for l in open("results/eval_summaries.jsonl")]
for m in ("sft", "ppo", "dpo"):
    cap = [r[m]["reward"] for r in rows if r[m]["length"] >= 48]
    ok = [r[m]["reward"] for r in rows if r[m]["length"] < 48]
    q = sum(r[m]["summary"].strip().endswith("?") for r in rows) / len(rows)
    print(f"{m}: capped n={len(cap)} mean r={st.mean(cap):.2f} | finished n={len(ok)} mean r={st.mean(ok):.2f} "
          f"| ends with '?': {q:.0%}")
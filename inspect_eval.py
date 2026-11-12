import json, random, statistics as st

rows = [json.loads(l) for l in open("results/eval_summaries.jsonl")]
print("posts:", len(rows))
for m in ["sft", "ppo", "dpo"]:
    L = [r[m]["length"] for r in rows]
    cap = sum(l >= 48 for l in L)
    print(f"{m}: mean len {st.mean(L):.1f} | at 48-token cap: {cap}/{len(rows)} "
          f"| mean reward {st.mean(r[m]['reward'] for r in rows):.2f}")


def show(r):
    print("-" * 80)
    print("POST (tail):", r["prompt_tail"][-250:].replace("\n", " "))
    print("REF :", r["reference"])
    for m in ["sft", "ppo", "dpo"]:
        print(f"{m.upper():4}: [{r[m]['reward']:.2f}] {r[m]['summary']}")


print("\n=== 3 random posts ===")
for r in random.Random(0).sample(rows, 3): show(r)
print("\n=== PPO reward >> SFT ===")
for r in sorted(rows, key=lambda r: r["ppo"]["reward"] - r["sft"]["reward"], reverse=True)[:3]: show(r)
print("\n=== PPO reward << SFT ===")
for r in sorted(rows, key=lambda r: r["ppo"]["reward"] - r["sft"]["reward"])[:3]: show(r)
print("\n=== DPO longest ===")
for r in sorted(rows, key=lambda r: r["dpo"]["length"], reverse=True)[:3]: show(r)

# Table 2 ke liye PPO run 2 ka final loss (pg + 0.1*value - 0.01*entropy, aakhri 10 iters)
log = json.load(open("results/ppo_run2_log.json"))["iters"][-10:]
loss = st.mean(r["pg_loss"] + 0.1 * r["value_loss"] - 0.01 * r["entropy"] for r in log)
print(f"\nPPO run2 final loss (last 10 iters): {loss:.4f}")
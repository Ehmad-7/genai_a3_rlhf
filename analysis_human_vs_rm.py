import csv, json

rows = [json.loads(l) for l in open("results/eval_summaries.jsonl")]
H = list(csv.DictReader(open("results/human_eval.csv")))

stats = {"member1": [0, 0.0], "member2": [0, 0.0]}   # [reward-top picks, sum of (picked reward - mean of 3 rewards)]
for h in H:
    r = rows[int(h["id"])]
    rew = {m: r[m]["reward"] for m in ("sft", "ppo", "dpo")}
    top = max(rew, key=rew.get)
    mean = sum(rew.values()) / 3
    for who in ("member1", "member2"):
        picked = h[f"{who}_model"]
        stats[who][0] += picked == top
        stats[who][1] += rew[picked] - mean

n = len(H)
for who, (hits, gap) in stats.items():
    print(f"{who}: picked the reward-model's top summary on {hits}/{n} posts "
          f"(chance = {n/3:.0f}); mean (picked reward - average reward) = {gap/n:+.2f}")

# PPO ko insaan ne kab pasand kiya jab RM ne usay sab se upar nahi rakha?
for h in H:
    r = rows[int(h["id"])]
    rew = {m: r[m]["reward"] for m in ("sft", "ppo", "dpo")}
    top = max(rew, key=rew.get)
    for who in ("member1", "member2"):
        if h[f"{who}_model"] == "ppo" and top != "ppo":
            print(f"{who} picked PPO on post id {h['id']} although RM ranked {top} first")
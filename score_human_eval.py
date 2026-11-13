import csv, json, collections

key = json.load(open("results/human_eval_key.json"))


def read(path):
    d = {}
    for r in csv.DictReader(open(path)):
        p = r["pick"].strip().upper()
        assert p in ("A", "B", "C"), f"{path}: id {r['id']} ka pick '{p}' galat hai"
        d[r["id"]] = p
    return d


m1, m2 = read("results/human_eval_member1.csv"), read("results/human_eval_member2.csv")
assert set(m1) == set(m2) == set(key), "ids match nahi karte"

out, c1, c2, agree = [], collections.Counter(), collections.Counter(), 0
for i in sorted(key, key=int):
    k = key[i]
    a, b = m1[i], m2[i]
    c1[k[a]] += 1
    c2[k[b]] += 1
    agree += a == b
    out.append({"id": i, "A_model": k["A"], "B_model": k["B"], "C_model": k["C"],
                "member1_pick": a, "member1_model": k[a],
                "member2_pick": b, "member2_model": k[b], "agree": int(a == b)})

with open("results/human_eval.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)

print("| | SFT picked | PPO picked | DPO picked |\n|---|---|---|---|")
print(f"| Member 1 | {c1['sft']} | {c1['ppo']} | {c1['dpo']} |")
print(f"| Member 2 | {c2['sft']} | {c2['ppo']} | {c2['dpo']} |")
print(f"| Agreement between members | {agree} / 30 |  |  |")
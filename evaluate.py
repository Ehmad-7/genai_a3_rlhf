import sys, json, argparse, os
sys.path.insert(0, "starter")

import torch
from rouge_score import rouge_scorer

from models import load_tokenizer, load_causal_lm
from data import load_eval_posts
from batching import collate_prompts, _pad
from common import token_logprobs, summary_mask, generate
from train_reward import load_reward_model

BS = 8


def load_policy(path, device):
    m = load_causal_lm(path, device=device).float().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def greedy_all(model, tok, posts, device):
    out = []
    for i in range(0, len(posts), BS):
        b = collate_prompts(posts[i:i + BS], tok.pad_token_id)
        out += generate(model, tok, b["input_ids"].to(device),
                        b["attention_mask"].to(device), max_new_tokens=48)   # greedy
    return out                                   # har post ke liye token ids (EOS samet)


@torch.no_grad()
def score_all(posts, resps, rm, policy, ref, tok, device):
    """Har post ka reward, aur KL to SFT = sum_t [log pi(y_t) - log pi_ref(y_t)] apni greedy summary par."""
    rewards, kls = [], []
    for i in range(0, len(posts), BS):
        P, R = posts[i:i + BS], resps[i:i + BS]
        ids, am = _pad([p["prompt_ids"] + r for p, r in zip(P, R)], tok.pad_token_id, "right")
        plen = torch.tensor([len(p["prompt_ids"]) for p in P])
        ids, am = ids.to(device), am.to(device)
        rewards += rm(ids, am).tolist()
        m = summary_mask(am, plen).float()
        lp = token_logprobs(policy, ids, am)
        lr = token_logprobs(ref, ids, am)
        kls += ((lp - lr) * m).sum(1).tolist()
    return rewards, kls


def mean(x):
    return sum(x) / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="checkpoints/sft")
    ap.add_argument("--ppo", default="checkpoints/ppo")
    ap.add_argument("--dpo", default="checkpoints/dpo")
    ap.add_argument("--reward", default="checkpoints/reward")
    ap.add_argument("--n", type=int, default=0)                  # smoke test
    ap.add_argument("--out", default="results/eval_summaries.jsonl")
    ap.add_argument("--metrics", default="results/eval_metrics.json")
    ap.add_argument("--table", default="results/table3.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device, torch.cuda.get_device_name(0) if device == "cuda" else "(CPU)")
    os.makedirs("results", exist_ok=True)

    tok = load_tokenizer()
    posts = load_eval_posts(tok)
    if args.n:
        posts = posts[:args.n]

    models = {"sft": load_policy(args.sft, device),
              "ppo": load_policy(args.ppo, device),
              "dpo": load_policy(args.dpo, device)}
    rm = load_reward_model(args.reward, base_path=args.sft, device=device).eval()
    for p in rm.parameters():
        p.requires_grad_(False)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    res = {}
    for name, m in models.items():
        print("generating:", name)
        resps = greedy_all(m, tok, posts, device)
        rew, kl = score_all(posts, resps, rm, m, models["sft"], tok, device)
        texts = [tok.decode(r, skip_special_tokens=True).strip() for r in resps]
        rl = [scorer.score(p["summary"].strip(), t)["rougeL"].fmeasure for p, t in zip(posts, texts)]
        res[name] = dict(text=texts, reward=rew, kl=kl, length=[len(r) for r in resps], rouge=rl)

    metrics = {}
    for name, r in res.items():
        win = None if name == "sft" else mean([a > b for a, b in zip(r["reward"], res["sft"]["reward"])])
        metrics[name] = {"mean_reward": mean(r["reward"]), "win_rate_vs_sft": win,
                         "kl_to_sft": mean(r["kl"]), "length": mean(r["length"]),
                         "rouge_l": mean(r["rouge"]), "n_posts": len(posts)}
    json.dump(metrics, open(args.metrics, "w"), indent=2)

    with open(args.out, "w") as f:
        for i, p in enumerate(posts):
            row = {"idx": i, "prompt_tail": p["prompt"][-500:], "reference": p["summary"].strip()}
            for name, r in res.items():
                row[name] = {"summary": r["text"][i], "reward": r["reward"][i], "kl": r["kl"][i],
                             "length": r["length"][i], "rouge_l": r["rouge"][i]}
            f.write(json.dumps(row) + "\n")

    lines = ["| Policy | Mean r_phi | Win rate vs. SFT | KL to SFT | Length | ROUGE-L |", "|---|---|---|---|---|---|"]
    for name in ["sft", "ppo", "dpo"]:
        m = metrics[name]
        w = "–" if m["win_rate_vs_sft"] is None else f"{m['win_rate_vs_sft']:.3f}"
        lines.append(f"| {name.upper()} | {m['mean_reward']:.3f} | {w} | {m['kl_to_sft']:.3f} | "
                     f"{m['length']:.1f} | {m['rouge_l']:.4f} |")
    open(args.table, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
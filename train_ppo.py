import sys, json, time, argparse, os, random
sys.path.insert(0, "starter")

import torch
import torch.nn as nn
from tqdm import tqdm

from config import SEED
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits
from batching import collate_prompts, _pad
from common import summary_mask, generate
from train_reward import load_reward_model

# Assignment ki fixed settings (gamma = 1: returns = reverse cumulative sum)
CLIP_EPS, BETA_KL, C_V, C_H = 0.2, 0.05, 0.1, 0.01
PPO_EPOCHS, LR = 4, 1e-5


def forward_tokens(model, ids, am, mask, vhead=None):
    """Sirf response positions par (mask: (B, L-1) bool) logprob nikalta hai.
    Position j ke logits token j+1 predict karte hain; value bhi usi position se.
    Policy ke liye (logp, value, entropy), ref ke liye sirf logp. Sab flat (N,)."""
    h = model.model(input_ids=ids, attention_mask=am).last_hidden_state[:, :-1][mask]
    logp_all = torch.log_softmax(model.lm_head(h).float(), dim=-1)
    tgt = ids[:, 1:][mask]
    logp = logp_all.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    if vhead is None:
        return logp
    entropy = -(logp_all.exp() * logp_all).sum(-1)
    return logp, vhead(h).squeeze(-1), entropy


def micro_batches(am, n):
    """Rows ko n-n ke tukron mein baantta hai, har tukre ki faltu right-padding hata kar."""
    for i in range(0, am.shape[0], n):
        sl = slice(i, i + n)
        yield sl, int(am[sl].sum(1).max())


@torch.no_grad()
def rollout(policy, vhead, ref, rm, tok, items, micro_bs, device):
    pad = tok.pad_token_id
    prompts = collate_prompts(items, pad)                      # left-padded, sirf generate ke liye
    resp = generate(policy, tok, prompts["input_ids"].to(device),
                    prompts["attention_mask"].to(device), max_new_tokens=48,
                    sample=True, temperature=1.0, top_k=0, top_p=1.0)
    # sequences ko right-padded dobara banao: [prompt ke asli tokens] + [response, pehle EOS tak]
    seqs = [x["prompt_ids"] + r for x, r in zip(items, resp)]
    ids, am = _pad(seqs, pad, "right")
    plen = torch.tensor([len(x["prompt_ids"]) for x in items])
    ids, am = ids.to(device), am.to(device)
    mask = summary_mask(am, plen).bool()                       # 1 sirf generated tokens par
    B, L = ids.shape

    old_logp = torch.zeros(B, L - 1, device=device)
    ref_logp, old_v = torch.zeros_like(old_logp), torch.zeros_like(old_logp)
    score = torch.zeros(B, device=device)
    for sl, Lc in micro_batches(am, micro_bs):
        i_, a_, m_ = ids[sl, :Lc], am[sl, :Lc], mask[sl, :Lc - 1]
        lp, v, _ = forward_tokens(policy, i_, a_, m_, vhead)
        old_logp[sl, :Lc - 1][m_] = lp
        old_v[sl, :Lc - 1][m_] = v
        ref_logp[sl, :Lc - 1][m_] = forward_tokens(ref, i_, a_, m_)
        score[sl] = rm(i_, a_)                                 # r_phi(x, y)

    # per-token reward (Eq. 41): -beta*(log pi_old - log pi_ref) + score at last token T
    kl_tok = (old_logp - ref_logp) * mask
    r = -BETA_KL * kl_tok
    last = (mask.long() * torch.arange(L - 1, device=device)).argmax(1)
    r[torch.arange(B, device=device), last] += score
    R = torch.flip(torch.cumsum(torch.flip(r, [1]), 1), [1])   # gamma = 1
    A = R - old_v
    Af = A[mask]
    A = (A - Af.mean()) / (Af.std() + 1e-8)                    # batch ke saare generated tokens par normalise

    return dict(ids=ids, am=am, mask=mask, old_logp=old_logp, R=R, A=A,
                score=score.mean().item(),
                kl=kl_tok.sum(1).mean().item(),
                length=mask.sum(1).float().mean().item(),
                sample=tok.decode(resp[0], skip_special_tokens=True))


def ppo_epoch(policy, vhead, opt, params, ro, micro_bs):
    opt.zero_grad(set_to_none=True)
    N = ro["mask"].sum().item()                                # poore batch ke generated tokens
    s = dict(pg=0.0, v=0.0, ent=0.0, clipped=0.0, dev=0.0)
    for sl, Lc in micro_batches(ro["am"], micro_bs):
        i_, a_, m_ = ro["ids"][sl, :Lc], ro["am"][sl, :Lc], ro["mask"][sl, :Lc - 1]
        lp, v, ent = forward_tokens(policy, i_, a_, m_, vhead)
        old = ro["old_logp"][sl, :Lc - 1][m_]
        A = ro["A"][sl, :Lc - 1][m_]
        R = ro["R"][sl, :Lc - 1][m_]
        ratio = torch.exp(lp - old)
        surr = torch.min(ratio * A, torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * A)
        pg = -surr.sum() / N                                   # -L_PPO
        vl = ((v - R) ** 2).sum() / N
        en = ent.sum() / N
        (pg + C_V * vl - C_H * en).backward()                  # L = -L_PPO + c_V*E(V-R)^2 - c_H*E[H]
        s["pg"] += pg.item(); s["v"] += vl.item(); s["ent"] += en.item()
        s["clipped"] += ((ratio - 1).abs() > CLIP_EPS).sum().item()
        s["dev"] = max(s["dev"], (ratio - 1).abs().max().item())
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step()
    s["clip_frac"] = s.pop("clipped") / N
    return s


def save_ckpt(path, policy, vhead, opt, it, log):
    os.makedirs(path, exist_ok=True)
    tmp = f"{path}/state.pt.tmp"
    torch.save({"policy": policy.state_dict(), "vhead": vhead.state_dict(),
                "opt": opt.state_dict(), "iter": it, "log": log}, tmp)
    os.replace(tmp, f"{path}/state.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", default="checkpoints/sft")
    ap.add_argument("--reward", default="checkpoints/reward")
    ap.add_argument("--out", default="checkpoints/ppo")
    ap.add_argument("--ckpt_dir", default="checkpoints/ppo_ckpt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--n_prompts", type=int, default=16)
    ap.add_argument("--micro_bs", type=int, default=4)
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--log_path", default="results/ppo_log.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device, torch.cuda.get_device_name(0) if device == "cuda" else "(CPU - STOP!)")
    os.makedirs("results", exist_ok=True)

    tok = load_tokenizer()
    sft_train, _ = load_sft_splits(tok)

    policy = load_causal_lm(args.sft, device=device).float()
    vhead = nn.Linear(policy.config.hidden_size, 1).to(device)
    nn.init.zeros_(vhead.weight); nn.init.zeros_(vhead.bias)  # V shuru mein 0
    ref = load_causal_lm(args.sft, device=device).float().eval()
    rm = load_reward_model(args.reward, base_path=args.sft, device=device).eval()
    for m in (ref, rm):
        for p in m.parameters():
            p.requires_grad_(False)
    assert not any(p.requires_grad for p in ref.parameters())
    assert not any(p.requires_grad for p in rm.parameters())
    print("frozen check: reference and reward model have requires_grad=False on all parameters")

    params = list(policy.parameters()) + list(vhead.parameters())
    opt = torch.optim.AdamW(params, lr=LR)
    log = {"gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "iters": []}

    start, prev_elapsed = 0, 0.0
    if args.resume and os.path.exists(f"{args.ckpt_dir}/state.pt"):
        s = torch.load(f"{args.ckpt_dir}/state.pt", map_location="cpu")
        policy.load_state_dict(s["policy"]); vhead.load_state_dict(s["vhead"])
        opt.load_state_dict(s["opt"])
        start, log = s["iter"], s["log"]
        prev_elapsed = log.get("elapsed_before", 0.0)
        print(f"RESUMED from iteration {start}")

    order = list(range(len(sft_train)))
    random.Random(SEED).shuffle(order)                         # har iteration ke naye 16 prompts
    t0 = time.time()
    for it in tqdm(range(start, args.iters), initial=start, total=args.iters):
        ti = time.time()
        torch.manual_seed(SEED + it)                           # resume ke baad bhi wohi sampling
        idx = order[it * args.n_prompts:(it + 1) * args.n_prompts]
        ro = rollout(policy, vhead, ref, rm, tok, [sft_train[i] for i in idx], args.micro_bs, device)

        eps = [ppo_epoch(policy, vhead, opt, params, ro, args.micro_bs) for _ in range(PPO_EPOCHS)]
        mean = lambda k: sum(e[k] for e in eps) / len(eps)
        rec = {"iter": it + 1, "reward": ro["score"], "kl": ro["kl"], "length": ro["length"],
               "clip_frac": mean("clip_frac"), "clip_frac_epoch1": eps[0]["clip_frac"],
               "ratio_dev_epoch1": eps[0]["dev"],
               "pg_loss": mean("pg"), "value_loss": mean("v"), "entropy": mean("ent"),
               "sec": time.time() - ti}
        log["iters"].append(rec)
        if it == 0:
            print(f"CHECK iter 1: KL={rec['kl']:.6f} (should be ~0) | epoch-1 max|ratio-1|="
                  f"{rec['ratio_dev_epoch1']:.2e} | epoch-1 clip fraction={rec['clip_frac_epoch1']:.4f} (should be 0)")
        if (it + 1) % 5 == 0 or it == 0:
            print(f"it {it+1} | r {rec['reward']:.3f} | KL {rec['kl']:.3f} | len {rec['length']:.1f} | "
                  f"clip {rec['clip_frac']:.3f} | pg {rec['pg_loss']:.3f} v {rec['value_loss']:.3f} "
                  f"H {rec['entropy']:.2f} | {rec['sec']:.1f}s | sample:{ro['sample'][:70]!r}")
        if (it + 1) % args.ckpt_every == 0:
            log["elapsed_before"] = prev_elapsed + time.time() - t0
            save_ckpt(args.ckpt_dir, policy, vhead, opt, it + 1, log)
            json.dump(log, open(args.log_path, "w"), indent=2)

    log["train_time_sec"] = prev_elapsed + time.time() - t0
    log["final_mean_reward_last10"] = sum(r["reward"] for r in log["iters"][-10:]) / len(log["iters"][-10:])
    json.dump(log, open(args.log_path, "w"), indent=2)
    policy.save_pretrained(args.out)                           # value head ke baghair
    tok.save_pretrained(args.out)
    print(f"FINAL mean reward (last 10 iters): {log['final_mean_reward_last10']:.3f} "
          f"| time {log['train_time_sec']/60:.1f} min")


if __name__ == "__main__":
    main()
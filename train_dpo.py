import sys, json, time, argparse, os, math
sys.path.insert(0, "starter")
from functools import partial

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import SEED
from models import load_tokenizer, load_causal_lm
from data import load_preference_splits
from batching import collate_pref
from common import token_logprobs, summary_mask
from train_reward import _fit


def split_pairs(batch, n, pad_id):
    """Pair-batch ko micro-batches mein todta hai: pehle n rows chosen, baad ke n rejected."""
    B = batch["chosen_ids"].shape[0]
    for i in range(0, B, n):
        ch, cm = batch["chosen_ids"][i:i + n], batch["chosen_mask"][i:i + n]
        rj, rjm = batch["rejected_ids"][i:i + n], batch["rejected_mask"][i:i + n]
        pl = batch["prompt_lens"][i:i + n]
        Lc = int(max(cm.sum(1).max(), rjm.sum(1).max()))
        ids = torch.cat([_fit(ch, Lc, pad_id), _fit(rj, Lc, pad_id)])
        mask = torch.cat([_fit(cm, Lc, 0), _fit(rjm, Lc, 0)])
        yield ids, mask, torch.cat([pl, pl]), ch.shape[0]


def seq_logps(model, ids, mask, pl):
    """log pi(y|x) = SUM of token logprobs over summary tokens only."""
    lp = token_logprobs(model, ids, mask)
    m = summary_mask(mask, pl).to(lp.dtype)
    return (lp * m).sum(dim=1)


@torch.no_grad()
def evaluate(policy, ref, val, pad_id, device, micro_bs, beta, cache):
    """Implicit reward accuracy: fraction where the term inside sigma(.) > 0.
    Reference logprobs har val pair ke liye ek hi baar nikalte hain (cache)."""
    policy.eval()
    loader = DataLoader(val, batch_size=8, shuffle=False,
                        collate_fn=partial(collate_pref, pad_id=pad_id))
    fill = "ref" not in cache
    if fill:
        cache["ref"] = []
    margins, k = [], 0
    for batch in loader:
        for ids, mask, pl, n in split_pairs(batch, micro_bs, pad_id):
            ids, mask = ids.to(device), mask.to(device)
            p = seq_logps(policy, ids, mask, pl)
            if fill:
                cache["ref"].append(seq_logps(ref, ids, mask, pl))
            r = cache["ref"][k]
            k += 1
            margins += (beta * ((p[:n] - r[:n]) - (p[n:] - r[n:]))).float().cpu().tolist()
    policy.train()
    m = torch.tensor(margins)
    return (m > 0).float().mean().item(), (-F.logsigmoid(m)).mean().item(), margins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/sft")
    ap.add_argument("--out", default="checkpoints/dpo")
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--micro_bs", type=int, default=2)    # pairs per micro-batch
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--max_steps", type=int, default=0)   # smoke test
    ap.add_argument("--val_limit", type=int, default=0)   # smoke test
    ap.add_argument("--log_path", default="results/dpo_log.json")
    ap.add_argument("--scores_path", default="results/dpo_val_scores.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    os.makedirs("results", exist_ok=True)

    tok = load_tokenizer()
    pad = tok.pad_token_id
    train, val = load_preference_splits(tok)
    if args.val_limit:
        val = val[: args.val_limit]

    policy = load_causal_lm(args.base, device=device).float()   # trainable
    ref = load_causal_lm(args.base, device=device).float().eval()  # frozen
    for p in ref.parameters():
        p.requires_grad_(False)
    assert not any(p.requires_grad for p in ref.parameters())
    print("frozen check: reference has requires_grad=False on all parameters")
    policy.train()

    loader = DataLoader(train, batch_size=args.bs, shuffle=True,
                        collate_fn=partial(collate_pref, pad_id=pad),
                        generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr)

    log = {"device": device,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "step_train_loss": [], "step_train_acc": [], "evals": []}
    cache = {}
    a0, l0, _ = evaluate(policy, ref, val, pad, device, args.micro_bs, args.beta, cache)
    log["evals"].append({"step": 0, "train_loss": None, "train_acc": None,
                         "val_acc": a0, "val_loss": l0})
    print(f"step 0 | val_acc {a0:.4f} | val_loss {l0:.4f}")

    t0, step = time.time(), 0
    pbar = tqdm(loader, total=args.max_steps or len(loader))
    for batch in pbar:
        step += 1
        B = batch["chosen_ids"].shape[0]
        opt.zero_grad(set_to_none=True)
        loss_val, correct = 0.0, 0
        for ids, mask, pl, n in split_pairs(batch, args.micro_bs, pad):
            ids, mask = ids.to(device), mask.to(device)
            with torch.no_grad():
                r = seq_logps(ref, ids, mask, pl)
            p = seq_logps(policy, ids, mask, pl)
            diff = args.beta * ((p[:n] - r[:n]) - (p[n:] - r[n:]))
            loss = -F.logsigmoid(diff).sum() / B
            loss.backward()
            loss_val += loss.item()
            correct += (diff > 0).sum().item()
        if step == 1:
            print(f"first-step loss = {loss_val:.5f}  (log 2 = {math.log(2):.5f})")
            log["first_step_loss"] = loss_val
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        log["step_train_loss"].append(loss_val)
        log["step_train_acc"].append(correct / B)
        pbar.set_description(f"loss {loss_val:.4f}")

        if step % args.log_every == 0:
            tr = sum(log["step_train_loss"][-args.log_every:]) / args.log_every
            ta = sum(log["step_train_acc"][-args.log_every:]) / args.log_every
            va, vl, _ = evaluate(policy, ref, val, pad, device, args.micro_bs, args.beta, cache)
            log["evals"].append({"step": step, "train_loss": tr, "train_acc": ta,
                                 "val_acc": va, "val_loss": vl})
            print(f"step {step} | train_loss {tr:.4f} train_acc {ta:.3f} | val_acc {va:.4f} val_loss {vl:.4f}")
        if args.max_steps and step >= args.max_steps:
            break

    log["train_time_sec"] = time.time() - t0
    policy.save_pretrained(args.out)
    tok.save_pretrained(args.out)

    va, vl, margins = evaluate(policy, ref, val, pad, device, args.micro_bs, args.beta, cache)
    if log["evals"][-1]["step"] != step:
        recent = log["step_train_loss"][-args.log_every:]
        log["evals"].append({"step": step, "train_loss": sum(recent) / len(recent),
                             "train_acc": None, "val_acc": va, "val_loss": vl})
    log["final_val_acc"] = va
    log["final_train_loss"] = sum(log["step_train_loss"][-50:]) / len(log["step_train_loss"][-50:])
    print(f"FINAL implicit reward accuracy: {va:.4f} | time {log['train_time_sec']/60:.1f} min")
    json.dump(margins, open(args.scores_path, "w"))
    json.dump(log, open(args.log_path, "w"), indent=2)


if __name__ == "__main__":
    main()
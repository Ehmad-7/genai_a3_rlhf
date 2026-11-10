import sys, json, time, argparse, os
sys.path.insert(0, "starter")
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import SEED
from models import load_tokenizer, load_causal_lm
from data import load_preference_splits
from batching import collate_pref


class RewardModel(nn.Module):
    """SFT body (no LM head) + linear head on the last real token."""
    def __init__(self, base_path="checkpoints/sft", device=None):
        super().__init__()
        lm = load_causal_lm(base_path, device="cpu").float()
        self.body = lm.model                      # transformer body, LM head dropped
        self.head = nn.Linear(lm.config.hidden_size, 1)
        del lm
        if device:
            self.to(device)

    def forward(self, input_ids, attention_mask):
        h = self.body(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        L = input_ids.shape[1]
        # last REAL token: attention_mask se, pad id se nahi (pad == eos hai!)
        idx = (attention_mask * torch.arange(L, device=input_ids.device)).argmax(dim=1)
        last = h[torch.arange(h.shape[0], device=h.device), idx]
        return self.head(last).squeeze(-1)        # (B,)


def save_reward_model(rm, path):
    os.makedirs(path, exist_ok=True)
    torch.save(rm.state_dict(), f"{path}/reward_model.pt")


def load_reward_model(path="checkpoints/reward", base_path="checkpoints/sft", device=None):
    rm = RewardModel(base_path)
    rm.load_state_dict(torch.load(f"{path}/reward_model.pt", map_location="cpu"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return rm.to(device)


def _fit(x, L, value):
    if x.shape[1] >= L:
        return x[:, :L]
    return F.pad(x, (0, L - x.shape[1]), value=value)


def split_pairs(batch, n, pad_id):
    """Pair-batch ko n-n pairs ke micro-batches mein todta hai. Har micro-batch
    mein chosen aur rejected ek hi lambai par pad hote hain aur concat ho jaate
    hain: pehle n rows = chosen, baad ke n rows = rejected."""
    B = batch["chosen_ids"].shape[0]
    for i in range(0, B, n):
        ch, cm = batch["chosen_ids"][i:i + n], batch["chosen_mask"][i:i + n]
        rj, rjm = batch["rejected_ids"][i:i + n], batch["rejected_mask"][i:i + n]
        Lc = int(max(cm.sum(1).max(), rjm.sum(1).max()))
        ids = torch.cat([_fit(ch, Lc, pad_id), _fit(rj, Lc, pad_id)])
        mask = torch.cat([_fit(cm, Lc, 0), _fit(rjm, Lc, 0)])
        yield ids, mask, ch.shape[0]


@torch.no_grad()
def evaluate(rm, val, pad_id, device, micro_bs):
    rm.eval()
    loader = DataLoader(val, batch_size=8, shuffle=False,
                        collate_fn=partial(collate_pref, pad_id=pad_id))
    rws, rls = [], []
    for batch in loader:
        for ids, mask, n in split_pairs(batch, micro_bs, pad_id):
            r = rm(ids.to(device), mask.to(device)).float().cpu()
            rws += r[:n].tolist()
            rls += r[n:].tolist()
    rm.train()
    rw, rl = torch.tensor(rws), torch.tensor(rls)
    acc = (rw > rl).float().mean().item()
    loss = -F.logsigmoid(rw - rl).mean().item()
    return acc, loss, rws, rls


@torch.no_grad()
def padding_check(rm, item, tok, device):
    """Correctness check: extra right-padding reward ko nahi badalni chahiye."""
    rm.eval()
    ids = torch.tensor([item["prompt_ids"] + item["chosen_ids"]]).to(device)
    am = torch.ones_like(ids)
    r1 = rm(ids, am)
    pad = torch.full((1, 7), tok.pad_token_id, device=device)
    r2 = rm(torch.cat([ids, pad], 1), torch.cat([am, torch.zeros_like(pad)], 1))
    print(f"padding check: r(no pad)={r1.item():.5f}  r(+7 pad)={r2.item():.5f}")
    assert torch.allclose(r1, r2, atol=1e-3), "reward padding token se badal rahi hai!"
    rm.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/sft")
    ap.add_argument("--out", default="checkpoints/reward")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=8)          # pairs per step
    ap.add_argument("--micro_bs", type=int, default=4)    # pairs per micro-batch
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--max_steps", type=int, default=0)   # smoke test
    ap.add_argument("--val_limit", type=int, default=0)   # smoke test
    ap.add_argument("--log_path", default="results/reward_log.json")
    ap.add_argument("--errors_path", default="results/reward_val_errors.json")
    ap.add_argument("--scores_path", default="results/reward_val_scores.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    os.makedirs("results", exist_ok=True)

    tok = load_tokenizer()
    pad = tok.pad_token_id
    train, val = load_preference_splits(tok)
    if args.val_limit:
        val = val[: args.val_limit]

    rm = RewardModel(args.base, device=device)
    rm.train()
    padding_check(rm, train[0], tok, device)

    loader = DataLoader(train, batch_size=args.bs, shuffle=True,
                        collate_fn=partial(collate_pref, pad_id=pad),
                        generator=torch.Generator().manual_seed(SEED))
    opt = torch.optim.AdamW(rm.parameters(), lr=args.lr)

    log = {"device": device,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "step_train_loss": [], "step_train_acc": [], "evals": []}

    acc0, vl0, _, _ = evaluate(rm, val, pad, device, args.micro_bs)
    log["evals"].append({"step": 0, "train_loss": None, "train_acc": None,
                         "val_acc": acc0, "val_loss": vl0})
    print(f"step 0 | val_acc {acc0:.4f} | val_loss {vl0:.4f}")

    t0, step = time.time(), 0
    pbar = tqdm(loader, total=args.max_steps or len(loader))
    for batch in pbar:
        step += 1
        B = batch["chosen_ids"].shape[0]
        opt.zero_grad(set_to_none=True)
        loss_val, correct = 0.0, 0
        for ids, mask, n in split_pairs(batch, args.micro_bs, pad):
            r = rm(ids.to(device), mask.to(device))
            rw, rl = r[:n], r[n:]
            loss = -F.logsigmoid(rw - rl).sum() / B       # mean over the full batch
            loss.backward()
            loss_val += loss.item()
            correct += (rw > rl).sum().item()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()

        log["step_train_loss"].append(loss_val)
        log["step_train_acc"].append(correct / B)
        pbar.set_description(f"loss {loss_val:.4f}")

        if step % args.log_every == 0:
            tr = sum(log["step_train_loss"][-args.log_every:]) / args.log_every
            ta = sum(log["step_train_acc"][-args.log_every:]) / args.log_every
            va, vl, _, _ = evaluate(rm, val, pad, device, args.micro_bs)
            log["evals"].append({"step": step, "train_loss": tr, "train_acc": ta,
                                 "val_acc": va, "val_loss": vl})
            print(f"step {step} | train_loss {tr:.4f} train_acc {ta:.3f} | val_acc {va:.4f} val_loss {vl:.4f}")
        if args.max_steps and step >= args.max_steps:
            break

    log["train_time_sec"] = time.time() - t0
    save_reward_model(rm, args.out)

    # final validation + per-pair scores (analysis 7.1 aur 7.5 ke liye)
    va, vl, rws, rls = evaluate(rm, val, pad, device, args.micro_bs)
    if log["evals"][-1]["step"] != step:
        recent = log["step_train_loss"][-args.log_every:]
        log["evals"].append({"step": step, "train_loss": sum(recent) / len(recent),
                             "train_acc": None, "val_acc": va, "val_loss": vl})
    log["final_val_acc"] = va
    log["final_train_loss"] = sum(log["step_train_loss"][-50:]) / len(log["step_train_loss"][-50:])
    print(f"FINAL pairwise val accuracy: {va:.4f}  | time {log['train_time_sec']/60:.1f} min")

    json.dump([{"r_chosen": a, "r_rejected": b} for a, b in zip(rws, rls)],
              open(args.scores_path, "w"))
    wrong = sorted([j for j in range(len(val)) if rws[j] <= rls[j]],
                   key=lambda j: rls[j] - rws[j], reverse=True)[:30]   # sab se confident galtiyan
    json.dump([{"prompt_tail": val[j]["prompt"][-400:], "chosen": val[j]["chosen"],
                "rejected": val[j]["rejected"], "r_chosen": rws[j], "r_rejected": rls[j]}
               for j in wrong], open(args.errors_path, "w"), indent=2)
    json.dump(log, open(args.log_path, "w"), indent=2)


if __name__ == "__main__":
    main()
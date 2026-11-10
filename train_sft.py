import sys, json, time, argparse, os
sys.path.insert(0, "starter")
from functools import partial

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import SEED
from models import load_tokenizer, load_causal_lm
from data import load_sft_splits, load_eval_posts
from batching import collate_sft, collate_prompts
from common import token_logprobs, summary_mask, generate


def sft_loss_parts(model, batch, device):
    """Returns (sum of NLL over summary tokens, number of summary tokens)."""
    ids = batch["input_ids"].to(device)
    am = batch["attention_mask"].to(device)
    lp = token_logprobs(model, ids, am)
    mask = summary_mask(am, batch["prompt_lens"]).to(lp.dtype)
    return -(lp * mask).sum(), mask.sum()

def split_batch(batch, n):
    """Batch ko n-n rows ke micro-batches mein todta hai aur har ek se
    faltu right-padding hata deta hai (memory bachane ke liye)."""
    B = batch["input_ids"].shape[0]
    for i in range(0, B, n):
        mb = {k: v[i:i + n] for k, v in batch.items()}
        Lc = int(mb["attention_mask"].sum(1).max())
        mb["input_ids"] = mb["input_ids"][:, :Lc]
        mb["attention_mask"] = mb["attention_mask"][:, :Lc]
        yield mb

@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total, count = 0.0, 0.0
    for batch in loader:
        s, c = sft_loss_parts(model, batch, device)
        total += s.item()
        count += c.item()
    model.train()
    return total / count


@torch.no_grad()
def sample_summaries(model, tok, posts, device):
    model.eval()
    batch = collate_prompts(posts, tok.pad_token_id)
    out = generate(model, tok, batch["input_ids"].to(device),
                   batch["attention_mask"].to(device), max_new_tokens=48)
    model.train()
    return [tok.decode(o, skip_special_tokens=True) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/sft")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--log_every", type=int, default=250)
    ap.add_argument("--max_steps", type=int, default=0)   # smoke test ke liye
    ap.add_argument("--val_limit", type=int, default=0)   # smoke test ke liye
    ap.add_argument("--log_path", default="results/sft_log.json")
    ap.add_argument("--samples_path", default="results/sft_samples.json")
    ap.add_argument("--micro_bs", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    os.makedirs("results", exist_ok=True)

    tok = load_tokenizer()
    pad = tok.pad_token_id
    train, val = load_sft_splits(tok)
    if args.val_limit:
        val = val[: args.val_limit]
    eval_posts = load_eval_posts(tok)[:5]

    model = load_causal_lm(device=device)
    model.train()

    # --- generations BEFORE SFT ---
    before = sample_summaries(model, tok, eval_posts, device)

    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        collate_fn=partial(collate_sft, pad_id=pad),
        generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(
        val, batch_size=8, shuffle=False,
        collate_fn=partial(collate_sft, pad_id=pad))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    log = {"device": device,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
           "step_train_loss": [], "evals": []}
    v0 = validate(model, val_loader, device)
    log["evals"].append({"step": 0, "train_loss": None, "val_loss": v0})
    print(f"step 0 | val_loss {v0:.4f}")

    t0 = time.time()
    step = 0
    pbar = tqdm(train_loader, total=args.max_steps or len(train_loader))
    for batch in pbar:
        step += 1
        
        # poore batch (16) ke summary tokens, taake loss poore batch par mean ho
        total_tokens = summary_mask(batch["attention_mask"], batch["prompt_lens"]).sum().item()
        opt.zero_grad(set_to_none=True)
        loss_val = 0.0
        for mb in split_batch(batch, args.micro_bs):
            s, _ = sft_loss_parts(model, mb, device)
            (s / total_tokens).backward()          # gradients jama hote jaate hain
            loss_val += s.item() / total_tokens
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()                                 # ek hi step poore batch ke liye

        log["step_train_loss"].append(loss_val)
        pbar.set_description(f"loss {loss_val:.4f}")
        
        if step % args.log_every == 0:
            recent = log["step_train_loss"][-args.log_every:]
            tr = sum(recent) / len(recent)
            v = validate(model, val_loader, device)
            log["evals"].append({"step": step, "train_loss": tr, "val_loss": v})
            print(f"step {step} | train_loss {tr:.4f} | val_loss {v:.4f}")
        if args.max_steps and step >= args.max_steps:
            break

    # agar aakhri step eval par nahi tha to ek final eval
    if log["evals"][-1]["step"] != step:
        recent = log["step_train_loss"][-args.log_every:]
        v = validate(model, val_loader, device)
        log["evals"].append({"step": step, "train_loss": sum(recent) / len(recent), "val_loss": v})
        print(f"step {step} (final) | val_loss {v:.4f}")

    log["train_time_sec"] = time.time() - t0
    log["final_train_loss"] = sum(log["step_train_loss"][-50:]) / len(log["step_train_loss"][-50:])
    log["final_val_loss"] = log["evals"][-1]["val_loss"]

    # --- save model + generations AFTER SFT ---
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    after = sample_summaries(model, tok, eval_posts, device)

    samples = [{"prompt_tail": p["prompt"][-300:], "reference": p["summary"],
                "before_sft": b, "after_sft": a}
               for p, b, a in zip(eval_posts, before, after)]
    json.dump(samples, open(args.samples_path, "w"), indent=2)
    json.dump(log, open(args.log_path, "w"), indent=2)
    print(f"saved model to {args.out}; time {log['train_time_sec']/60:.1f} min")


if __name__ == "__main__":
    main()
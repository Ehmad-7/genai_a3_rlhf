"""FastAPI front end: paste a Reddit post, get the SFT, PPO and DPO summaries
side by side, each with the score of the frozen reward model r_phi.

Run from the repository root:

    uvicorn app.main:app --port 8000

Checkpoint folders (default: checkpoints/sft, ppo, dpo, reward) can be changed
with the environment variables SFT_PATH, PPO_PATH, DPO_PATH and RM_PATH.
"""
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "starter"))
sys.path.insert(0, str(ROOT))

from config import MAX_PROMPT_TOKENS, MAX_SUMMARY_TOKENS  # noqa: E402
from models import load_tokenizer, load_causal_lm  # noqa: E402
from data import clean_post  # noqa: E402
from common import generate  # noqa: E402
from train_reward import load_reward_model  # noqa: E402


def _path(env_name, default):
    p = Path(os.environ.get(env_name, default))
    return str(p if p.is_absolute() else ROOT / p)


POLICY_PATHS = {
    "sft": _path("SFT_PATH", "checkpoints/sft"),
    "ppo": _path("PPO_PATH", "checkpoints/ppo"),
    "dpo": _path("DPO_PATH", "checkpoints/dpo"),
}
RM_PATH = _path("RM_PATH", "checkpoints/reward")

STATE = {}
LOCK = threading.Lock()  # one generation at a time (the models are shared)


@asynccontextmanager
async def lifespan(app):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STATE["device"] = device
    print("device:", device)
    STATE["tok"] = load_tokenizer(POLICY_PATHS["sft"])  # tokenizer saved with the SFT checkpoint
    STATE["policies"] = {}
    for name, path in POLICY_PATHS.items():
        t0 = time.time()
        STATE["policies"][name] = load_causal_lm(path, device=device).float().eval()
        print(f"loaded {name} from {path} ({time.time() - t0:.1f}s)")
    STATE["rm"] = load_reward_model(RM_PATH, base_path=POLICY_PATHS["sft"], device=device).eval()
    print("loaded reward model from", RM_PATH)
    for m in [*STATE["policies"].values(), STATE["rm"]]:
        for p in m.parameters():
            p.requires_grad_(False)
    yield
    STATE.clear()


app = FastAPI(title="RLHF summarisation demo", lifespan=lifespan)


class SummariseRequest(BaseModel):
    policy: str
    post: str = Field(max_length=30000)
    subreddit: str = "relationships"
    title: str = ""


def build_prompt_ids(req, tok):
    """Same prompt format as the training data:
    SUBREDDIT: r/<name> / TITLE: <title> / POST: <text> / TL;DR:"""
    sub = req.subreddit.strip()
    if sub.lower().startswith("r/"):
        sub = sub[2:]
    raw = f"SUBREDDIT: r/{sub or 'relationships'}\nTITLE: {req.title.strip()}\nPOST: {req.post.strip()}"
    prompt = clean_post(raw)  # drops any 'TL;DR:' the user pasted and ends with "\nTL;DR:"
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    truncated = False
    if len(ids) > MAX_PROMPT_TOKENS:  # the models were trained on prompts of at most 448 tokens
        tail = tok("\nTL;DR:", add_special_tokens=False)["input_ids"]
        ids = ids[: MAX_PROMPT_TOKENS - len(tail)] + tail
        truncated = True
    return ids, truncated


@app.get("/")
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"device": STATE.get("device"), "policies": list(STATE.get("policies", {})),
            "reward_model": "rm" in STATE}


@app.post("/api/summarise")
def summarise(req: SummariseRequest):
    name = req.policy.lower()
    if name not in POLICY_PATHS:
        raise HTTPException(400, f"policy must be one of {sorted(POLICY_PATHS)}")
    if not req.post.strip():
        raise HTTPException(400, "Paste a post first.")
    tok, dev = STATE["tok"], STATE["device"]
    ids, truncated = build_prompt_ids(req, tok)

    t0 = time.time()
    with LOCK, torch.inference_mode():
        x = torch.tensor([ids], device=dev)
        resp = generate(STATE["policies"][name], tok, x, torch.ones_like(x),
                        max_new_tokens=MAX_SUMMARY_TOKENS)[0]  # greedy, like evaluate.py
        seq = torch.tensor([ids + resp], device=dev)
        reward = STATE["rm"](seq, torch.ones_like(seq)).item()  # r_phi(post, summary)

    return {
        "policy": name,
        "summary": tok.decode(resp, skip_special_tokens=True).strip(),
        "reward": reward,
        "tokens": len(resp),
        "finished": bool(resp) and resp[-1] == tok.eos_token_id,  # False = cut off at the token limit
        "input_tokens": len(ids),
        "input_truncated": truncated,
        "seconds": round(time.time() - t0, 1),
    }
import sys
sys.path.insert(0, "starter")
from functools import partial
import torch
from torch.utils.data import DataLoader

from models import load_tokenizer, load_causal_lm
from data import load_sft_splits
from batching import collate_sft
from common import token_logprobs, summary_mask

tok = load_tokenizer()
sft_train, _ = load_sft_splits(tok)
model = load_causal_lm(device="cpu").eval()

items = sft_train[:4]
b = collate_sft(items, pad_id=tok.pad_token_id)

with torch.no_grad():
    lp = token_logprobs(model, b["input_ids"], b["attention_mask"])
mask = summary_mask(b["attention_mask"], b["prompt_lens"])
print("logprobs shape:", tuple(lp.shape), "| mask shape:", tuple(mask.shape))
assert lp.shape == mask.shape == (4, b["input_ids"].shape[1] - 1)

for i, item in enumerate(items):
    ids = b["input_ids"][i, 1:][mask[i].bool()]
    decoded = tok.decode(ids)
    print(f"\n[{i}] masked tokens decoded: {decoded!r}")
    assert ids.tolist() == item["summary_ids"], "mask does not match summary tokens!"
    print(f"[{i}] mask sum = {mask[i].sum().item()} | summary tokens (with EOS) = {len(item['summary_ids'])}")

print("\nAll checks passed: mask is 1 exactly on summary tokens (+ EOS).")
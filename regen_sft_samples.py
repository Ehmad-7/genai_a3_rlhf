import sys, json
sys.path.insert(0, "starter")

from models import load_tokenizer, load_causal_lm
from data import load_eval_posts
from batching import collate_prompts
from common import generate

tok = load_tokenizer()
posts = load_eval_posts(tok)[:5]
batch = collate_prompts(posts, tok.pad_token_id)


def run(path):
    model = load_causal_lm(path).eval()
    dev = next(model.parameters()).device
    out = generate(model, tok, batch["input_ids"].to(dev),
                   batch["attention_mask"].to(dev), max_new_tokens=48)
    return [tok.decode(o, skip_special_tokens=True) for o in out]


before = run("HuggingFaceTB/SmolLM2-135M")
after = run("checkpoints/sft")

samples = [{"prompt_tail": p["prompt"][-300:], "reference": p["summary"],
            "before_sft": b, "after_sft": a}
           for p, b, a in zip(posts, before, after)]
json.dump(samples, open("results/sft_samples.json", "w"), indent=2)
for s in samples[:2]:
    print("REF   :", s["reference"])
    print("BEFORE:", s["before_sft"])
    print("AFTER :", s["after_sft"], "\n")
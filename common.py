import torch
import torch.nn.functional as F


def token_logprobs(model, input_ids, attention_mask):
    """log pi(x_t | x_<t) for every position t >= 1.
    Returns shape (B, L-1). Entry j is the logprob of token j+1."""
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    targets = input_ids[:, 1:]
    B, Lm1, V = logits.shape
    lp = -F.cross_entropy(
        logits.reshape(-1, V).float(), targets.reshape(-1), reduction="none"
    )
    return lp.view(B, Lm1)


def summary_mask(attention_mask, prompt_lens):
    """(B, L-1) mask, 1 exactly on summary tokens (incl. EOS), 0 on prompt
    and padding. Aligned with token_logprobs: entry j <-> token j+1.
    Assumes RIGHT-padded batches (collate_sft / collate_pref)."""
    L = attention_mask.shape[1]
    pos = torch.arange(1, L, device=attention_mask.device).unsqueeze(0)  # target positions
    is_summary = pos >= prompt_lens.to(attention_mask.device).unsqueeze(1)
    is_real = attention_mask[:, 1:].bool()
    return (is_summary & is_real).long()


@torch.no_grad()
def generate(model, tok, input_ids, attention_mask, max_new_tokens=48,
             sample=False, temperature=1.0, top_k=0, top_p=1.0):
    """Generate from LEFT-padded prompts. Returns a list (one per prompt) of
    new token ids, cut at the first EOS (EOS included if produced)."""
    kwargs = dict(max_new_tokens=max_new_tokens,
                  pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if sample:
        kwargs.update(do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p)
    else:
        kwargs.update(do_sample=False)
    out = model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    new = out[:, input_ids.shape[1]:].tolist()
    res = []
    for seq in new:
        if tok.eos_token_id in seq:
            seq = seq[: seq.index(tok.eos_token_id) + 1]
        res.append(seq)
    return res
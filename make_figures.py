import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(p):
    return json.load(open(p))


def smooth(x, k=50):
    x = np.array(x, dtype=float)
    return x if len(x) < k else np.convolve(x, np.ones(k) / k, mode="valid")


# Figure 1: SFT train aur validation loss
ev = load("results/sft_log.json")["evals"]
tr = [e for e in ev if e["train_loss"] is not None]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([e["step"] for e in tr], [e["train_loss"] for e in tr], "o-", label="train (mean of previous 250 steps)")
ax.plot([e["step"] for e in ev], [e["val_loss"] for e in ev], "s-", label="validation")
ax.set_xlabel("step"); ax.set_ylabel("NLL per summary token"); ax.set_title("SFT loss"); ax.legend()
fig.tight_layout(); fig.savefig("results/fig1_sft.png", dpi=150); plt.close(fig)

# Figure 2: reward model train loss aur validation accuracy
r = load("results/reward_log.json")
k = 50
sm = smooth(r["step_train_loss"], k)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(np.arange(len(sm)) + k, sm)
axs[0].set_xlabel("step"); axs[0].set_ylabel("loss"); axs[0].set_title("Reward model: training loss (50-step avg)")
axs[1].plot([e["step"] for e in r["evals"]], [e["val_acc"] for e in r["evals"]], "o-")
axs[1].axhline(0.5, ls="--", c="gray", label="chance")
axs[1].set_xlabel("step"); axs[1].set_ylabel("pairwise accuracy"); axs[1].set_title("Reward model: validation accuracy"); axs[1].legend()
fig.tight_layout(); fig.savefig("results/fig2_reward.png", dpi=150); plt.close(fig)

# Figure 3: PPO (run 1 aur run 2): reward, KL, length, clip fraction
runs = {"run 1 (value bias 0)": load("results/ppo_run1_log.json")["iters"],
        "run 2 (value bias = mean return)": load("results/ppo_run2_log.json")["iters"]}
panels = [("reward", "mean $r_\\phi$ score"), ("kl", "mean KL to reference"),
          ("length", "mean summary length (tokens)"), ("clip_frac", "clip fraction")]
fig, axs = plt.subplots(2, 2, figsize=(11, 7))
for ax, (key, title) in zip(axs.ravel(), panels):
    for name, it in runs.items():
        ax.plot([x["iter"] for x in it], [x[key] for x in it], label=name, alpha=0.85)
    ax.set_xlabel("iteration"); ax.set_title(title)
axs[0, 0].legend()
fig.tight_layout(); fig.savefig("results/fig3_ppo.png", dpi=150); plt.close(fig)

# Figure 4: DPO training loss aur validation implicit reward accuracy
d = load("results/dpo_log.json")
sm = smooth(d["step_train_loss"], k)
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot(np.arange(len(sm)) + k, sm)
axs[0].axhline(0.6931, ls="--", c="gray", label="log 2")
axs[0].set_xlabel("step"); axs[0].set_ylabel("loss"); axs[0].set_title("DPO: training loss (50-step avg)"); axs[0].legend()
axs[1].plot([e["step"] for e in d["evals"]], [e["val_acc"] for e in d["evals"]], "o-")
axs[1].set_xlabel("step"); axs[1].set_ylabel("implicit reward accuracy"); axs[1].set_title("DPO: validation implicit reward accuracy")
fig.tight_layout(); fig.savefig("results/fig4_dpo.png", dpi=150); plt.close(fig)

print("saved fig1_sft.png, fig2_reward.png, fig3_ppo.png, fig4_dpo.png in results/")
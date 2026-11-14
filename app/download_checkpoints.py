"""Download the four checkpoints from Hugging Face into checkpoints/.

Interrupted downloads resume, so it is safe to run this again.

    HF_TOKEN=hf_xxx python app/download_checkpoints.py            # everything that is missing
    HF_TOKEN=hf_xxx python app/download_checkpoints.py ppo dpo    # only these

HF_TOKEN is only needed while the repositories are private.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
USER = os.environ.get("HF_USER", "ehmad7")

# name: (local folder, Hugging Face repo, file that proves the folder is complete)
TARGETS = {
    "sft": ("checkpoints/sft", f"{USER}/genai-a3-sft", "model.safetensors"),
    "ppo": ("checkpoints/ppo", f"{USER}/genai-a3-ppo-run2", "model.safetensors"),
    "dpo": ("checkpoints/dpo", f"{USER}/genai-a3-dpo", "model.safetensors"),
    "reward": ("checkpoints/reward", f"{USER}/genai-a3-reward", "reward_model.pt"),
}

wanted = sys.argv[1:] or list(TARGETS)
for name in wanted:
    if name not in TARGETS:
        sys.exit(f"unknown checkpoint '{name}', choose from {list(TARGETS)}")
    folder, repo, marker = TARGETS[name]
    local = ROOT / folder
    if (local / marker).exists():
        print(f"{name}: already in {folder}")
        continue
    print(f"{name}: downloading {repo} -> {folder}")
    snapshot_download(repo, local_dir=str(local), token=os.environ.get("HF_TOKEN"))
    print(f"{name}: done")
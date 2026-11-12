import json, sys

path = sys.argv[1] if len(sys.argv) > 1 else "results/ppo_run1_log.json"
log = json.load(open(path))
it = log["iters"]
print("file:", path)
print("iterations:", len(it), "| time (min):", round(log["train_time_sec"] / 60, 1))

f = it[0]
print(f"CHECK iter 1: KL={f['kl']:.6f} | epoch-1 max|ratio-1|={f['ratio_dev_epoch1']:.2e} "
      f"| epoch-1 clip fraction={f['clip_frac_epoch1']:.4f}")
print("max epoch-1 |ratio-1| over all iters:",
      f"{max(r['ratio_dev_epoch1'] for r in it):.2e}")
print("max epoch-1 clip fraction over all iters:",
      max(r['clip_frac_epoch1'] for r in it))

print("\niter | reward |    KL |  len | clip | pg_loss | v_loss | entropy")
for k in [1, 10, 25, 50, 75, 100, 125, 150, 175, 200]:
    if k <= len(it):
        r = it[k - 1]
        print(f"{k:4d} | {r['reward']:6.3f} | {r['kl']:5.2f} | {r['length']:4.1f} | "
              f"{r['clip_frac']:.3f} | {r['pg_loss']:7.4f} | {r['value_loss']:6.3f} | {r['entropy']:.2f}")
"""Measure the official loop's own reproducibility floor.

If re-running model.inference under the same seed already differs by ~1e-4 A,
then the split loop matching to 1e-4 is exact agreement, not a discrepancy.
"""
import argparse, os, sys
os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")
p = argparse.ArgumentParser()
p.add_argument("--repo", required=True); p.add_argument("--ckpt", required=True)
p.add_argument("--pdb", required=True); p.add_argument("--seed", type=int, default=307)
p.add_argument("--sde-step", type=int, default=10); p.add_argument("--particles", type=int, default=4)
a = p.parse_args()
sys.path.insert(0, a.repo)
import torch
from data import make_batch
device = torch.device("cuda:0")
model = torch.load(a.ckpt, map_location="cpu", weights_only=False).to(device).eval()
batch, _ = make_batch(a.pdb, a.particles)
batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
x0 = batch["x0"].clone()
outs = []
with torch.no_grad():
    for _ in range(2):
        batch["x0"] = x0.clone()
        torch.manual_seed(a.seed)
        outs.append(model.inference(batch, sde_step=a.sde_step))
print("official vs official, same seed: max abs diff = %.3e A" % (outs[0]-outs[1]).abs().max().item())
print("coordinate magnitude: mean |x| = %.2f A" % outs[0].abs().mean().item())

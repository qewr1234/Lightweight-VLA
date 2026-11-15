"""train_multires.py: what is trained, at which resolution, and that the result can be evaluated.

The LIBERO dataset and the public checkpoint come from the Hugging Face Hub, so they are not
used here. A random-weight policy and a synthetic dataset in the same layout as lerobot/libero
(two cameras, 8-dim state, 7-dim actions, a task string) go through the real code: lerobot's
processors, the policy's training forward, the optimizer step and the checkpoint round trip.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "experiments"))

import torch

import train_multires as T
from lerobot.utils.constants import ACTION, OBS_STATE

E = T.E
results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")


class FakeLibero(torch.utils.data.Dataset):
    """Frames in the layout of lerobot/libero after LeRobotDataset: images in [0, 1], CHW."""

    def __init__(self, chunk_size: int, n: int = 8):
        g = torch.Generator().manual_seed(0)
        self.images = torch.rand(n, 2, 3, 64, 64, generator=g)
        self.state = torch.randn(n, 8, generator=g)
        self.action = torch.randn(n, chunk_size, 7, generator=g)

    def __len__(self):
        return len(self.state)

    def __getitem__(self, i):
        return {"observation.images.image": self.images[i, 0], "observation.images.image2": self.images[i, 1],
                OBS_STATE: self.state[i], ACTION: self.action[i],
                "action_is_pad": torch.zeros(self.action.shape[1], dtype=torch.bool),
                "task": "put the bowl on the plate"}


print("=" * 70)
print("Building random-weight policy + real processors ...")
torch.manual_seed(0)
args = SimpleNamespace(random=True, policy_path=None, rename_map=None, num_steps=2, n_action_steps=None,
                       dtype="float32", device="cpu")
policy, preprocessor, postprocessor = E.load_policy(args)
cfg = policy.config
train_res = cfg.resize_imgs_with_padding[0]
low_res = train_res - 64

# 1) Only the action expert (and the small projections around it) is trained.
trainable = T.freeze_all_but_expert(policy)
policy.train()
names = [n for n, p in policy.named_parameters() if p.requires_grad]
vlm = policy.model.vlm_with_expert.vlm
check("frozen: no trainable parameter in the VLM or the vision encoder",
      names and not any(p.requires_grad for p in vlm.parameters()),
      f"{sum(p.numel() for p in trainable) / 1e6:.1f}M trainable in {len(names)} tensors")
check("frozen parts stay in eval mode while the policy trains", policy.training and not vlm.training)

# 2) Resolution schedule: equal shares, reproducible.
sched = T.resolution_schedule([train_res, low_res, 256], 30, seed=7)
blocks = [sorted(sched[i:i + 3]) for i in range(0, 30, 3)]
check("schedule: every block of 3 steps holds each resolution once",
      all(b == sorted([train_res, low_res, 256]) for b in blocks)
      and sched == T.resolution_schedule([train_res, low_res, 256], 30, seed=7)
      and sched != T.resolution_schedule([train_res, low_res, 256], 30, seed=8))
lrs = [T.learning_rate(s, 100, 1.0, 10) for s in range(100)]
check("learning rate: warms up to the peak, decays to a tenth",
      abs(lrs[0] - 0.1) < 1e-9 and abs(max(lrs) - 1.0) < 1e-9 and 0.1 <= lrs[-1] < 0.11)

# 3) A training step at each resolution.
dataset = FakeLibero(cfg.chunk_size)
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
batch = preprocessor(next(iter(loader)))
check("dataset batch passes through the checkpoint's own preprocessor",
      batch[ACTION].shape == (2, cfg.chunk_size, 7) and batch[OBS_STATE].shape == (2, 8)
      and "observation.language.tokens" in batch)

T.set_resolution(policy, low_res, train_res)
check("below the training resolution: smaller input, posmap on the training grid",
      tuple(cfg.resize_imgs_with_padding) == (low_res, low_res) and cfg.posmap_ref_resolution == train_res)
T.set_resolution(policy, train_res, train_res)
check("at the training resolution: posmap off", cfg.posmap_ref_resolution is None)

noise = torch.randn(2, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(1))
t_fixed = torch.tensor([0.3, 0.7])


def fixed_loss(res):
    T.set_resolution(policy, res, train_res)
    with torch.no_grad():
        return float(T.compute_loss(policy, batch, "cpu", noise=noise.clone(), time_=t_fixed.clone()))


before = {r: fixed_loss(r) for r in (train_res, low_res)}
check("the loss depends on the input resolution", before[train_res] != before[low_res],
      f"{before[train_res]:.4f} vs {before[low_res]:.4f}")

vlm_before = [p.detach().clone() for p in list(vlm.parameters())[:20]]
expert_before = trainable[0].detach().clone()
optimizer = torch.optim.AdamW(trainable, lr=1e-3)
optimizer.zero_grad()
torch.manual_seed(3)
losses = []
for i, res in enumerate(T.resolution_schedule([train_res, low_res], 12, seed=0)):
    T.set_resolution(policy, res, train_res)
    losses.append(T.train_step(policy, batch, optimizer, trainable, "cpu"))
after = {r: fixed_loss(r) for r in (train_res, low_res)}
check("training steps: finite losses, expert weights move, VLM weights do not",
      all(torch.isfinite(torch.tensor(losses)))
      and not torch.equal(trainable[0], expert_before)
      and all(torch.equal(a, b) for a, b in zip(vlm_before, list(vlm.parameters())[:20])))
check("the loss falls at both resolutions on the batch it trained on",
      after[train_res] < before[train_res] and after[low_res] < before[low_res],
      f"{train_res}px {before[train_res]:.3f} -> {after[train_res]:.3f}, "
      f"{low_res}px {before[low_res]:.3f} -> {after[low_res]:.3f}")

# Gradient accumulation: the optimizer steps on the second of two passes.
snapshot = trainable[0].detach().clone()
T.train_step(policy, batch, optimizer, trainable, "cpu", accum_index=0, accum=2)
held = torch.equal(trainable[0], snapshot)
T.train_step(policy, batch, optimizer, trainable, "cpu", accum_index=1, accum=2)
check("accumulation: weights change only on the last pass of a group",
      held and not torch.equal(trainable[0], snapshot))

# 4) The saved checkpoint is one eval_libero.py can load, and plans the same actions.
policy.eval()
T.set_resolution(policy, low_res, train_res)
obs = {k: v for k, v in batch.items() if k != ACTION}
plan_noise = torch.randn(2, cfg.chunk_size, cfg.max_action_dim, generator=torch.Generator().manual_seed(9))
ref = E.plan_chunk(policy, obs, obs, noise=plan_noise.clone())
with tempfile.TemporaryDirectory() as tmp:
    T.save_checkpoint(policy, preprocessor, postprocessor, Path(tmp), train_res)
    check("save leaves the running policy at its current resolution",
          tuple(cfg.resize_imgs_with_padding) == (low_res, low_res) and cfg.posmap_ref_resolution == train_res)
    saved_cfg = json.loads((Path(tmp) / "config.json").read_text())
    check("saved config: training resolution, posmap off",
          saved_cfg["resize_imgs_with_padding"] == [train_res, train_res]
          and saved_cfg.get("posmap_ref_resolution") is None)
    loaded, pre2, post2 = E.load_policy(SimpleNamespace(
        random=False, policy_path=tmp, rename_map=None, num_steps=2, n_action_steps=None, dtype="float32",
        device="cpu"))
    E.apply_setting(loaded, {"resolution": low_res, "posmap": True}, train_res)
    got = E.plan_chunk(loaded, obs, obs, noise=plan_noise.clone())
    batch2 = pre2(next(iter(loader)))
check("reloaded checkpoint plans the same actions at the lower resolution",
      (ref - got).abs().max().item() < 1e-5, f"max|diff| = {(ref - got).abs().max().item():.2e}")
check("reloaded preprocessor produces the same batch",
      torch.equal(batch2[OBS_STATE], batch[OBS_STATE]) and torch.equal(batch2[ACTION], batch[ACTION]))

# 5) main(): the whole run with the dataset swapped out.
T.make_dataset = lambda a, chunk_size: FakeLibero(chunk_size)
with tempfile.TemporaryDirectory() as tmp:
    sys.argv = ["train_multires.py", "--random", "--resolutions", str(train_res), str(low_res),
                "--steps", "2", "--batch", "2", "--accum", "2", "--warmup", "1", "--workers", "0",
                "--device", "cpu", "--out", tmp]
    T.main()
    summary = json.loads((Path(tmp) / "train_summary.json").read_text())
    log = [json.loads(line) for line in (Path(tmp) / "train_log.jsonl").read_text().splitlines()]
    files = {p.name for p in Path(tmp).iterdir()}
check("main: 2 optimizer steps of 2 passes each, both resolutions seen",
      [r["step"] for r in log] == [1, 2] and set(summary["final_loss"]) == {str(train_res), str(low_res)},
      f"final loss {summary['final_loss']}")
check("main: writes a loadable checkpoint directory",
      {"config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"} <= files)

print("=" * 70)
failed = [r for r in results if not r[1]]
print(f"{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)

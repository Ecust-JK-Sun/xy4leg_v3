from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

from depth_bar_estimator import DepthBarEstimator


parser = argparse.ArgumentParser(description="Train a heightmap-to-bar-estimate model for Truss V3.")
parser.add_argument(
    "--dataset",
    type=str,
    nargs="+",
    required=True,
    help="One or more dataset .pt files from collect_depth_bar_dataset_v3.py.",
)
parser.add_argument("--output", type=str, required=True, help="Output TorchScript .pt path.")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--learning_rate", type=float, default=1.0e-3)
parser.add_argument("--val_fraction", type=float, default=0.1)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()


def _load_dataset(path: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    data = torch.load(path, map_location="cpu")
    if "heightmap" in data:
        heightmap = data["heightmap"].float()
    else:
        heightmap = data["depth"].float()
    target = data["bar_delta_b"].float()
    if heightmap.ndim == 3:
        heightmap = heightmap.unsqueeze(1)
    if heightmap.ndim != 4:
        raise ValueError(f"Expected heightmap shape [N,C,H,W], got {tuple(heightmap.shape)}")
    if target.ndim != 2 or target.shape[1] < 3:
        raise ValueError(f"Expected target shape [N,3+], got {tuple(target.shape)}")
    levels = data.get("level")
    if levels is not None:
        levels = levels.long()
    return heightmap, target[:, :3], levels


heightmap_chunks = []
target_chunks = []
level_chunks = []
for dataset_path in args.dataset:
    heightmap_i, target_i, levels_i = _load_dataset(dataset_path)
    heightmap_chunks.append(heightmap_i)
    target_chunks.append(target_i)
    if levels_i is not None:
        level_chunks.append(levels_i)
    print(f"loaded {dataset_path}: {heightmap_i.shape[0]} samples")
heightmap = torch.cat(heightmap_chunks, dim=0)
target = torch.cat(target_chunks, dim=0)
if level_chunks:
    levels = torch.cat(level_chunks, dim=0)
    level_counts = torch.bincount(levels)
    print("level sample counts:")
    for level, count in enumerate(level_counts.tolist()):
        if count > 0:
            print(f"  level {level}: {count}")
dataset = TensorDataset(heightmap, target)
val_len = max(1, int(len(dataset) * args.val_fraction))
train_len = len(dataset) - val_len
train_set, val_set = random_split(
    dataset,
    [train_len, val_len],
    generator=torch.Generator().manual_seed(0),
)
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
model = DepthBarEstimator(input_channels=heightmap.shape[1]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1.0e-4)

best_val = float("inf")
best_state = None
for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0
    train_count = 0
    for heightmap_b, target_b in train_loader:
        heightmap_b = heightmap_b.to(device, non_blocking=True)
        target_b = target_b.to(device, non_blocking=True)
        pred = model(heightmap_b)
        loss = F.mse_loss(pred, target_b)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        train_loss += float(loss.detach()) * heightmap_b.shape[0]
        train_count += heightmap_b.shape[0]

    model.eval()
    val_loss = 0.0
    val_l1 = torch.zeros(3, device=device)
    val_count = 0
    with torch.inference_mode():
        for heightmap_b, target_b in val_loader:
            heightmap_b = heightmap_b.to(device, non_blocking=True)
            target_b = target_b.to(device, non_blocking=True)
            pred = model(heightmap_b)
            val_loss += float(F.mse_loss(pred, target_b, reduction="sum"))
            val_l1 += torch.sum(torch.abs(pred - target_b), dim=0)
            val_count += heightmap_b.shape[0]

    train_mse = train_loss / max(train_count, 1)
    val_mse = val_loss / max(val_count * 3, 1)
    val_l1_m = val_l1 / max(val_count, 1)
    print(
        f"epoch={epoch:03d} train_mse={train_mse:.6f} val_mse={val_mse:.6f} "
        f"val_l1_m=({val_l1_m[0]:.4f}, {val_l1_m[1]:.4f}, {val_l1_m[2]:.4f})"
    )
    if val_mse < best_val:
        best_val = val_mse
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

if best_state is not None:
    model.load_state_dict(best_state)
model.eval()

output = Path(args.output).expanduser().resolve()
output.parent.mkdir(parents=True, exist_ok=True)
example = torch.zeros((1, heightmap.shape[1], heightmap.shape[-2], heightmap.shape[-1]), device=device)
scripted = torch.jit.trace(model, example)
scripted.save(str(output))
print(f"saved TorchScript estimator to {output}")

#!/usr/bin/env python
"""
Run ONE DEX benchmark episode with the ShowUI-pi policy, draw the predicted
drag trajectory on the intermediate screenshots, and save the annotated
frames as PNGs.

Everything is drawn in 1920x1080 pixel space, which is exactly the stored
image resolution, so predicted coordinates map 1:1 onto the screenshots.

Usage:
    HF_HOME=hf_cache PYTHONPATH=lerobot/src \
    .venv/bin/python scripts1025/visualize_episode.py \
        --episode 77 --out outputs/viz_episode

If --episode is omitted, the first episode of --domain is used.
"""

import argparse
import io
import json
import math
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from eval_dex_local import (
    DOMAINS,
    THRESHOLD_PX,
    load_policy,
)

SCREEN_W, SCREEN_H = 1920, 1080


def find_episode_parquet(ds_root: Path, ep_idx: int) -> Path | None:
    """Return the first parquet file whose episode_index column contains ep_idx."""
    for p in sorted((ds_root / "data").glob("chunk-*/file-*.parquet")):
        eps = pd.read_parquet(p, columns=["episode_index"])["episode_index"].unique()
        if ep_idx in eps:
            return p
    return None


def load_episode(ds_root: Path, ep_idx: int) -> pd.DataFrame:
    p = find_episode_parquet(ds_root, ep_idx)
    if p is None:
        raise FileNotFoundError(f"episode {ep_idx} not found under {ds_root}/data")
    df = pd.read_parquet(p)
    ep = df[df["episode_index"] == ep_idx].sort_values("frame_index").reset_index(drop=True)
    if len(ep) == 0:
        raise RuntimeError(f"episode {ep_idx} vanished after full read of {p}")
    return ep


def decode_image(img_cell) -> Image.Image:
    raw = img_cell.get("bytes") if isinstance(img_cell, dict) else img_cell
    return Image.open(io.BytesIO(raw)).convert("RGB")


def draw_frame(
    img: Image.Image,
    step: int,
    n_steps: int,
    traj: list[tuple[float, float]],   # all predicted points so far, incl. current
    gt: tuple[float, float],
    err: float,
    task_text: str,
) -> Image.Image:
    """Annotate one screenshot with the trajectory up to this step."""
    canvas = img.copy()
    d = ImageDraw.Draw(canvas)
    # Slightly larger default font for labels.
    try:
        font = ImageFont.load_default(size=28)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()

    gt_x, gt_y = gt

    # 20px threshold circle around the ground-truth endpoint.
    r = THRESHOLD_PX
    d.ellipse([gt_x - r, gt_y - r, gt_x + r, gt_y + r],
              outline=(0, 200, 0), width=3)
    # Ground-truth endpoint: green cross.
    d.line([gt_x - 14, gt_y, gt_x + 14, gt_y], fill=(0, 200, 0), width=4)
    d.line([gt_x, gt_y - 14, gt_x, gt_y + 14], fill=(0, 200, 0), width=4)

    # Predicted trajectory polyline so far.
    if len(traj) >= 2:
        d.line(traj, fill=(255, 60, 60), width=5)
    # Past points: small red dots.
    for (x, y) in traj[:-1]:
        d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(255, 60, 60))
    # Current prediction: larger yellow dot with outline.
    if traj:
        cx, cy = traj[-1]
        d.ellipse([cx - 10, cy - 10, cx + 10, cy + 10],
                  fill=(255, 220, 0), outline=(0, 0, 0), width=3)

    # Info panel top-left.
    label = f"step {step}/{n_steps - 1}   err={err:.1f}px"
    d.rectangle([8, 8, 8 + 620, 8 + 100], fill=(0, 0, 0))
    d.text((16, 14), label, fill=(255, 255, 255), font=font)
    d.text((16, 50), task_text[:70], fill=(200, 200, 200), font=font)

    # Legend top-right.
    d.rectangle([SCREEN_W - 430, 8, SCREEN_W - 8, 8 + 74], fill=(0, 0, 0))
    d.text((SCREEN_W - 420, 14), "yellow: current pred", fill=(255, 220, 0), font=font)
    d.text((SCREEN_W - 420, 46), "green: GT endpoint (r=20px)", fill=(0, 200, 0), font=font)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/showui-pi")
    ap.add_argument("--ds_root", required=True)
    ap.add_argument("--domain", default="premiere", choices=DOMAINS)
    ap.add_argument("--episode", type=int, default=None,
                    help="episode_index; defaults to the first episode of --domain")
    ap.add_argument("--out", default="outputs/viz_episode")
    ap.add_argument("--every", type=int, default=1, help="save every Nth frame")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ds_root = Path(args.ds_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve episode.
    with (ds_root / "meta" / "domain_map.json").open() as f:
        domain_map = {int(k): str(v) for k, v in json.load(f).items()}
    ep_idx = args.episode
    if ep_idx is None:
        ep_idx = min(e for e, d in domain_map.items() if d == args.domain)
    elif domain_map.get(ep_idx) != args.domain:
        print(f"note: episode {ep_idx} is domain '{domain_map.get(ep_idx)}', "
              f"not '{args.domain}'")

    tasks_df = pd.read_parquet(ds_root / "meta" / "tasks.parquet")
    tasks_by_id = {
        int(tasks_df.iloc[i]["task_index"]): str(tasks_df.index[i])
        for i in range(len(tasks_df))
    }

    ep = load_episode(ds_root, ep_idx)
    task_text = tasks_by_id.get(int(ep.iloc[0]["task_index"]), "")
    last_action = np.asarray(ep.iloc[-1]["action"], dtype=np.float64)
    gt = (float(last_action[1]), float(last_action[2]))
    print(f"episode {ep_idx} ({domain_map.get(ep_idx)}): {len(ep)} steps, "
          f"task='{task_text}', GT endpoint=({gt[0]:.1f}, {gt[1]:.1f})", flush=True)

    # Load policy (uses only meta/, not the data files).
    policy, preprocessor, postprocessor = load_policy(args.ckpt, ds_root, args.device)

    # Closed-loop roll-out, same protocol as eval_dex_local.run_episode.
    prev_state = torch.tensor([[0.0, -1.0, -1.0]], dtype=torch.float32)
    traj: list[tuple[float, float]] = []
    errs: list[float] = []
    frames: list[tuple[Image.Image, float]] = []  # (annotated canvas, err)

    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.device == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        for i in range(len(ep)):
            row = ep.iloc[i]
            img = decode_image(row["observation.images.screen"])
            arr = np.asarray(img, dtype=np.uint8).copy()
            batch_img = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            mb = preprocessor({
                "observation.images.screen": batch_img,
                "observation.state": prev_state,
                "task": task_text,
            })
            act = postprocessor(policy.select_action(mb))
            a = act.numpy()[0]

            x_pred = float(np.clip(a[1], 0, SCREEN_W - 1))
            y_pred = float(np.clip(a[2], 0, SCREEN_H - 1))
            traj.append((x_pred, y_pred))
            err = math.hypot(x_pred - gt[0], y_pred - gt[1])
            errs.append(err)

            btn_pred = 1.0 if float(a[0]) > 0.5 else 0.0
            prev_state = torch.tensor([[btn_pred, x_pred, y_pred]], dtype=torch.float32)
            policy._queues["action"].clear()

            frames.append((draw_frame(img, i, len(ep), traj, gt, err, task_text), err))
            print(f"  step {i:2d}: pred=({x_pred:7.1f},{y_pred:7.1f})  err={err:7.1f}px",
                  flush=True)

    # Save annotated frames.
    for i, (canvas, err) in enumerate(frames):
        if i % args.every == 0 or i == len(frames) - 1:
            p = out_dir / f"ep{ep_idx}_step{i:02d}_err{err:.0f}px.png"
            canvas.save(p)
    print(f"saved {len(frames)} annotated frames to {out_dir} "
          f"(every={args.every})", flush=True)

    summary = {
        "episode_index": int(ep_idx),
        "domain": domain_map.get(ep_idx),
        "task": task_text,
        "gt_endpoint": [gt[0], gt[1]],
        "trajectory": [[x, y] for (x, y) in traj],
        "errors_px": [round(e, 2) for e in errs],
        "endpoint_error_px_final": round(errs[-1], 2),
        "endpoint_error_px_best": round(min(errs), 2),
        "endpoint_success": errs[-1] <= THRESHOLD_PX,
        "endpoint_success_best": min(errs) <= THRESHOLD_PX,
    }
    with (out_dir / f"ep{ep_idx}_trajectory.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"final_err={errs[-1]:.1f}px  best_err={min(errs):.1f}px  "
          f"success_best={min(errs) <= THRESHOLD_PX}", flush=True)


if __name__ == "__main__":
    main()

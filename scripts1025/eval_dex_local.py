#!/usr/bin/env python
"""
Local (macOS / MPS) variant of eval_dex.py.

Differences from scripts1025/eval_dex.py:
- No download step: reads whatever parquet files already exist under
  --ds_root (HF_HOME/hf_cache snapshot), so we can evaluate a subset of
  the benchmark without pulling all 4.4 GB.
- --device defaults to "mps"; bfloat16 autocast is only enabled for cuda.
- --limit N caps the number of episodes evaluated per domain.
"""

import argparse
import io
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy, make_policy_config, make_pre_post_processors


THRESHOLD_PX = 20.0
DOMAINS = ["captcha", "premiere", "desktop", "powerpoint", "handwriting"]


def load_benchmark_data(ds_root: Path) -> tuple[pd.DataFrame, dict[int, str], dict[int, str]]:
    """Load locally present parquet data, the domain map, and the task lookup."""
    parquet_dir = ds_root / "data"
    paths = sorted(parquet_dir.glob("chunk-*/file-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found under {parquet_dir}")
    print(f"Reading {len(paths)} local parquet files:", flush=True)
    for p in paths:
        print(f"  {p.name}", flush=True)
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)

    with (ds_root / "meta" / "domain_map.json").open() as f:
        dm_raw = json.load(f)
    domain_map = {int(k): str(v) for k, v in dm_raw.items()}

    tasks_df = pd.read_parquet(ds_root / "meta" / "tasks.parquet")
    tasks_by_id: dict[int, str] = {
        int(tasks_df.iloc[i]["task_index"]): str(tasks_df.index[i])
        for i in range(len(tasks_df))
    }
    return df, domain_map, tasks_by_id


def load_policy(ckpt_dir: str, ds_root: Path, device: str):
    ckpt_path = Path(ckpt_dir)
    with (ckpt_path / "config.json").open() as f:
        raw_cfg = json.load(f)
    ptype = raw_cfg.pop("type", None)
    if not ptype:
        train_cfg_path = ckpt_path / "train_config.json"
        if train_cfg_path.is_file():
            with train_cfg_path.open() as f:
                tc = json.load(f)
            ptype = (tc.get("policy") or {}).get("type")
    if not ptype:
        raise ValueError(f"Could not determine policy type from {ckpt_path}")

    policy_cfg = make_policy_config(ptype, **raw_cfg)
    policy_cfg.pretrained_path = str(ckpt_path)
    policy_cfg.device = device

    ds_meta = LeRobotDatasetMetadata("ScreenDrag", root=ds_root, revision="v3.0")

    policy = make_policy(cfg=policy_cfg, ds_meta=ds_meta)
    if device != "cuda":
        # The VLM backbone is loaded in bf16 (torch_dtype in smolvlm_with_expert.py)
        # while the checkpoint's new modules are fp32. Mixed-dtype matmuls crash on
        # MPS and are rejected on CPU, so cast everything to fp32.
        policy = policy.float()
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(ckpt_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
        },
    )
    return policy, preprocessor, postprocessor


def _decode_image(img_cell) -> torch.Tensor:
    if isinstance(img_cell, dict):
        raw = img_cell.get("bytes")
    else:
        raw = img_cell
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def run_episode(policy, preprocessor, postprocessor, episode_df, tasks_by_id, device):
    ep = episode_df.sort_values("frame_index").reset_index(drop=True)
    last_action = np.asarray(ep.iloc[-1]["action"], dtype=np.float64)
    tgt_x = float(last_action[1])
    tgt_y = float(last_action[2])

    task_index = int(ep.iloc[0]["task_index"])
    task_text = tasks_by_id.get(task_index, "")

    prev_state = torch.tensor([[0.0, -1.0, -1.0]], dtype=torch.float32)
    best_err = float("inf")
    final_err = float("inf")

    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), amp_ctx:
        for i in range(len(ep)):
            row = ep.iloc[i]
            img = _decode_image(row["observation.images.screen"])

            batch = {
                "observation.images.screen": img,
                "observation.state": prev_state,
                "task": task_text,
            }
            mb = preprocessor(batch)
            act = policy.select_action(mb)
            act = postprocessor(act)
            a = act.numpy()[0]

            x_pred = float(np.clip(a[1], 0, 1919))
            y_pred = float(np.clip(a[2], 0, 1079))

            err = math.sqrt((x_pred - tgt_x) ** 2 + (y_pred - tgt_y) ** 2)
            if err < best_err:
                best_err = err
            final_err = err

            btn_pred = 1.0 if float(a[0]) > 0.5 else 0.0
            prev_state = torch.tensor(
                [[btn_pred, x_pred, y_pred]], dtype=torch.float32
            )

            policy._queues["action"].clear()

    return {
        "endpoint_error_px_final": round(final_err, 2),
        "endpoint_error_px_best": round(best_err, 2),
        "endpoint_success": final_err <= THRESHOLD_PX,
        "endpoint_success_best": best_err <= THRESHOLD_PX,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/showui-pi")
    ap.add_argument("--ds_root", required=True,
                    help="Local path to the benchmark's test/parquet directory")
    ap.add_argument("--output_dir", default="outputs/eval_dex_local")
    ap.add_argument("--domain", default=None, choices=DOMAINS)
    ap.add_argument("--limit", type=int, default=None,
                    help="Max episodes per domain (None = all locally present)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df, domain_map, tasks_by_id = load_benchmark_data(Path(args.ds_root))
    print(f"Loaded {len(df)} frames across {df['episode_index'].nunique()} episodes", flush=True)

    print("Loading policy ...", flush=True)
    policy, preprocessor, postprocessor = load_policy(args.ckpt, Path(args.ds_root), args.device)

    domains = [args.domain] if args.domain else DOMAINS
    all_results: dict[str, dict] = {}

    for domain in domains:
        domain_eps = sorted(ep for ep, d in domain_map.items() if d == domain)
        # Keep only episodes that actually have frames in the local parquet files.
        present = set(df["episode_index"].unique().tolist())
        domain_eps = [e for e in domain_eps if e in present]
        if args.limit:
            domain_eps = domain_eps[: args.limit]
        if not domain_eps:
            print(f"\n=== {domain}: no local episodes, skipping ===", flush=True)
            continue
        print(f"\n=== {domain}: {len(domain_eps)} episodes ===", flush=True)

        success = 0
        best_success = 0
        total = 0
        per_episode = []
        t0 = time.time()

        for ep_idx in domain_eps:
            episode_df = df[df["episode_index"] == ep_idx]
            if len(episode_df) == 0:
                continue
            r = run_episode(policy, preprocessor, postprocessor, episode_df, tasks_by_id, args.device)
            success += int(r["endpoint_success"])
            best_success += int(r["endpoint_success_best"])
            total += 1
            per_episode.append({"episode_index": int(ep_idx), **r})
            print(f"  ep {ep_idx}: err_final={r['endpoint_error_px_final']}px "
                  f"err_best={r['endpoint_error_px_best']}px "
                  f"success_best={r['endpoint_success_best']}", flush=True)

            if total % 10 == 0 or total == len(domain_eps):
                elapsed = time.time() - t0
                sr = 100 * best_success / total
                print(f"  [{domain}] {total}/{len(domain_eps)}  sr_best={sr:.2f}%  elapsed={elapsed:.0f}s",
                      flush=True)

        summary = {
            "domain": domain,
            "evaluated": total,
            "success": success,
            "success_best": best_success,
            "endpoint_accuracy": f"{100 * success / total:.2f}%" if total else "N/A",
            "endpoint_accuracy_best": f"{100 * best_success / total:.2f}%" if total else "N/A",
        }
        all_results[domain] = summary

        out = Path(args.output_dir) / domain
        out.mkdir(parents=True, exist_ok=True)
        with (out / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        with (out / "per_episode.json").open("w") as f:
            json.dump(per_episode, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for domain, r in all_results.items():
        print(f"  {domain}: best={r['endpoint_accuracy_best']}  final={r['endpoint_accuracy']}  "
              f"({r['success_best']}/{r['evaluated']})", flush=True)


if __name__ == "__main__":
    main()

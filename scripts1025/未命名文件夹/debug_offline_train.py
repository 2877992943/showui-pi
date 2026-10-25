#!/usr/bin/env python
"""训练集离线排查 — 一步一步验证模型在分布内数据上的行为。

模式:
  A) open-loop: 第 0 帧 (state=[0,-1,-1]) 一次预测整段 21 步, 与 GT 逐点比。
  B) closed-loop: eval_dex 协议 — 每步重新推理, state 用上一步预测反馈,
     官方评测即此模式 (THRESHOLD_PX=20)。

用法:
  PYTHONPATH=lerobot/src python scripts1025/debug_offline_train.py \
      --episodes 5 --domain handwriting --out outputs/debug_train
"""

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "lerobot" / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts1025"))

from demo_live_drag import load_policy  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402


def download_train_meta(dataset_id: str) -> Path:
    """下载 train split 的 meta/（不含 data parquet），返回 train/parquet 根目录。"""
    api = HfApi()
    wanted = [p for p in api.list_repo_files(dataset_id, repo_type="dataset")
              if p.startswith("train/parquet/meta/")]
    root = None
    for rel in wanted:
        local = Path(hf_hub_download(dataset_id, rel, repo_type="dataset"))
        parts = local.parts
        root = Path(*parts[: parts.index("train")]) / "train" / "parquet"
    if not (root / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Expected meta at {root}/meta/info.json")
    return root


def load_episode(ds_root: Path, ep_idx: int):
    eps = pd.read_parquet(ds_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    row = eps[eps["episode_index"] == ep_idx].iloc[0]
    f = f"train/parquet/data/chunk-{int(row['data/chunk_index']):03d}/file-{int(row['data/file_index']):03d}.parquet"
    df = pd.read_parquet(hf_hub_download("h-siyuan/ScreenDrag", f, repo_type="dataset"))
    ep = df[df["episode_index"] == ep_idx].sort_values("frame_index").reset_index(drop=True)
    return ep


def decode_frame(cell) -> torch.Tensor:
    img = Image.open(io.BytesIO(cell["bytes"])).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0


def open_loop(policy, pre, post, img_t, task) -> np.ndarray:
    policy._queues["action"].clear()
    batch = {"observation.images.screen": img_t,
             "observation.state": torch.tensor([[0.0, -1.0, -1.0]]), "task": task}
    mb = pre(batch)
    acts = []
    with torch.no_grad():
        for _ in range(21):
            acts.append(post(policy.select_action(mb)).cpu().numpy()[0])
    return np.asarray(acts)[:21]


def closed_loop(policy, pre, post, img_t, task) -> np.ndarray:
    """eval_dex 协议: 每步重规划 + 预测 state 反馈 (图像始终用第 0 帧, 与 eval_dex 一致)。"""
    prev_state = torch.tensor([[0.0, -1.0, -1.0]])
    acts = []
    with torch.no_grad():
        for _ in range(21):
            policy._queues["action"].clear()
            batch = {"observation.images.screen": img_t,
                     "observation.state": prev_state, "task": task}
            a = post(policy.select_action(pre(batch))).cpu().numpy()[0]
            acts.append(a)
            btn = 1.0 if float(a[0]) > 0.5 else 0.0
            prev_state = torch.tensor([[btn, float(np.clip(a[1], 0, 1919)),
                                        float(np.clip(a[2], 0, 1079))]])
    return np.asarray(acts)


def teacher_forced(policy, pre, img_t, task, gt, ts=(0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 0.999),
                   n_seeds: int = 3) -> dict[float, float]:
    """Mode C: 训练目标直检. 固定 (x_t, t), 看 a_hat 是否还原 GT 动作 (训练分布内应几 px 内)."""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    actions = gt.copy()
    actions[:, 1] /= 1920.0
    actions[:, 2] /= 1080.0
    act_t = torch.tensor(actions[None], dtype=torch.float32, device="mps")

    batch = {"observation.images.screen": img_t,
             "observation.state": torch.tensor([[0.0, -1.0, -1.0]]), "task": task}
    mb = pre(batch)
    lang_tokens = mb[OBS_LANGUAGE_TOKENS]
    lang_masks = mb[OBS_LANGUAGE_ATTENTION_MASK]

    images, img_masks = policy.prepare_images(
        {"observation.images.screen": img_t.to("mps")})
    state = policy.prepare_state({"observation.state": torch.tensor([[0.0, -1.0, -1.0]]).to("mps")})

    out = {}
    for t in ts:
        errs = []
        for seed in range(n_seeds):
            g = torch.Generator(device="cpu").manual_seed(seed)
            noise = (torch.randn(act_t.shape, generator=g) * policy.config.noise_std).to("mps")
            x_t = (1 - t) * act_t + t * noise
            time = torch.full((1,), t, device="mps")
            with torch.no_grad():
                _, _, _, _, a_hat, _, _ = policy.model.forward(
                    images, img_masks, lang_tokens, lang_masks, state,
                    act_t, noise, time, is_grounding=None)
            a_hat = a_hat[..., :3]
            errs.append(((a_hat - act_t).abs()
                         * torch.tensor([1.0, 1920.0, 1080.0], device="mps"))
                        .sum(dim=-1).mean().item())
        out[t] = float(np.mean(errs))
    return out


def custom_sample(policy, img_t, task, t_start=0.1, n_int_steps=20, restarts=1,
                  use_ahat=True) -> np.ndarray:
    """自定义采样: 正常积分到 t=t_start, (可选 restart 再去噪), 返回低 t 处的 a_hat。
    依据: teacher-forced 显示 a_hat 误差随 t 线性放大 (t=0.1→43px, t=0.5→161px),
    而 Euler 积分终点被中途误差带偏 ~200px。"""
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    dev = "mps"
    mb = pre({"observation.images.screen": img_t,
              "observation.state": torch.tensor([[0.0, -1.0, -1.0]]), "task": task})
    images, img_masks = policy.prepare_images(
        {"observation.images.screen": img_t.to(dev)})
    state = policy.prepare_state(
        {"observation.state": torch.tensor([[0.0, -1.0, -1.0]]).to(dev)})
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.model.embed_prefix(
        images, img_masks, mb[OBS_LANGUAGE_TOKENS], mb[OBS_LANGUAGE_ATTENTION_MASK], state=state)
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    with torch.no_grad():
        _, past_kv = policy.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks, position_ids=prefix_position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, None],
            use_cache=True, fill_kv_cache=True)

    shape = (1, policy.config.chunk_size, policy.config.max_action_dim)
    std = policy.config.noise_std
    x = torch.randn(shape, device=dev) * std
    T = policy.config.chunk_size

    def integrate(x, t_hi, t_lo, n):
        dt = (t_hi - t_lo) / n
        for i in range(n):
            t_cur = t_hi - i * dt
            time = torch.tensor([t_cur], device=dev)
            with torch.no_grad():
                v = policy.model.denoise_step(prefix_pad_masks, past_kv, x, time)
            x = x - dt * v
        return x, t_lo

    a_final = None
    t_cur = 1.0
    for r in range(restarts):
        x, t_cur = integrate(x, t_cur, t_start, n_int_steps)
        time = torch.tensor([t_cur], device=dev)
        with torch.no_grad():
            v = policy.model.denoise_step(prefix_pad_masks, past_kv, x, time)
        a_final = x - t_cur * v  # 低 t 处的 x0 估计 (最准)
        if r + 1 < restarts:  # restart: 重新加噪到 t_start*2 再积分回来
            t_re = min(t_start * 3, 0.3)
            x = (1 - t_re) * a_final + t_re * torch.randn(shape, device=dev) * std
            t_cur = t_re

    if not use_ahat:
        x, _ = integrate(x, t_start, 0.0, max(1, int(n_int_steps * t_start)))
        a_final = x
    return a_final[0, :, :3].cpu().numpy() * np.array([1.0, 1920.0, 1080.0])


def viz(img_t, gt, pred_a, pred_b, out: Path) -> None:
    canvas = Image.fromarray((img_t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    d = ImageDraw.Draw(canvas)
    d.line([tuple(p) for p in gt[:, 1:]], fill=(200, 200, 200), width=3)          # GT 灰
    if pred_a is not None:
        d.line([tuple(p) for p in pred_a[:, 1:]], fill=(255, 140, 0), width=3)    # 开环 橙
    if pred_b is not None:
        d.line([tuple(p) for p in pred_b[:, 1:]], fill=(0, 200, 80), width=3)     # 闭环 绿
    p0 = (gt[0, 1], gt[0, 2])
    d.ellipse([p0[0] - 6, p0[1] - 6, p0[0] + 6, p0[1] + 6], outline=(0, 0, 255), width=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/showui-pi")
    ap.add_argument("--episodes", type=int, nargs="+", default=[265])
    ap.add_argument("--modes", nargs="+", default=["A", "B"], choices=["A", "B", "C", "D", "E"])
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--out", default="outputs/debug_train")
    ap.add_argument("--sampler", default="noise",
                    choices=["noise", "zero", "ahat_median", "zero_ahat_median"])
    args = ap.parse_args()

    ds_root = download_train_meta("h-siyuan/ScreenDrag")
    eps_meta = pd.read_parquet(ds_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    tasks = pd.read_parquet(ds_root / "meta" / "tasks.parquet")
    tid = {int(tasks.iloc[i]["task_index"]): str(tasks.index[i]) for i in range(len(tasks))}
    dm = {int(k): str(v) for k, v in json.load(open(ds_root / "meta" / "domain_map.json")).items()}

    policy, pre, post = load_policy(args.ckpt, ds_root, "mps")
    policy.config.inference_zero_noise = "zero" in args.sampler
    policy.config.inference_ahat_median = "ahat_median" in args.sampler
    policy.config.num_steps = args.num_steps
    if args.ensemble > 1:
        policy.config.inference_ensemble = args.ensemble

    for ep_idx in args.episodes:
        ep = load_episode(ds_root, ep_idx)
        gt = np.stack(ep["action"].values)
        img_t = decode_frame(ep.iloc[0]["observation.images.screen"])
        task = tid[int(eps_meta[eps_meta["episode_index"] == ep_idx].iloc[0]["task_index"])]
        domain = dm.get(ep_idx, "?")

        pred_a = open_loop(policy, pre, post, img_t, task) if "A" in args.modes else None
        pred_b = closed_loop(policy, pre, post, img_t, task) if "B" in args.modes else None
        pred_d = (custom_sample(policy, img_t, task, restarts=1)   # 积分到 t=0.1 + 末端 a_hat
                  if "D" in args.modes else None)
        pred_e = (custom_sample(policy, img_t, task, restarts=3)   # 加 restart 精化
                  if "E" in args.modes else None)

        if "C" in args.modes:
            errs = teacher_forced(policy, pre, img_t, task, gt)
            print("C teacher-forced a_hat err (px, mean over chunk, 3 seeds):")
            for t, e in errs.items():
                print(f"  t={t:<6} -> {e:.1f}px")

        print(f"\n=== ep{ep_idx} [{domain}] {task!r}")
        print(f"GT xy: start {gt[0,1:].astype(int)} end {gt[-1,1:].astype(int)} "
              f"span {np.linalg.norm(gt[-1,1:]-gt[0,1:]):.0f}px")
        for name, pred in [("A open-loop", pred_a), ("B closed-loop", pred_b),
                           ("D ahat@t0.1", pred_d), ("E restart+ahat", pred_e)]:
            if pred is None:
                continue
            se = np.linalg.norm(pred[0, 1:] - gt[0, 1:])
            ee = np.linalg.norm(pred[-1, 1:] - gt[-1, 1:])
            nn = np.linalg.norm(pred[:, None, 1:] - gt[None, :, 1:], axis=-1).min(axis=1).mean()
            print(f"{name}: start_err={se:.0f}px end_err={ee:.0f}px "
                  f"mean_nn={nn:.0f}px smooth={np.abs(np.diff(pred[:,1:],axis=0)).sum():.0f}")

        viz(img_t, gt, pred_a, pred_b, Path(args.out) / f"ep{ep_idx}.png")
        print(f"viz -> {args.out}/ep{ep_idx}.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
